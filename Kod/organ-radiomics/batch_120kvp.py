"""Tüm 120 kVp serilerini işleyen toplu, KESİNTİYE DAYANIKLI runner.

Tasarım ilkeleri (kullanıcı gereksinimleri):
  * KLASÖR-TABANLI kimlik: kaynaktaki her seri klasörü = ayrı bir seri. Hiçbir
    şey birleştirilmez → test-retest'ler ve kopyalar ayrı ayrı korunur.
  * GERÇEK KOPYA (aynı SeriesInstanceUID iki klasörde): bir kez işlenir, diğerleri
    rapor edilir. Kaynak dosyalara DOKUNULMAZ (silme yok — güvenlik).
  * TEST-RETEST (aynı SeriesDescription, farklı çekim oturumu = farklı Study
    UID/Date/Time): ayrı çekimlerdir → 'take 1' / 'take 2' eki verilir, tümü tutulur.
  * %100 IBSI: MIRP ayarları değiştirilmez (crop yok). Öznitelikler tek-çağrı
    referansıyla bit-birebir.
  * RESUMABLE: her seri için ct.nii.gz / seg.nii.gz / features.csv varsa atlanır.
    Çökme/yeniden başlatmada kaldığı yerden devam eder.
  * KOPYALAMAZ: NIfTI doğrudan kaynak DICOM'lardan okunur (disk tasarrufu).

Kullanım:
    C:\\Python314\\python.exe batch_120kvp.py "C:\\...\\29062026-EID"
    C:\\Python314\\python.exe batch_120kvp.py "<kaynak>" --limit 1        # smoke test
    C:\\Python314\\python.exe batch_120kvp.py "<kaynak>" --only SCAN_1_1_0_BR40_0010
    C:\\Python314\\python.exe batch_120kvp.py "<kaynak>" --plan-only      # sadece analiz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom

import config
import sort_spatially as ss
import overlay_qc as qc
import run_totalsegmentator as ts
import feature_extraction_mirp as fx

_DICOM_EXTS = {".ima", ".dcm"}

# TotalSegmentator TEK GPU'yu kullanıyor ve `total` görevi kartın belleğini
# neredeyse dolduruyor. Seriler paralel işlendiğinde iki TS'in aynı anda
# başlaması VRAM'i taşırır ve ikisi de çöker. Bu kilit segmentasyon adımını
# seri tutar; NIfTI dönüşümü, QC ve MIRP paralel akmaya devam eder.
_TS_LOCK = threading.Lock()

# Paralel seriler aynı anda yazdırdığında satırlar iç içe geçiyor. Bu kilit
# yalnız ÇOK SATIRLI blokları bütün tutar (tek satırlık print'ler zaten atomik).
_PRINT_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Keşif — her klasör bir seri
# --------------------------------------------------------------------------- #
def _read_header(path: Path) -> pydicom.Dataset | None:
    """DICOM üstbilgisini okur. Preamble'lı dosyalar için force=False; başarısızsa
    preamble'sız (PACS/CD, ham) DICOM olabilir → force=True dene ama gerçek DICOM
    olduğunu (SeriesInstanceUID + Modality) doğrula, rastgele dosyayı kabul etme."""
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception:  # noqa: BLE001
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            if getattr(ds, "SeriesInstanceUID", None) and getattr(ds, "Modality", None):
                return ds
        except Exception:  # noqa: BLE001
            pass
        return None


def _is_dicom_candidate(name: str) -> bool:
    """DICOM olabilecek dosya: .ima/.dcm uzantılı VEYA uzantısız (PACS/CD dışa aktarımı)."""
    suf = Path(name).suffix.lower()
    return suf in _DICOM_EXTS or suf == ""


def discover_series(source_root: Path) -> list[dict[str, Any]]:
    """Kaynak ağacındaki 120 kVp CT serilerini bulur.

    Tek os.walk geçişi. Uzantısız DICOM'lar da aday (PACS/CD dışa aktarımları).

    Bir SERİ = bir SeriesInstanceUID; gruplama KLASÖRDEN BAĞIMSIZDIR. PACS/CD
    dışa aktarımı bir seriyi numaralı klasörlere BÖLEBİLİYOR — bu veri setinde
    ölçüldü: 376 slice'lık bir seri 109 + 267 olarak iki klasöre dağılmış, SOP
    kesişimi 0, z aralıkları bitişik, hiçbir rekonstrüksiyon tag'i farklı değil.
    Klasör bazlı gruplamada bunlar iki ayrı kayıt oluyor, ardından 'aynı UID'
    diye kopya sayılıp biri ATILIYORDU → hacmin bir kısmı sessizce kayboluyordu
    (bir seride 376 slice'ın 181'i).

    Slice kimliği SOPInstanceUID'dir: aynı SOP ikinci kez görülürse (gerçek
    kopya dosya) bir kez alınır ve raporlanır. Böylece hem bölünmüş seri
    birleşir hem gerçek kopya slice iki kez sayılmaz.
    """
    by_uid: dict[str, dict[str, Path]] = defaultdict(dict)      # uid -> {sop: dosya}
    folder_counts: dict[str, Counter] = defaultdict(Counter)    # uid -> klasör sayacı
    dup_files: dict[str, list[Path]] = defaultdict(list)        # uid -> atılan kopyalar

    for dirpath, _dirnames, filenames in os.walk(source_root):
        folder = Path(dirpath)
        for name in filenames:
            if not _is_dicom_candidate(name):
                continue
            f = folder / name
            ds = _read_header(f)
            if ds is None:
                continue
            uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            if not uid:
                continue
            # SOP yoksa yol kimlik olarak kullanılır (kopya sayılmasın).
            sop = str(getattr(ds, "SOPInstanceUID", "") or "") or str(f)
            slices = by_uid[uid]
            if sop in slices:
                dup_files[uid].append(f)
                continue
            slices[sop] = f
            folder_counts[uid][folder] += 1

    records: list[dict[str, Any]] = []
    for uid, slices in by_uid.items():
        files = sorted(slices.values())
        # Referans başlık DETERMİNİSTİK seçilir (os.walk sırası garantisiz):
        # sıralı listenin ilki. Seri geneli tag'ler tüm slice'larda aynı.
        ds = _read_header(files[0])
        if ds is None:
            continue
        # Cihaz-bağımsız filtre (YALNIZ DICOM metadata): CT + LOCALIZER değil +
        # etkin kVp hedefe uyuyor + rekon izinliyse.
        ok, kvp, rt = ss.is_target_series(ds)
        if not ok:
            continue
        counted = folder_counts[uid].most_common()
        records.append(
            {
                # Birincil klasör = en çok slice'ı olan (anahtar/ad üretimi için).
                "folder": counted[0][0],
                "folders": [f for f, _ in counted],
                "folder_counts": {str(f): n for f, n in counted},
                "n_dup_files": len(dup_files[uid]),
                "files": files,   # bu serinin (UID) TÜM slice'ları, klasör farkı önemsiz
                "series_uid": uid,
                "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
                "series_desc": str(getattr(ds, "SeriesDescription", "") or ""),
                "series_number": str(getattr(ds, "SeriesNumber", "") or ""),
                # Tekrar çekim ekseni. Bir dışa aktarımda test ve retest AYNI
                # StudyInstanceUID altında olabiliyor (ölçüldü: 20260312 seti,
                # 202 seri, tek çalışma) — o durumda oturum karşılaştırması
                # tekrarı göremez ve ayrım yalnız burada kalır.
                "acq_number": str(getattr(ds, "AcquisitionNumber", "") or ""),
                "series_time": str(getattr(ds, "SeriesTime", "") or ""),
                "patient_id": ss._sanitize(getattr(ds, "PatientID", None), "unknown_patient"),
                "study_date": str(getattr(ds, "StudyDate", "") or ""),
                "study_time": str(getattr(ds, "StudyTime", "") or ""),
                "acq_time": str(getattr(ds, "AcquisitionTime", "") or ""),
                "kvp": kvp,
                "recon_type": rt,   # native / VMI_45keV / VMI_70keV
                "scanner": ss.scanner_model(ds),
                "photon_counting": ss.is_photon_counting(ds),
            }
        )
    # Deterministik: klasör + UID (os.walk sırası ve dict sırası garantisiz)
    records.sort(key=lambda r: (str(r["folder"]), r["series_uid"]))
    return records


# --------------------------------------------------------------------------- #
# Dedup (gerçek kopya) + test-retest (take1/take2)
# --------------------------------------------------------------------------- #
def dedup_and_takes(
    records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """UID kopyalarını tekiller (rapor) ve protokol+StudyDate/Time ile take atar."""
    # 1) Gerçek kopya: aynı SeriesInstanceUID → en çok slice'lısını tut, diğerlerini raporla.
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uid[r["series_uid"]].append(r)

    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, str]] = []
    for uid, group in by_uid.items():
        group.sort(key=lambda r: -len(r["files"]))
        unique.append(group[0])
        for dup in group[1:]:
            duplicates.append(
                {
                    "kept": group[0]["folder"].name,
                    "duplicate": dup["folder"].name,
                    "series_uid": uid,
                }
            )

    # 2) Test-retest: AYNI HASTA + aynı SeriesDescription, farklı çekim oturumu
    #    (Study UID/Date/Time). Grup anahtarı (patient_id, series_desc) olmalı; aksi
    #    halde farklı hastalar aynı protokol adı yüzünden karışır ve yanlış take alır.
    #    Boş açıklamalı seriler protokol eşleştirmesine girmez (sahte take almasın).
    by_proto: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in unique:
        if not r["series_desc"]:
            r["take"] = None
            continue
        by_proto[(r["patient_id"], r["series_desc"])].append(r)
    for _proto, group in by_proto.items():
        sessions = sorted(
            {(r["study_uid"], r["study_date"], r["study_time"]) for r in group},
            key=lambda s: (s[1], s[2], s[0]),
        )
        if len(sessions) > 1:
            take_of = {s: i + 1 for i, s in enumerate(sessions)}
            for r in group:
                r["take"] = take_of[(r["study_uid"], r["study_date"], r["study_time"])]
        else:
            # TEK OTURUM — ama tekrar çekim yine de olabilir. Bu veri setinde
            # ölçüldü (20260312, 202 seri): test ve retest AYNI
            # StudyInstanceUID/Date/Time altında dışa aktarılmış ve tek ayrım
            # AcquisitionNumber (çiftler N ve N+10: 2/12, 6/16, ... 11/21).
            # Oturum karşılaştırması burada hiçbir tekrarı göremiyordu.
            #
            # Sıralama numeriktir ve +10 gibi bir ofset VARSAYMAZ: aynı
            # protokolün çekimleri artan AcquisitionNumber sırasıyla
            # 1, 2, ... diye numaralanır (oturum dalıyla aynı mantık).
            acqs = sorted({r["acq_number"] for r in group if r.get("acq_number")},
                          key=_acq_sort_key)
            if len(acqs) > 1:
                take_of_acq = {a: i + 1 for i, a in enumerate(acqs)}
                for r in group:
                    r["take"] = take_of_acq.get(r.get("acq_number"))
            else:
                for r in group:
                    r["take"] = None

    if getattr(config, "DROP_DUPLICATE_RECONSTRUCTIONS", False):
        unique, duplicates = _drop_duplicate_reconstructions(unique, duplicates)
    return unique, duplicates


def _acq_sort_key(value: Any) -> tuple[int, Any]:
    """AcquisitionNumber'ı sayısal sırala; sayı değilse metin olarak (sonda)."""
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _drop_duplicate_reconstructions(
    unique: list[dict[str, Any]], duplicates: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Aynı çekimin ikinci kez rekonstrükte edilmiş kopyalarını eler.

    KİMLİK: (hasta, SeriesDescription, AcquisitionNumber). Aynı protokolün aynı
    çekimi demektir — farklı SeriesInstanceUID ve SeriesTime taşısalar bile
    içerik aynıdır. Ölçüldü (20260312 setinde 62 çift): dilim/instans düzeyi
    alanlar dışında tam DICOM başlıkları BİREBİR aynı, 0 farklı etiket.

    TUTULAN: en küçük SeriesNumber (özgün dışa aktarım), eşitlikte UID sırası —
    deterministiktir, koşumdan koşuma değişmez.

    Atılanlar `duplicates` listesine `reason="recon"` ile eklenir ve
    DUPLICATES_report.txt'te AYRI bir başlıkta raporlanır: bu bir UID kopyası
    değil, ayrı bir seridir ve neden atlandığı görünür olmalı. Kaynak
    dosyalara DOKUNULMAZ (taşınmaz, silinmez) — yalnız işlenmez.
    """
    by_acq: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for r in unique:
        # Açıklaması ya da çekim numarası olmayan seri eşleştirmeye girmez:
        # kimlik kurulamıyorsa eleme yapmak veri atmak olur.
        if not r.get("series_desc") or not r.get("acq_number"):
            passthrough.append(r)
            continue
        by_acq[(r["patient_id"], r["series_desc"], r["acq_number"])].append(r)

    kept: list[dict[str, Any]] = list(passthrough)
    for _key, group in by_acq.items():
        group.sort(key=lambda r: (_acq_sort_key(r.get("series_number")),
                                  r["series_uid"]))
        kept.append(group[0])
        for dup in group[1:]:
            duplicates.append({
                "kept": group[0].get("series_number") or group[0]["folder"].name,
                "duplicate": dup.get("series_number") or dup["folder"].name,
                "series_uid": dup["series_uid"],
                "series_desc": dup.get("series_desc", ""),
                "acq_number": dup.get("acq_number", ""),
                "series_time": dup.get("series_time", ""),
                "reason": "recon",
            })
    kept.sort(key=lambda r: (str(r["folder"]), r["series_uid"]))
    return kept, duplicates


def assign_series_names(records: list[dict[str, Any]]) -> None:
    """Her kayda okunur `series_name` yazar (çıktı klasörü adı olacak).

    Ad, tablodaki `series` sütunuyla AYNI şemadan üretilir; klasör adı ile
    tablo birbirini tutsun. Çakışma olursa seri numarası eklenir — iki farklı
    seri asla aynı klasöre yazamaz (sessiz üzerine yazma olmaz).
    """
    import series_naming as sn

    kurumlar = [f["institution"] for f in
                (sn.tidy_fields_from_ds(_read_header(r["files"][0])) for r in records)
                if f["institution"] not in ("NA", "")]
    varsayilan = max(set(kurumlar), key=kurumlar.count) if kurumlar else "NA"

    kullanilan: dict[str, dict] = {}
    for r in records:
        ds = _read_header(r["files"][0])
        fields = sn.tidy_fields_from_ds(
            ds, kvp=r.get("kvp"), recon=r.get("recon_type"),
            take=r.get("take"), institution_default=varsayilan)
        ad = sn.build_series_name(fields)
        if ad in kullanilan and kullanilan[ad]["series_uid"] != r["series_uid"]:
            ad = f"{ad}_s{r.get('series_number') or 'NA'}"
            if ad in kullanilan:                      # hâlâ çakışıyorsa UID kuyruğu
                ad = f"{ad}_{hashlib.sha1(r['series_uid'].encode()).hexdigest()[:6]}"
        kullanilan[ad] = r
        r["series_name"] = ad


def move_dicoms(records: list[dict[str, Any]], dest_root: Path) -> dict[str, int]:
    """Seçilen serilerin DICOM'larını `dest_root/<series_name>/` altına TAŞIR.

    Yalnız hedef serilerin dosyaları taşınır; filtreye takılanlar (140 kVp,
    topogram, rapor dosyaları) yerinde kalır.

    Taşıma GERİ ALINAMAZ olduğu için her dosyanın eski ve yeni yolu
    `MOVE_LOG.json`e yazılır — sonradan hangi dosyanın nereden geldiği
    kanıtlanabilir ve gerekirse elle geri alınabilir. Kayıt dosyası taşımadan
    ÖNCE değil, her dosya taşındıkça büyütülür ve sonunda atomik yazılır.

    Kayıtların `files` listesi yeni yollara güncellenir; sonraki adımlar
    (NIfTI üretimi) taşınmış dosyaları okur.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, str]] = []
    stats = {"tasinan": 0, "zaten_yerinde": 0, "hata": 0}

    for r in records:
        hedef = dest_root / r["series_name"]
        hedef.mkdir(parents=True, exist_ok=True)
        yeni_yollar: list[Path] = []
        for f in r["files"]:
            if f.parent == hedef:                     # daha önce taşınmış
                yeni_yollar.append(f)
                stats["zaten_yerinde"] += 1
                continue
            target = hedef / f.name
            if target.exists():                       # aynı ad, farklı kaynak
                target = hedef / f"{f.stem}__{hashlib.sha1(str(f).encode()).hexdigest()[:6]}{f.suffix}"
            try:
                os.replace(f, target)                 # aynı birimde atomik
            except OSError:
                try:
                    shutil.move(str(f), str(target))  # birim farklıysa
                except Exception as exc:              # noqa: BLE001
                    print(f"[dicom] HATA taşınamadı {f}: {exc!r}")
                    yeni_yollar.append(f)
                    stats["hata"] += 1
                    continue
            log.append({"eski": str(f), "yeni": str(target)})
            yeni_yollar.append(target)
            stats["tasinan"] += 1
        r["files"] = yeni_yollar

    if log:
        onceki = []
        log_path = dest_root / "MOVE_LOG.json"
        if log_path.exists():
            try:
                onceki = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                onceki = []
        fx._atomic_write_text(log_path,
                              json.dumps(onceki + log, indent=2, ensure_ascii=False))
    return stats


def _series_key(r: dict[str, Any]) -> str:
    """Serinin kimliği = çıktı klasörünün adı.

    `assign_series_names` çalıştıysa OKUNUR ad kullanılır (protokolü doğrudan
    gösterir ve çakışmaya karşı çözülmüştür). Böylece klasör adı, sidecar'daki
    anahtar ve tablodaki `series` sütunu aynı şeydir — resume, yeniden
    adlandırma ve raporlar tek bir kimlik üzerinden yürür.

    Ad yoksa (eski kayıtlar, testler) hasta__klasör[__takeN]__uidkuyruğu
    şemasına düşülür; UID kuyruğu farklı serilerin çakışmasını engeller.
    """
    if r.get("series_name"):
        return r["series_name"]
    key = f"{r['patient_id']}__{ss._sanitize(r['folder'].name)}"
    if r.get("take"):
        key += f"__take{r['take']}"
    uid_tag = hashlib.sha1(str(r["series_uid"]).encode("utf-8")).hexdigest()[:8]
    return f"{key}__{uid_tag}"


def build_meta(r: dict[str, Any]) -> dict[str, Any]:
    """Mekânsal sıralama (kopyalamadan) + nifti/TS/MIRP için meta."""
    pairs: list[tuple[Path, pydicom.Dataset]] = []
    n_unreadable = 0
    for f in r["files"]:
        ds = _read_header(f)
        if ds is not None:
            pairs.append((f, ds))
        else:
            n_unreadable += 1
    pairs = ss.spatial_sort(pairs)   # deterministik mekânsal sıra
    if n_unreadable:
        # Slice-düzeyi kayıp da sessiz kalmasın (bir kısmı bozuk olabilir).
        print(f"[batch]   UYARI {r['folder'].name}: {n_unreadable} slice okunamadı, "
              f"NIfTI {len(pairs)} slice'tan üretilecek.")
    ordered_paths = [str(f) for f, _ in pairs]
    # Bir seri birden çok klasöra bölünmüş olabildiği için TAM YOLLAR taşınır
    # (`series_dir` + dosya adı birleştirmesi böyle serilerde çözülemez).
    # `series_dir` yine de ilk slice'ın klasörü olarak verilir: DICOM metadata
    # toplayıcısı (dicom_metadata.harvest) `series_dir / ordered_files[0]`
    # bekliyor ve o eşleşme bu sayede korunur.
    first_dir = str(Path(ordered_paths[0]).parent) if ordered_paths else str(r["folder"])
    return {
        "n_unreadable": n_unreadable,
        "series_key": _series_key(r),
        "patient_id": r["patient_id"],
        "series_dir": first_dir,                 # KAYNAK klasör — kopya yok
        "ordered_files": [f.name for f, _ in pairs],
        "ordered_paths": ordered_paths,          # yetkili kaynak (klasör aşan seriler)
        "n_folders": len(r.get("folders", [r["folder"]])),
        "kvp": r["kvp"],
        "recon_type": r.get("recon_type", "native"),   # native / VMI_45keV / VMI_70keV
        "series_uid": r["series_uid"],
        "series_description": r["series_desc"],
        "study_date": r["study_date"],
        "take": r.get("take"),
        "num_slices": len(pairs),
    }


# --------------------------------------------------------------------------- #
# Tek seri işleme (resumable)
# --------------------------------------------------------------------------- #
def process_series(meta: dict[str, Any], label_map: dict[str, str]) -> str:
    """Bir seriyi işler: NIfTI → TS → MIRP. Her aşama varsa atlanır."""
    key = meta["series_key"]
    ct_path = config.NIFTI / key / "ct.nii.gz"
    seg_dir = config.SEG / key
    seg_path = seg_dir / "seg.nii.gz"
    lm_path = seg_dir / "label_map.json"
    log_path = seg_dir / "totalsegmentator.log"

    # RESUME: tamamlanma İŞARETİ (features.done.json) esas alınır. Ham features.csv
    # varlığı değil — yarım/atomik-olmayan CSV yanlışlıkla 'tamam' sayılmasın.
    if fx.series_is_complete(key):
        return "atlandı-tamamlanmış"

    # 1) DICOM -> NIfTI (kaynaktan, kopyasız, atomik yazım)
    if not ct_path.exists():
        ts.dicom_series_to_nifti(meta, ct_path)

    # 2) TotalSegmentator
    seg_dir.mkdir(parents=True, exist_ok=True)
    if not lm_path.exists():
        fx._atomic_write_text(lm_path, json.dumps(label_map, indent=2, ensure_ascii=False))
    if not seg_path.exists():
        # TEK GPU: segmentasyon seri kalır (bkz. _TS_LOCK). Kilit yalnız bu
        # adımı kapsar — kilidi bekleyen bir seri, başka bir serinin MIRP'ini
        # engellemez.
        with _TS_LOCK:
            # Kilidi beklerken başka bir iş parçacığı bu seriyi bitirmiş
            # olabilir (aynı seri iki kez sıraya girmez ama savunma ucuz).
            if not seg_path.exists():
                if not ts.run_totalsegmentator(ct_path, seg_path, log_path):
                    return "seg-hatası"

    # 2b) Segmentasyonun görsel kontrolü. Maskenin DOĞRU YERDE olduğunu doğrulayan
    # tek şey bu: geometri testleri ızgaraların uyuştuğunu söylüyor, nereye
    # düştüğünü söylemiyor. Yalnızca görüntüleme; hiçbir çıktısı çıkarıma girmiyor.
    # Her kesit ayrı PNG; seri başına bir klasör.
    overlay_dir = config.OVERLAYS / key
    if config.WRITE_OVERLAYS and not any(overlay_dir.glob("*.png")):
        # Öznitelik çıkarımıyla AYNI organ seçimi çizilir; yoksa kontrol edilen
        # görüntü ile ölçülen organlar birbirini tutmaz.
        qc.write_overlays(ct_path, seg_path, label_map, overlay_dir,
                          keep_labels=fx.target_labels(label_map))

    # 3) MIRP (chunk + RAM-farkında + OOM backoff; %100 IBSI)
    meta2 = dict(meta)
    meta2.update(
        {"ct_path": str(ct_path), "seg_path": str(seg_path), "label_map_path": str(lm_path)}
    )
    df = fx.extract_for_series(meta2)
    return "tamam" if df is not None else "mirp-boş"


# --------------------------------------------------------------------------- #
# Raporlama
# --------------------------------------------------------------------------- #
def _write_reports(unique, duplicates, source_root) -> None:
    config.FEATURES.mkdir(parents=True, exist_ok=True)
    # Kopya / bölünmüş seri raporu
    dup_txt = config.FEATURES / "DUPLICATES_report.txt"
    split = [r for r in unique if len(r.get("folders", [])) > 1]
    dup_slices = [r for r in unique if r.get("n_dup_files")]
    lines = [f"Kaynak: {source_root}", ""]

    # 1) Dışa aktarımda klasörlere BÖLÜNMÜŞ seriler — birleştirildi.
    lines.append(f"BİRDEN ÇOK KLASÖRE BÖLÜNMÜŞ SERİ (birleştirildi): {len(split)}")
    if split:
        lines.append("  Aynı SeriesInstanceUID, ayrık SOP'lar, bitişik z aralıkları:")
        lines.append("  tek seridir, TÜM parçalar birleştirilerek işlenir.")
        for r in split:
            lines.append(f"    {_series_key(r)}  ({len(r['files'])} slice)")
            for folder, n in r["folder_counts"].items():
                lines.append(f"      {n:>5} slice  <- {folder}")
    else:
        lines.append("  (yok)")
    lines.append("")

    # 2) Aynı SOPInstanceUID iki dosyada — gerçek kopya slice, biri kullanıldı.
    total_dup = sum(r["n_dup_files"] for r in dup_slices)
    lines.append(f"GERÇEK KOPYA SLICE DOSYASI (aynı SOPInstanceUID): {total_dup}")
    if dup_slices:
        for r in dup_slices:
            lines.append(f"    {_series_key(r)}: {r['n_dup_files']} kopya dosya "
                         f"atlandı ({len(r['files'])} benzersiz slice kullanıldı)")
        lines.append("  -> Kopya dosyaları istersen SEN silebilirsin; kaynağa dokunmam.")
    else:
        lines.append("  (yok)")
    lines.append("")

    # 3) UID düzeyinde kopya (aynı UID iki kayıt) — artık oluşmamalı; güvenlik ağı.
    uid_dups = [d for d in duplicates if d.get("reason", "uid") == "uid"]
    lines.append(f"UID DÜZEYİNDE KOPYA KAYIT: {len(uid_dups)}")
    if uid_dups:
        for d in uid_dups:
            lines.append(f"    TUTULAN: {d['kept']}  /  ATILAN: {d['duplicate']}  "
                         f"(UID ...{d['series_uid'][-16:]})")
    else:
        lines.append("  (yok — seriler UID'ye göre klasörden bağımsız gruplanıyor)")
    lines.append("")

    # 4) Aynı çekimin ikinci rekonstrüksiyonu — işlenmedi (config ile açılır/kapanır).
    recon_dups = [d for d in duplicates if d.get("reason") == "recon"]
    lines.append(f"ATLANAN ÇİFT REKONSTRÜKSİYON: {len(recon_dups)}")
    if recon_dups:
        lines.append("  Aynı protokol + aynı AcquisitionNumber, farklı SeriesTime:")
        lines.append("  içerik özdeş olduğu için her ikiliden yalnız biri işlendi")
        lines.append("  (config.DROP_DUPLICATE_RECONSTRUCTIONS = False ile hepsi işlenir).")
        lines.append("  KAYNAK DOSYALARA DOKUNULMADI — yalnız işlenmediler.")
        for d in recon_dups:
            lines.append(
                f"    TUTULAN SeriesNo {d['kept']}  /  ATLANAN SeriesNo "
                f"{d['duplicate']}  (çekim {d.get('acq_number','?')}, "
                f"SeriesTime {d.get('series_time','?')}, "
                f"UID ...{d['series_uid'][-16:]})")
            lines.append(f"        {d.get('series_desc','')}")
    elif getattr(config, "DROP_DUPLICATE_RECONSTRUCTIONS", False):
        lines.append("  (yok)")
    else:
        lines.append("  (eleme kapalı: config.DROP_DUPLICATE_RECONSTRUCTIONS = False)")
    dup_txt.write_text("\n".join(lines), encoding="utf-8")

    # Manifest
    manifest = []
    for r in unique:
        manifest.append(
            {
                "series_key": _series_key(r),
                "folder": r["folder"].name,
                "series_uid": r["series_uid"],
                "series_desc": r["series_desc"],
                "scanner": r.get("scanner", ""),
                "photon_counting": r.get("photon_counting", False),
                "kvp": r.get("kvp"),
                "recon_type": r.get("recon_type", "native"),
                "study_date": r["study_date"],
                "take": r.get("take"),
                "n_files": len(r["files"]),
                # Bölünmüş seri izlenebilirliği: kaç klasörden birleştirildi ve
                # hangi klasörden kaç slice geldi (sonradan doğrulanabilsin).
                "n_folders": len(r.get("folders", [r["folder"]])),
                "folder_counts": r.get("folder_counts", {}),
                "n_duplicate_files": r.get("n_dup_files", 0),
            }
        )
    fx._atomic_write_text(config.FEATURES / "MANIFEST.json",
                          json.dumps(manifest, indent=2, ensure_ascii=False))
    takes = [m for m in manifest if m["take"]]
    print(f"[batch] Rapor: {dup_txt.name}, MANIFEST.json ({len(manifest)} seri, "
          f"{len(takes)} test-retest take, {len(split)} bölünmüş seri birleştirildi, "
          f"{total_dup} kopya slice atlandı, "
          f"{len(recon_dups)} çift rekonstrüksiyon işlenmedi).")


def _feature_csvs(keys: list[str] | None, name: str = "features.csv") -> list[Path]:
    """Toplanacak seri tabloları. keys verilirse YALNIZ o serilerinki (eski/yabancı
    çıktı kirletmesin); None ise geriye-uyum için tüm ağaç (eski çıktıları da alır).

    `name` mod dosyasını seçer: 3B için `features.csv`, 2B için `features_2d.csv`.
    """
    if keys is not None:
        return [config.FEATURES / k / name
                for k in keys if (config.FEATURES / k / name).exists()]
    # rglob deseni ADI TAM eşleştirir; `features.csv` ararken `features_2d.csv`
    # yakalanmaz (ve tersi), yani iki mod birbirine karışmaz.
    return sorted(config.FEATURES.rglob(name))


def mode_combined_path(mode: str) -> Path:
    """Bir modun birleşik tablosunun yolu.

    Ana mod (3D) yolunu KORUR — `all_features_combined.csv`; dış betikler,
    testler ve daha önce üretilmiş çıktılar o adı bekliyor.
    """
    ana = config.COMBINED_FEATURES_CSV
    if mode == config.PRIMARY_MODE:
        return ana
    return ana.with_name(f"{ana.stem}_{mode.lower()}{ana.suffix}")


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomik yaz; hedef başka programda açıksa (Excel) yedeğe yaz, orijinali koru."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
    except OSError as exc:
        alt = path.with_name(path.stem + "_yeni" + path.suffix)
        os.replace(tmp, alt)
        print(f"[batch] UYARI: {path.name} yazılamadı ({exc}); {alt.name} olarak kaydedildi "
              f"(dosya başka programda açık olabilir).")


def _safe_write_xlsx(df: "pd.DataFrame | dict[str, pd.DataFrame]",
                     csv_path: Path) -> None:
    """CSV'nin yanına biçimlendirilmiş bir Excel çalışma kitabı yazar (atomik).

    Ham `to_excel` dökümü 200+ sütunda okunamıyordu; excel_report dondurulmuş
    başlık, filtre, katlanmış MIRP meta sütunları ve bir beyan sayfası ekliyor.
    Yazma başarısız olsa bile CSV çoktan kaydedilmiş olur → yalnız uyarı verilir.

    Excel satır/sütun sınırları (1.048.576 × 16.384) içindeyiz.
    """
    try:
        import excel_report
    except Exception as exc:  # noqa: BLE001
        print(f"[batch] UYARI: excel_report yüklenemedi ({exc}); xlsx atlandı.")
        return
    excel_report.write_workbook(df, csv_path)


def _enrich_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Okunur `series` adı + ayrıştırılmış protokol sütunlarını tablonun başına ekler.

    `series_key` (hasta__klasör__uidkuyruğu) izlenebilirlik için gerekli ama
    tabloda okunmuyor: klasör adı protokolü göstermiyor. Buradaki `series`,
    satırın hangi çekim parametrelerinden geldiğini doğrudan okutur; yanındaki
    ayrık sütunlar da Excel'de süzme/gruplamayı mümkün kılar.

    Ham DICOM sütunlarından (`dcm_*`) türetilir — DICOM dosyaları yeniden
    okunmaz. İki cihazı da kapsar: SOMATOM'da rekon gücü çok-değerli kernel
    etiketinden, NAEOTOM'da SeriesDescription'daki QIR seviyesinden gelir.
    """
    import series_naming as sn

    # Kurum kodu: NAEOTOM serilerinde InstitutionName boş geliyor. Aynı
    # tablodaki dolu değerlerden en sık olanı yedek olarak kullanılır.
    codes: list[str] = []
    for value in df.get("dcm_InstitutionName", pd.Series(dtype=object)).dropna():
        found = re.findall(r"\b[A-Z]{2,6}\b", str(value))
        if found:
            codes.append(found[0])
    default_institution = max(set(codes), key=codes.count) if codes else "NA"

    kayitlar = df.to_dict("records")

    # `take` KLASÖR ADINDAN kurtarılır. Seri tabloları bu sütunu taşımıyor
    # (`_finalize_series_frame` yalnız series_key/patient_id/recon_type/kvp
    # ekliyor), dolayısıyla `tidy_fields` içindeki `row.get("take") or 1`
    # varsayılanı devreye giriyor ve TÜM satırlar take=1 oluyordu — retest
    # serileri dahil. Test-retest çalışmasında en çok gruplanacak sütun bu:
    # ölçüldü, 139 serinin 69'u take2 olduğu hâlde tabloda hepsi take1
    # görünüyordu ve take1/take2 eşleştirmesi hiç kurulamıyordu.
    #
    # Klasör adı bu projede TEK YETKİLİ KİMLİK (bkz. `_combine_mode`): çakışma
    # çözümü de take eki de oraya yazılıyor, yani doğru kaynak orası.
    if "series_dir" in df.columns:
        for kayit in kayitlar:
            if not kayit.get("take"):
                eslesme = re.search(r"_take(\d+)", str(kayit.get("series_dir") or ""))
                if eslesme:
                    kayit["take"] = int(eslesme.group(1))

    rows = [sn.tidy_fields(row, default_institution) for row in kayitlar]
    tidy = pd.DataFrame(rows, index=df.index)
    # `series` KLASÖR adından gelir; yalnız klasör yoksa yeniden hesaplanır.
    # Yeniden hesaplama çakışma ekini (`_s<no>`) taşımadığı için tek başına
    # kullanılırsa aynı protokolün tekrar çekimleri tek satıra çöker.
    uretilen = [sn.build_series_name(f) for f in rows]
    if "series_dir" in df.columns:
        tidy["series"] = [d if isinstance(d, str) and d else u
                          for d, u in zip(df["series_dir"], uretilen)]
    else:
        tidy["series"] = uretilen
    tidy = tidy.rename(columns={"mas": "mAs", "fov": "FOV"})

    # Zaten varsa üzerine yazma (yeniden çalıştırmada sütun ikizlenmesin).
    yeni = [c for c in tidy.columns if c not in df.columns]
    out = pd.concat([tidy[yeni], df], axis=1)
    return out


def combine_features(keys: list[str] | None = None) -> pd.DataFrame | None:
    """Her modun seri tablolarını birleştirir; ANA modun tablosunu döndürür.

    3B ve 2B satırları AYNI TABLOYA konamaz: `by_slice=True` yalnız doku
    toplamasını değil yeniden örneklemeyi ve morfolojiyi de değiştiriyor, yani
    aynı adlı sütunlar (`morph_volume`, `stat_mean`, ...) farklı bir işlemden
    geliyor. Birleştirmek iki farklı ölçümü tek sütuna yığmak olurdu. Bu yüzden
    ayrı CSV, Excel'de ayrı sayfa.
    """
    tablolar: dict[str, pd.DataFrame] = {}
    for mod in config.MIRP_MODES:
        df = _combine_mode(keys, mod)
        if df is not None:
            tablolar[mod] = df

    ana = tablolar.get(config.PRIMARY_MODE)
    if ana is None:
        return None

    for mod, df in tablolar.items():
        yol = mode_combined_path(mod)
        _safe_write_csv(df, yol)
        print(f"[batch] Birleşik [{mod}]: {len(df)} satır, {len(df.columns)} sütun -> {yol}")
    _safe_write_xlsx(tablolar, config.COMBINED_FEATURES_CSV)
    return ana


def _combine_mode(keys: list[str] | None, mode: str) -> pd.DataFrame | None:
    """Tek bir modun seri tablolarını birleştirir (CSV yazmaz)."""
    frames = []
    for csv in _feature_csvs(keys, fx.mode_csv_name(mode)):
        try:
            f = pd.read_csv(csv)
            # KLASÖR ADI tek yetkili kimliktir. `assign_series_names` çakışmayı
            # `_s<seri_no>` ekiyle çözüp klasör adına yazıyor; `_enrich_identity`
            # ise adı DICOM'dan YENİDEN hesaplıyor ve o ek orada yok. İki yol
            # ayrıştığı için aynı protokolün tekrarlanan çekimleri tabloda tek
            # ada düşüyordu — ölçüldü: 32 işlenmiş seri, tabloda 28 benzersiz ad,
            # 4 seri görünmez. Klasör adını burada sabitliyoruz.
            f["series_dir"] = csv.parent.name
            frames.append(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[batch] okunamadı {csv}: {exc}")
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)

    # Seri tabloları, o seri işlendiği ANDAKİ organ seçimiyle yazıldı. Seçim
    # sonradan daraltıldıysa birleşik tablo GÜNCEL seçimi yansıtmalı; yoksa
    # artık istenmeyen organlar tabloda kalır. Daraltma yeniden çıkarım
    # gerektirmez: MIRP her ROI'yi bağımsız hesaplıyor, bir organın değerleri
    # yanında kaç organ işlendiğinden etkilenmiyor (bkz.
    # test_integration_mirp_parallel_identity).
    selected = fx.selected_organs()
    if selected is not None and "image_mask_label" in combined.columns:
        before = len(combined)
        combined = combined[combined["image_mask_label"].astype(str).isin(selected)]
        combined = combined.reset_index(drop=True)
        if len(combined) != before:
            print(f"[batch] Organ seçimi uygulandı: {before} -> {len(combined)} satır "
                  f"({', '.join(sorted(selected))}).")
    if combined.empty:
        print("[batch] UYARI: seçilen organlar tabloda yok; birleşik tablo boş.")
        return None

    # Daha önce yazılmış seri tabloları bu sütunları içeriyor olabilir; birleşik
    # tablo her durumda temiz kalsın (yeniden çıkarım gerektirmeden).
    onceki = len(combined.columns)
    combined = fx.drop_unreportable(combined)
    if len(combined.columns) != onceki:
        print(f"[batch] Raporlanamaz öznitelik düşürüldü "
              f"({onceki - len(combined.columns)} sütun): kestirilen "
              f"{', '.join(fx.APPROXIMATED_FEATURES)}; sabitlenmiş IVH hacim "
              f"kesirleri {', '.join(fx.IVH_VOLUME_FRACTION_FEATURES)}")

    try:
        combined = _enrich_identity(combined)
    except Exception as exc:  # noqa: BLE001 — ad üretimi çıktıyı engellemesin
        print(f"[batch] UYARI: düzenli seri adları üretilemedi ({exc!r}).")
    return combined


def write_completeness_report(keys: list[str] | None = None) -> int:
    """Bu çalışmanın features.done.json işaretlerini tarayıp eksik organları raporlar.

    'Hiçbir organ sessizce kaybolmasın' garantisi: eksikler burada görünür kılınır.
    keys verilirse yalnız o seriler (eski/yabancı çıktı raporu kirletmesin).
    Döndürür: eksik organı olan seri sayısı.
    """
    if keys is not None:
        sidecars = [config.FEATURES / k / fx.DONE_SIDECAR
                    for k in keys if (config.FEATURES / k / fx.DONE_SIDECAR).exists()]
    else:
        sidecars = sorted(config.FEATURES.rglob(fx.DONE_SIDECAR))
    incomplete = []
    absent = []          # hedeflenip maskede HİÇ bulunmayan organı olan seriler
    for sidecar in sidecars:
        try:
            d = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if d.get("missing_organs"):
            incomplete.append(d)
        if d.get("absent_target_organs"):
            absent.append(d)

    report = config.FEATURES / "INCOMPLETE_report.txt"
    lines: list[str] = []
    if incomplete:
        lines += [f"ÇIKARIMI BAŞARISIZ ORGANI OLAN SERİLER: {len(incomplete)}",
                  "(organ maskede vardı ama özniteliği hesaplanamadı — RAM/OOM,"
                  " takılma ya da MIRP hatası)", ""]
        for d in incomplete:
            lines.append(f"  {d['series_key']}: {d['n_present_organs']}/{d['n_expected_organs']} "
                         f"organ; eksik: {', '.join(d['missing_organs'])}")
        lines.append("")
    # Bu ayrı bir kategori: TS o yapıyı hiç üretmemiş ya da taranan bölge onu
    # kapsamıyor. Nedeni ve çözümü farklı olduğu için aynı listeye karışmıyor.
    if absent:
        lines += [f"HEDEFLENİP MASKEDE BULUNMAYAN ORGANI OLAN SERİLER: {len(absent)}",
                  "(TotalSegmentator bu yapıyı üretmedi — taranan anatomik bölge"
                  " onu kapsamıyor olabilir)", ""]
        for d in absent:
            lines.append(f"  {d['series_key']}: yok -> "
                         f"{', '.join(d['absent_target_organs'])}")
        lines.append("")
    if not lines:
        lines = ["Tüm serilerde hedeflenen tüm organlar başarıyla çıkarıldı.", ""]
    report.write_text("\n".join(lines), encoding="utf-8")

    if incomplete:
        print(f"[batch] UYARI: {len(incomplete)} seride çıkarımı başarısız organ var "
              f"-> {report.name}")
    if absent:
        print(f"[batch] UYARI: {len(absent)} seride hedef organ maskede yok "
              f"-> {report.name}")
    return len(incomplete)


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def main(source_root: Path, limit: int | None = None,
         only: str | None = None, plan_only: bool = False) -> int:
    # Çıktı kökü KAYNAK klasördür. Bu satır olmadan `batch_120kvp.py` tek başına
    # çalıştırıldığında (README'nin ana komutu) çıktı kod klasörüne düşüyor ve
    # `move_dicoms` ham DICOM'ları oraya TAŞIYOR — taşıma `os.replace` ile
    # yapıldığı için geri alınamaz. `run_and_verify` bunu kendi çağırdığı için
    # arıza yalnız doğrudan çalıştırmada görünüyordu.
    config.set_output_root(source_root)
    config.ensure_dirs()
    t0 = time.time()

    print(f"[batch] Kaynak taranıyor: {source_root}")
    records = discover_series(source_root)
    # Otomatik algılanan cihaz(lar)ı bildir (kVp kaynağı buna göre seçilir).
    if records:
        from collections import Counter as _C
        scanners = _C((r.get("scanner", "?"),
                       "foton-sayan" if r.get("photon_counting") else "konvansiyonel")
                      for r in records)
        for (model, kind), n in scanners.most_common():
            # kVp kaynağı config.KVP_FROM_DESCRIPTION'a göre (bkz. sort_spatially.effective_kvp):
            # PCCT + flag True ise SeriesDescription, aksi halde (varsayılan) KVP etiketi.
            src = ("SeriesDescription" if (kind == "foton-sayan" and config.KVP_FROM_DESCRIPTION)
                   else "KVP etiketi")
            print(f"[batch] Cihaz algılandı: {model} ({kind}) — kVp kaynağı: {src} [{n} seri]")
    unique, duplicates = dedup_and_takes(records)
    n_uid = sum(1 for d in duplicates if d.get("reason", "uid") == "uid")
    n_recon = sum(1 for d in duplicates if d.get("reason") == "recon")
    print(f"[batch] {len(records)} adet 120 kVp CT serisi bulundu -> {len(unique)} benzersiz, "
          f"{n_uid} gerçek kopya, {n_recon} çift rekonstrüksiyon elendi.")
    n_take = sum(1 for r in unique if r.get("take"))
    if n_take:
        print(f"[batch] Test-retest: {n_take} seriye take atandı "
              f"(en yüksek take{max(r['take'] for r in unique if r.get('take'))}).")

    # Okunur adlar: çıktı klasörlerinin ve tablodaki `series` sütununun kaynağı.
    assign_series_names(unique)
    _write_reports(unique, duplicates, source_root)

    # Filtre (smoke test)
    if only:
        unique = [r for r in unique if only.lower() in r["folder"].name.lower()]
        print(f"[batch] --only '{only}' -> {len(unique)} seri.")
    if limit:
        unique = unique[:limit]
        print(f"[batch] --limit {limit} -> {len(unique)} seri.")

    if plan_only:
        print("[batch] --plan-only: yalnızca analiz yapıldı, işleme yok.")
        return 0

    # Seçilen serilerin DICOM'ları kaynak klasördeki `dicoms/<seri>/` altına
    # toplanır. Bölünmüş seriler burada tek klasörde birleşir.
    stats_move = move_dicoms(unique, config.DICOMS)
    if stats_move["tasinan"] or stats_move["hata"]:
        print(f"[dicom] {stats_move['tasinan']} dosya taşındı, "
              f"{stats_move['zaten_yerinde']} zaten yerindeydi, "
              f"{stats_move['hata']} hata -> {config.DICOMS}")

    label_map = ts.load_total_label_map()
    print(f"[batch] TotalSegmentator '{config.TS_TASK}' — {len(label_map)} yapı.")
    # Organ seçimi burada, işlemeden ÖNCE doğrulanır: geçersiz bir ad varsa
    # 60 seri işlendikten sonra değil, ilk saniyede öğrenilsin.
    selected = fx.selected_organs()
    if selected is None:
        print("[batch] Organ seçimi: TÜM yapılar (config.TARGET_ORGANS = None).")
    else:
        print(f"[batch] Organ seçimi: {len(selected)} yapı -> {', '.join(sorted(selected))}")

    stats = defaultdict(int)

    # RESUME: tamamlanmış serilerde build_meta'yı (tüm slice başlıklarını okur)
    # ATLA — yeniden başlatmada binlerce gereksiz header okumasını önler.
    bekleyen: list[tuple[int, dict[str, Any]]] = []
    for i, r in enumerate(unique, 1):
        if fx.series_is_complete(_series_key(r)):
            stats["atlandı-tamamlanmış"] += 1
            continue
        bekleyen.append((i, r))

    esz = max(1, int(getattr(config, "SERIES_PARALLELISM", 1) or 1))
    esz = min(esz, len(bekleyen)) or 1
    if esz > 1:
        print(f"[batch] {len(bekleyen)} seri {esz} paralel işlenecek "
              f"(segmentasyon tek GPU'da seri kalır).")

    def _isle(sira: int, r: dict[str, Any]) -> tuple[str, str]:
        """Tek seriyi işler; (anahtar, durum) döndürür. İstisna yutulmaz-raporlanır."""
        key = _series_key(r)
        try:
            meta = build_meta(r)
        except Exception as exc:  # noqa: BLE001
            return key, f"HATA: meta üretilemedi {exc!r}"
        take_str = f" [take {meta['take']}]" if meta.get("take") else ""
        with _PRINT_LOCK:
            print(f"\n[batch] ({sira}/{len(unique)}) {key}{take_str} "
                  f"({meta['num_slices']} slice) — {time.strftime('%H:%M:%S')}")
        try:
            return key, process_series(meta, label_map)
        except Exception as exc:  # noqa: BLE001 — bir seri patlarsa diğerleri sürsün
            return key, f"HATA: {exc!r}"

    if esz == 1:
        for sira, r in bekleyen:
            key, status = _isle(sira, r)
            stats[status.split(":")[0]] += 1
            print(f"[batch]   -> {status}")
    else:
        with ThreadPoolExecutor(max_workers=esz) as havuz:
            futures = {havuz.submit(_isle, sira, r): sira for sira, r in bekleyen}
            for fut in as_completed(futures):
                key, status = fut.result()
                stats[status.split(":")[0]] += 1
                with _PRINT_LOCK:
                    print(f"[batch]   -> {key[-46:]}: {status}")

    # Birleşik tablo ve eksik raporu YALNIZCA bu çalışmanın serilerinden derlenir
    # (eski anahtar şeması / farklı kaynak kalıntıları çift saymasın/kirletmesin).
    run_keys = [_series_key(r) for r in unique]
    combined = combine_features(run_keys)
    n_incomplete = write_completeness_report(run_keys)

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print("[batch] ÖZET")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    if n_incomplete:
        print(f"  eksik organı olan seri: {n_incomplete} (bkz. INCOMPLETE_report.txt)")
    if combined is not None:
        print(f"  toplam öznitelik satırı: {len(combined)}")
    print(f"  süre: {dt/60:.1f} dk")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="120 kVp toplu radiomics pipeline")
    ap.add_argument("source", help="Ham DICOM kök klasörü")
    ap.add_argument("--limit", type=int, default=None, help="İlk N seriyi işle (smoke test)")
    ap.add_argument("--only", default=None, help="Adında bu geçen serileri işle")
    ap.add_argument("--plan-only", action="store_true", help="Sadece analiz/rapor, işleme yok")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    raise SystemExit(main(Path(args.source), args.limit, args.only, args.plan_only))
