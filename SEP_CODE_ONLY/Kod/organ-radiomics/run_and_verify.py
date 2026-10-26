"""Çok kaynaklı koşum + çıktı doğrulama aracı.

NEDEN AYRI BİR ARAÇ: `batch_120kvp.py` TEK bir kaynağı işler ve koşum sonunda
birleşik tabloyu YALNIZ o koşumun serilerinden derler (`combine_features(run_keys)`).
İki cihazı arka arkaya çalıştırmak, ikinci koşumun `all_features_combined.csv`,
`MANIFEST.json` ve `DUPLICATES_report.txt` dosyalarını ilkinin ÜZERİNE yazması
demek — yani birinci cihazın kaydı kaybolur. Bu araç:

  1) ÖN KONTROL — ortam gerçekten hazır mı (TS, GPU, paketler, disk, organ seçimi).
     Bir eksiği 6 saatlik koşumun ortasında değil, ilk saniyede öğrenmek için.
  2) Her kaynağı sırayla işler ve kaynağa ÖZEL raporları saklar
     (`data/features/_runs/<kaynak>/`), üzerine yazılmasın.
  3) Tüm kaynakların serilerini TEK birleşik tabloda toplar.
  4) DOĞRULAR: "bir hata var mı" sorusunu, çıktıyı okuyup net bir kararla
     yanıtlar ve `VERIFY_report.txt`e yazar. Çıkış kodu 0 ise temiz.

Doğrulama, pipeline'ın kendi ürettiği kanıtları tüketir (features.done.json,
MANIFEST.json, features.csv, NIfTI başlıkları) — yeniden hesap yapmaz.

Kullanım:
    python run_and_verify.py                          # varsayılan kaynaklar
    python run_and_verify.py --limit 1                # her kaynaktan 1 seri (duman testi)
    python run_and_verify.py --plan-only              # yalnız tarama + ön kontrol
    python run_and_verify.py --verify-only            # işleme yok, mevcut çıktıyı denetle
    python run_and_verify.py "C:\\yol\\A" "C:\\yol\\B"  # kaynakları elle ver

Argümansız koşumun kaynakları PHANTOMX_SOURCES ortam değişkeninden okunur
(yol ayırıcısıyla, Windows'ta ';'); yoksa `_FALLBACK_SOURCES`'a düşülür:
    setx PHANTOMX_SOURCES "C:\\veri\\EID;C:\\veri\\PCCT"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

import batch_120kvp as batch
import config
import feature_extraction_mirp as fx

# Türkçe konsolda (cp1254) `✓`, `→` gibi karakterler yazılamıyor ve HATASIZ bir
# koşum bile son satırda UnicodeEncodeError ile 1 kodunu döndürüyordu — yani
# başarı, başarısızlıktan ayırt edilemiyordu. Kodlamayı burada sabitliyoruz;
# çağıranın PYTHONIOENCODING ayarlamış olmasına bel bağlanamaz.
for _akis in (sys.stdout, sys.stderr):
    try:
        _akis.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                      # noqa: BLE001 — yönlendirilmiş akış
        pass
import run_totalsegmentator as ts

# Argümansız çalıştırmanın varsayılan kaynakları.
#
# TAŞINABİLİR TESPİT — config.py'nin TotalSegmentator yolları için kullandığı
# sırayla aynı: (1) ortam değişkeni, (2) yoksa bu makinedeki bilinen yollar.
# Buradaki iki yol tek bir makinenin masaüstüne sabitti ve projenin geri kalanı
# taşınabilirken bu satırlar değildi: başka bir bilgisayarda argümansız koşum
# ön kontrolde "Kaynak klasör yok" ile duruyordu ve kullanıcının düzeltmek için
# kodu açması gerekiyordu.
#
# Birden çok kaynak, yol ayırıcısıyla verilir (Windows'ta ';'):
#   setx PHANTOMX_SOURCES "C:\veri\EID;C:\veri\PCCT"
_FALLBACK_SOURCES = [
    Path(r"C:\Users\Administrator\Desktop\14072026-EID"),
    Path(r"C:\Users\Administrator\Desktop\14072026-PCCT"),
]


def default_sources() -> list[Path]:
    """Argümansız koşumun kaynakları (PHANTOMX_SOURCES varsa oradan)."""
    env = os.environ.get("PHANTOMX_SOURCES", "").strip()
    if env:
        return [Path(p.strip()) for p in env.split(os.pathsep) if p.strip()]
    return list(_FALLBACK_SOURCES)


# Geriye uyum: dış betikler bu adı içe aktarıyor olabilir.
DEFAULT_SOURCES = default_sources()

VERIFY_REPORT = "VERIFY_report.txt"


# --------------------------------------------------------------------------- #
# 1) Ön kontrol
# --------------------------------------------------------------------------- #
def preflight(sources: list[Path]) -> list[str]:
    """Ortamı denetler; ENGELLEYİCİ sorunların listesini döndürür (boş = hazır)."""
    problems: list[str] = []

    for src in sources:
        if not src.is_dir():
            problems.append(f"Kaynak klasör yok: {src}")

    # TotalSegmentator: config PATH/ortam değişkeni/bilinen yol sırasıyla arar.
    if config.TOTALSEG_EXE is None or not Path(config.TOTALSEG_EXE).exists():
        problems.append(
            "TotalSegmentator çalıştırılabiliri bulunamadı. PHANTOMX_TOTALSEG_EXE "
            "ortam değişkenini ayarlayın ya da TS'i PATH'e ekleyin."
        )
    if config.PY311 is None or not Path(config.PY311).exists():
        problems.append(
            "TotalSegmentator'ın Python'u bulunamadı (PHANTOMX_PY311). Etiket "
            "haritası statik dosyadan okunacak; yine de ayarlanması önerilir."
        )

    # Etiket haritası + organ seçimi: geçersiz bir ad koşumun tamamını boşa çıkarır.
    try:
        label_map = ts.load_total_label_map()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Etiket haritası yüklenemedi: {exc!r}")
        label_map = {}
    if label_map:
        try:
            selected = fx.selected_organs()
        except ValueError as exc:
            problems.append(str(exc))
            selected = None
        if selected is not None:
            missing_ids = sorted(selected - set(label_map.values()))
            if missing_ids:
                problems.append(f"Seçilen organ TS haritasında yok: {missing_ids}")

    # Ana ortamın paketleri (MIRP hesabı bunlara bağlı).
    import importlib.metadata as md
    for package in ("mirp", "numpy", "pandas", "SimpleITK", "nibabel", "pydicom"):
        try:
            md.version(package)
        except Exception:  # noqa: BLE001
            problems.append(f"Ana ortamda paket yok: {package}")
    # openpyxl requirements.txt'de listelenmiyor ama excel_report onu kullanıyor;
    # yokluğu ölümcül değil (CSV yazılır), o yüzden yalnız uyarı olarak yazdırılır.
    try:
        md.version("openpyxl")
    except Exception:  # noqa: BLE001
        print("[ön] UYARI: openpyxl yok → .xlsx üretilmez, yalnız CSV yazılır.")

    # GPU: TS ayrı ortamda koşuyor, CUDA'yı ORADA sormak gerekir.
    if config.TS_DEVICE == "gpu" and config.PY311 and Path(config.PY311).exists():
        import subprocess
        try:
            out = subprocess.run(
                [str(config.PY311), "-c",
                 "import torch;print('OK' if torch.cuda.is_available() else 'NO')"],
                capture_output=True, text=True, timeout=300,
            )
            if "OK" not in out.stdout:
                print("[ön] UYARI: TS ortamında CUDA görünmüyor → segmentasyon CPU'da "
                      "çok yavaş olur. config.TS_DEVICE='cpu' yapın ya da PyTorch'u "
                      "CUDA sürümüyle kurun.")
        except Exception as exc:  # noqa: BLE001
            print(f"[ön] UYARI: CUDA denetlenemedi ({exc!r}).")

    # Disk: ara dosyalar (nifti + seg) girdinin katı kadar yer ister.
    try:
        # Boş alan KAYNAK klasörün diskinde ölçülür, kodun bulunduğu diskte
        # değil: çıktı kaynağın İÇİNE yazılıyor (`config.set_output_root(src)`).
        # İkisi farklı sürücü olabilir — paket C:'de, DICOM'lar harici diskte.
        # PROJECT_ROOT'a bakmak o durumda iki yönde de yanlış karar veriyordu:
        # dolu bir hedef diski "hazır" saymak ya da bol yeri olan bir hedefte
        # koşumu hiç başlatmamak.
        for src in sources:
            if not src.is_dir():
                continue
            free_gb = shutil.disk_usage(src).free / 1e9
            src_gb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e9
            print(f"[ön] {src.name}: girdi ~{src_gb:.1f} GB, boş disk ~{free_gb:.1f} GB.")
            if free_gb < src_gb:
                problems.append(
                    f"Disk yetersiz olabilir ({src.name}): girdi ~{src_gb:.0f} GB, "
                    f"boş ~{free_gb:.0f} GB.")
    except Exception as exc:  # noqa: BLE001
        print(f"[ön] UYARI: disk denetlenemedi ({exc!r}).")

    return problems


# --------------------------------------------------------------------------- #
# 2) Kaynağa özel raporları sakla
# --------------------------------------------------------------------------- #
def _manifest_keys_at(features_dir: Path) -> list[str]:
    """Bir kaynağın MANIFEST'indeki seri anahtarları.

    Her kaynak kendi çıktı ağacına yazdığı için raporlar birbirini ezmiyor;
    ayrı bir anlık görüntü klasörüne gerek kalmadı.
    """
    path = features_dir / "MANIFEST.json"
    if not path.exists():
        return []
    try:
        return [m["series_key"] for m in json.loads(path.read_text(encoding="utf-8"))]
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# 3) Doğrulama
# --------------------------------------------------------------------------- #
def _read_sidecar(key: str) -> dict | None:
    path = config.FEATURES / key / fx.DONE_SIDECAR
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _geometry(path: Path) -> tuple | None:
    """NIfTI'nin ızgarası (piksel okumadan): (boyut, spacing, origin)."""
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return (tuple(reader.GetSize()),
                tuple(round(v, 6) for v in reader.GetSpacing()),
                tuple(round(v, 4) for v in reader.GetOrigin()))
    except Exception:  # noqa: BLE001
        return None


def _mask_geometry_checks(key: str, targets: set[str] | None,
                          label_map: dict[str, str]) -> list[str]:
    """Maske düzeyinde bilimsel geçerlilik kontrolleri → uyarı satırları.

    Bir organ hacmin İLK ya da SON diliminde ise taranan alan onu tamamen
    kapsamıyor: ölçülen şey organ değil, organın rastgele bir kesitidir ve
    şekil öznitelikleri (morph_*) geçersizdir. Bu sessizce olur — tablo dolu
    görünür, değerler makul aralıkta çıkar. Maske sıkıştırılmış ve çoğu sıfır
    olduğu için okuması ucuz.
    """
    import numpy as np

    notes: list[str] = []
    seg_path = config.SEG / key / "seg.nii.gz"
    if not seg_path.exists():
        return notes
    try:
        image = sitk.ReadImage(str(seg_path))
        arr = sitk.GetArrayFromImage(image)          # (z, y, x)
    except Exception as exc:  # noqa: BLE001
        return [f"{key}: maske geometri kontrolü yapılamadı ({exc!r})."]

    label_of = {name: int(lab) for lab, name in label_map.items()}
    sx, sy, sz = image.GetSpacing()
    voxel_ml = sx * sy * sz / 1000.0
    n_z = arr.shape[0]

    for name in sorted(targets or set()):
        lab = label_of.get(name)
        if lab is None:
            continue
        mask = arr == lab
        if not mask.any():
            continue
        zs = np.flatnonzero(mask.any(axis=(1, 2)))
        z0, z1 = int(zs[0]), int(zs[-1])
        volume_ml = float(mask.sum()) * voxel_ml
        if z0 == 0 or z1 == n_z - 1:
            edge = "alt" if z0 == 0 else "üst"
            notes.append(f"{key}/{name}: organ hacmin {edge} sınırına dokunuyor "
                         f"(z={z0}..{z1}/{n_z - 1}) → KESİK; şekil öznitelikleri "
                         f"bu organ için geçersiz.")
        if volume_ml < 10.0:
            notes.append(f"{key}/{name}: hacim çok küçük (~{volume_ml:.1f} mL) → "
                         f"segmentasyon kısmi/hatalı olabilir; QC overlay'e bakın.")
    return notes


def verify(expected_keys: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Çıktıyı denetler → (engelleyici_hatalar, uyarılar, bilgi_satırları)."""
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    targets = fx.selected_organs()
    # Seçim daraltıldığında seri tablolarında kalan eski organlar; hata değil,
    # bilgi olarak raporlanır (birleşik tablo güncel seçime göre süzülür).
    extras: set[str] = set()
    hashes: dict[str, list[str]] = {}
    versions: dict[str, list[str]] = {}
    per_series_rows = 0
    volume_rows: list[tuple[str, str, float]] = []

    processed = 0
    # Tabloya SATIR VERMESİ beklenen seriler. Hedef organı maskede hiç bulunmayan
    # bir seri (taranan bölge onu kapsamıyor) tanım gereği satır üretemez; onu da
    # beklemek yanlış pozitif üretiyordu.
    #
    # ÖLÇÜLDÜ (20260312 seti): 78 slice'lık bir seri fantomun yalnız en kraniyal
    # 78 mm'sini kapsıyor (eşi 373 mm), dolayısıyla dalak ve aorta görüntüde YOK.
    # Hat bunu doğru şekilde `absent_target_organs` diye raporluyordu, ama
    # aşağıdaki üç denetim (birleşik tabloda eksik seri, benzersiz `series`
    # sayısı, 2B tablosunda eksik seri) onu yine de bekleyip ENGELLEYİCİ HATA
    # veriyordu — 140/140 işlenmiş, sağlam bir çıktı "HATA VAR" damgası alıyordu.
    #
    # Çıkarımı BAŞARISIZ olan organ (missing_organs) bu muafiyete girmez: orası
    # gerçek kayıptır ve hata olarak kalmalıdır.
    satir_bekleyen: set[str] = set()
    for key in expected_keys:
        sidecar = _read_sidecar(key)
        if sidecar is None:
            errors.append(f"{key}: features.done.json yok → seri hiç işlenmemiş.")
            continue
        processed += 1
        if sidecar.get("n_present_organs", 0) > 0:
            satir_bekleyen.add(str(key))

        # Maskede VARDI ama özniteliği hesaplanamadı → gerçek kayıp.
        if sidecar.get("missing_organs"):
            errors.append(f"{key}: çıkarım başarısız organ(lar): "
                          f"{', '.join(sidecar['missing_organs'])}")
        # TS o yapıyı hiç üretmemiş → veri/kapsam sorunu, kod hatası değil.
        if sidecar.get("absent_target_organs"):
            warnings.append(f"{key}: maskede bulunmayan hedef organ: "
                            f"{', '.join(sidecar['absent_target_organs'])}")
        if sidecar.get("failed_units"):
            warnings.append(f"{key}: atlanan iş birimi: "
                            f"{', '.join(sidecar['failed_units'])}")

        # Ayar/sürüm kayması: farklı ayarla üretilmiş satırlar karşılaştırılamaz.
        hashes.setdefault(str(sidecar.get("settings_hash")), []).append(key)
        versions.setdefault(json.dumps(sidecar.get("library_versions"), sort_keys=True),
                            []).append(key)

        # Maske ile görüntü aynı ızgarada mı — kayma sessizce yanlış öznitelik üretir.
        ct_geom = _geometry(config.NIFTI / key / "ct.nii.gz")
        seg_geom = _geometry(config.SEG / key / "seg.nii.gz")
        if ct_geom is None:
            errors.append(f"{key}: ct.nii.gz okunamadı.")
        elif seg_geom is None:
            errors.append(f"{key}: seg.nii.gz okunamadı.")
        elif ct_geom != seg_geom:
            errors.append(f"{key}: CT ve maske ızgarası UYUŞMUYOR "
                          f"(ct={ct_geom} seg={seg_geom}).")
        else:
            # Kesik organ / şüpheli küçük hacim: değerler makul görünse de
            # geçersiz olabilir, o yüzden maske düzeyinde ayrıca bakılır.
            lm_path = config.SEG / key / "label_map.json"
            if lm_path.exists():
                try:
                    series_label_map = json.loads(lm_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    series_label_map = {}
                if series_label_map:
                    warnings.extend(
                        _mask_geometry_checks(key, targets, series_label_map))

        # Seri tablosu: organ adları seçimin içinde mi, çekirdek öznitelik geçerli mi.
        csv_path = config.FEATURES / key / "features.csv"
        if not csv_path.exists():
            if sidecar.get("n_expected_organs", 0) > 0:
                errors.append(f"{key}: features.csv yok (sidecar organ bekliyordu).")
            continue
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{key}: features.csv okunamadı ({exc!r}).")
            continue
        if "image_mask_label" not in frame.columns:
            errors.append(f"{key}: features.csv'de image_mask_label sütunu yok.")
            continue
        organs = set(frame["image_mask_label"].astype(str))
        if targets is not None:
            # Seri tablosu, o seri işlendiği ANDAKİ seçimle yazıldı ve seçim
            # sonradan daraltılmış olabilir. Fazlalık organ hata DEĞİLDİR;
            # birleşik tablo güncel seçime göre süzülüyor (combine_features).
            # Asıl hata, seçilen bir organın maskede olmasına rağmen tabloda
            # bulunmaması olurdu — onu sidecar'daki missing_organs yakalıyor.
            extra = sorted(organs - targets)
            if extra:
                extras.update(extra)
            frame = frame[frame["image_mask_label"].astype(str).isin(targets)]
        if frame["image_mask_label"].duplicated().any():
            dup = sorted(frame.loc[frame["image_mask_label"].duplicated(),
                                   "image_mask_label"].astype(str).unique())
            errors.append(f"{key}: aynı organ birden çok satırda: {', '.join(dup)}")
        # Süzmeden SONRA sayılır: birleşik tablo da güncel seçime göre süzülü,
        # ikisi karşılaştırılabilir olsun.
        per_series_rows += len(frame)

        if "morph_volume" in frame.columns:
            for _, row in frame.iterrows():
                organ, volume = str(row["image_mask_label"]), row["morph_volume"]
                if pd.isna(volume) or float(volume) <= 0:
                    errors.append(f"{key}/{organ}: morph_volume geçersiz ({volume}).")
                else:
                    volume_rows.append((key, organ, float(volume)))

    info.append(f"İşlenmiş seri: {processed}/{len(expected_keys)}")
    n_muaf = processed - len(satir_bekleyen)
    if n_muaf:
        info.append(f"Bunların {n_muaf} tanesi tabloya satır vermiyor: hedef organ "
                    f"görüntüde yok (taranan bölge kapsamıyor). Eksik sayılmadılar; "
                    f"yukarıdaki uyarılarda seri adlarıyla listeliler.")
    if targets is not None:
        info.append(f"Organ seçimi: {', '.join(sorted(targets))}")
    if extras:
        info.append(f"Seri tablolarında duran ama seçim dışı kalan organlar: "
                    f"{', '.join(sorted(extras))} (birleşik tabloya alınmadı; "
                    f"seçimi genişletmek yeniden çıkarım gerektirmez)")

    if len(hashes) > 1:
        errors.append("MIRP ayar hash'i seriler arasında FARKLI (çıktılar "
                      "karşılaştırılamaz): " +
                      "; ".join(f"{h[:8]}→{len(k)} seri" for h, k in hashes.items()))
    elif hashes:
        info.append(f"MIRP ayar hash'i (tüm serilerde aynı): {next(iter(hashes))}")
    if len(versions) > 1:
        errors.append(f"Kütüphane sürümleri seriler arasında FARKLI "
                      f"({len(versions)} ayrı küme) → çıktılar karşılaştırılamaz.")

    # Birleşik tablo, seri tablolarının toplamıyla uyuşmalı.
    combined_path = config.COMBINED_FEATURES_CSV
    if not combined_path.exists():
        errors.append(f"Birleşik tablo yok: {combined_path.name}")
    else:
        try:
            combined = pd.read_csv(combined_path)
            info.append(f"Birleşik tablo: {len(combined)} satır, "
                        f"{len(combined.columns)} sütun")
            if len(combined) != per_series_rows:
                errors.append(f"Birleşik tablo {len(combined)} satır ama seri "
                              f"tablolarının toplamı {per_series_rows}.")
            if {"series_key", "image_mask_label"} <= set(combined.columns):
                dups = combined.duplicated(["series_key", "image_mask_label"]).sum()
                if dups:
                    errors.append(f"Birleşik tabloda {dups} yinelenen "
                                  f"(seri, organ) satırı var.")
                # Satır SAYISI değil, KÜMESİ denetlenmeli. Sayı denetimi
                # yanıltıcı: MANIFEST dışı bir seri tabloya sızarken başka biri
                # düşerse toplam tutar ve doğrulama "TEMİZ" der. Kum havuzunda
                # tam bu oldu — MANIFEST'te olmayan bir seri hatasız onaylandı.
                tablo_k = set(combined["series_key"].astype(str))
                beklenen_k = set(map(str, expected_keys))
                fazla = sorted(tablo_k - beklenen_k)
                # Yalnız SATIR VERMESİ beklenen seriler aranır (bkz.
                # satir_bekleyen): hedef organı görüntüde olmayan seri satır
                # üretemez ve eksik sayılmamalı.
                eksik = sorted(satir_bekleyen - tablo_k)
                if fazla:
                    errors.append(
                        f"Birleşik tabloda MANIFEST dışı {len(fazla)} seri var "
                        f"(ilk: {fazla[0]}).")
                if eksik:
                    errors.append(
                        f"Birleşik tabloda {len(eksik)} işlenmiş seri eksik "
                        f"(ilk: {eksik[0]}).")
                info.append("Organ başına satır: " + ", ".join(
                    f"{organ}={n}" for organ, n in
                    combined["image_mask_label"].value_counts().sort_index().items()))
            # `series` sütunu KLASÖR SAYISI kadar benzersiz olmak zorunda.
            # Bu denetim eksikti ve gerçek bir kusuru gizledi: aynı protokolün
            # tekrar çekimlerinde klasör adı `_s<no>` ekiyle ayrılırken sütun
            # ayrılmıyordu; 32 işlenmiş seri tabloda 28 ada düşüyor, 4 seri
            # görünmez oluyordu. Yukarıdaki `series_key` denetimi bunu
            # yakalayamaz, çünkü o sütun tekil kalıyor.
            if "series" in combined.columns:
                n_ad = combined["series"].nunique()
                # Karşılaştırma `processed` ile DEĞİL, satır vermesi beklenen
                # seri sayısıyla yapılır: kapsam dışı kalan seri tabloda
                # görünemez ve onu beklemek sağlam bir çıktıyı hatalı gösterir.
                if n_ad != len(satir_bekleyen):
                    kayip = sorted(satir_bekleyen - set(combined["series"]))
                    errors.append(
                        f"Birleşik tabloda {n_ad} benzersiz `series` var ama "
                        f"{len(satir_bekleyen)} seri satır vermeliydi — "
                        f"{len(kayip)} seri tabloda görünmüyor "
                        f"(ilk: {kayip[0] if kayip else '?'}).")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Birleşik tablo okunamadı ({exc!r}).")

    # İKİNCİL MODLARIN (2B) tablosu da denetlenir. Denetlenmezse 2B sessizce
    # eksik kalabilir: ana tablo tamdır, doğrulama "TEMİZ" der, Excel'de
    # `2D_base` sayfası ya hiç olmaz ya da serilerin bir kısmını içerir.
    for mod in config.MIRP_MODES:
        if mod == config.PRIMARY_MODE:
            continue
        yol = batch.mode_combined_path(mod)
        if not yol.exists():
            errors.append(f"{mod} birleşik tablosu yok: {yol.name}")
            continue
        try:
            ikincil = pd.read_csv(yol)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{mod} birleşik tablosu okunamadı ({exc!r}).")
            continue
        info.append(f"Birleşik tablo [{mod}]: {len(ikincil)} satır, "
                    f"{len(ikincil.columns)} sütun")
        if len(ikincil) != per_series_rows:
            errors.append(f"{mod} tablosu {len(ikincil)} satır ama seri "
                          f"tablolarının toplamı {per_series_rows}.")
        if "series_key" in ikincil.columns:
            # Ana tabloyla aynı ölçüt: yalnız satır vermesi beklenen seriler.
            eksik_seri = sorted(satir_bekleyen -
                                set(ikincil["series_key"].astype(str)))
            if eksik_seri:
                errors.append(f"{mod} tablosunda {len(eksik_seri)} işlenmiş seri "
                              f"eksik (ilk: {eksik_seri[0]}).")

    # QC görüntüleri: maskenin DOĞRU YERDE olduğunu doğrulayan tek şey.
    if config.WRITE_OVERLAYS:
        missing_png = [k for k in expected_keys
                       if not any((config.OVERLAYS / k).glob("*.png"))]
        if missing_png:
            warnings.append(f"{len(missing_png)} seride QC overlay PNG üretilmemiş "
                            f"(ilk örnekler: {', '.join(missing_png[:3])}).")
        else:
            n_png = sum(len(list((config.OVERLAYS / k).glob("*.png")))
                        for k in expected_keys)
            info.append(f"QC overlay: {len(expected_keys)} seri için {n_png} PNG "
                        f"-> {config.OVERLAYS}")

    # Hacim özeti: gözle mantıklılık kontrolü (gross yanlış etiketleme buradan görünür).
    if volume_rows:
        frame = pd.DataFrame(volume_rows, columns=["series", "organ", "volume"])
        info.append("Organ hacmi (morph_volume) özeti — medyan [min–maks]:")
        for organ, group in frame.groupby("organ"):
            info.append(f"    {organ:18} n={len(group):4d}  "
                        f"{group['volume'].median():12.1f}  "
                        f"[{group['volume'].min():.1f} – {group['volume'].max():.1f}]")

    return errors, warnings, info


def write_verify_report(errors: list[str], warnings: list[str],
                        info: list[str]) -> Path:
    lines = ["PhantomXPro — ÇIKTI DOĞRULAMA RAPORU",
             time.strftime("Tarih: %Y-%m-%d %H:%M:%S"), ""]
    lines.append("KARAR: " + ("TEMİZ — engelleyici hata bulunamadı."
                              if not errors else
                              f"HATA — {len(errors)} engelleyici bulgu."))
    lines.append("")
    for title, items in (("ENGELLEYİCİ HATALAR", errors),
                         ("UYARILAR", warnings),
                         ("BİLGİ", info)):
        lines.append(f"== {title} ({len(items)}) ==")
        if items:
            lines.extend(f"  {item}" for item in items)
        else:
            lines.append("  (yok)")
        lines.append("")
    path = config.FEATURES / VERIFY_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def completed_keys() -> list[str]:
    """Tamamlanma işareti yazılmış seriler (koşum yarıda bırakılmış olabilir)."""
    return sorted(p.parent.name for p in
                  config.FEATURES.glob(f"*/{fx.DONE_SIDECAR}"))


def main(sources: list[Path], limit: int | None = None, only: str | None = None,
         plan_only: bool = False, verify_only: bool = False,
         interim: bool = False) -> int:
    t0 = time.time()
    print("=" * 70)
    print(f"[koşum] {len(sources)} kaynak: " + ", ".join(s.name for s in sources))
    try:
        selected = fx.selected_organs()
    except ValueError as exc:
        print(f"[koşum] YAPILANDIRMA HATASI: {exc}")
        return 2
    print("[koşum] Organ seçimi: " +
          ("TÜM yapılar" if selected is None
           else f"{len(selected)} yapı -> {', '.join(sorted(selected))}"))
    print("=" * 70)

    if not (verify_only or interim):
        problems = preflight(sources)
        if problems:
            print("\n[ön] ENGELLEYİCİ SORUN(LAR):")
            for p in problems:
                print(f"  - {p}")
            print("[ön] Koşum başlatılmadı.")
            return 2
        print("[ön] Ortam hazır.\n")

    # 1) Her kaynağı SIRAYLA işle. Çıktı kökü kaynağın içine bağlanır, yani
    #    her veri seti kendi dicoms/nifti/segmentation/overlay/features ağacını
    #    yanında taşır; kaynaklar birbirinin raporunu ezemez.
    per_source: dict[Path, list[str]] = {}
    for index, src in enumerate(sources, 1):
        config.set_output_root(src)
        config.ensure_dirs()
        fx._KEY_INDEX = None
        if interim or verify_only:
            keys = completed_keys() if interim else _manifest_keys_at(config.FEATURES)
            print(f"[koşum] ({index}/{len(sources)}) {src.name}: {len(keys)} seri "
                  f"(işleme yok) -> {config.DATA}")
        else:
            print("\n" + "-" * 70)
            print(f"[koşum] ({index}/{len(sources)}) KAYNAK: {src}")
            print(f"[koşum] Çıktı kökü: {config.DATA}")
            print("-" * 70)
            code = batch.main(src, limit=limit, only=only, plan_only=plan_only)
            if code != 0:
                print(f"[koşum] UYARI: {src.name} sıfırdan farklı kod döndürdü ({code}).")
            keys = _manifest_keys_at(config.FEATURES)
        per_source[src] = keys

    if plan_only:
        print("\n[koşum] --plan-only: tarama ve ön kontrol yapıldı, işleme yok.")
        return 0

    # 2) Her kaynak KENDİ tablosunu derler ve KENDİ raporuyla doğrulanır.
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    reports: list[Path] = []

    for src, keys in per_source.items():
        config.set_output_root(src)
        fx._KEY_INDEX = None
        if limit or only or interim:
            # Filtreli/yarım koşumda MANIFEST tüm serileri listeler ama yalnız
            # bir kısmı işlendi; derleme ve denetim işlenenleri kapsamalı.
            keys = [k for k in keys if fx.series_is_complete(k)]

        print("\n" + "=" * 70)
        print(f"[koşum] {src.name}: {len(keys)} seri birleştiriliyor...")
        batch.combine_features(keys)

        print(f"[doğrulama] {src.name} denetleniyor...")
        e, w, i = verify(keys)
        reports.append(write_verify_report(e, w, i))
        errors += [f"[{src.name}] {x}" for x in e]
        warnings += [f"[{src.name}] {x}" for x in w]
        info += [f"[{src.name}] {x}" for x in i]

    # Kaynaklar arası TEK tablo üretilmez: her kaynak klasörü tek bir cihazın
    # tek bir tarihteki çekimidir, dolayısıyla birleşik tablonun kapsamı da
    # o klasörle sınırlıdır. Cihazları karşılaştırmak isteyen, iki klasörün
    # all_features_combined dosyasını analiz tarafında birleştirir.

    for line in info:
        print(f"  {line}")
    if warnings:
        print(f"\n[doğrulama] {len(warnings)} UYARI:")
        for w in warnings[:20]:
            print(f"  ! {w}")
        if len(warnings) > 20:
            print(f"  ... ve {len(warnings) - 20} uyarı daha (bkz. {VERIFY_REPORT})")
    if errors:
        print(f"\n[doğrulama] {len(errors)} ENGELLEYİCİ HATA:")
        for e in errors[:20]:
            print(f"  X {e}")
        if len(errors) > 20:
            print(f"  ... ve {len(errors) - 20} hata daha (bkz. {VERIFY_REPORT})")

    print("\n" + "=" * 70)
    print("[doğrulama] KARAR: " + ("TEMİZ ✓" if not errors else "HATA VAR ✗"))
    for r in reports:
        print(f"[doğrulama] Rapor: {r}")
    print(f"[koşum] Toplam süre: {(time.time() - t0) / 60:.1f} dk")
    return 0 if not errors else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="İki cihazı işleyip çıktıyı doğrulayan koşum aracı")
    ap.add_argument("sources", nargs="*", default=None,
                    help="Ham DICOM kök klasörleri (boşsa varsayılan iki kaynak)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Kaynak başına ilk N seri (duman testi)")
    ap.add_argument("--only", default=None, help="Adında bu geçen serileri işle")
    ap.add_argument("--plan-only", action="store_true",
                    help="Yalnız ön kontrol + tarama/rapor, işleme yok")
    ap.add_argument("--verify-only", action="store_true",
                    help="İşleme yok; mevcut çıktıyı denetle")
    ap.add_argument("--interim", action="store_true",
                    help="Yarıda bırakılmış koşumun ara durumu: yalnız "
                         "tamamlanmış serileri topla ve denetle")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    srcs = [Path(s) for s in args.sources] if args.sources else default_sources()
    raise SystemExit(main(srcs, args.limit, args.only, args.plan_only,
                          args.verify_only, args.interim))
