"""Adım 3 — MIRP ile IBSI-uyumlu radiomics öznitelik çıkarımı (organ-paralel, RAM-güvenli).

Strateji:
 * Seriler SIRALI işlenir (biri bitince diğerine) → her seri mümkün olan en kısa
   sürede tamamlanır ve CSV'si hemen yazılır.
 * Bir serinin organları (ROI) işçi süreçlere BÖLÜNÜR → tek bir serinin 117 organı
   tüm çekirdeklerde paralel hesaplanır (MIRP kendi başına organları paralel etmez).
 * İşçi sayısı RAM bütçesine göre kısılır (her işçi CT'nin 1 mm kopyasını tutar).
 * OOM (işçi çökmesi) olursa işçi sayısı yarıya inip yeniden denenir; en sonda
   organlar tek tek izole edilir, yalnızca sorunlu organ atlanıp raporlanır.

Python 3.14 (mirp 2.5.0) altında koşar.
"""

from __future__ import annotations

# İşçi başına BLAS/SITK iş parçacıklarını sınırla (bellek + CPU aşırı-abonelik
# kontrolü). numpy/mirp import edilmeden ÖNCE ayarlanmalı; spawn'da çocuk süreçler
# bu ortamı miras alır.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import hashlib
import json
import subprocess
import multiprocessing as mp
import queue as _queue
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import mirp

import config


# --------------------------------------------------------------------------- #
# Etiket / ROI yardımcıları
# --------------------------------------------------------------------------- #
def present_labels(seg_path: Path) -> list[int]:
    """Multilabel maskede bulunan sıfırdan farklı etiketleri döndürür."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
    return sorted(int(v) for v in np.unique(arr) if v != 0)


_CANONICAL_NAMES: set[str] | None = None


def _canonical_structure_names() -> set[str]:
    """TS görevinin geçerli yapı adları (repo ile gelen statik haritadan)."""
    global _CANONICAL_NAMES
    if _CANONICAL_NAMES is None:
        try:
            shipped = config.PROJECT_ROOT / "totalseg_total_labels.json"
            _CANONICAL_NAMES = set(
                json.loads(shipped.read_text(encoding="utf-8")).values())
        except Exception:  # noqa: BLE001 — harita yoksa doğrulama atlanır
            _CANONICAL_NAMES = set()
    return _CANONICAL_NAMES


def selected_organs() -> set[str] | None:
    """config.TARGET_ORGANS — doğrulanmış; None ise (tüm yapılar) None döner.

    Yapılandırmadaki bir ad TS görevinde YOKSA burada net bir hata verilir.
    Sessizce filtrelenmesi çok daha kötü olurdu: tek bir yazım hatası 0 organlık
    bir çıkarıma yol açar, seri 'tamamlandı' işaretlenir ve resume onu bir daha
    hiç denemez — yani veri sessizce kaybolur.

    Doğrulama çağıranın verdiği haritaya değil KANONİK listeye bakar: kısmi bir
    harita (testler, tek organlı çağrılar) yanlış alarm vermesin.
    """
    targets = getattr(config, "TARGET_ORGANS", None)
    if not targets:
        return None
    known = _canonical_structure_names()
    unknown = sorted(set(targets) - known) if known else []
    if unknown:
        raise ValueError(
            f"config.TARGET_ORGANS bilinmeyen yapı adı içeriyor: {', '.join(unknown)}. "
            f"Geçerli adlar totalseg_total_labels.json içinde (TS görevi: {config.TS_TASK})."
        )
    return set(targets)


def target_labels(label_map: dict[str, str]) -> set[int] | None:
    """Seçilen yapıların etiket numaraları (None → tümü). QC overlay için."""
    selected = selected_organs()
    if selected is None:
        return None
    return {int(k) for k, v in label_map.items() if v in selected}


def build_roi_map(seg_path: Path, label_map: dict[str, str]) -> dict[str, str]:
    """{etiket(str): yapı_adı} — maskede bulunan VE seçilmiş yapılar için.

    Seçim config.TARGET_ORGANS ile ve AD üzerinden yapılır (etiket numarası
    üzerinden değil), böylece TS sürümü etiketleri yeniden numaralandırsa bile
    yanlış organa öznitelik yazılamaz.
    """
    selected = selected_organs()
    roi_map: dict[str, str] = {}
    for label in present_labels(seg_path):
        key = str(label)
        name = label_map.get(key, f"label_{key}")
        if selected is not None and name not in selected:
            continue
        roi_map[key] = name
    return roi_map


def _split_roi_map(roi_map: dict[str, str], n_chunks: int) -> list[dict[str, str]]:
    """roi_map'i ~eşit büyüklükte n öbeğe böler."""
    items = list(roi_map.items())
    n = max(1, min(n_chunks, len(items)))
    base, extra = divmod(len(items), n)
    chunks: list[dict[str, str]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        chunks.append(dict(items[start : start + size]))
        start += size
    return chunks


# --------------------------------------------------------------------------- #
# Bellek (RAM) bütçesi
# --------------------------------------------------------------------------- #
def available_memory_bytes() -> int:
    """RAM bütçesi (byte).

    psutil varsa: min(TOPLAM×MEM_TOTAL_FRACTION, BOŞ×MEM_AVAIL_FRACTION) — hem
    OS'e pay bırakır hem anlık boş RAM'e uyum sağlar. psutil yoksa MAX_MEMORY_GB.

    Bütçe `config.SERIES_PARALLELISM`'e BÖLÜNÜR: birden çok seri aynı anda
    işlendiğinde her biri psutil'e ayrı ayrı sorup aynı boş RAM'i kendi payı
    sanıyor ve toplam bütçe eşzamanlılık katına çıkıyordu (aşırı-abonelik →
    OOM zinciri). Bölme, her serinin yalnız kendi payını görmesini sağlar.
    """
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        budget = min(vm.total * config.MEM_TOTAL_FRACTION,
                     vm.available * config.MEM_AVAIL_FRACTION)
    except Exception:
        budget = config.MAX_MEMORY_GB * (1024 ** 3)
    pay = max(1, int(getattr(config, "SERIES_PARALLELISM", 1) or 1))
    return int(budget / pay)


def _resampled_bytes(ct_path: Path, new_spacing: float) -> int:
    """CT'yi 1 mm'ye yeniden örnekleyince oluşacak float32 hacmin ~byte boyutu.

    Piksel yüklemeden yalnızca başlıktan okunur.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(ct_path))
    reader.ReadImageInformation()
    size = reader.GetSize()
    spacing = reader.GetSpacing()
    voxels = 1
    for s, sp in zip(size, spacing):
        voxels *= max(1, int(round(s * sp / float(new_spacing))))
    return voxels * 4  # float32


def _plan_workers(ct_path: Path) -> tuple[int, int, int]:
    """(işçi_sayısı, RAM_bütçesi, resample_boyutu) — header ve psutil TEK sefer okunur."""
    spacing = float(config.MIRP_SETTINGS.get("new_spacing", 1.0))
    resampled = _resampled_bytes(ct_path, spacing)
    budget = available_memory_bytes()
    per_worker = config.WORKER_MEM_MULTIPLIER * resampled
    workers = max(1, min(int(config.PARALLEL_WORKERS),
                         int(budget // max(1.0, per_worker))))
    return workers, budget, resampled


def _safe_worker_count(ct_path: Path) -> int:
    """RAM bütçesine göre bu seri için güvenli işçi sayısı."""
    return _plan_workers(ct_path)[0]


# --------------------------------------------------------------------------- #
# İşçi süreç (top-level olmalı — Windows spawn için picklable)
# --------------------------------------------------------------------------- #
def _extract_chunk_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Tek bir (seri, organ-öbeği) iş birimi için MIRP çağrısı."""
    series_key = task["series_key"]
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(task.get("sitk_threads", 1)))
    except Exception:
        pass
    try:
        # Mod ayarları ORTAK ayarların ÜSTÜNE yazılır. MIRP_SETTINGS mod-bağımsız
        # tutuluyor (yeniden örnekleme, ayrıklaştırma, resegmentation); modun
        # değiştirdiği tek şey `by_slice` ve doku toplama yöntemleri.
        mod = task.get("mode", config.PRIMARY_MODE)
        result = mirp.extract_features(
            image=task["ct_path"],
            mask=task["seg_path"],
            sample_name=series_key,
            roi_name=task["roi_map"],
            export_features=True,
            write_features=False,
            num_cpus=config.MIRP_NUM_CPUS,
            parallel_backend=config.MIRP_PARALLEL_BACKEND,
            **config.MIRP_SETTINGS,
            **config.MIRP_MODES[mod],
        )
        if not result:
            return {"series_key": series_key, "df": None, "error": "öznitelik üretilmedi"}
        df = pd.concat(result, ignore_index=True)
        return {"series_key": series_key, "df": df, "error": None}
    except MemoryError:
        return {"series_key": series_key, "df": None, "error": "MemoryError"}
    except Exception as exc:  # noqa: BLE001
        return {"series_key": series_key, "df": None, "error": repr(exc)}


# --------------------------------------------------------------------------- #
# Yürütme — RAM-farkında geri çekilme (backoff) + izolasyon
# --------------------------------------------------------------------------- #
def _watchdog_seconds(tasks: list[dict[str, Any]], floor: float = 1800.0) -> float:
    """Takılma-koruması süresi (organ sayısına göre ölçeklenir).

    `floor`: alt sınır. Havuz koşusunda cömert (1800 sn) çünkü tüm öbekler
    paylaşımlı çalışır. İZOLASYONDA (tek organ) bu taban çok cömert kalıyordu
    (30 dk/organ → 6 organ = 3 saat); izolasyon daha sıkı bir taban kullanır.
    """
    n_organs = sum(len(t.get("roi_map", {})) for t in tasks) or 1
    return max(floor, config.MIRP_WATCHDOG_PER_ORGAN_S * n_organs)


def _run_pool(tasks: list[dict[str, Any]], workers: int) -> tuple[list[dict], list[dict]]:
    """Görevleri `workers` işçiyle çalıştırır.

    Döndürür: (tamamlanan_sonuçlar, tamamlanamayan_görevler). Havuz çökerse (OOM),
    watchdog süresi aşılırsa (takılma) veya çok-organlı bir öbek HATA/MemoryError
    dönerse → o görevler `incomplete`'e gider ve _execute tarafından daha az
    eşzamanlılıkla / tek tek izole edilerek yeniden denenir. Yalnızca TEK-organlı
    hata sonuçları terminal (df=None) kabul edilir; hiçbiri sessizce kaybolmaz.
    """
    results: list[dict] = []
    done: set[int] = set()
    timed_out = False
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        fut_to_idx = {executor.submit(_extract_chunk_worker, t): i for i, t in enumerate(tasks)}
        try:
            for fut in as_completed(fut_to_idx, timeout=_watchdog_seconds(tasks)):
                i = fut_to_idx[fut]
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    continue  # işçi öldü → incomplete
                except Exception as exc:  # noqa: BLE001
                    res = {"series_key": tasks[i]["series_key"], "df": None, "error": repr(exc)}
                # Başarısız (df=None) HER öbek — tek-organlı dahil — terminal sayılmaz;
                # incomplete'e gider → _execute daha az eşzamanlılıkla / izole ederek dener.
                # (Tek-organlı öbek de havuzda eşzamanlı çalışırken OOM olabilir; izolasyon
                #  tek işçiyle başarabilir.) Terminal karar YALNIZCA izolasyonda verilir.
                if res.get("df") is None:
                    continue
                results.append(res)
                done.add(i)
        except FuturesTimeout:
            timed_out = True
            print("[mirp]   UYARI: watchdog süresi aşıldı (MIRP takılması olası); "
                  "takılan işçiler sonlandırılıp kalan öbekler izolasyonla denenecek.")
        except BrokenProcessPool:
            pass
    finally:
        # Watchdog takılmasında: cancel_futures çalışan işçiyi durdurmaz; belleği geri
        # almak için çocuk süreçleri AÇIKÇA öldür (yoksa ~5.8 GB sızar → sonraki OOM zinciri).
        # (BrokenProcessPool/OOM'da CPython işçileri zaten sonlandırır.)
        if timed_out:
            _kill_pool_workers(executor)
        executor.shutdown(wait=False, cancel_futures=True)
    incomplete = [t for i, t in enumerate(tasks) if i not in done]
    return results, incomplete


def _kill_pool_workers(executor: ProcessPoolExecutor) -> None:
    """Takılan havuzun çocuk süreçlerini zorla sonlandırır (belleği geri al)."""
    try:
        for proc in list(getattr(executor, "_processes", {}).values()):
            if proc.is_alive():
                try:
                    proc.kill()      # terminate değil: takılan MIRP'i kesin durdur
                except Exception:  # noqa: BLE001
                    proc.terminate()
        for proc in list(getattr(executor, "_processes", {}).values()):
            try:
                proc.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def _worker_entry(task: dict[str, Any], q) -> None:
    """İzole süreç giriş noktası: sonucu Queue'ya koyar (spawn için modül düzeyinde)."""
    try:
        res = _extract_chunk_worker(task)
    except MemoryError:
        res = {"series_key": task.get("series_key"), "df": None, "error": "MemoryError"}
    except Exception as exc:  # noqa: BLE001
        res = {"series_key": task.get("series_key"), "df": None, "error": repr(exc)}
    try:
        q.put(res)
    except Exception:  # noqa: BLE001 — parent kapanmış olabilir
        pass


def _run_one_process(task: dict[str, Any], timeout_s: float) -> tuple[dict | None, bool]:
    """Tek görevi KENDİ sürecinde çalıştırır — ProcessPoolExecutor KULLANMADAN.

    NEDEN (gerçek arıza): ProcessPoolExecutor'ün işçisini terminate/kill etmek
    havuzun iç çağrı-kuyruğu ve kilitlerini bozuk bırakıyor; sonraki submit()
    parent'ı KALICI kilitliyordu (bir koşuda 11 saat askıda kaldı, watchdog dahil
    hiçbir şey kurtaramadı). Burada süreç doğrudan yönetilir ve HER çağrI TAZE
    Process+Queue kullanır → bozulma bir sonraki çağrıya SIZAMAZ.

    Çöken süreci beklemeden farkeder (is_alive kontrolü), böylece OOM'da tüm
    timeout boyunca boşuna beklemez.

    Döndürür (res, hung): hung=True → süre doldu, süreç öldürüldü.
                          res None & hung False → süreç çöktü (OOM).
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_entry, args=(task, q), daemon=False)
    p.start()
    res: dict | None = None
    hung = False
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                res = q.get(timeout=0.5)
                break
            except _queue.Empty:
                if not p.is_alive():          # sonuç yok + süreç ölü → çöktü (OOM)
                    break
                if time.monotonic() >= deadline:
                    hung = True               # takıldı → sert öldür
                    break
    finally:
        if p.is_alive():
            try:
                p.kill()                      # terminate değil: sert öldür
            except Exception:  # noqa: BLE001
                pass
        try:
            p.join(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        try:
            q.cancel_join_thread()            # feeder thread'i bekleme → kilitlenme yok
            q.close()
        except Exception:  # noqa: BLE001
            pass
    return res, hung


def _run_isolated(task: dict[str, Any]) -> tuple[dict | None, bool]:
    """Tek görevi izole süreçte çalıştırır (çökme/takılma izolasyonu).

    Döndürür (res, ok): res None ise İŞÇİ ÇÖKTÜ (OOM) veya TAKILDI; res bir hata
    sözlüğü ise MIRP hata döndürdü (df=None); aksi halde başarı.
    """
    floor = float(getattr(config, "MIRP_ISOLATED_FLOOR_S", 600.0))
    res, _hung = _run_one_process(task, _watchdog_seconds([task], floor=floor))
    if res is None:
        return None, False          # çöktü ya da takıldı (ikisi de izolasyonda terminal)
    return res, res.get("df") is not None


def _execute(tasks: list[dict[str, Any]], initial_workers: int) -> tuple[list[dict], list[str]]:
    """Görevleri RAM-farkında geri çekilmeyle çalıştırır.

    Döndürür: (sonuçlar, atlanan_organ_açıklamaları).
    """
    results: list[dict] = []
    failed: list[str] = []
    remaining = list(tasks)
    workers = max(1, initial_workers)

    while remaining:
        if workers >= 2:
            got, incomplete = _run_pool(remaining, workers)
            results.extend(got)
            if not incomplete:
                break
            new_workers = max(1, workers // 2)
            print(
                f"[mirp]   {len(incomplete)} iş birimi tamamlanamadı (muhtemelen RAM/OOM); "
                f"işçi {workers} -> {new_workers}, yeniden deneniyor."
            )
            workers = new_workers
            remaining = incomplete
        else:
            # Tek işçi: her görevi izole et; çöken/hatalı çok-organlı öbeği tek organlara böl.
            for task in remaining:
                res, ok = _run_isolated(task)
                organs = list(task["roi_map"].items())
                if ok:
                    results.append(res)
                elif len(organs) == 1:
                    _record_terminal(organs[0][1], res, task["series_key"], results, failed)
                else:
                    print(f"[mirp]   {task['series_key']}: {len(organs)}'lı öbek izole ediliyor "
                          f"({'çökme/takılma' if res is None else res.get('error')}).")
                    for k, v in organs:
                        r2, ok2 = _run_isolated({**task, "roi_map": {k: v}})
                        if ok2:
                            results.append(r2)
                        else:
                            _record_terminal(v, r2, task["series_key"], results, failed)
            remaining = []
    return results, failed


def _record_terminal(organ: str, res: dict | None, key: str,
                     results: list, failed: list) -> None:
    """İzolasyonda da başarısız tek organı raporla; çökme (RAM) ile MIRP hatasını ayır."""
    if res is None:
        reason = "işlem çöktü/takıldı (RAM/OOM ya da MIRP takılması olası)"
    else:
        reason = res.get("error", "bilinmeyen MIRP hatası")
        results.append(res)  # error-dict; df=None, missing-set de yakalar
    failed.append(f"{key}/{organ}")
    print(f"[mirp]   ATLANDI: {key} / {organ}  ({reason})")


# --------------------------------------------------------------------------- #
# Sonuç birleştirme yardımcıları
# --------------------------------------------------------------------------- #
def _dicom_columns_for(key: str, meta: dict[str, Any]) -> dict[str, str]:
    """Serinin DICOM başlık sütunları; kaynak yoksa boş sözlük.

    Metadata eksikliği bir çıkarımı durdurmamalı — öznitelikler NIfTI'den
    geliyor, DICOM yalnızca bağlam ekliyor.
    """
    if not meta.get("series_dir"):
        return {}
    try:
        import dicom_metadata
        return dicom_metadata.as_columns(dicom_metadata.load_or_harvest(key, meta))
    except Exception as exc:  # noqa: BLE001
        print(f"[mirp] {key}: DICOM metadata okunamadı ({exc}); sütunlar atlanıyor.")
        return {}


def _finalize_series_frame(df: pd.DataFrame, key: str, patient_id: str | None,
                          recon_type: str = "native", kvp: float | None = None,
                          dicom_columns: dict[str, str] | None = None) -> pd.DataFrame:
    df = drop_unreportable(df).copy()
    if "image_mask_label" in df.columns:
        df = df.sort_values("image_mask_label").reset_index(drop=True)
    df.insert(0, "series_key", key)
    df.insert(1, "patient_id", patient_id or key.split("__", 1)[0])
    df.insert(2, "recon_type", recon_type)   # native / VMI_45keV / VMI_70keV (PCCT)
    df.insert(3, "kvp", kvp)                  # etkin (rekonstrüksiyon) kVp

    # Çekim parametreleri (kernel, mAs, FOV, cihaz...) doğrudan tabloda dursun ki
    # bir satırın hangi protokolden geldiği ham DICOM'a dönmeden görülebilsin.
    # Sıra dicom_metadata.harvest()'ten geliyor (önce en çok bakılanlar); alfabetik
    # sıralamak onu bozar.
    for offset, (column, value) in enumerate((dicom_columns or {}).items()):
        df.insert(4 + offset, column, value)
    return df


# MIRP bu iki özniteliği 1000 vokselin üstünde Monte-Carlo ile KESTİRİR
# (no_approximation=False). Tam hesap O(n²) ve bu boyutlarda pratik değil.
# İki sonucu var: (1) değerler IBSI'nin tanımladığı kesin hesap değil,
# (2) tohum dosya YOLUNU içerdiği için aynı veriden yeniden çıkarım farklı
# sayılar üretir — klasör yeniden adlandırıldığında ya da veri taşındığında.
# Tekrarlanamayan bir sayıyı tabloda tutmak, onu kullanılabilir sanmaya yol
# açıyor; bu yüzden çıkarılıyor. Kalan öznitelikler kesin hesaplanıyor.
APPROXIMATED_FEATURES = ("morph_moran_i", "morph_geary_c")

# IVH HACİM KESİRLERİ — tabloya alınmıyor.
#
# `ivh_discretisation_method="none"` iken MIRP, IVH'nin yoğunluk eksenini
# RESEGMENTASYON aralığından alıyor (mirp/_images/base_mask.py:428). Bizim
# aralık [-1300, 3100] HU; kesim noktaları v10=-860, v25=-200, v50=+900,
# v75=+2000, v90=+2660 HU'ya düşüyor. Yumuşak dokuda gözlenen span çok daha
# dar (ölçüldü, 284 satır: stat_min -729..-29, stat_max 136..1100), yani beş
# kesimin dördü veriye hiç değmiyor.
#
# ÖLÇÜLDÜ (284 satır): ivh_v10 ≡ 1.0, ivh_v75 ≡ 0.0, ivh_v90 ≡ 0.0,
# ivh_diff_v10_v90 ≡ 1.0, ivh_v50 ≈ 0.0; ayrıca ivh_diff_v10_v90 ≡ ivh_v10 ve
# ivh_diff_v25_v75 ≡ ivh_v25 (284/284 satırda max|fark| = 0.0).
#
# Bu sütunlarda ICC, CV, korelasyon ve model katsayısı tanımsız ya da
# anlamsız. Kardeş hat (pcct-radiomics-mirp) da bunları raporlamıyor; referans
# çalışma kitabında yalnız yedi `ivh_i*` sütunu var. Yoğunluk kesirleri
# (ivh_i10..i90, ivh_diff_i*) TABLODA KALIYOR — onlar bilgi taşıyor.
IVH_VOLUME_FRACTION_FEATURES = (
    "ivh_v10", "ivh_v25", "ivh_v50", "ivh_v75", "ivh_v90",
    "ivh_diff_v10_v90", "ivh_diff_v25_v75",
)

DROPPED_FEATURES = APPROXIMATED_FEATURES + IVH_VOLUME_FRACTION_FEATURES


def drop_unreportable(df: pd.DataFrame) -> pd.DataFrame:
    """Raporlanabilir olmayan öznitelikleri düşürür.

    İki ayrı gerekçe, tek çıkış: (1) Monte-Carlo ile kestirilenler
    tekrarlanamıyor, (2) IVH hacim kesirleri bu resegmentasyon aralığı altında
    sabitlenmiş durumda ve iki tanesi başka bir sütunun birebir kopyası.
    """
    atilacak = [c for c in DROPPED_FEATURES if c in df.columns]
    return df.drop(columns=atilacak) if atilacak else df


# Eski ad — dış betikler ve daha önce yazılmış çağrılar kırılmasın.
drop_approximated = drop_unreportable


DONE_SIDECAR = "features.done.json"   # tamamlanma işareti (resume bunu kontrol eder)


def _atomic_write_text(path: Path, text: str) -> None:
    """Geçici dosyaya yazıp atomik taşır (yarım dosya bırakmaz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def mode_csv_name(mode: str) -> str:
    """Bir modun seri tablosunun dosya adı.

    Ana mod (3D) adını KORUYOR — `features.csv`. Daha önce üretilmiş bütün
    çıktılar, testler ve dış betikler bu adı bekliyor; 2B'yi eklemek için onu
    yeniden adlandırmak gereksiz bir kırılma olurdu.
    """
    return "features.csv" if mode == config.PRIMARY_MODE else f"features_{mode.lower()}.csv"


def _write_series_csv(df: pd.DataFrame, key: str,
                      mode: str | None = None) -> Path:
    """Seri tablosunu ATOMİK yazar (yazım kesilirse yarım dosya kalmaz)."""
    out_csv = config.FEATURES / key / mode_csv_name(mode or config.PRIMARY_MODE)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_name(out_csv.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    print(f"[mirp] {key}: {len(df)} satır -> {out_csv}")
    return out_csv


def _library_versions() -> dict[str, str]:
    """Öznitelik değerlerini belirleyen kütüphanelerin sürümleri."""
    import importlib.metadata as md

    versions: dict[str, str] = {}
    for package in ("mirp", "numpy", "scipy", "pandas", "SimpleITK", "nibabel", "pydicom"):
        try:
            versions[package] = md.version(package)
        except Exception:
            versions[package] = "not-installed"
    versions.update(totalsegmentator_versions())
    return versions


_TS_VERSIONS: dict[str, str] | None = None


def totalsegmentator_versions() -> dict[str, str]:
    """TotalSegmentator ve torch sürümleri — maskeyi ÜRETEN yazılım.

    Bunlar başka bir Python ortamında kurulu olduğu için subprocess ile okunur.
    Provenansta olmaları şart: maske sınırı değişirse HER öznitelik değişir,
    yani segmentasyon sürümü öznitelik değerlerinin bir parçası. Beyanda
    yalnız 'TotalSegmentator' yazıyordu, sürüm hiçbir yerde kayıtlı değildi.
    Okunamazsa 'unknown' yazılır — TAHMİN EDİLMEZ.
    """
    global _TS_VERSIONS
    if _TS_VERSIONS is not None:
        return _TS_VERSIONS
    sonuc = {"totalsegmentator": "unknown", "torch": "unknown"}
    py = config.PY311
    if py is not None and Path(py).exists():
        kod = ("import json,importlib.metadata as m\n"
               "d={}\n"
               "for p in ('totalsegmentator','torch'):\n"
               "    try: d[p]=m.version(p)\n"
               "    except Exception: d[p]='not-installed'\n"
               "print(json.dumps(d))")
        try:
            r = subprocess.run([str(py), "-c", kod], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
            if r.returncode == 0 and r.stdout.strip():
                sonuc.update(json.loads(r.stdout.strip().splitlines()[-1]))
        except Exception:  # noqa: BLE001 — provenans okunamaması koşumu düşürmemeli
            pass
    _TS_VERSIONS = sonuc
    return sonuc


def _settings_hash() -> str:
    """MIRP ayarlarının parmak izi.

    Ayarların sessizce kayması, sonuçları karşılaştırılamaz hale getiriyor:
    ölçüldü, yalnızca anti_aliasing'i değiştirmek 177 özelliğin 145'ini
    değiştiriyor. Hash sayesinde iki çıktının aynı yapılandırmadan gelip
    gelmediği sonradan da doğrulanabiliyor.
    """
    # Mod ayarları da hash'e GİRER. Girmezse `3d_average`'ın eklenmesi ya da 2B
    # modunun açılıp kapanması sessizce geçer ve eski çıktılar "güncel" sayılır.
    payload = json.dumps({"base": config.MIRP_SETTINGS, "modes": config.MIRP_MODES},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _input_checksum(path: Path) -> str:
    """Girdi dosyasının sha256'sı (kısaltılmış).

    NIfTI önbelleği içerik-adresli değil: anahtar klasör adı + seri UID'sinden
    üretiliyor, dolayısıyla kaynak seri değişse bile ct.nii.gz yeniden
    üretilmiyor. Checksum, bir sonucun hangi girdiden geldiğini sonradan
    kanıtlanabilir kılıyor.
    """
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _write_done_sidecar(key: str, expected: set[str], present: set[str],
                        failed: list[str], *,
                        absent_targets: list[str] | None = None) -> None:
    """Seri sonucunu (beklenen/üretilen/eksik organlar) kalıcı olarak yazar.

    Bu dosya hem RESUME işareti hem de 'hiçbir organ sessizce kaybolmasın'
    garantisidir: eksik organlar burada ve toplu raporda görünür.

    `absent_targets`: hedeflenmiş ama maskede hiç bulunmayan organlar. Bunlar
    `missing_organs`'tan AYRI tutuluyor: orası çıkarımın başarısız olduğu
    organlar, burası TS'in hiç üretmediği organlar — nedenleri ve çözümleri
    farklı, karıştırılırsa hangisi olduğu anlaşılmaz.
    """
    missing = sorted(expected - present)
    targets = getattr(config, "TARGET_ORGANS", None)
    payload = {
        "series_key": key,
        "n_expected_organs": len(expected),
        "n_present_organs": len(present),
        "missing_organs": missing,
        "complete": not missing,
        "failed_units": failed,
        # Hangi organ kümesi hedeflendi (None → tümü). Bir çıktının dar bir
        # seçimle mi tüm yapılarla mı üretildiği sonradan da görülebilsin.
        "target_organs": sorted(targets) if targets else None,
        "absent_target_organs": sorted(absent_targets or []),
        # Provenans: bir öznitelik değeri ancak onu üreten kod ve ayarlarla
        # birlikte tekrarlanabilir. Bu proje birden fazla sitede koştuğu için
        # sürüm ve ayar kaymasının sonradan tespit edilebilmesi şart.
        "mirp_version": _library_versions().get("mirp", "?"),
        "library_versions": _library_versions(),
        "settings_hash": _settings_hash(),
        "mirp_settings": {k: v for k, v in sorted(config.MIRP_SETTINGS.items())},
        # Hangi modlar üretildi ve hangi dosyaya yazıldı. Resume bunu okuyup
        # yeni bir mod eklendiğinde seriyi yeniden işliyor.
        "modes": {m: mode_csv_name(m) for m in config.MIRP_MODES},
        "mode_settings": {m: dict(sorted(s.items()))
                          for m, s in config.MIRP_MODES.items()},
        "input_checksum": {
            "ct": _input_checksum(config.NIFTI / key / "ct.nii.gz"),
            "seg": _input_checksum(config.SEG / key / "seg.nii.gz"),
        },
    }
    _atomic_write_text(config.FEATURES / key / DONE_SIDECAR,
                       json.dumps(payload, indent=2, ensure_ascii=False))
    if missing:
        print(f"[mirp] {key}: UYARI — {len(missing)} organ eksik: {', '.join(missing)}")


_KEY_INDEX: dict[str, Path] | None = None


def _key_index() -> dict[str, Path]:
    """series_key -> tamamlanma sidecar'ını içeren klasör (bir kez taranır, önbelleklenir).

    apply_naming klasörü <series_key>'ten insan-okunur ada TAŞIDIĞINDA bile resume
    çalışsın diye: klasör adı değişse de sidecar içindeki 'series_key' ile bulunur.
    Yalnız doğrudan yol bulunamayınca (miss) danışılır → normal akışı yavaşlatmaz.
    """
    global _KEY_INDEX
    if _KEY_INDEX is None:
        idx: dict[str, Path] = {}
        try:
            for sc in config.FEATURES.glob(f"*/{DONE_SIDECAR}"):
                try:
                    d = json.loads(sc.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                sk = d.get("series_key")
                if sk:
                    idx[sk] = sc.parent
        except Exception:  # noqa: BLE001
            pass
        _KEY_INDEX = idx
    return _KEY_INDEX


def series_is_complete(key: str) -> bool:
    """Resume için: seri daha önce işlenip *anlamlı sonuç* kaydedilmiş mi.

    Yalnızca işaret dosyasının VARLIĞINA bakmak yetmez: HİÇ organ üretmemiş
    (present=0, expected>0) bir seri geçici bir OOM/takılma yüzünden başarısız
    olmuş olabilir → yeniden denenmelidir. Bu yüzden içerik okunur:
      - sidecar yok           → tamamlanmadı (işle)
      - present=0 ve expected>0 → geçici tam başarısızlık → tamamlanmadı (yeniden dene)
      - aksi halde (tam, kısmi ya da boş-maske) → tamamlandı (atla; eksikler raporlu)

    Klasör apply_naming ile yeniden adlandırılmışsa sidecar series_key ile bulunur.
    """
    side = config.FEATURES / key / DONE_SIDECAR
    if not side.exists():
        folder = _key_index().get(key)          # yeniden adlandırılmış olabilir
        if folder is None:
            return False
        side = folder / DONE_SIDECAR
        if not side.exists():
            return False
    try:
        d = json.loads(side.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — bozuk sidecar → yeniden işle
        return False
    if d.get("n_present_organs", 0) == 0 and d.get("n_expected_organs", 0) > 0:
        return False

    # PROVENANS DENETİMİ. Sidecar bu üç alanı zaten yazıyordu ama hiçbir yerde
    # geri okunmuyordu; sonuç: ayarlar ya da organ seçimi değişince eski
    # sonuçlar sessizce "tamamlanmış" sayılıp tabloya giriyordu. Ölçüldü —
    # 142 seri `e1dd5f4cb457fe6e` hash'iyle duruyor, canlı yapılandırma
    # `95aeeb0498213b8c`; sidecar 5 organ diyor, config 2.
    neden = _stale_reason(d)
    if neden:
        print(f"[fx] {key[-46:]}: yeniden işlenecek — {neden}")
        return False

    # Her modun tablosu DİSKTE duruyor mu? Sidecar atomik yazılıyor ama modlar
    # sırayla yazıldığı için ikinci mod yazılmadan kesilme mümkün.
    klasor = side.parent
    for mod in config.MIRP_MODES:
        if not (klasor / mode_csv_name(mod)).exists():
            print(f"[fx] {key[-46:]}: yeniden işlenecek — "
                  f"{mode_csv_name(mod)} yok ({mod} modu)")
            return False
    return True


def _stale_reason(d: dict[str, Any]) -> str | None:
    """Sidecar bugünkü yapılandırmayla uyuşuyor mu? Uyuşmuyorsa sebebi döner."""
    eski = d.get("settings_hash")
    yeni = _settings_hash()
    if eski and eski != yeni:
        return f"MIRP ayarları değişti ({eski} -> {yeni})"

    # Bir mod SONRADAN eklendiyse (2B gibi) ayar hash'i zaten değişir; ama
    # hash aynı kalsa bile dosyanın DİSKTE olduğunu doğrulamak gerekiyor:
    # yarım kalmış bir koşumda sidecar yazılmış ama ikinci modun CSV'si
    # yazılmamış olabilir.
    kayitli_modlar = d.get("modes")
    if kayitli_modlar is not None:
        eksik_mod = sorted(set(config.MIRP_MODES) - set(kayitli_modlar))
        if eksik_mod:
            return f"yeni mod eklendi: {', '.join(eksik_mod)}"

    # MIRP SÜRÜMÜ. Ayar hash'i yalnız BİZİM verdiğimiz parametreleri kapsıyor;
    # öznitelik tanımlarını uygulayan kod değişirse hash aynı kalır ve iki
    # farklı sürümle üretilmiş satırlar tek tabloda birleşir. Ölçüldü: diskte
    # 176 seri mirp 2.5.0 ile, 3 seri 2.6.0 ile üretilmişti ve resume farkı
    # görmüyordu. IBSI referans doğrulaması (1804/1804) 2.5.0'a ait.
    eski_mirp = (d.get("library_versions") or {}).get("mirp")
    yeni_mirp = _library_versions().get("mirp")
    if eski_mirp and yeni_mirp and eski_mirp != yeni_mirp:
        return f"MIRP sürümü değişti ({eski_mirp} -> {yeni_mirp})"

    # GİRDİ DEĞİŞTİ Mİ. Sidecar `input_checksum`'ı yazıyordu ama hiçbir yerde
    # geri okunmuyordu. Sonucu: bir seriye slice eklenir ya da seri yeniden
    # dışa aktarılırsa ct.nii.gz/seg.nii.gz değişir, öznitelikler ESKİ hacme
    # ait kalır ve resume seriyi "tamamlanmış" sayar. Meslektaşın makinesinde
    # bu, sessizce yanlış bir tabloya dönüşür.
    #
    # Maliyet: seri başına iki dosyanın sha256'sı (~95 MB + ~1 MB). Ölçüldü,
    # seri başına saniyenin onda birleri; yalnız resume yolunda ödeniyor.
    kayitli_ck = d.get("input_checksum") or {}
    anahtar = d.get("series_key")
    if kayitli_ck and anahtar:
        simdiki = {"ct": _input_checksum(config.NIFTI / anahtar / "ct.nii.gz"),
                   "seg": _input_checksum(config.SEG / anahtar / "seg.nii.gz")}
        for ad in ("ct", "seg"):
            eski_v, yeni_v = kayitli_ck.get(ad), simdiki.get(ad)
            # "missing" karşılaştırmaya girmez: dosya taşınmış/yeniden
            # adlandırılmış olabilir, bu bir içerik değişikliği değildir.
            if (eski_v and yeni_v and "missing" not in (eski_v, yeni_v)
                    and eski_v != yeni_v):
                return f"{ad} girdisi değişti ({eski_v[:8]} -> {yeni_v[:8]})"

    hedef = selected_organs()
    if hedef is not None:
        kayitli = d.get("target_organs")
        if kayitli is not None and set(kayitli) != set(hedef):
            eksik = sorted(set(hedef) - set(kayitli))
            if eksik:                       # yeni organ istendi, o seride yok
                return f"organ seçimi genişledi (eksik: {', '.join(eksik)})"
    return None


# --------------------------------------------------------------------------- #
# Bir seri için organ-paralel çıkarım
# --------------------------------------------------------------------------- #
def extract_for_series(meta: dict[str, Any]) -> pd.DataFrame | None:
    """Tek seriyi organ-paralel + RAM-güvenli işler; DataFrame döndürür ve CSV yazar."""
    key = meta["series_key"]
    ct_path = Path(meta["ct_path"])
    label_map = json.loads(Path(meta["label_map_path"]).read_text(encoding="utf-8"))
    roi_map = build_roi_map(Path(meta["seg_path"]), label_map)
    expected = set(roi_map.values())

    # Hedeflenip maskede HİÇ bulunmayan organlar. Bu bir çıkarım hatası değil
    # (TS yapıyı bulamamış ya da taranan bölge onu kapsamıyor) ama sessiz
    # kalmamalı: seçim daraldığında eksik bir organ tablonun yarısı demek.
    absent = sorted((selected_organs() or set()) - expected)
    if absent:
        print(f"[mirp] {key}: UYARI — hedef organ maskede yok: {', '.join(absent)}")

    if not roi_map:
        print(f"[mirp] {key}: istenen yapı maskede bulunamadı, atlanıyor.")
        # işaret yaz → resume tekrar denemesin
        _write_done_sidecar(key, set(), set(), [], absent_targets=absent)
        return None

    workers, budget, resampled = _plan_workers(ct_path)   # header/psutil TEK sefer
    n_chunks = min(workers, len(roi_map))
    sitk_threads = max(1, min(4, (os.cpu_count() or 4) // max(1, workers)))
    print(f"[mirp] {key}: {len(roi_map)} organ, {n_chunks} öbek, {workers} işçi, "
          f"{len(config.MIRP_MODES)} mod ({', '.join(config.MIRP_MODES)}) "
          f"(RAM bütçesi ~{budget / 1e9:.1f} GB, CT ~{resampled / 1e6:.0f} MB/kopya).")

    # Modlar SIRAYLA koşar. Paralel koşmak cazip ama her mod CT'nin kendi
    # yeniden örneklenmiş kopyasını tutuyor; ikisini birden açmak RAM bütçesini
    # ikiye katlar ve `_plan_workers` o bütçeye göre işçi sayısı seçmiş oluyor.
    primary_df: pd.DataFrame | None = None
    tum_failed: list[str] = []
    present: set[str] = set()
    for mod in config.MIRP_MODES:
        tasks = [
            {
                "series_key": key,
                "ct_path": str(ct_path),
                "seg_path": str(meta["seg_path"]),
                "roi_map": chunk,
                "sitk_threads": sitk_threads,
                "mode": mod,
            }
            for chunk in _split_roi_map(roi_map, n_chunks)
        ]
        results, failed = _execute(tasks, workers)
        for r in results:
            if r.get("error"):
                print(f"[mirp]   HATA {r['series_key']} [{mod}]: {r['error']}")
        tum_failed.extend(f"{mod}:{u}" for u in failed)
        dfs = [r["df"] for r in results if r.get("df") is not None]
        if not dfs:
            print(f"[mirp] {key} [{mod}]: sonuç üretilemedi.")
            if mod == config.PRIMARY_MODE:
                _write_done_sidecar(key, expected, set(), tum_failed,
                                    absent_targets=absent)
                return None
            continue

        merged = _finalize_series_frame(pd.concat(dfs, ignore_index=True), key,
                                        meta.get("patient_id"),
                                        recon_type=meta.get("recon_type", "native"),
                                        kvp=meta.get("kvp"),
                                        dicom_columns=_dicom_columns_for(key, meta))
        _write_series_csv(merged, key, mod)
        if mod == config.PRIMARY_MODE:
            primary_df = merged
            present = _present_organs(merged)

    # tamamlanma + eksik raporu (ana moda göre; sidecar hangi modların
    # yazıldığını `modes` alanında listeliyor)
    _write_done_sidecar(key, expected, present, tum_failed, absent_targets=absent)
    if tum_failed:
        print(f"[mirp] {key}: {len(tum_failed)} birim atlandı (bkz. features.done.json).")
    return primary_df


def _present_organs(merged: pd.DataFrame) -> set[str]:
    """GEÇERLİ öznitelik üreten organları döndürür.

    Bir organ etiketli satır üretmiş ama çekirdek özniteliği (morph_volume) NaN ise
    (dejenere ROI: örn. tüm voksseller reseg aralığı dışında) 'üretilmiş' sayılmaz →
    eksik olarak raporlanır (sessizce geçerli sanılmasın)."""
    if "image_mask_label" not in merged.columns:
        return set()
    valid = merged
    for core in ("morph_volume", "stat_mean"):
        if core in merged.columns:
            valid = merged[merged[core].notna()]
            break
    return set(valid["image_mask_label"].astype(str))
