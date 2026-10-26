"""Adım 2 — 120 kVp serilerini NIfTI'ye çevir ve TotalSegmentator ile segmente et.

Bu modül Python 3.14 altında koşar ve SimpleITK ile DICOM→NIfTI dönüşümünü
yapar. TotalSegmentator ayrı bir ortamda (Python 3.11 + CUDA) kurulu olduğu
için `TotalSegmentator.exe` bir *subprocess* olarak çağrılır.

Maske ile görüntünün geometrisi birebir aynı olsun diye TotalSegmentator'a ham
DICOM değil, buradan üretilen `ct.nii.gz` verilir. Bu dosya MIRP adımında da
aynen kullanılır.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import SimpleITK as sitk

import config

# TotalSegmentator "total" görevinin etiket haritası tek sefer üretilip cache'lenir.
_LABEL_MAP_CACHE: dict[str, str] | None = None


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def dicom_series_to_nifti(meta: dict[str, Any], out_path: Path) -> Path:
    """Mekânsal sıralı DICOM dosyalarını tek bir NIfTI hacme dönüştürür.

    `ordered_paths` sırası korunur; SimpleITK slice'lar arası konumdan
    spacing/origin/direction'ı doğru hesaplar.

    Yollar `ordered_paths`ten alınır: bir seri dışa aktarımda birden çok
    klasöra bölünmüş olabiliyor, o durumda `series_dir` + dosya adı
    birleştirmesi var olmayan yollar üretir. Eski meta'lar için (yalnız
    `ordered_files` içeren) geriye-uyumlu olarak birleştirmeye düşülür.
    """
    series_dir = Path(meta["series_dir"])
    file_names = [str(p) for p in meta.get("ordered_paths") or
                  [series_dir / name for name in meta["ordered_files"]]]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    image = reader.Execute()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ATOMİK yazım: geçici dosyaya yaz, tamamlanınca taşı → yarım ct.nii.gz kalmaz
    # (kesinti sonrası resume bozuk CT'yi 'geçerli' saymasın diye kritik).
    tmp = out_path.with_name(out_path.name + ".tmp.nii.gz")
    sitk.WriteImage(image, str(tmp))
    os.replace(tmp, out_path)
    return out_path


def load_total_label_map() -> dict[str, str]:
    """TotalSegmentator 'total' görevinin {etiket: yapı_adı} haritasını verir.

    class_map yalnızca Python 3.11 ortamında import edilebildiği için değer
    tek sefer subprocess ile alınır ve cache'lenir. JSON anahtarları string'e
    döner (MIRP roi_name dict'i zaten string anahtar bekler).
    """
    global _LABEL_MAP_CACHE
    if _LABEL_MAP_CACHE is not None:
        return _LABEL_MAP_CACHE

    # (1) Yetkili kaynak: TS'in kurulu olduğu python'dan canlı class_map.
    if config.PY311 is not None and Path(config.PY311).exists():
        code = (
            "import json;"
            "from totalsegmentator.map_to_binary import class_map;"
            "print(json.dumps(class_map['total']))"
        )
        try:
            result = subprocess.run(
                [str(config.PY311), "-c", code],
                capture_output=True, text=True, check=True,
                encoding="utf-8", errors="replace",   # bkz. aşağıdaki not
            )
            _LABEL_MAP_CACHE = json.loads(result.stdout.strip())
            return _LABEL_MAP_CACHE
        except Exception as exc:  # noqa: BLE001 — statik dosyaya düş
            print(f"[seg] class_map alınamadı ({exc}); statik etiket haritasına düşülüyor.")

    # (2) Taşınabilir yedek: repo ile gelen statik JSON (117 'total' yapısı).
    shipped = config.PROJECT_ROOT / "totalseg_total_labels.json"
    if shipped.exists():
        _LABEL_MAP_CACHE = json.loads(shipped.read_text(encoding="utf-8"))
        return _LABEL_MAP_CACHE

    raise RuntimeError(
        "TotalSegmentator etiket haritası bulunamadı. Ya PHANTOMX_PY311 ortam "
        "değişkenini TS'in python'una ayarlayın ya da 'totalseg_total_labels.json' "
        "dosyasını proje köküne koyun."
    )


def _resolve_totalseg_cmd(ct_path: Path, seg_path: Path) -> list[str]:
    """Kullanılabilir TotalSegmentator giriş noktasına göre komutu kurar."""
    base_args = [
        "-i", str(ct_path),
        "-o", str(seg_path),
        "-ml",                      # tek multilabel çıktı
        "-ta", config.TS_TASK,
        "-d", config.TS_DEVICE,
    ]
    # SIRA ÖNEMLİ: python'u biliyorsak MODÜL olarak çağır, konsol exe'sini değil.
    #
    # `Scripts\TotalSegmentator.exe` bir setuptools sarmalayıcısıdır ve içine
    # KENDİSİNİ KURAN python'un MUTLAK yolu gömülüdür. Taşınabilir bir kopyada
    # o yol başka makinede yoktur; sarmalayıcı ya hiç açılmaz ya da — bu
    # makinede olduğu gibi — paketin içindeki değil, SİSTEMDEKİ python'u
    # başlatır. Ölçüldü: paketteki 91 konsol exe'sinin 91'inde de gömülü yol
    # C:\Users\Administrator\AppData\...\Python311\python.exe idi.
    #
    # Modül biçimi bu tuzağı tamamen atlar: hangi python'un koşacağını biz
    # söyleriz, dolayısıyla TS sürümü de yorumlayıcı da kesin bilinir.
    if config.PY311 is not None and Path(config.PY311).exists():
        return [str(config.PY311), "-m", "totalsegmentator.bin.TotalSegmentator", *base_args]
    if config.TOTALSEG_EXE is not None and Path(config.TOTALSEG_EXE).exists():
        return [str(config.TOTALSEG_EXE), *base_args]
    raise RuntimeError(
        "TotalSegmentator çalıştırılabiliri bulunamadı. TotalSegmentator'ı kurun ve "
        "PATH'e ekleyin, ya da PHANTOMX_TOTALSEG_EXE / PHANTOMX_PY311 ortam "
        "değişkenlerini ayarlayın."
    )


def _sureç_agacini_oldur(surec: "subprocess.Popen") -> None:
    """Süreci ve BÜTÜN torunlarını öldürür (Windows: taskkill /T).

    `Popen.kill()` yalnız doğrudan çocuğu öldürür. TotalSegmentator kendi
    içinde nnU-Net işçi süreçleri açıyor; onlar öksüz kalıp CPU'yu ve RAM'i
    tutmaya devam ediyor, sıradaki serinin işçi bütçesini de bozuyorlar.
    """
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(surec.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=60)
    except Exception:  # noqa: BLE001 — taskkill yoksa (Windows dışı) yedeğe düş
        pass
    try:
        surec.kill()
    except Exception:  # noqa: BLE001
        pass
    try:
        surec.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass


def run_totalsegmentator(ct_path: Path, seg_path: Path, log_path: Path) -> bool:
    """CT NIfTI'sini segmente eder; başarılıysa True döndürür.

    ATOMİK + doğrulamalı: TS geçici bir dosyaya yazar, başarı + okunabilirlik
    doğrulandıktan sonra os.replace ile seg_path'e taşınır. Böylece kesintide
    yarım/bozuk seg.nii.gz asla 'geçerli' sayılmaz (resume güvenliği).
    Ayrıca subprocess'e zaman sınırı (GPU/CUDA kilitlenmesi batch'i dondurmasın).
    """
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_seg = seg_path.with_name("seg.tmp.nii.gz")
    if tmp_seg.exists():
        tmp_seg.unlink()
    cmd = _resolve_totalseg_cmd(ct_path, tmp_seg)
    print(f"[seg] çalıştırılıyor: {' '.join(cmd)}")

    # ÇIKTI BORUYA DEĞİL DOSYAYA YAZILIR. `capture_output=True` kullanmak
    # Windows'ta KALICI KİLİTLENME üretiyor ve bu saha koşumunda gerçekleşti:
    #
    #   1. subprocess.run zaman aşımına uğrar (TimeoutExpired)
    #   2. subprocess.py:552 `process.kill()` — yalnız DOĞRUDAN çocuğu öldürür
    #   3. subprocess.py:559 Windows'ta ardından `process.communicate()` çağırır,
    #      zaman aşımı VERMEDEN
    #   4. communicate() -> endtime=None -> _remaining_time(None) -> None
    #   5. subprocess.py:1628 `stdout_thread.join(None)` -> SONSUZA KADAR bekler
    #   6. Okuyucu thread borunun kapanmasını bekler; nnU-Net'in TORUN işçi
    #      süreçleri borunun yazma ucunu hâlâ tuttuğu için kapanma hiç gelmez.
    #
    # Ölçüldü: bir seri 20:53'te başladı, 22:53'te zaman aşımı tetiklendi ve
    # koşum o noktada ölü kaldı — hiçbir zaman aşımı kurtarmıyor.
    #
    # Dosyaya yazınca Popen'ın stdout/stderr'i None olur, okuyucu thread hiç
    # kurulmaz, communicate() anında döner. Ek fayda: log CANLI yazılır, yani
    # takılan bir seride ne olduğu koşum sürerken görülebiliyor.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as günlük:
        günlük.write("== CMD ==\n" + " ".join(cmd) + "\n\n== ÇIKTI ==\n")
        günlük.flush()
        # `run` yerine `Popen`: zaman aşımında SÜREÇ AĞACININ tamamını öldürmek
        # için PID gerekiyor. `Popen.kill()` yalnız doğrudan çocuğu öldürür;
        # nnU-Net'in işçi süreçleri hayatta kalıp CPU'yu yemeye devam eder.
        surec = subprocess.Popen(cmd, stdout=günlük, stderr=subprocess.STDOUT)
        try:
            donus = surec.wait(timeout=config.TS_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _sureç_agacini_oldur(surec)
            with open(log_path, "a", encoding="utf-8") as ek:
                ek.write(f"\n== ZAMAN AŞIMI ({config.TS_TIMEOUT_S}s) ==\n")
            print(f"[seg] HATA: TS zaman aşımı ({config.TS_TIMEOUT_S}s) — seri atlanıyor.")
            if tmp_seg.exists():
                tmp_seg.unlink()
            return False

    if donus != 0 or not tmp_seg.exists():
        print(f"[seg] HATA (returncode={donus}) — bkz. {log_path}")
        if tmp_seg.exists():
            tmp_seg.unlink()
        return False
    # Okunabilirlik doğrulaması (bozuk gzip vb. yakala)
    try:
        sitk.ReadImage(str(tmp_seg))
    except Exception as exc:  # noqa: BLE001
        print(f"[seg] HATA: TS çıktısı okunamadı ({exc}); atlanıyor.")
        tmp_seg.unlink()
        return False
    os.replace(tmp_seg, seg_path)   # atomik: ancak tam+geçerli dosya görünür olur
    return True


# NOT: Serilerin keşfi, resume ve TS orkestrasyonu artık batch_120kvp.py'de
# yapılır (kanonik giriş noktası). Bu modül yalnız yeniden kullanılan yardımcıları
# sağlar: dicom_series_to_nifti, run_totalsegmentator, load_total_label_map.
