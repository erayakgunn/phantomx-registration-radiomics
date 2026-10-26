"""DICOM seri sınıflandırma + mekânsal slice sıralama yardımcıları.

batch_120kvp.py'nin kullandığı çekirdek mantık: cihaz tespiti (EID/PCCT),
etkin kVp (KVP etiketinden), rekonstrüksiyon tipi, hedef-seri filtresi ve
DETERMİNİSTİK mekânsal slice sıralaması. Klasörleme/kopyalama YAPMAZ — batch
ham DICOM'ları doğrudan kaynak klasörden okur.

Python 3.14 (pydicom 3.0.1) altında koşar.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pydicom

import config


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _sanitize(text: Any, fallback: str = "NA") -> str:
    """Dosya/klasör adı için güvenli bir dize üretir."""
    s = str(text).strip() if text not in (None, "") else fallback
    s = re.sub(r"[^\w.\-]+", "_", s)          # harf/rakam/._- dışını _ yap
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or fallback


def _get_kvp(ds: pydicom.Dataset) -> float | None:
    """(0018,0060) KVP değerini float olarak döndürür (yoksa None)."""
    kvp = getattr(ds, "KVP", None)
    if kvp is None or kvp == "":
        return None
    try:
        return float(kvp)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Cihaz-bağımsız seri sınıflandırma (EID + PCCT)
# --------------------------------------------------------------------------- #
# "120kVp" / "80 kV" gibi kalıplar; "keV" (VMI enerjisi) EŞLEŞMEZ (k-e-V ≠ k-v).
_KVP_RE = re.compile(r"(\d{2,3})\s*kv", re.IGNORECASE)


def _parse_kvp_from_text(text: Any) -> float | None:
    if not text:
        return None
    m = _KVP_RE.search(str(text))
    return float(m.group(1)) if m else None


def scanner_model(ds: pydicom.Dataset) -> str:
    """ManufacturerModelName (ör. 'NAEOTOM Alpha', 'SOMATOM Edge Plus')."""
    return str(getattr(ds, "ManufacturerModelName", "") or "").strip()


def is_photon_counting(ds: pydicom.Dataset) -> bool:
    """Foton-sayan CT (PCCT) mı? — DICOM metadata'sından OTOMATİK algılanır.

    İki bağımsız işaret: (1) model adı NAEOTOM içerir, (2) ImageType'ta 'COUNT'
    öğesi (foton-sayan). İkisinden biri yeterli.
    """
    if "NAEOTOM" in scanner_model(ds).upper():
        return True
    it = [str(x).upper() for x in (getattr(ds, "ImageType", []) or [])]
    return "COUNT" in it


def effective_kvp(ds: pydicom.Dataset) -> float | None:
    """Rekonstrüksiyonun ETKİN kVp'si — YALNIZCA DICOM METADATA'sından, cihaza göre
    OTOMATİK kaynak seçimiyle (klasör/dosya adına ASLA bakılmaz).

    - FOTON-SAYAN (PCCT, NAEOTOM): DICOM KVP etiketi (0018,0060) tüp voltajıdır ve
      rekonstrüksiyon kVp'sini yansıtmaz (80/100/120 hepsi 120 raporlar). Gerçek kVp
      SeriesDescription'dadır (0008,103E, ör. "120kVp_..."). Önce oradan, yoksa etiket.
    - KONVANSİYONEL (EID, SOMATOM Edge Plus): KVP etiketi güvenilirdir → doğrudan etiket.
    config.KVP_FROM_DESCRIPTION=False ile PCCT'de de etiket zorlanabilir.
    """
    if getattr(config, "KVP_FROM_DESCRIPTION", True) and is_photon_counting(ds):
        v = _parse_kvp_from_text(getattr(ds, "SeriesDescription", ""))
        if v is not None:
            return v
    return _get_kvp(ds)


def recon_type(ds: pydicom.Dataset) -> str:
    """Rekonstrüksiyon tipi (ImageType'tan): 'localizer', 'VMI_<N>keV' veya 'native'.

    Siemens ImageType örnekleri:
      LOCALIZER (topogram/scout)  → 'localizer'  (HER ZAMAN hariç)
      VMI + ME70KEV               → 'VMI_70keV'
      THRESHOLD/DET3D (native/PolyE) → 'native'
    """
    it = [str(x).upper() for x in (getattr(ds, "ImageType", []) or [])]
    if "LOCALIZER" in it or "TOPOGRAM" in it:
        return "localizer"
    if "VMI" in it:
        kev = next((x for x in it if x.startswith("ME") and x.endswith("KEV")), None)
        if kev:
            return f"VMI_{kev[2:-3]}keV"   # ME70KEV -> VMI_70keV
        return "VMI"
    return "native"


def is_target_series(ds: pydicom.Dataset) -> tuple[bool, float | None, str]:
    """(hedef_mi, etkin_kvp, rekon_tipi) — filtreleme kararı tek yerde (yalnız DICOM).

    Hedef: CT + LOCALIZER değil + etkin kVp hedefe uyuyor + rekon tipi izinliyse.
    """
    rt = recon_type(ds)
    kvp = effective_kvp(ds)
    if str(getattr(ds, "Modality", "")) != "CT":
        return False, kvp, rt
    if getattr(config, "EXCLUDE_LOCALIZER", True) and rt == "localizer":
        return False, kvp, rt
    if kvp is None or abs(kvp - config.TARGET_KVP) > config.KVP_TOLERANCE:
        return False, kvp, rt
    allowed = getattr(config, "TARGET_RECON_TYPES", None)
    if allowed is not None and rt not in allowed:
        return False, kvp, rt
    return True, kvp, rt


def _kvp_bucket(kvp: float | None) -> tuple[str, bool]:
    """kVp değerini klasör adına ve hedef (120) bayrağına çevirir."""
    if kvp is None:
        return "unknown_kVp", False
    is_target = abs(kvp - config.TARGET_KVP) <= config.KVP_TOLERANCE
    return f"{int(round(kvp))}kVp", is_target


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _has_geometry(ds: pydicom.Dataset) -> bool:
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    return iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3


def _instance_number(ds: pydicom.Dataset) -> int:
    try:
        return int(getattr(ds, "InstanceNumber", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _spatial_key(ds: pydicom.Dataset) -> float:
    """Slice'ı eksenel konumuna göre sıralamak için fiziksel izdüşüm anahtarı.

    ImageOrientationPatient'tan slice normali hesaplanır ve ImagePositionPatient
    bu normale izdüşürülür. Geometri yoksa InstanceNumber'a düşer.
    """
    if _has_geometry(ds):
        iop = ds.ImageOrientationPatient
        ipp = ds.ImagePositionPatient
        row = [float(x) for x in iop[:3]]
        col = [float(x) for x in iop[3:]]
        normal = _cross(row, col)
        pos = [float(x) for x in ipp]
        return pos[0] * normal[0] + pos[1] * normal[1] + pos[2] * normal[2]
    return float(_instance_number(ds))


def _orientation(ds: pydicom.Dataset) -> tuple:
    return tuple(round(float(x), 4) for x in ds.ImageOrientationPatient)


def spatial_sort(items: list[tuple[Path, pydicom.Dataset]]) -> list[tuple[Path, pydicom.Dataset]]:
    """(dosya, header) listesini mekânsal ve DETERMİNİSTİK sıralar.

    Fiziksel izdüşüm stratejisi YALNIZCA tüm slice'larda geometri varsa VE
    ImageOrientationPatient TUTARLIYSA kullanılır — aksi halde (scout karışmış,
    gantry-tilt değişmiş) izdüşümler farklı normallere olur ve karşılaştırılamaz;
    bu durumda InstanceNumber'a düşülür (yanlış sıra üretilmez).
    İkincil kararlı anahtarlar (InstanceNumber, dosya adı) tekrar-üretilebilir sıra sağlar.
    """
    geom_ok = bool(items) and all(_has_geometry(ds) for _, ds in items)
    if geom_ok and len({_orientation(ds) for _, ds in items}) == 1:
        return sorted(items, key=lambda it: (_spatial_key(it[1]),
                                             _instance_number(it[1]), it[0].name))
    return sorted(items, key=lambda it: (_instance_number(it[1]), it[0].name))
