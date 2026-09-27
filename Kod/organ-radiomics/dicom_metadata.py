"""Seri başına DICOM başlık bilgilerini toplar ve öznitelik tablosuna taşır.

Öznitelik tablosu şimdiye kadar yalnızca dört kimlik sütunu taşıyordu
(series_key, patient_id, recon_type, kvp). Geri kalan çekim parametreleri —
kernel, mAs, FOV, dilim kalınlığı, cihaz — yalnızca kaynak DICOM'larda vardı,
yani bir sonucun hangi protokolden geldiğini görmek için ham veriye dönmek
gerekiyordu. Protokol karşılaştırması yapan bir çalışmada bu, tablonun en çok
ihtiyaç duyulan sütunlarının eksik olması demek.

Değerler bir kez okunup NIfTI önbelleğinin yanına yazılıyor: kaynak DICOM'lar
taşınsa bile tablo yeniden üretilebiliyor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config

# Başlıktaki HER etiket tabloya giriyor. Bu liste yalnızca SIRAYI belirliyor:
# radyomik açıdan en çok bakılanlar başa gelsin, gerisi alfabetik olarak onları
# izlesin. Böylece hiçbir bilgi kaybolmuyor ama tablo okunabilir kalıyor.
PRIORITY_TAGS: tuple[str, ...] = (
    "Manufacturer", "ManufacturerModelName", "StationName",
    "StudyDate", "SeriesNumber", "AcquisitionNumber",
    "ProtocolName", "SeriesDescription", "BodyPartExamined",
    "KVP", "XRayTubeCurrent", "Exposure", "ExposureTime",
    "CTDIvol", "ConvolutionKernel", "FilterType",
    "SliceThickness", "SpacingBetweenSlices", "PixelSpacing",
    "ReconstructionDiameter", "DataCollectionDiameter",
    "ImageType", "SeriesInstanceUID", "StudyInstanceUID",
)

# Tablo sütunları bu önekle yazılıyor: MIRP'in kendi ``image_*`` meta sütunlarıyla
# ve öznitelik adlarıyla asla çakışmasın.
COLUMN_PREFIX = "dcm_"

_SIDECAR = "dicom_metadata.json"


def _stringify(value: Any) -> str:
    """pydicom değerini tabloya yazılabilir tek bir metne indirger."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "\\".join(str(v) for v in value)
    return str(value)


# Tabloya taşınmayacak veri tipleri: ikili bloklar ve iç içe diziler. Bir hücreye
# sığmazlar ve dosyayı gereksiz şişirirler.
_SKIP_VR = {"OB", "OW", "OF", "OD", "OL", "OV", "UN", "SQ"}

# Tek bir hücrede okunabilir kalmasının sınırı; bunun üstü kırpılır.
_MAX_VALUE_CHARS = 240


def _sanitise(keyword: str, tag) -> str:
    """Sütun adı. Anahtar kelimesi olmayan (özel) etiketler grup_eleman olur."""
    if keyword:
        return keyword
    return f"{tag.group:04X}_{tag.element:04X}"


def harvest(series_dir: Path, first_file: str | None = None) -> dict[str, str]:
    """Serinin ilk diliminden başlıktaki TÜM etiketleri okur.

    Seri boyunca sabit olan bilgiler hedeflendiği için tek dilim yeterli;
    dilimden dilime değişenler (InstanceNumber, ImagePositionPatient) zaten
    öznitelik tablosunda anlam taşımıyor.

    Çıktı PRIORITY_TAGS sırasında başlar, ardından kalan standart etiketler
    alfabetik, en sonda özel (private) etiketler gelir.
    """
    import pydicom

    series_dir = Path(series_dir)
    if not series_dir.is_dir():
        # Kaynak taşınmış ya da bağlı değil. Metadata yalnızca bağlam olduğu için
        # bu bir hata değil; çıkarım DICOM'a değil NIfTI'ye dayanıyor.
        return {}

    if first_file:
        candidate = series_dir / first_file
    else:
        files = sorted(p for p in series_dir.iterdir() if p.is_file())
        candidate = files[0] if files else None

    if candidate is None or not candidate.is_file():
        return {}

    try:
        ds = pydicom.dcmread(str(candidate), stop_before_pixels=True, force=True)
    except Exception:
        return {}

    standard: dict[str, str] = {}
    private: dict[str, str] = {}
    for element in ds:
        if element.VR in _SKIP_VR or element.keyword == "PixelData":
            continue
        try:
            text = _stringify(element.value)
        except Exception:
            continue
        if not text:
            continue
        name = _sanitise(element.keyword, element.tag)
        target = standard if element.keyword else private
        target[name] = text[:_MAX_VALUE_CHARS]

    ordered: dict[str, str] = {}
    for tag in PRIORITY_TAGS:                       # en çok bakılanlar önce
        if tag in standard:
            ordered[tag] = standard.pop(tag)
    for tag in sorted(standard):                    # kalan standart, alfabetik
        ordered[tag] = standard[tag]
    for tag in sorted(private):                     # özel etiketler en sonda
        ordered[tag] = private[tag]
    return ordered


def load_or_harvest(key: str, meta: dict[str, Any]) -> dict[str, str]:
    """Önbellekten okur, yoksa kaynaktan toplayıp önbelleğe yazar.

    Önbellek NIfTI'nin yanında durur, çünkü ikisi de aynı seriyi tanımlıyor ve
    aynı anda geçersizleşiyor.
    """
    sidecar = config.NIFTI / key / _SIDECAR
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            pass  # bozuk önbellek: yeniden topla

    ordered = meta.get("ordered_files") or []
    harvested = harvest(Path(meta["series_dir"]), ordered[0] if ordered else None)

    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tmp = sidecar.with_name(sidecar.name + ".tmp")
        tmp.write_text(json.dumps(harvested, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(sidecar)
    except Exception:
        pass  # önbellek yazılamadıysa akış sürsün; değerler zaten elde

    return harvested


def as_columns(harvested: dict[str, str]) -> dict[str, str]:
    """{'KVP': '120'} -> {'dcm_KVP': '120'}, harvest() sırasını koruyarak."""
    return {f"{COLUMN_PREFIX}{tag}": value for tag, value in harvested.items()}
