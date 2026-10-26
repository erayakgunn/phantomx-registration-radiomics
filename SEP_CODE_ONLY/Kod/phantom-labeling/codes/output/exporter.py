"""Export ROI data as DICOM (.dcm) files and JSON metadata.

Each ROI patch is written as a minimal CT DICOM file containing:
  * The patch HU values rescaled to int16 (RescaleSlope=1, Intercept=-1024)
  * NaN pixels (outside the circular ROI mask) replaced with a sentinel
    stored value (-32768) and tagged via PixelPaddingValue so any DICOM
    viewer that respects the padding tag renders them as background.
  * Provenance in ImageComments (label, slice index, sub-pixel centre,
    angle, accepted flag) so each file is self-describing.

Folder layout under ``<output_dir>/<series_name>/``::

    Textures/<TextureName>/slice_XXXX_<TextureName>.dcm
    Stars/<Star16|Star8>/slice_XXXX_Star.dcm
    Inserts/<5mm|10mm>/Insert_NN/slice_XXXX_Insert_NN_Xmm.dcm   (15mm only with --insert-sizes 5,10,15)

The size-first inserts layout puts each sub-ROI diameter at the top of
the Inserts/ tree so all 40 inserts at a given size can be inspected
together without traversing into 40 individual insert folders.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from core.roi_extractor import ExtractionResult, ROI
from core.validator import ValidationReport
from utils.logger import log
from utils.file_utils import ensure_dir
import config


# Matches "Insert_NN" and the size suffix on insert sub-ROIs.
_INSERT_FOLDER_RE = re.compile(r"^(Insert_\d+)(?:_(\d+mm))?$")

# Pixel stored value used to mark NaN ROI pixels (= "outside the circle").
# Equivalent HU under RescaleSlope=1/Intercept=-1024 is -33792, well below
# the air HU floor (-1024), so any viewer that honours PixelPaddingValue
# (or simply clips to a reasonable window) will render these pixels as
# background. Stored as signed int16, so PixelRepresentation=1 is required.
_DICOM_PADDING_STORED = -32768
_DICOM_RESCALE_SLOPE = 1.0
_DICOM_RESCALE_INTERCEPT = -1024.0

# One ImplementationClassUID for everything this exporter writes — required
# by the DICOM file meta header but doesn't have to be a registered UID for
# private/derived data.
_IMPL_CLASS_UID = "1.2.826.0.1.3680043.10.1234.5678.9999.1"


def _get_texture_names():
    names = set()
    for sec_cfg in config.SECTIONS.values():
        if "textures" in sec_cfg:
            names.update(sec_cfg["textures"].values())
    return names


_TEXTURE_NAMES = _get_texture_names()


def _hu_to_int16(pixels_hu: np.ndarray) -> np.ndarray:
    """Convert an HU float array (with NaN for out-of-ROI pixels) to int16
    DICOM stored values under RescaleSlope=1, RescaleIntercept=-1024.

    NaN pixels become ``_DICOM_PADDING_STORED`` so they round-trip via
    PixelPaddingValue when the file is reloaded.
    """
    out = np.empty(pixels_hu.shape, dtype=np.int16)
    valid_mask = ~np.isnan(pixels_hu)
    stored_valid = pixels_hu[valid_mask] - _DICOM_RESCALE_INTERCEPT
    # Clip to int16 range (HU ≈ [-1024, +3071] under usual CT; clamp anyway
    # so any out-of-range HU doesn't overflow).
    stored_valid = np.clip(stored_valid, -32767, 32767)
    out[valid_mask] = stored_valid.astype(np.int16)
    out[~valid_mask] = _DICOM_PADDING_STORED
    return out


def _save_roi_as_dicom(
    roi: ROI,
    dst_path: str,
    series_name: str = "",
    source_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one ROI patch as a minimal CT-Image DICOM file.

    The created file is parseable by ``pydicom.dcmread`` and any viewer
    that accepts standard CT-Image SOP class objects. The pixel data is
    int16 with RescaleSlope/Intercept set so the values reload as HU.
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = _IMPL_CLASS_UID

    ds = FileDataset(dst_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # ---- Patient / Study / Series tags ---------------------------------
    src = source_metadata or {}
    ds.PatientName = src.get("PatientName", "PCCT-Phantom")
    ds.PatientID = src.get("PatientID", "PHANTOM-001")
    ds.PatientBirthDate = src.get("PatientBirthDate", "")
    ds.PatientSex = src.get("PatientSex", "")
    ds.StudyInstanceUID = src.get("StudyInstanceUID", generate_uid())
    # Different SeriesInstanceUID per ROI label, so a viewer that groups
    # by series shows each insert/texture/star as its own series.
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "CT"
    ds.Manufacturer = "PCCT-Annotations"
    now = datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime
    ds.SeriesDescription = (
        f"{series_name}|{roi.section}|{roi.label}|slice{roi.slice_idx}"
    )[:64]  # DICOM LO max length

    ds.SliceLocation = float(roi.slice_idx)
    ds.InstanceNumber = int(roi.slice_idx)

    # ---- Pixel data ----------------------------------------------------
    pixel_int16 = _hu_to_int16(roi.pixels)
    h, w = pixel_int16.shape
    ds.Rows = int(h)
    ds.Columns = int(w)
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_int16.tobytes()

    # HU rescale + padding
    ds.RescaleSlope = str(int(_DICOM_RESCALE_SLOPE))
    ds.RescaleIntercept = str(int(_DICOM_RESCALE_INTERCEPT))
    ds.RescaleType = "HU"
    ds.PixelPaddingValue = _DICOM_PADDING_STORED

    # ROI provenance
    ds.ImageComments = (
        f"label={roi.label};section={roi.section};slice={roi.slice_idx};"
        f"cx={roi.center_x:.3f};cy={roi.center_y:.3f};"
        f"shape={roi.shape_type};size_cm={roi.size_cm};"
        f"angle={roi.angle_deg:.2f};accepted={roi.accepted}"
    )[:1024]

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(dst_path, write_like_original=False)


def _save_to_organized_dir(
    roi: ROI,
    dcm_name: str,
    textures_dir: str,
    stars_dir: str,
    inserts_dir: str,
    series_name: str,
    source_metadata: Optional[Dict[str, Any]],
) -> None:
    """Route an ROI to its destination folder and write the DICOM file.

    Layout:
      Textures/<TextureName>/slice_XXXX_<TextureName>.dcm
      Stars/<Star16|Star8>/slice_XXXX_Star.dcm
      Inserts/<5mm|10mm>/Insert_NN/slice_XXXX_Insert_NN_Xmm.dcm   (15mm only with --insert-sizes 5,10,15)
    """
    if roi.label in _TEXTURE_NAMES:
        out_dir = ensure_dir(os.path.join(textures_dir, roi.label))
    elif roi.label in ("Star", "Star8", "Star16"):
        # Star ROIs are now labelled "Star8" or "Star16" per section
        # (legacy "Star" still accepted for backwards compatibility).
        star_sub = "Star16" if roi.section == "STAR16" else "Star8"
        out_dir = ensure_dir(os.path.join(stars_dir, star_sub))
    elif roi.label.startswith("Insert_"):
        # Size-first layout: Inserts/<size>/Insert_NN/...
        m = _INSERT_FOLDER_RE.match(roi.label)
        if m and m.group(2):
            base_folder = m.group(1)          # "Insert_NN"
            size_folder = m.group(2)          # "5mm" / "10mm" / "15mm"
            out_dir = ensure_dir(
                os.path.join(inserts_dir, size_folder, base_folder)
            )
        elif m:
            # No size suffix (legacy code path) — fall back to no-size layout.
            out_dir = ensure_dir(os.path.join(inserts_dir, m.group(1)))
        else:
            out_dir = ensure_dir(os.path.join(inserts_dir, roi.label))
    else:
        return

    try:
        _save_roi_as_dicom(
            roi,
            os.path.join(out_dir, dcm_name),
            series_name=series_name,
            source_metadata=source_metadata,
        )
    except Exception as e:  # pydicom/OS errors all caught here
        log.error(f"Failed to save {dcm_name} to {out_dir}: {e}")


def export_results(
    extraction: ExtractionResult,
    validation: ValidationReport,
    output_dir: str,
    series_name: str,
    source_metadata: Optional[Dict[str, Any]] = None,
):
    """Export ROI patches as DICOM files + a metadata JSON summary."""
    log.info("Exporting results...")

    base_dir = os.path.join(output_dir, series_name)

    # Group ROIs by section
    by_section: Dict[str, List[ROI]] = {}
    for roi in extraction.rois:
        by_section.setdefault(roi.section, []).append(roi)

    metadata_all: Dict[str, Any] = {
        "series": series_name,
        "total_rois": len(extraction.rois),
        "accepted_rois": sum(1 for r in extraction.rois if r.accepted),
        "rejected_rois": sum(1 for r in extraction.rois if not r.accepted),
        "output_format": "DICOM (.dcm)",
        "dicom_rescale": {
            "slope": _DICOM_RESCALE_SLOPE,
            "intercept": _DICOM_RESCALE_INTERCEPT,
            "padding_stored_value": _DICOM_PADDING_STORED,
        },
        "sections": {},
    }

    textures_dir = os.path.join(base_dir, "Textures")
    stars_dir = os.path.join(base_dir, "Stars")
    inserts_dir = os.path.join(base_dir, "Inserts")

    for section_name, rois in by_section.items():
        sec_meta = {"rois": []}
        for roi in rois:
            dcm_name = f"slice_{roi.slice_idx:04d}_{roi.label}.dcm"

            _save_to_organized_dir(
                roi, dcm_name,
                textures_dir, stars_dir, inserts_dir,
                series_name, source_metadata,
            )

            valid_pixels = roi.pixels[~np.isnan(roi.pixels)]
            roi_meta = {
                "label": roi.label,
                "slice_idx": roi.slice_idx,
                "center_x": round(roi.center_x, 2),
                "center_y": round(roi.center_y, 2),
                "shape_type": roi.shape_type,
                "size_cm": roi.size_cm,
                "angle_deg": round(roi.angle_deg, 2),
                "accepted": roi.accepted,
                "rejection_reason": roi.rejection_reason,
                "mean_hu": round(float(np.mean(valid_pixels)), 2) if valid_pixels.size > 0 else None,
                "std_hu": round(float(np.std(valid_pixels)), 2) if valid_pixels.size > 0 else None,
                "dcm_file": dcm_name,
            }
            sec_meta["rois"].append(roi_meta)
        metadata_all["sections"][section_name] = sec_meta

    metadata_all["validation"] = {
        "warnings": validation.warnings,
        "errors": validation.errors,
    }

    json_path = os.path.join(base_dir, "metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_all, f, indent=2, ensure_ascii=False)

    log.info(f"Exported to {base_dir}")
    log.info(f"  Metadata: {json_path}")
    dcm_count = sum(len(rois) for rois in by_section.values())
    log.info(f"  DICOM files: {dcm_count}")
