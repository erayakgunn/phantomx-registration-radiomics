"""DICOM header metadata extraction for standardized labeling.

Reads a single DICOM file (header-only, fast) and extracts all fields
needed to build a standardized series name. Missing fields are filled
with UNKNOWN and recorded in the `warnings` list so reports can audit
what was inferred vs. present.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pydicom


UNKNOWN = "UNKNOWN"


# Siemens NAEOTOM Alpha / PCCT recon SeriesDescription pattern. Example match:
#   "120kVp_100mAs_1mm_300FOV 1,00 Qr40 Q3"
#   "80kVp_200mAs_3mm_500FOV 3,00 Qr64 Q1"
# VMI kVp (80/100/120/140) lives here — the physical DICOM KVP tag on
# photon-counting CTs is fixed (120 or 140) and is NOT what we want in the
# standardized filename.
_PCCT_RECON_DESC = re.compile(
    r"(?P<vmi>\d+)\s*kVp_(?P<mas>\d+)\s*mAs_(?P<slice>\d+)\s*mm_(?P<fov>\d+)\s*FOV"
    r"\s+(?P<slice_dec>[\d,\.]+)\s+(?P<kernel>[A-Za-z]+\d*[A-Za-z]*)\s+Q(?P<q>\d+)",
    re.IGNORECASE,
)


def _parse_pcct_recon_desc(desc: str) -> Optional[Dict[str, str]]:
    """Return parsed VMI/mAs/slice/fov/kernel/q from a PCCT SeriesDescription.

    Returns ``None`` if the description does not match the PCCT recon
    pattern (e.g. Philips / GE scans, or topograms on Siemens).
    """
    m = _PCCT_RECON_DESC.search(desc or "")
    if not m:
        return None
    return {k: v for k, v in m.groupdict().items() if v}


@dataclass
class LabelMetadata:
    """All fields needed for standardized filename generation.

    The first 10 fields become the filename; the last 3 are audit info.
    """

    institution: str          # Merkez       -> InstitutionName
    manufacturer: str         # Marka        -> Manufacturer
    device_model: str         # Cihaz        -> ManufacturerModelName
    study_date: str           # Tarih        -> StudyDate (YYYYMMDD)
    recon_kernel: str         # Recon kernel -> ReconstructionAlgorithm / SeriesDescription
    conv_kernel: str          # Conv kernel  -> ConvolutionKernel
    kvp: str                  # KVP          -> KVP
    mas: str                  # mAs          -> Exposure / (XRayTubeCurrent * ExposureTime / 1000)
    fov: str                  # FOV          -> ReconstructionDiameter (mm)
    slice_thickness: str      # Slice Th.    -> SliceThickness (mm)
    recon_spacing: str = UNKNOWN  # Recon spacing -> SpacingBetweenSlices (mm)
                                  #               or median(diff(SliceLocations))

    # --- audit / debug info (NOT used in filename) ---
    source_path: str = ""
    series_description: str = ""
    warnings: List[str] = field(default_factory=list)

    # --- Siemens PCCT recon-type + take_idx inputs (NOT used in build_name) ---
    # Consumed by name_builder.compute_recon_type and labeler take-grouping.
    image_type: List[str] = field(default_factory=list)
    image_comments: str = ""
    acquisition_datetime: str = ""
    series_number: str = ""


def _safe_get(
    ds: pydicom.Dataset,
    tag: str,
    fallback_tags: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> str:
    """Fetch a DICOM attribute with fallbacks.

    Returns UNKNOWN (and appends a warning) if neither the primary tag
    nor any fallback produces a non-empty value.
    """
    val = getattr(ds, tag, None)
    if val in (None, ""):
        if fallback_tags:
            for fb in fallback_tags:
                val = getattr(ds, fb, None)
                if val not in (None, ""):
                    break
    if val in (None, ""):
        if warnings is not None:
            warnings.append(f"Missing: {tag}")
        return UNKNOWN
    return str(val).strip()


def _infer_recon_kernel(
    ds: pydicom.Dataset,
    warnings: List[str],
    pcct: Optional[Dict[str, str]] = None,
) -> str:
    """Determine the reconstruction kernel.

    Preference order:
      1. ReconstructionAlgorithm (DICOM tag 0018,9315)
      2. Parse SeriesDescription for well-known Philips keywords
         (IMR, iDose, FBP)
      3. Siemens PCCT: iteration level token (``Q1``..``Q4``) from
         SeriesDescription — this is what distinguishes one PCCT recon
         from another (same base kernel, different Q-level).
      4. UNKNOWN (only then is a warning appended)

    Note: we deliberately do NOT pass ``warnings`` down into the first
    ``_safe_get`` call — otherwise a successful SeriesDescription
    fallback would still leave a misleading "Missing:
    ReconstructionAlgorithm" entry in the report.
    """
    recon = _safe_get(ds, "ReconstructionAlgorithm")  # silent probe
    if recon != UNKNOWN:
        return recon

    desc = str(getattr(ds, "SeriesDescription", "") or "").upper()
    if "IMR" in desc:
        return "IMR"
    if "IDOSE" in desc:
        return "iDose"
    if "FBP" in desc:
        return "FBP"

    # Siemens NAEOTOM Alpha / PCCT: no ReconstructionAlgorithm tag is set,
    # but the SeriesDescription carries the quantum iteration level which
    # differentiates otherwise-identical reconstructions (same kernel,
    # different Q-level).
    if pcct and pcct.get("q"):
        return f"Q{pcct['q']}"

    warnings.append("Missing: ReconstructionAlgorithm")
    return UNKNOWN


def _infer_mas(ds: pydicom.Dataset, warnings: List[str]) -> str:
    """Return mAs as a string.

    Prefers the Exposure tag (already mAs). Falls back to computing it
    from tube current and exposure time (mA * ms / 1000 = mAs).
    """
    mas = _safe_get(ds, "Exposure", warnings=warnings)
    if mas != UNKNOWN:
        return mas

    cur = getattr(ds, "XRayTubeCurrent", None)
    t = getattr(ds, "ExposureTime", None)
    if cur not in (None, "") and t not in (None, ""):
        try:
            computed = float(cur) * float(t) / 1000.0
            return str(int(round(computed)))
        except (TypeError, ValueError):
            pass
    return UNKNOWN


def estimate_recon_spacing(dcm_paths: List[str], sample_n: int = 60) -> Optional[float]:
    """Estimate physical reconstruction spacing (mm) from SliceLocations.

    Used as a fallback when ``SpacingBetweenSlices`` is absent (Siemens
    NAEOTOM PCCT scans don't populate it, for example).

    Strategy: read ``sample_n`` files from the head of ``dcm_paths``,
    collect their SliceLocations, sort + deduplicate, and return the
    SMALLEST non-zero diff between any two adjacent sorted positions.

    Why SMALLEST and not median:
      File-name sort is usually lexicographic (1, 10, 100, 101, ...
      not 1, 2, 3, ...), so the first N files do NOT correspond to
      the first N slices in z. The collected SliceLocations end up
      scattered across the volume with irregular gaps. The MEDIAN of
      sorted diffs reflects the average gap (= N × recon spacing for
      some N), not the true single-step recon spacing. The MIN, on
      the other hand, equals the recon spacing whenever any two
      sampled slices happen to be slice-adjacent — which is highly
      likely once we sample ~20+ files from a ~600-file series.

    Multi-slice-per-z is handled by dedup before computing diffs.
    """
    if len(dcm_paths) < 2:
        return None
    sampled = list(dcm_paths)[: sample_n + 1]
    locs: List[float] = []
    for p in sampled:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            sl = getattr(ds, "SliceLocation", None)
            if sl is not None:
                locs.append(float(sl))
        except Exception:
            pass
    if len(locs) < 2:
        return None
    unique_sorted = sorted({round(x, 4) for x in locs})
    if len(unique_sorted) < 2:
        return None
    diffs = [
        unique_sorted[i + 1] - unique_sorted[i]
        for i in range(len(unique_sorted) - 1)
    ]
    pos = [d for d in diffs if d > 1e-6]
    if not pos:
        return None
    return min(pos)


def extract_from_file(dcm_path: str) -> LabelMetadata:
    """Read a single DICOM file and extract label metadata.

    Uses ``stop_before_pixels=True`` so pixel data is NOT loaded — this
    makes header-only inspection dramatically faster, which matters
    when walking thousands of files.

    This function never raises on missing fields; every missing field
    becomes ``UNKNOWN`` and is appended to ``meta.warnings``.

    Siemens NAEOTOM Alpha / Photon-Counting CT:
      * The physical ``KVP`` tag is fixed (always 120 or 140) and does not
        correspond to the Virtual Monoenergetic image's kVp that readers
        actually care about. When SeriesDescription matches the known PCCT
        recon pattern, VMI kVp (80/100/120/140) is used instead.
      * ``ReconstructionAlgorithm`` is not populated — we fall back to the
        quantum iteration level (``Q1``..``Q4``) parsed from SeriesDescription
        so reconstructions that differ only by Q-level get distinct names.
    """
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    warnings: List[str] = []

    series_desc = str(getattr(ds, "SeriesDescription", "") or "")
    pcct = _parse_pcct_recon_desc(series_desc)

    # KVP: prefer VMI kVp from SeriesDescription on PCCT (physical KVP is
    # constant on photon-counting scanners, so it can't differentiate recons).
    if pcct and pcct.get("vmi"):
        kvp = pcct["vmi"]
    else:
        kvp = _safe_get(ds, "KVP", warnings=warnings)

    image_type = list(getattr(ds, "ImageType", []) or [])
    image_comments = str(getattr(ds, "ImageComments", "") or "")
    acq_dt = str(getattr(ds, "AcquisitionDateTime", "") or "")
    if not acq_dt:
        # Fallback: combine AcquisitionDate + AcquisitionTime when the merged
        # tag isn't populated (older Siemens exports).
        adate = str(getattr(ds, "AcquisitionDate", "") or "")
        atime = str(getattr(ds, "AcquisitionTime", "") or "")
        if adate or atime:
            acq_dt = f"{adate}{atime}"
    series_number = str(getattr(ds, "SeriesNumber", "") or "")

    return LabelMetadata(
        institution=_safe_get(ds, "InstitutionName", warnings=warnings),
        manufacturer=_safe_get(ds, "Manufacturer", warnings=warnings),
        device_model=_safe_get(
            ds, "ManufacturerModelName",
            fallback_tags=["StationName"],
            warnings=warnings,
        ),
        study_date=_safe_get(
            ds, "StudyDate",
            fallback_tags=["AcquisitionDate", "ContentDate"],
            warnings=warnings,
        ),
        recon_kernel=_infer_recon_kernel(ds, warnings, pcct=pcct),
        conv_kernel=_safe_get(ds, "ConvolutionKernel", warnings=warnings),
        kvp=kvp,
        mas=_infer_mas(ds, warnings),
        fov=_safe_get(
            ds, "ReconstructionDiameter",
            fallback_tags=["DataCollectionDiameter"],
            warnings=warnings,
        ),
        slice_thickness=_safe_get(ds, "SliceThickness", warnings=warnings),
        recon_spacing=_safe_get(
            ds, "SpacingBetweenSlices",
            # Some scanners only populate SliceLocation per slice; the
            # labeler computes the median diff later as a fallback. Keep
            # silent here (no warning) so the audit log isn't noisy when
            # we'll recover it downstream.
        ),
        source_path=dcm_path,
        series_description=series_desc,
        warnings=warnings,
        image_type=image_type,
        image_comments=image_comments,
        acquisition_datetime=acq_dt,
        series_number=series_number,
    )
