"""Phase 0: DICOM loading, HU calibration, and 3D volume construction."""

import hashlib
from collections import defaultdict
import numpy as np
import pydicom
from typing import Dict, List, Tuple, Any

from utils.logger import log
from utils.file_utils import find_dicom_files
import config


def _deduplicate_slices(
    slices: List[pydicom.Dataset], warnings: List[str]
) -> List[pydicom.Dataset]:
    """Drop DICOM files that are duplicates of one another.

    Some input datasets contain multiple DICOM files per physical z
    position — typically because the source data was anonymized /
    re-exported more than once, producing fresh files with identical
    pixel arrays but slightly different metadata (e.g. updated
    ``AttributeModificationDateTime``). This inflates the per-series
    file count by 2-3x without adding any physical information.

    Algorithm:
      1. Group input slices by ``SliceLocation`` (rounded to 0.1 µm).
      2. For each group with > 1 file, compute MD5 of the raw
         ``PixelData`` bytes for every file in the group.
      3. If ALL files share the same pixel hash → keep the FIRST one,
         drop the rest. Log how many were dropped.
      4. If files at the same z have DIFFERING pixel hashes (real
         dual-energy / dual-detector reconstruction), KEEP ALL of
         them and emit a warning so the operator knows to investigate.

    The returned list is re-sorted by SliceLocation so downstream code
    sees the same ordering convention regardless of how many duplicates
    were dropped.
    """
    if len(slices) < 2:
        return list(slices)

    by_z: Dict[float, List[pydicom.Dataset]] = defaultdict(list)
    no_z: List[pydicom.Dataset] = []
    for ds in slices:
        try:
            z = round(float(ds.SliceLocation), 4)
        except (AttributeError, TypeError, ValueError):
            no_z.append(ds)
            continue
        by_z[z].append(ds)

    kept: List[pydicom.Dataset] = list(no_z)
    n_identical_dropped = 0
    n_differing_z = 0
    differing_z_examples: List[float] = []
    for z, group in by_z.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # Hash pixel data for every file in this z-group.
        hashes: List[str] = []
        for ds in group:
            try:
                hashes.append(hashlib.md5(ds.PixelData).hexdigest())
            except Exception:
                hashes.append(f"unhashable-{id(ds)}")
        unique_hashes = set(hashes)
        if len(unique_hashes) == 1:
            # All identical — keep the first, drop the rest.
            kept.append(group[0])
            n_identical_dropped += len(group) - 1
        else:
            # Different pixel data — possible real multi-energy.
            # Keep ALL files; downstream code (section trim) already
            # handles multi-slice-per-z via unique-z counting.
            kept.extend(group)
            n_differing_z += 1
            if len(differing_z_examples) < 3:
                differing_z_examples.append(z)

    if n_identical_dropped > 0:
        msg = (
            f"Dropped {n_identical_dropped} duplicate DICOM file(s) "
            f"with identical pixel data at the same SliceLocation "
            f"(re-anonymization / re-export artefact)"
        )
        warnings.append(msg)
        log.info(msg)
    if n_differing_z > 0:
        msg = (
            f"WARNING: {n_differing_z} SliceLocation(s) have multiple "
            f"files with DIFFERING pixel data (possible dual-energy / "
            f"multi-detector reconstruction). KEEPING ALL — review with "
            f"diagnostic/spectral_detector.py. Examples: "
            f"{differing_z_examples}"
        )
        warnings.append(msg)
        log.warning(msg)

    kept.sort(
        key=lambda s: float(getattr(s, "SliceLocation", 0.0)) or 0.0,
    )
    return kept


class DicomVolume:
    """Container for a loaded DICOM series volume."""

    def __init__(
        self,
        volume: np.ndarray,
        pixel_spacing: Tuple[float, float],
        slice_locations: np.ndarray,
        slice_thickness: float,
        metadata: Dict[str, Any],
        warnings: List[str],
    ):
        self.volume = volume  # (N, H, W) float32 in HU
        self.pixel_spacing = pixel_spacing  # (row_spacing, col_spacing) in mm
        self.slice_locations = slice_locations
        self.slice_thickness = slice_thickness
        self.metadata = metadata
        self.warnings = warnings

    @property
    def num_slices(self) -> int:
        return self.volume.shape[0]

    @property
    def height(self) -> int:
        return self.volume.shape[1]

    @property
    def width(self) -> int:
        return self.volume.shape[2]

    def cm_to_pixels(self, cm: float) -> float:
        """Convert centimeters to pixels using pixel spacing."""
        mm = cm * 10.0
        avg_spacing = (self.pixel_spacing[0] + self.pixel_spacing[1]) / 2.0
        return mm / avg_spacing


def load_series(series_dir: str) -> DicomVolume:
    """Load a DICOM series from a directory and build a calibrated HU volume."""
    log.info(f"Loading DICOM series from: {series_dir}")

    dcm_files = find_dicom_files(series_dir)
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    log.info(f"Found {len(dcm_files)} DICOM files")

    # Read all slices
    slices = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp)
            if hasattr(ds, "SliceLocation") and hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception as e:
            log.warning(f"Skipping {fp}: {e}")

    if not slices:
        raise ValueError(f"No valid DICOM slices in {series_dir}")

    # Sort by SliceLocation ascending
    slices.sort(key=lambda s: float(s.SliceLocation))

    # Deduplicate identical-pixel files at the same SliceLocation.
    # See _deduplicate_slices for the algorithm and rationale. The
    # warnings list is shared with the rest of the loader so the audit
    # surface (DicomVolume.warnings) records how many were dropped.
    warnings: List[str] = []
    n_before = len(slices)
    slices = _deduplicate_slices(slices, warnings)
    n_after = len(slices)
    if n_after < n_before:
        log.info(
            f"DICOM dedup: {n_before} files -> {n_after} files "
            f"(removed {n_before - n_after} identical-pixel duplicates)"
        )

    # Extract metadata from first slice
    ds0 = slices[0]
    if not hasattr(ds0, "PixelSpacing"):
        raise ValueError(f"DICOM missing PixelSpacing attribute in {series_dir}")
    pixel_spacing = (
        float(ds0.PixelSpacing[0]),
        float(ds0.PixelSpacing[1]),
    )
    # SliceThickness may be present-but-None on some scanners (Siemens
    # NAEOTOM topograms, for example, carry the DICOM tag set to an empty
    # value). Fall back to 1.0 mm so downstream math doesn't explode; the
    # scout-filter in labeler.py should keep topograms from reaching here
    # in normal use, but this guard makes the loader robust on its own.
    _st = getattr(ds0, "SliceThickness", 1.0)
    slice_thickness = float(_st) if _st is not None else 1.0

    if not hasattr(ds0, "Rows") or not hasattr(ds0, "Columns"):
        raise ValueError(f"DICOM missing Rows/Columns attribute in {series_dir}")
    metadata = {
        "patient_id": str(getattr(ds0, "PatientID", "")),
        "study_description": str(getattr(ds0, "StudyDescription", "")),
        "series_description": str(getattr(ds0, "SeriesDescription", "")),
        "series_number": str(getattr(ds0, "SeriesNumber", "")),
        "kvp": str(getattr(ds0, "KVP", "")),
        "rows": int(ds0.Rows),
        "columns": int(ds0.Columns),
    }

    # Build volume with HU calibration. `warnings` is already initialised
    # above before the dedup step; reusing it preserves dedup messages.
    slice_locations = []
    arrays = []

    # Determine the most common image shape across all slices
    from collections import Counter
    shape_counts = Counter()
    for ds in slices:
        shape_counts[(int(ds.Rows), int(ds.Columns))] += 1
    expected_shape = shape_counts.most_common(1)[0][0]
    expected_rows, expected_cols = expected_shape
    if len(shape_counts) > 1:
        log.warning(
            f"Mixed image shapes detected: {dict(shape_counts)}. "
            f"Using most common: {expected_rows}x{expected_cols}"
        )
        # Update metadata and pixel_spacing from a slice with the expected shape
        for ds in slices:
            if (int(ds.Rows), int(ds.Columns)) == expected_shape:
                pixel_spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
                metadata["rows"] = expected_rows
                metadata["columns"] = expected_cols
                break

    for ds in slices:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixel_data = ds.pixel_array.astype(np.float32)
        if pixel_data.shape != (expected_rows, expected_cols):
            continue
        hu = pixel_data * slope + intercept
        arrays.append(hu)
        slice_locations.append(float(ds.SliceLocation))

    if not arrays:
        raise ValueError(f"No valid slices found in {series_dir}")

    volume = np.stack(arrays, axis=0)
    slice_locations = np.array(slice_locations, dtype=np.float64)

    # Check slice spacing consistency
    if len(slice_locations) > 1:
        spacings = np.diff(slice_locations)
        median_spacing = np.median(spacings)
        if median_spacing > 0:
            deviations = np.abs(spacings - median_spacing) / median_spacing
            bad_indices = np.where(deviations > config.SLICE_SPACING_TOLERANCE)[0]
            if len(bad_indices) > 0:
                msg = (
                    f"Slice spacing anomalies at indices {bad_indices.tolist()} "
                    f"(median spacing={median_spacing:.3f}mm)"
                )
                warnings.append(msg)
                log.warning(msg)

    log.info(
        f"Volume built: {volume.shape}, pixel_spacing={pixel_spacing}, "
        f"slice_thickness={slice_thickness}"
    )

    return DicomVolume(
        volume=volume,
        pixel_spacing=pixel_spacing,
        slice_locations=slice_locations,
        slice_thickness=slice_thickness,
        metadata=metadata,
        warnings=warnings,
    )
