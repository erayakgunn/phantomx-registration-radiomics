"""Shared fixtures for the pcct-radiomics-mirp test suite.

The pipeline is a standalone script whose filename is not a valid module name,
so it is loaded through importlib rather than imported.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
PIPELINE_PATH = PROJECT_DIR / "extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py"

# Real labelled dataset. Read-only; tests that use it are marked `slow` and skip
# when it is absent so the suite still runs on a machine without the data.
REAL_DATASET = Path.home() / "Desktop" / "20260402_USZ_PCCT"

# The three acquisitions this pipeline is run on, one per scanner. They differ in
# ways that each broke something once (see test_multidevice.py), so the suite
# checks all three rather than treating one as representative:
#
#   USZ_PCCT  photon-counting, .dcm raw CT, IQ level in a Siemens private tag,
#             air noise reaching -1203 HU, and test-retest repeats
#   KSW       Philips spectral, IMR/iDose levels in Philips private tags,
#             air HARD-CLIPPED at exactly -1024 HU
#   EID       energy-integrating, .IMA raw CT, ADMIRE strength 1..5 carried as
#             the second element of a multi-valued ConvolutionKernel
DEVICE_DATASETS = {
    "USZ_PCCT": Path.home() / "Desktop" / "20260402_USZ_PCCT",
    "KSW": Path.home() / "Desktop" / "20260506_KSW",
    "EID": Path.home() / "Desktop" / "29062026-EID_LABELED",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the real dataset or a full MIRP run")


def sibling_project(*names: str) -> Path:
    """Locate a sibling project by any of the names it has been known under.

    These paths went stale and took five tests with them. The cross-project
    contract tests were pinned to ``../../labeling`` and ``../phantomxpro-mirp``;
    both projects were later renamed (``phantom-labeling``, ``organ-radiomics``)
    and the tests began skipping silently — the exact "dead safety net" this
    project warns about elsewhere, because a skip reads as "not applicable here"
    rather than "this check has not run since the rename".

    Accepting the historical names as aliases means a rename degrades to a
    lookup miss in ONE place instead of quietly disarming the checks. Searching
    both the parent and grandparent directory covers the two layouts the tests
    were originally written against.
    """
    for root in (PROJECT_DIR.parent, PROJECT_DIR.parent.parent):
        for name in names:
            candidate = root / name
            if candidate.is_dir():
                return candidate
    pytest.skip(f"kardeş proje bulunamadı (denenen adlar: {', '.join(names)})")


def labeling_project() -> Path:
    """The labelling pipeline that writes the ROI patches this project reads."""
    return sibling_project("phantom-labeling", "labeling")


def organ_radiomics_project() -> Path:
    """The sibling radiomics pipeline whose workbook conventions are shared."""
    return sibling_project("organ-radiomics", "phantomxpro-mirp")


def device_root(name: str) -> Path:
    """Dataset root for one scanner, skipping the test when it is not present."""
    root = DEVICE_DATASETS[name]
    if not root.is_dir():
        pytest.skip(f"{name} dataset not present at {root}")
    return root


def series_dirs(root: Path):
    """Series folders of a dataset, excluding the pipeline's own output folder."""
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name != "radiomics_output")


def real_hu(path: Path):
    """(hu_of_real_voxels, n_sentinel) for one patch file, or None without rescale.

    "Real" means everything except the exporter's out-of-ROI padding, which is
    stored as -32768. Those voxels are cropped away by the reader and must not be
    mistaken for data when characterising a scanner's air floor.
    """
    import numpy as np
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    if ds.get("RescaleSlope") is None or ds.get("RescaleIntercept") is None:
        return None
    stored = ds.pixel_array.astype(np.int32)
    hu = stored * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    sentinel = stored == -32768
    return hu[~sentinel], int(sentinel.sum())


@pytest.fixture(scope="session")
def pipeline():
    """The extraction script, loaded as a module."""
    spec = importlib.util.spec_from_file_location("pcct_pipeline", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pcct_pipeline"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def real_dataset():
    if not REAL_DATASET.is_dir():
        pytest.skip(f"real dataset not present at {REAL_DATASET}")
    return REAL_DATASET


def resolve_mirp_settings(**kwargs) -> dict:
    """Flatten MIRP's fully-resolved settings into a comparable dict.

    The point is to surface the parameters the pipeline never states — MIRP
    fills them from its own defaults, and they govern the result just as much
    as the ones that are written down.
    """
    from mirp.settings.import_config_parameters import import_configuration_settings

    s = import_configuration_settings(compute_features=True, **kwargs)[0]

    def spacing(value):
        # new_spacing is nested lists; make it hashable/comparable.
        if value is None:
            return None
        return tuple(tuple(float(x) for x in row) for row in value)

    return {
        "by_slice": s.general.by_slice,
        "ibsi_compliant": s.general.ibsi_compliant,
        "no_approximation": s.general.no_approximation,
        "interpolate": s.img_interpolate.interpolate,
        "new_spacing": spacing(s.img_interpolate.new_spacing),
        "spline_order": s.img_interpolate.spline_order,
        "anti_aliasing": s.img_interpolate.anti_aliasing,
        "smoothing_beta": s.img_interpolate.smoothing_beta,
        "roi_spline_order": s.roi_interpolate.spline_order,
        "resegmentation_method": tuple(s.roi_resegment.resegmentation_method),
        "resegmentation_range": tuple(s.roi_resegment.intensity_range)
        if s.roi_resegment.intensity_range is not None else None,
        "discretisation_method": tuple(s.feature_extr.discretisation_method),
        "discretisation_bin_width": tuple(s.feature_extr.discretisation_bin_width)
        if s.feature_extr.discretisation_bin_width is not None else None,
        "ivh_discretisation_method": s.feature_extr.ivh_discretisation_method,
        "glcm_distance": tuple(s.feature_extr.glcm_distance),
        "glcm_spatial_method": tuple(s.feature_extr.glcm_spatial_method),
        "glrlm_spatial_method": tuple(s.feature_extr.glrlm_spatial_method),
        "glszm_spatial_method": tuple(s.feature_extr.glszm_spatial_method),
        "ngldm_distance": tuple(s.feature_extr.ngldm_dist),
        "ngldm_difference_level": tuple(s.feature_extr.ngldm_diff_lvl),
        "stat_percentile": tuple(s.feature_extr.stat_percentile),
        "spatial_filters": s.img_transform.spatial_filters,
        "intensity_normalisation": s.post_process.intensity_normalisation,
        "denoise_method": s.post_process.image_denoise_method,
    }


def make_patch_dicom(
    path: Path,
    hu: np.ndarray,
    *,
    slice_index: int,
    slope: float = 1.0,
    intercept: float = -1024.0,
    padding_stored: int | None = -32768,
    include_rescale: bool = True,
) -> None:
    """Write one synthetic ROI patch that mimics the labelling exporter.

    ``hu`` may contain NaN; those voxels become ``padding_stored``, exactly as
    ``output/exporter.py::_hu_to_int16`` does.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    stored = np.empty(hu.shape, dtype=np.int16)
    valid = ~np.isnan(hu)
    stored[valid] = np.clip(hu[valid] - intercept, -32767, 32767).astype(np.int16)
    if padding_stored is not None:
        stored[~valid] = padding_stored

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.1234.5678.9999.1"

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.Modality = "CT"
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SliceLocation = float(slice_index)
    ds.InstanceNumber = int(slice_index)
    ds.Rows, ds.Columns = int(hu.shape[0]), int(hu.shape[1])
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = stored.tobytes()
    if include_rescale:
        ds.RescaleSlope = str(int(slope))
        ds.RescaleIntercept = str(int(intercept))
        ds.RescaleType = "HU"
    if padding_stored is not None:
        ds.PixelPaddingValue = padding_stored
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def make_disc_patch_dir(
    directory: Path,
    *,
    size: int = 19,
    n_slices: int = 5,
    fill_hu: float = 50.0,
    radius: float | None = None,
) -> Path:
    """A patch folder shaped like the exporter's circular ROIs: disc of real HU
    inside a square, corners left as padding."""
    if radius is None:
        radius = size / 2.0
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    inside = (xx - centre) ** 2 + (yy - centre) ** 2 <= radius ** 2

    for k in range(n_slices):
        hu = np.full((size, size), np.nan, dtype=np.float64)
        hu[inside] = fill_hu + k
        make_patch_dicom(directory / f"slice_{k:04d}.dcm", hu, slice_index=k)
    return directory
