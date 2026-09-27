"""
Radiomics extraction for the PCCT CalibraMics PhantomXPro INSERT-only dataset.

Modified version:
    - CT image input is read directly from DICOM files in <series>/dicoms/**/*.dcm.
    - Insert ROI masks/labels are read directly from DICOM files under
      <series>/Inserts/5mm/Insert_XX/*.dcm and 10mm/Insert_XX/*.dcm
      (see INSERT_SIZES; a 15mm folder on disk is ignored).
    - .npy patches and metadata.json are not required.
    - Inserts, Stars and Textures are all processed; see GROUP_FEATURE_POLICY
      for the per-group feature families.
    - Inserts are extracted separately for each size in INSERT_SIZES.

FIDELITY TO THE ACQUIRED DATA is the governing rule here. Nothing is invented
and nothing real is thrown away between the DICOM and MIRP:

    * No value is ever substituted. Missing RescaleSlope/Intercept,
      SliceThickness or slice-ordering tags raise instead of defaulting, so a
      gap in the data surfaces rather than being papered over.
    * Acquisition metadata comes from the DICOM tags only, never from folder
      names — those are human-typed and demonstrably wrong on some series
      (see _audit_series_name_against_tags).
    * No voxel is removed by resegmentation. MIRP IS given an intensity range,
      because it is the only way to fix the fixed-bin-size grey-level origin,
      but the range brackets everything a CT can produce, so it excludes
      nothing. If a voxel ever did fall below the anchor the run raises rather
      than letting the value be dropped or clamped (see RESEG note below).
    * No exporter padding: circular ROIs are cropped to a padding-free square
      rather than masked, so the -33792 sentinel never enters the image.
    * NO FILTER LAYER. IBSI-2 response-map features are not computed. Every
      convolution has to invent values beyond the image edge — MIRP offers only
      reflect / constant / nearest / mirror / wrap, all of which synthesise
      data, and there is no "valid region" mode. On these cropped patches the
      invented border dominates: at FOV 300 the LoG kernel spans 21x21 px on a
      33x33 px star patch, so only a 13x13 core would be free of it. A feature
      that exists only because the edge was fabricated is not a measurement of
      the phantom, so it is not reported. Filters can be restored the moment
      the labelling export includes a margin of REAL surrounding tissue at
      least as wide as the largest kernel radius.

Pipeline (per series, per ROI):
    1. Read the cropped DICOM patches for the ROI.
    2. Convert patch pixels to HU using RescaleSlope/RescaleIntercept.
    3. Crop circular ROIs to their largest padding-free square; write the
       NIfTI image + an all-ones mask.
    4. Run MIRP (IBSI-1 compliant) for the group's base feature families.

Output is one combined Excel including an insert_ranking sheet (rank = insert number, i.e. the LABELING order, so the sheet never diverges from the overlay/`label` numbering; mean HU across series and insert sizes is reported alongside); rows are sorted by insert number, then protocol/series, then insert size.

Self-contained: all IBSI parameter sets and MIRP wrappers are defined inline;
no external project imports.

Usage:
    py -3.14 extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py --dry-run
    py -3.14 extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py --max-series 1
    py -3.14 extract_pcct_phantom_radiomics_dicom_inserts_only_v4.py
"""

import argparse
import functools
import hashlib
import json
import math
import multiprocessing
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# MIRP imports matplotlib in its core image module. In a frozen console .exe
# matplotlib would otherwise probe for a GUI backend on a machine that may have
# none, which either slows startup or raises. Agg needs no display and nothing
# here draws anything.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

warnings.filterwarnings("ignore")

# HU floor used to distinguish real ROI voxels from the labelling pipeline's
# out-of-ROI padding. The exporter stores padding as int16 -32768, which
# reloads (RescaleSlope=1, Intercept=-1024) to HU ~ -33792. Real values are
# clipped to stored -32767 (HU -33791), so any threshold strictly between
# -33791 and the lowest real CT value separates the two populations.
#
# This is deliberately NOT the discretisation anchor. The two were one constant
# until 2026-07-29 and that coupling was a silent multi-site hazard: at a site
# whose air reconstructs below the anchor, real voxels were reclassified as
# padding, which moves the largest-inscribed-square crop and therefore changes
# the ROI geometry itself. Measured on a synthetic 19x19 disc with a -2100 HU
# air pocket: the emitted ROI collapsed from 13x13 to 7x7 (845 -> 245 voxels)
# and the air population vanished, with no warning. Placing this floor far
# below any achievable CT value removes that failure mode entirely; it costs
# nothing, because this constant only classifies, it never enters a feature.
_PADDING_HU_FLOOR = -20000.0

# Insert ROI sizes that are extracted, in report order. A 15mm folder on disk is
# ignored: all three sizes sample the same cylinder, so 15mm added runtime and a
# third correlated row per insert without adding information.
INSERT_SIZES: Tuple[str, ...] = ("5mm", "10mm")

# Stamped into every ``_per_series`` checkpoint. Bump it whenever a change makes
# previously written rows mean something different — ROI geometry, the feature
# set, the discretisation anchor, the meta columns. A checkpoint carrying a
# different version is re-extracted rather than reused, so one workbook can
# never mix rows from two pipeline versions. Checkpoints written before
# stamping existed report None and are therefore always re-extracted.
#
# v5: recon_kernel / k_strength are now parsed from a multi-valued
#     ConvolutionKernel instead of stringified (see _split_convolution_kernel).
#     A v4 checkpoint from a SOMATOM series carries recon_k "['Br40f', '3']" and
#     an empty k_strength, so reusing it would mix two spellings of the same
#     kernel in one workbook.
# v6: k_strength additionally falls back to the Philips iDose level and IMR label
#     (see _reconstruction_identity), so a v5 Philips checkpoint carries an empty
#     strength where this version reports 1..4.
CHECKPOINT_SCHEMA_VERSION = 6
_CHECKPOINT_SCHEMA_SHEET = "pipeline"

# Grey-level origin for fixed-bin-size discretisation, and the reason it is
# expressed as a "resegmentation" range at all.
#
# MIRP has no separate anchor parameter: it takes the fixed-bin-size origin from
# the lower bound of the resegmentation range
# (mirp/_image_processing/discretise_image.py, "fixed_bin_size" branch). The
# range therefore has to be given. Nothing is removed by it — see
# _assert_within_discretisation_range, which refuses the ROI rather than letting
# MIRP silently drop or clamp a voxel that falls outside.
#
# WHY -2000, measured on this dataset rather than assumed:
#
#   Too shallow and real data is destroyed. MIRP falls back to
#   CTImage.get_default_lowest_intensity() = -1000.0 when no range is given, and
#   fixed-bin-size CLAMPS everything at or below the anchor into bin 1. Air
#   reconstructs with noise well under -1000: across the 80 series of
#   20260402_USZ_PCCT the lowest real ROI voxel is -1246 HU (sharp kernel
#   Qr64f; the softer Qr40f reaches -1168). An anchor at -1000 collapses 36.8%
#   of a honeycomb texture — its entire air population — into one grey level;
#   -1100 would clamp data in 23 of those 80 series, -1024 in all 80.
#
#   Too deep and discrimination is diluted. Every bin below the data is a
#   permanently empty leading bin, and the intensity-index-weighted families
#   lose separation as the offset grows. Measured, separating Honeylarge from
#   Diagonal texture: ih_mean 0.283 -> 0.117 -> 0.054 and szm_lgze
#   0.387 -> 0.168 -> 0.079 for anchors -2000 -> -4000 -> -8000. Difference-based
#   features such as cm_contrast are unaffected (1.092 throughout).
#
# -2000 keeps 754 HU of headroom over the worst observed value while staying a
# fixed declared constant rather than something fitted to one dataset, so it
# holds across scanners, kernels and sites. It is NOT tied to _PADDING_HU_FLOOR;
# conflating the two silently corrupted ROI geometry (see that constant).
_DISCRETISATION_ANCHOR_HU = -2000.0
_HU_ENCODING_CEILING = 31743.0          # 32767 stored + (-1024) intercept
DISCRETISATION_RANGE_HU = (_DISCRETISATION_ANCHOR_HU, _HU_ENCODING_CEILING)


# =========================================================================
# IBSI-COMPLIANT MIRP PARAMETERS (validated against IBSI-1 / IBSI-2)
# =========================================================================

def get_3d_base_params(bin_width=25.0):
    """IBSI-compliant 3D feature extraction parameters."""
    return dict(
        by_slice=False,
        base_feature_families="all",
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=float(bin_width),
        resegmentation_intensity_range=list(DISCRETISATION_RANGE_HU),
        ibsi_compliant=True,
        glcm_spatial_method=["3d_average", "3d_volume_merge"],
        glrlm_spatial_method=["3d_average", "3d_volume_merge"],
        glszm_spatial_method=["3d"],
        gldzm_spatial_method=["3d"],
        ngtdm_spatial_method=["3d"],
        ngldm_spatial_method=["3d"],
    )


def get_2d_base_params(bin_width=25.0):
    """IBSI-compliant 2D feature extraction parameters."""
    return dict(
        by_slice=True,
        base_feature_families="all",
        base_discretisation_method="fixed_bin_size",
        base_discretisation_bin_width=float(bin_width),
        resegmentation_intensity_range=list(DISCRETISATION_RANGE_HU),
        ibsi_compliant=True,
        glcm_spatial_method=["2d_average", "2d_slice_merge",
                             "2.5d_direction_merge", "2.5d_volume_merge"],
        glrlm_spatial_method=["2d_average", "2d_slice_merge",
                              "2.5d_direction_merge", "2.5d_volume_merge"],
        glszm_spatial_method=["2d", "2.5d"],
        gldzm_spatial_method=["2d", "2.5d"],
        ngtdm_spatial_method=["2d", "2.5d"],
        ngldm_spatial_method=["2d", "2.5d"],
    )


# =========================================================================
# MIRP wrappers
# =========================================================================

# Dropped from every output table. MIRP estimates Moran's I and Geary's C by
# Monte Carlo for ROIs of 1000+ voxels (mirp/_features/morph_3d_data.py:626) and
# seeds the sampler from a hash that folds in the file location, so the same
# voxels produce different values from a different output directory. They are
# therefore not reproducible and must not be reported. MIRP's public API cannot
# deselect individual features (generate_features is called without a `features`
# list in mirp/_workflows/standardWorkflow.py:487), so they are computed and
# discarded here; the sampler is cheap, the point is to keep them out of results.
# Setting no_approximation=True would make them exact instead, but that is an
# O(n^2) pairwise computation over 30k-70k voxel ROIs and is not tractable here.
_DROPPED_FEATURES = (
    # Monte-Carlo estimated by MIRP above 1000 voxels, seeded from a hash that
    # folds in the file location, so the same voxels give different values from
    # a different output directory (mirp/_features/morph_3d_data.py:626).
    "morph_moran_i", "morph_geary_c",
    # Intensity-volume-histogram VOLUME FRACTIONS. MIRP reuses the
    # resegmentation range as the IVH intensity axis
    # (mirp/_features/ivh_features.py:50), and this pipeline sets that range to
    # the int16 encoding limits so discretisation has a fixed anchor. The axis
    # therefore spans -2000..31743 HU while real data reaches ~400 HU, which
    # puts the 10% level at +1374 HU — above everything — and every volume
    # fraction collapses to exactly 0.0. Not a measurement, so not reported.
    # The ivh_i* threshold features are unaffected (they are real HU values)
    # and are kept. Giving IVH its own axis is possible only via
    # ivh_discretisation_method="fixed_bin_number", which derives the range
    # per ROI and so destroys comparability between ROIs — the very thing the
    # fixed anchor exists to protect.
    "ivh_v10", "ivh_v25", "ivh_v50", "ivh_v75", "ivh_v90",
    "ivh_diff_v10_v90", "ivh_diff_v25_v75",
    # ivh_auc integrates over the same distorted axis (a factor ~10 on a real
    # texture ROI). MIRP happens to withhold it today because it is flagged
    # ibsi_compliant=False and this pipeline asks for compliant features only —
    # but that is an unrelated switch, so the artefact is named here explicitly
    # rather than left depending on it.
    "ivh_auc",
    # Local-intensity peaks. For ROIs above 300 voxels MIRP convolves a 1 cm^3
    # sphere over the image with mode="constant", cval=-1000
    # (mirp/_features/local_intensity_features.py:96-105), i.e. it fabricates
    # air outside the patch — the same boundary invention that cost this
    # pipeline its filter layer. The kernel radius is 6.2 mm (10.6 px in-plane,
    # 6.2 slices), so on a 33x33x30 star patch almost every voxel is within
    # reach of the fabricated border. Measured on the real ROIs: replacing the
    # invented fill with edge replication moves BOTH features in 9 of 10
    # star/texture ROIs, by up to 159%. Not a measurement of the phantom.
    "loc_peak_loc", "loc_peak_glob",
)


def _drop_nonreproducible(df: pd.DataFrame) -> pd.DataFrame:
    """Remove feature columns that MIRP cannot compute reproducibly."""
    drop = [c for c in df.columns
            if any(c == f or c.startswith(f + "_") for f in _DROPPED_FEATURES)]
    return df.drop(columns=drop) if drop else df


def extract_base_features(image, mask, roi_name, modality, params):
    """Extract base radiomics features using MIRP."""
    from mirp import extract_features

    result = extract_features(
        image=str(image),
        mask=str(mask),
        roi_name=roi_name,
        image_modality=modality,
        **params,
    )
    if result and result[0] is not None:
        return _drop_nonreproducible(result[0])
    return pd.DataFrame()


# =========================================================================
# Feature classification + long-form table builder
# =========================================================================

def classify_feature(col):
    """Classify feature column into family name."""
    prefixes = [
        ("morph_", "Morphology"),
        ("li_", "Local Intensity"),
        ("stat_", "Statistics"),
        ("ih_", "Intensity Histogram"),
        ("ivh_", "Intensity Volume Histogram"),
        ("cm_", "Co-occurrence Matrix (GLCM)"),
        ("rlm_", "Run Length Matrix (GLRLM)"),
        ("szm_", "Size Zone Matrix (GLSZM)"),
        ("dzm_", "Distance Zone Matrix (GLDZM)"),
        ("ngt_", "Neighbourhood Grey Tone Diff (NGTDM)"),
        ("ngl_", "Neighbouring Grey Level Dep (NGLDM)"),
    ]
    for prefix, name in prefixes:
        if col.startswith(prefix):
            if "_2.5d_" in col or "_2_5d_" in col:
                return f"{name} (2.5D)"
            elif "_2d_" in col:
                return f"{name} (2D)"
            elif "_3d_" in col:
                return f"{name} (3D)"
            return name
    return "Other"


def build_feature_table(base_df):
    """Build clean feature table: Feature ID | Family | Source | Value."""
    rows = []

    feat_cols = [
        c for c in base_df.columns
        if not c.startswith("image_") and not c.startswith("sample_")
    ]
    for col in sorted(feat_cols):
        val = base_df[col].iloc[0] if not base_df.empty else None
        family = classify_feature(col)
        rows.append({
            "Feature ID": col,
            "Family": family,
            "Source": "Base",
            "Value": val,
        })

    return pd.DataFrame(rows)


# =========================================================================
# Dataset paths
# =========================================================================
#
# There is deliberately NO default dataset path. One used to live here — the
# author's own folder — and it made the .exe unusable for anybody else: a
# double-click scanned a directory that does not exist on their disk and
# reported "0 series" with nothing pointing at the real cause. The path now
# comes from the command line, from a folder dropped onto the .exe, or from a
# prompt (see _ask_for_dataset_root).
#
# The output folder is always ``<dataset root>/radiomics_output`` unless
# --output-dir overrides it, so results sit next to the data they came from and
# the checkpoints are found again on a resume.


# =========================================================================
# Series discovery
# =========================================================================

# Where each scanner records the reconstruction it used, and how strongly.
#
# There is no single standard tag for this, and all three devices in the project
# put it somewhere different. Reading only the Siemens private tag left
# k_strength empty on 172 of 252 series, which hid the primary experimental
# variable of two of the three acquisitions. None of these is the folder name:
# names are not a trusted source here (see _audit_series_name_against_tags), and
# the SOMATOM set literally spells "UNKNOWN" where the level would sit.
#
#   NAEOTOM Alpha      (0029,1043)                     IQ level 1..4
#   SOMATOM Edge Plus  ConvolutionKernel ['Br40f','3'] ADMIRE strength 1..5
#                      ConvolutionKernel 'Br40f'       no iterative strength
#   Philips 7500       (01F7,109B) "iDose Level"       iDose level (4 here)
#   Philips 7500       (00E1,1040) "Image Label"       IMR_BR_1 / IMR_LSh_3 /
#                                                      iDose4_Y-Sharp
_Q_STRENGTH_TAG = (0x0029, 0x1043)
_PHILIPS_IDOSE_LEVEL_TAG = (0x01F7, 0x109B)
_PHILIPS_IMAGE_LABEL_TAG = (0x00E1, 0x1040)

# Philips IMR labels end in the algorithm's level: IMR_BR_1, IMR_LSh_3.
# Deliberately anchored on the IMR prefix rather than "the last digit in the
# string": a bare trailing-digit rule reads 'Qr40f' as strength 40.
_IMR_LEVEL_RE = re.compile(r"^IMR.*?_(\d+)$", re.IGNORECASE)


def _split_convolution_kernel(value) -> Tuple[str, str]:
    """Split ConvolutionKernel into ``(kernel_name, strength)``.

    ConvolutionKernel is a multi-valued DICOM element, and the three scanners in
    this project populate it three different ways:

        NAEOTOM Alpha      'Qr40f'                 single value
        SOMATOM Edge Plus  ['Br40f', '3']          name + iterative strength
        SOMATOM Edge Plus  'Br36f'                 single value (no IR strength)
        Philips 7500       'IMR1,Routine'          ONE string that contains a comma

    Two things went wrong before this existed. First, ``str()`` on pydicom's
    MultiValue produces a PYTHON LIST REPR, so 70 of the 100 SOMATOM series wrote
    ``['Br40f', '3']`` into the recon_k column while the other 30 wrote ``Br40f``
    — the same kernel under two spellings, which silently breaks grouping and
    filtering in Excel on the column a protocol comparison is keyed on. Second,
    the strength element was thrown away, so k_strength was empty for all 100
    SOMATOM series even though the strength varies 1..5 across them and is the
    primary experimental variable of that acquisition.

    Only a purely numeric trailing element is treated as a strength; the Philips
    'IMR1,Routine' form is a single value and is returned unchanged, comma and
    all, because splitting it would invent a distinction the tag does not make.
    An empty strength means "not recorded by this scanner", which on the SOMATOM
    distinguishes the non-iterative reconstruction from the ADMIRE ones.
    """
    if value is None:
        return "", ""
    if isinstance(value, str):
        return value.strip(), ""
    try:
        parts = [str(v).strip() for v in value]
    except TypeError:
        return str(value).strip(), ""
    if not parts:
        return "", ""
    if len(parts) >= 2 and parts[-1].isdigit():
        return "\\".join(parts[:-1]), parts[-1]
    return "\\".join(parts), ""


def _reconstruction_identity(ds) -> Tuple[str, str]:
    """Return ``(kernel_name, strength)`` for a series, from DICOM tags only.

    One function so the ``series`` sheet and the feature sheets cannot end up
    disagreeing about the reconstruction — they read it through the same
    precedence:

      1. (0029,1043)              Siemens NAEOTOM IQ level
      2. ConvolutionKernel        SOMATOM ADMIRE strength (trailing element)
      3. (01F7,109B)              Philips iDose level
      4. (00E1,1040)              Philips IMR level, from the label suffix

    An empty strength means no scanner recorded one. That is meaningful rather
    than missing: on the SOMATOM it is what separates the non-iterative
    reconstruction from ADMIRE 1..5 of the same acquisition.
    """
    kernel_name, kernel_strength = _split_convolution_kernel(
        getattr(ds, "ConvolutionKernel", None))

    def tag_text(tag) -> str:
        elem = ds.get(tag)
        if elem is None or elem.value is None:
            return ""
        if isinstance(elem.value, bytes):
            return ""
        return str(elem.value).strip()

    def level(tag) -> str:
        """A private tag's value, accepted only if it looks like a level.

        Private tags are addressed by raw group/element, and the same group means
        different things to different vendors: group 0029 is "SIEMENS CT EXAM
        IMAGE" on the NAEOTOM but "SIEMENS CSA HEADER" on the SOMATOM. Element
        1043 happens to exist only in the former, so reading it by number is safe
        on these three devices — but a fourth scanner could put anything there.
        Requiring a small integer means a foreign value is ignored and the next
        source is tried, rather than a reconstruction strength of "b'\\x01\\x00'"
        reaching the workbook.
        """
        text = tag_text(tag)
        return text if text.isdigit() and len(text) <= 2 else ""

    for candidate in (level(_Q_STRENGTH_TAG),
                      kernel_strength,
                      level(_PHILIPS_IDOSE_LEVEL_TAG)):
        if candidate:
            return kernel_name, candidate

    imr = _IMR_LEVEL_RE.match(tag_text(_PHILIPS_IMAGE_LABEL_TAG))
    return kernel_name, (imr.group(1) if imr else "")


# The raw CT source DICOMs come in different file extensions depending on the
# scanner / export tool: Philips (and most PACS exports) use ``.dcm`` while
# Siemens SOMATOM / NAEOTOM native exports use ``.IMA``. (The ROI patch files
# the labelling pipeline writes are always ``.dcm`` — this helper is used ONLY
# for the raw ``dicoms/`` CT folder, which may be either.) Matching is
# case-insensitive; results are de-duplicated (a case-insensitive filesystem
# can otherwise return the same file under several patterns).
_CT_DICOM_PATTERNS = ("*.dcm", "*.DCM", "*.ima", "*.IMA")


def _has_dicom_magic(path: Path) -> bool:
    """True if the file carries the DICOM preamble marker.

    A DICOM part-10 file has a 128-byte preamble followed by the literal
    ``DICM``. Checking it is a positive identification, not a guess: no
    reasonable JSON, PNG or log file matches.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


@functools.lru_cache(maxsize=None)
def _list_ct_dicom_files(directory: Path, recursive: bool = True) -> Tuple[Path, ...]:
    """Return the raw CT DICOM files under *directory*.

    Extensions first, then CONTENT. Some exports drop the extension entirely —
    the 20260714 NAEOTOM set stores its slices as bare accession numbers
    (``33804459``) — and matching on ``*.dcm`` / ``*.IMA`` alone rejected all 76
    of its series with "no CT DICOM folder found". The whole dataset was
    invisible because of a filename convention.

    The content fallback runs ONLY when the extension patterns find nothing, so
    the common case costs exactly what it did before, and it identifies files by
    the DICOM marker rather than assuming that whatever sits in the folder must
    be a slice. That matters here: a series folder also holds metadata.json,
    overlays/*.png and painted/*.png, none of which pass.

    Cached because the answer cannot change during a run and the callers ask
    repeatedly — spacing, tags, full metadata and discovery each re-list the same
    folder. Without the cache the content fallback re-read the magic bytes of
    every slice on every call: a --dry-run over the 20260714 set went from 6
    seconds to 3m18s.
    """
    globber = directory.rglob if recursive else directory.glob
    seen: Dict[str, Path] = {}
    for pattern in _CT_DICOM_PATTERNS:
        for f in globber(pattern):
            seen[str(f).lower()] = f
    if seen:
        return tuple(seen[k] for k in sorted(seen))

    for f in sorted(globber("*")):
        if f.is_file() and _has_dicom_magic(f):
            seen[str(f).lower()] = f
    return tuple(seen[k] for k in sorted(seen))


def _first_dicom_file(dicom_dir: Path) -> Optional[Path]:
    files = _list_ct_dicom_files(dicom_dir)
    return files[0] if files else None


def _dicom_series_meta_from_header(series_dir: Path) -> Dict:
    """Read acquisition metadata from the CT DICOM header.

    This is the ONLY source of acquisition metadata — see
    ``_audit_series_name_against_tags`` for why folder names are not trusted.
    """
    meta = {
        "prefix": "",
        "study_date": "",
        "recon_kernel": "",
        "conv_kernel": "",
        "k_strength": "",
        "kvp": "",
        "mas": "",
        "fov": "",
        "slice_thickness_label": "",
        "repeat": "",
    }
    dicom_dir = _find_dicoms_dir(series_dir)
    if dicom_dir is None:
        return meta
    f = _first_dicom_file(dicom_dir)
    if f is None:
        return meta

    try:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
    except Exception:
        return meta

    meta["study_date"] = str(getattr(ds, "StudyDate", "") or "")
    kernel_name, kernel_strength = _reconstruction_identity(ds)
    meta["recon_kernel"] = kernel_name
    # NOTE: conv_kernel reads the same tag as recon_kernel, so it carries no
    # extra information — it is kept only so older checkpoints keep loading, and
    # it is dropped from every user-facing sheet by _COMPACT_DROP. The column
    # reported to users is k_strength, below.
    meta["conv_kernel"] = kernel_name
    meta["k_strength"] = kernel_strength
    meta["kvp"] = getattr(ds, "KVP", "") or ""
    # Exposure (mAs) ONLY. The previous fallback to XRayTubeCurrent silently put
    # a current in milliamps under a column headed mAs — a different physical
    # quantity, not a substitute. Verified across all 80 series of the reference
    # dataset: both tags are always present and they always differ (e.g. 100 mAs
    # vs 200 mA), so the fallback could only ever have reported the wrong
    # number. If Exposure is absent the column stays empty and says so; the tube
    # current remains available as a dcm_* column on the feature sheets.
    exposure = getattr(ds, "Exposure", None)
    if exposure in (None, ""):
        print(f"  WARN: no Exposure (mAs) tag in {series_dir.name}; the mAs "
              f"column will be empty. XRayTubeCurrent is a current in mA and is "
              f"NOT substituted.")
        meta["mas"] = ""
    else:
        meta["mas"] = exposure
    meta["fov"] = getattr(ds, "ReconstructionDiameter", "") or ""
    meta["slice_thickness_label"] = getattr(ds, "SliceThickness", "") or ""
    meta["repeat"] = ""
    return meta


# Acquisition metadata comes from the DICOM tags ONLY. The folder name is a
# human-typed label and is demonstrably wrong: on the 20260402 PCCT set 16 of 80
# series carry 80 or 100 kVp in the folder name — and in SeriesDescription —
# while the KVP tag says 120, because those were acquisition logging errors. The
# name is used solely as the series identifier.
#
# The tokens are still parsed, but ONLY to warn: the misleading name stays in
# column A of every sheet, so a reader seeing "..._100_100_300_..." next to a
# kVp column reading 120 needs to know which one is real.
# Matches the _<kVp>_<mAs>_<FOV>_<thickness>_ run of numbers that every naming
# convention in this project shares, with the scanner-specific tokens before it
# left loose. Anchoring on the numbers rather than on one vendor's layout keeps
# the audit alive for the Philips/KSW and SOMATOM/SEP series too — a regex tuned
# to NAEOTOM names would silently never fire on them, which is exactly when a
# safety net is most needed.
# ``\d+(?:p\d+)?`` also accepts this project's 'p'-for-decimal spelling
# (0p67, 1p5, 250p5). Without it a fractional thickness or FOV made the whole
# audit silently no-op — and a dead safety net is worse than none, because it
# looks alive.
_NAME_AUDIT_RE = re.compile(
    r"_(?P<kvp>\d{2,3})_(?P<mas>\d{1,4})_(?P<fov>\d{3,4}(?:p\d+)?)_"
    r"(?P<thick>\d(?:p\d+)?)(?:_|$)")
_Q_TOKEN_AUDIT_RE = re.compile(r"_Q(?P<q>\d+)_")

# (series name, field, name value, tag value) for every disagreement found.
_SERIES_META_CONFLICTS: List[Tuple[str, str, str, str]] = []

# Folders that look like a series but were rejected during discovery, as
# (folder, reason). Reported before extraction starts so a shortfall in the
# series count is explained rather than merely visible.
_REJECTED_SERIES: List[Tuple[str, str]] = []


def _audit_series_name_against_tags(series_name: str, tag_meta: Dict) -> None:
    """Record folder-name tokens that contradict the DICOM tags.

    Purely diagnostic — nothing here feeds a reported value.
    """
    m = _NAME_AUDIT_RE.search(series_name)
    if not m:
        return
    tokens = dict(m.groupdict())
    q = _Q_TOKEN_AUDIT_RE.search(series_name)
    tokens["q"] = q.group("q") if q else None
    # The kernel token is deliberately not audited: every convention spells it
    # differently from the DICOM tag (Qr40f vs Br36f-3 vs a Python list repr
    # ['Br40f', '3']), so comparing them would raise a permanent false alarm.
    for field, token in (("kvp", "kvp"), ("mas", "mas"), ("fov", "fov"),
                         ("slice_thickness_label", "thick"),
                         ("k_strength", "q")):
        name_value = tokens.get(token)
        tag_value = tag_meta.get(field, "")
        if name_value in ("", None) or tag_value in ("", None):
            continue
        try:
            same = float(str(name_value).replace("p", ".")) == float(tag_value)
        except (TypeError, ValueError):
            same = str(name_value).strip().lower() == str(tag_value).strip().lower()
        if not same:
            _SERIES_META_CONFLICTS.append(
                (series_name, field, str(name_value), str(tag_value)))


def _print_rejected_series() -> None:
    """Report folders that look like a series but were not picked up."""
    if not _REJECTED_SERIES:
        return
    seen = {}
    for name, reason in _REJECTED_SERIES:
        seen[name] = reason
    print("\n" + "!" * 70)
    print(f"REJECTED FOLDERS: {len(seen)} folder(s) look like a series but were "
          f"NOT processed.")
    print("They are absent from the workbook and from the series count.")
    print("!" * 70)
    for name, reason in sorted(seen.items()):
        print(f"  {name[-58:]}\n      {reason}")
    print("!" * 70)


def _print_series_meta_conflicts() -> None:
    """Report folder-name tokens that contradict the DICOM tags."""
    if not _SERIES_META_CONFLICTS:
        return
    by_field: Dict[str, List[Tuple[str, str, str]]] = {}
    for series_name, field, name_value, tag_value in _SERIES_META_CONFLICTS:
        by_field.setdefault(field, []).append((series_name, name_value, tag_value))
    print("\n" + "-" * 70)
    print(f"METADATA WARNING: folder names contradict the DICOM tags in "
          f"{len(_SERIES_META_CONFLICTS)} place(s).")
    print("Every reported value comes from the DICOM tags. The folder name is")
    print("used only as the series identifier, so column A may look misleading.")
    print("-" * 70)
    for field, items in sorted(by_field.items()):
        pairs: Dict[Tuple[str, str], int] = {}
        for _, name_value, tag_value in items:
            pairs[(name_value, tag_value)] = pairs.get((name_value, tag_value), 0) + 1
        for (name_value, tag_value), n in sorted(pairs.items()):
            print(f"  {field:22s} name says {name_value:>8s}, tag says "
                  f"{tag_value:<8s} x{n} series")
        for series_name, _, _ in items[:2]:
            print(f"      e.g. {series_name[-58:]}")
        if len(items) > 2:
            print(f"      ... and {len(items) - 2} more")
    print("-" * 70)


def _has_insert_dicom_masks(series_dir: Path) -> bool:
    """True if the ROI reader will actually find insert patches in this series.

    Supported layouts (defined by ``discover_insert_roi_dirs``, which this asks
    rather than reimplementing):
      Inserts/5mm/*.dcm
      Inserts/5mm/Insert_01/*.dcm
      Inserts/10mm/Insert_02/*.dcm

    Only ``INSERT_SIZES`` counts, so a series holding nothing but 15mm folders
    is correctly reported as having no insert ROIs rather than being queued and
    then yielding none.

    DELEGATING rather than deciding is the whole point. This gate used to have
    its own idea of "has insert DICOMs" — a recursive ``rglob`` plus a fuzzy
    ``(5|10)\\s*_?\\s*mm`` match on the folder name — while the reader looks for
    an EXACT ``Inserts/<size>`` name and a non-recursive ``*.dcm`` one level
    below it. The gate was therefore strictly more permissive, and the gap was
    silent in the worst way: a series whose patches sat one directory deeper
    (``Inserts/5mm/Insert_01/extra/*.dcm``), or whose folder was spelled
    ``5 mm``, passed the gate, produced ZERO insert ROIs, and was recorded
    nowhere — not in _REJECTED_SERIES, not in _SKIPPED_ROIS, because no ROI was
    ever enumerated to skip. With readable Stars/Textures the run still exited 0
    while 80 insert ROIs per series were missing. Deriving the answer from the
    reader makes that divergence impossible to reintroduce.
    """
    return bool(discover_insert_roi_dirs(series_dir))


def _find_dicoms_dir(series_dir: Path) -> Optional[Path]:
    """Return the directory that contains CT DICOM slices for a series.

    Tries (in order):
      1. <series>/dicoms/   — standard layout
      2. <series>/DICOM/    — alternate capitalisation
      3. <series>/          — flat layout (DICOMs directly in series folder)

    Returns None when no CT DICOM file (.dcm or .IMA) can be found under the
    candidate.
    """
    for candidate_name in ("dicoms", "DICOM", "Dicom"):
        candidate = series_dir / candidate_name
        if candidate.is_dir() and _list_ct_dicom_files(candidate):
            return candidate
    # Flat layout: DICOMs directly inside the series folder (not inside
    # Inserts/, which belongs to mask discovery).
    flat = _list_ct_dicom_files(series_dir, recursive=False)
    if flat:
        return series_dir
    return None


def discover_series(dataset_root: Path) -> List[Dict]:
    """Find series folders containing CT DICOMs + DICOM insert masks.

    Supported layouts (searched up to 3 levels deep):
      <root>/<series>/dicoms/**/*.dcm          + <series>/Inserts/…
      <root>/<group>/<series>/dicoms/**/*.dcm  + <series>/Inserts/…
      <root>/<date>/<group>/<series>/…

    The ``dicoms`` sub-folder may also be named ``DICOM`` or the DICOMs
    may sit directly inside the series folder (flat layout).

    metadata.json is no longer required; ROI masks come from DICOM files
    under Inserts/<size> for the sizes in INSERT_SIZES.
    """
    def _try_series(entry: Path) -> Optional[Dict]:
        """Return a series dict if *entry* looks like a valid series folder."""
        if not entry.is_dir():
            return None
        # A folder carrying any of the ROI/CT sub-folders was clearly MEANT to
        # be a series. If such a folder is then rejected, that is a data problem
        # the user has to see: otherwise an 80-series dataset quietly yields 77
        # and the count is the only clue.
        looks_intended = any((entry / n).exists()
                             for n in ("dicoms", "DICOM", "Inserts", "Stars", "Textures"))
        dicom_dir = _find_dicoms_dir(entry)
        if dicom_dir is None:
            if looks_intended:
                _REJECTED_SERIES.append((entry.name, "no CT DICOM folder found"))
            return None
        if not _has_insert_dicom_masks(entry):
            if not looks_intended:
                # Holds CT DICOMs but none of the marker sub-folders, so there is
                # no evidence it was ever meant to be a series. Reporting it
                # would be noise — and specifically self-inflicted noise: when
                # no series is found at level 0 the scan descends, so a rejected
                # series' own ``dicoms/`` folder is visited as a candidate and
                # would be listed as a second rejection of the same problem.
                # The same guard already protects the CT branch above.
                return None
            # Distinguish "not a series" from a near miss, because the two need
            # different actions from the reader: the first is expected noise,
            # the second is a layout typo that costs every insert ROI.
            if (entry / "Inserts").exists():
                reason = (
                    f"Inserts/ exists but holds no readable patches. *.dcm must "
                    f"sit directly in Inserts/<{'|'.join(INSERT_SIZES)}>/ or one "
                    f"Insert_NN level below it, and the size folder name must "
                    f"match exactly — '5 mm' and deeper nesting are not read."
                )
            else:
                reason = f"no Inserts/{'|'.join(INSERT_SIZES)} folder"
            _REJECTED_SERIES.append((entry.name, reason))
            return None
        # Acquisition metadata comes from the DICOM tags only — never from the
        # folder name (see _audit_series_name_against_tags for why).
        meta = _dicom_series_meta_from_header(entry)
        _audit_series_name_against_tags(entry.name, meta)
        meta["series_name"] = entry.name
        meta["series_dir"] = entry
        # Store the resolved dicoms path so downstream code can use it.
        meta["dicoms_dir"] = dicom_dir
        return meta

    def _scan(root: Path) -> List[Dict]:
        out = []
        try:
            children = sorted(root.iterdir())
        except PermissionError:
            return out
        for entry in children:
            result = _try_series(entry)
            if result is not None:
                out.append(result)
        return out

    # Level 0: series directly inside dataset_root
    series = _scan(dataset_root)
    if series:
        return series

    # Level 1: one intermediate folder (e.g. root/<date>/<series>)
    try:
        top_entries = sorted(dataset_root.iterdir())
    except PermissionError:
        return series

    for entry in top_entries:
        if entry.is_dir():
            series.extend(_scan(entry))

    if series:
        return series

    # Level 2: two intermediate folders (e.g. root/<date>/<group>/<series>)
    for entry in top_entries:
        if not entry.is_dir():
            continue
        try:
            for sub in sorted(entry.iterdir()):
                if sub.is_dir():
                    series.extend(_scan(sub))
        except PermissionError:
            continue

    return series


# =========================================================================
# DICOM spacing + metadata
# =========================================================================

def read_dicom_spacing(dicom_dir: Path) -> Tuple[float, float, float]:
    """Return voxel spacing (sx, sy, sz) in mm — SimpleITK X/Y/Z order."""
    files = _list_ct_dicom_files(dicom_dir)
    if not files:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")
    ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    sy, sx = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])
    # No fallback: z spacing scales volume and every morphological feature, so a
    # substituted value would silently produce wrong geometry. If the scanner
    # did not record it, that is a fact about the data and must surface.
    raw_sz = getattr(ds, "SliceThickness", None)
    try:
        sz = float(raw_sz)
    except (TypeError, ValueError):
        raise ValueError(
            f"SliceThickness is {raw_sz!r} in {dicom_dir} — cannot determine "
            f"z spacing. Nothing is assumed; fix the source data."
        ) from None
    if sz <= 0.0:
        raise ValueError(
            f"SliceThickness is {sz} in {dicom_dir} — cannot determine z spacing."
        )
    return (sx, sy, sz)


_ME_KEV_RE = re.compile(r"ME[_ ]?(\d+)\s*keV", re.IGNORECASE)
# Philips Spectral CT: ImageType has MONO_ENERGY, ImageComments like
# "MonoE45keV , IMR_BR_3, Spectral (3)".
_MONOE_KEV_RE = re.compile(r"MonoE\s*[_ ]?(\d+)\s*keV", re.IGNORECASE)


def _parse_me_kev(comments: str) -> str:
    for rx in (_ME_KEV_RE, _MONOE_KEV_RE):
        m = rx.search(comments or "")
        if m:
            return m.group(1)
    return ""


def _derive_recon_type(image_type: List[str], comments: str) -> Tuple[str, str]:
    """Derive (recon_type_label, me_keV_str) from ImageType + ImageComments."""
    if "VMI" in image_type:
        kev = _parse_me_kev(comments)
        if kev:
            return f"MonoE{kev}keV", kev
        return "VMI_unknown", ""
    if "MONO_ENERGY" in image_type:
        kev = _parse_me_kev(comments)
        if kev:
            return f"MonoE{kev}keV", kev
        return "MonoE_unknown", ""
    if "THRESHOLD" in image_type:
        return "PolyE", ""
    # Neither spectral nor a photon-counting threshold image: a conventional
    # reconstruction. Label it by whatever marker ImageType carries in position 3
    # ("CT_SOM5 SPI" on the SOMATOM), and say "conventional" rather than emitting
    # a bare "OTHER_" when there is no marker at all — the Philips 7500 writes
    # only ORIGINAL,PRIMARY,AXIAL, which used to produce a trailing-underscore
    # placeholder on 54 series that reads like a truncation bug.
    other_marker = image_type[3] if len(image_type) > 3 else ""
    return (f"OTHER_{other_marker}" if other_marker else "conventional"), ""


def read_dicom_tags(dicom_dir: Path) -> Dict[str, str]:
    """Read the small set of DICOM tags shown on the 'series' summary sheet."""
    empty = {
        "image_type": "", "recon_type": "", "me_keV": "",
        "acq_dt": "", "series_no": "",
        "recon_diam_mm": "", "q_strength": "",
    }
    files = _list_ct_dicom_files(dicom_dir)
    if not files:
        return empty
    ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    image_type = list(getattr(ds, "ImageType", []) or [])
    comments = str(getattr(ds, "ImageComments", "") or "")
    recon_type, me_keV = _derive_recon_type(image_type, comments)
    # Through the same helper as _dicom_series_meta_from_header, so the series
    # sheet and the feature sheets cannot disagree about the reconstruction.
    _, q_strength = _reconstruction_identity(ds)
    return {
        "image_type": ",".join(image_type),
        "recon_type": recon_type,
        "me_keV": me_keV,
        "acq_dt": str(getattr(ds, "AcquisitionDateTime", "") or ""),
        "series_no": str(getattr(ds, "SeriesNumber", "") or ""),
        "recon_diam_mm": str(getattr(ds, "ReconstructionDiameter", "") or ""),
        "q_strength": q_strength,
    }


# Binary VRs we skip — too large and not human-readable.
_BINARY_VRS = {"OB", "OW", "OD", "OF", "OL", "OV", "UN"}


def _format_dicom_value(elem) -> str:
    """Render a DICOM element's value as a compact string for Excel cells."""
    vr = getattr(elem, "VR", "")
    if vr in _BINARY_VRS:
        try:
            n = len(elem.value) if elem.value is not None else 0
        except TypeError:
            n = 0
        return f"<binary:{vr}, {n} bytes>"
    if vr == "SQ":
        try:
            return f"<sequence: {len(elem.value)} items>"
        except TypeError:
            return "<sequence>"
    v = elem.value
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "\\".join("" if x is None else str(x) for x in v)
    if isinstance(v, bytes):
        return f"<bytes:{len(v)}>"
    return str(v)


def _column_for_element(elem) -> str:
    """Stable, readable column name for a DICOM element."""
    kw = getattr(elem, "keyword", "") or ""
    if kw:
        return kw
    tag = elem.tag
    name = (getattr(elem, "name", "") or "").strip()
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    base = f"({tag.group:04X},{tag.element:04X})"
    return f"{base}_{name}" if name and name != "Private_tag_data" else base


def read_dicom_full_metadata(dicom_dir: Path) -> Dict[str, str]:
    """Return ALL DICOM header tags from the first slice of a series."""
    files = _list_ct_dicom_files(dicom_dir)
    if not files:
        return {}
    ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    out: Dict[str, str] = {}
    for elem in ds:
        if elem.tag == 0x7FE00010:
            continue
        col = _column_for_element(elem)
        out[col] = _format_dicom_value(elem)
    return out


# =========================================================================
# DICOM image volume + DICOM insert mask discovery
# =========================================================================

# ROIs that were discovered on disk but produced no rows, as
# (series, group, roi, reason). A dropped ROI is invisible in the workbook —
# it is simply a missing row among hundreds — so every skip is recorded here
# and reported at the end of the run instead of scrolling past as one WARN.
_SKIPPED_ROIS: List[Tuple[str, str, str, str]] = []


def _note_skipped_roi(group: str, roi: str, reason: str) -> None:
    _SKIPPED_ROIS.append(("", group, roi, str(reason).splitlines()[0][:120]))


def _rescale_to_hu(arr: np.ndarray, ds, source) -> np.ndarray:
    """Apply RescaleSlope/RescaleIntercept, requiring both to be present.

    No defaults: assuming slope 1 / intercept 0 for a file that simply omits
    them shifts the whole ROI on the HU scale — a 1024 HU error that looks like
    perfectly ordinary data. If a scanner did not write them, that has to stop
    the run rather than quietly change the numbers.
    """
    slope = getattr(ds, "RescaleSlope", None)
    intercept = getattr(ds, "RescaleIntercept", None)
    missing = [n for n, v in (("RescaleSlope", slope),
                              ("RescaleIntercept", intercept))
               if v in (None, "")]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} missing in {source} — HU values cannot be "
            f"derived and nothing is assumed."
        )
    return arr.astype(np.float32) * float(slope) + float(intercept)


def _slice_sort_key(ds) -> Tuple[float, int]:
    """Sort DICOM slices in physical order.

    Tries ImagePositionPatient z, then SliceLocation, then InstanceNumber. No
    silent default: if none of the three is present the slice order would be
    arbitrary, and an arbitrary z order silently corrupts every 3D feature.
    """
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        try:
            return (float(ipp[2]), 0)
        except (TypeError, ValueError):
            pass
    loc = getattr(ds, "SliceLocation", None)
    if loc not in (None, ""):
        try:
            return (float(loc), 0)
        except (TypeError, ValueError):
            pass
    inst = getattr(ds, "InstanceNumber", None)
    if inst not in (None, ""):
        return (float(int(inst)), 0)
    raise ValueError(
        f"DICOM slice has no ImagePositionPatient, SliceLocation or "
        f"InstanceNumber ({getattr(ds, 'SOPInstanceUID', '?')}) — slice order "
        f"cannot be established and nothing is assumed."
    )


def _load_dicom_slices(dicom_dir: Path, with_pixels: bool = True):
    files = _list_ct_dicom_files(dicom_dir)
    if not files:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=not with_pixels)
            slices.append((f, ds))
        except Exception as e:
            print(f"  WARN: could not read DICOM {f}: {e}")

    if not slices:
        raise FileNotFoundError(f"No readable DICOM files in {dicom_dir}")

    slices.sort(key=lambda item: _slice_sort_key(item[1]))
    return slices


def read_ct_dicom_volume(dicom_dir: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Read CT DICOM slices into HU volume (Z,H,W) + spacing (X,Y,Z)."""
    slices = _load_dicom_slices(dicom_dir, with_pixels=True)

    arrays = []
    for path, ds in slices:
        arrays.append(_rescale_to_hu(ds.pixel_array, ds, path))

    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent CT slice shapes in {dicom_dir}: {shapes}")

    volume = np.stack(arrays, axis=0).astype(np.float32)
    spacing = read_dicom_spacing(dicom_dir)
    return volume, spacing


def _normalise_insert_size_label(name: str) -> Optional[str]:
    """Map folder/file text to a 5mm/10mm/15mm label (15mm is filtered later)."""
    m = re.search(r"(?<!\d)(5|10|15)\s*_?\s*mm(?!\w)", name, re.IGNORECASE)
    if not m:
        m = re.search(r"(?<!\d)(5|10|15)(?!\d)", name, re.IGNORECASE)
    if not m:
        return None
    return f"{m.group(1)}mm"


def discover_insert_roi_dirs(series_dir: Path) -> List[Tuple[str, str, Path]]:
    """Return individual insert DICOM folders.

    Expected current layout:
      Inserts/5mm/Insert_01/*.dcm
      Inserts/5mm/Insert_02/*.dcm
      Inserts/10mm/Insert_01/*.dcm

    Also tolerates the older layout where DICOMs are directly in
    Inserts/5mm/*.dcm. In that case the ROI label becomes Insert_5mm.

    Only the sizes in ``INSERT_SIZES`` are processed; a 15mm folder on disk is
    ignored.
    """
    inserts_dir = series_dir / "Inserts"
    out: List[Tuple[str, str, Path]] = []
    if not inserts_dir.exists():
        return out

    for size in INSERT_SIZES:
        size_dir = inserts_dir / size
        if not size_dir.exists():
            continue

        # Old/flat layout: DICOMs directly under the size folder.
        direct = sorted(size_dir.glob("*.dcm"))
        if direct:
            out.append((size, f"Insert_{size}", size_dir))

        # Current layout: each insert has its own subfolder.
        for sub in sorted(size_dir.iterdir()):
            if not sub.is_dir():
                continue
            if list(sub.glob("*.dcm")):
                out.append((size, sub.name, sub))

    return out


def discover_insert_mask_groups(series_dir: Path) -> Dict[str, Path]:
    """Backward-compatible summary: first folder found for each size."""
    out: Dict[str, Path] = {}
    for size, _label, path in discover_insert_roi_dirs(series_dir):
        out.setdefault(size, path)
    return out


def _largest_inscribed_square(footprint: np.ndarray) -> Tuple[int, int, int]:
    """Largest axis-aligned all-True square in a 2D boolean footprint.

    Returns ``(row0, col0, size)``. Classic largest-square dynamic program;
    among equally large squares the one whose centre is closest to the
    footprint centroid wins, so the result is deterministic and stays on the
    ROI axis rather than drifting to a corner.
    """
    fp = footprint.astype(bool)
    if not fp.any():
        return 0, 0, 0

    ny, nx = fp.shape
    dp = np.zeros((ny, nx), dtype=np.int32)
    dp[0, :] = fp[0, :]
    dp[:, 0] = fp[:, 0]
    for i in range(1, ny):
        row, prev = dp[i], dp[i - 1]
        line = fp[i]
        for j in range(1, nx):
            if line[j]:
                row[j] = min(prev[j], row[j - 1], prev[j - 1]) + 1
    size = int(dp.max())
    if size == 0:
        return 0, 0, 0

    ys, xs = np.nonzero(fp)
    cy, cx = ys.mean(), xs.mean()
    best, best_d = None, None
    for i, j in zip(*np.nonzero(dp == size)):
        r0, c0 = int(i) - size + 1, int(j) - size + 1
        d = (r0 + (size - 1) / 2.0 - cy) ** 2 + (c0 + (size - 1) / 2.0 - cx) ** 2
        if best_d is None or d < best_d:
            best, best_d = (r0, c0), d
    return best[0], best[1], size


def _assert_within_discretisation_range(hu: np.ndarray, patch_dir: Path) -> None:
    """Refuse an ROI that MIRP would silently trim or clamp.

    The resegmentation range doubles as the fixed-bin-size grey-level origin, so
    a voxel outside it is not merely re-binned: below the lower bound MIRP first
    resegments it out of the intensity mask, and anything surviving at the
    boundary is clamped into bin 1. Either way a real measurement disappears
    without a trace in the workbook.

    The anchor carries 754 HU of headroom over the lowest value seen in the
    reference dataset, so reaching this is a signal that a new site, kernel or
    protocol reconstructs air deeper than anything used to choose it. Raising is
    the point: the anchor must be re-declared and everything re-extracted, and
    that decision belongs to the operator rather than to a silent default.
    """
    lower, upper = DISCRETISATION_RANGE_HU
    below = int((hu < lower).sum())
    above = int((hu > upper).sum())
    if not (below or above):
        return
    raise ValueError(
        f"{patch_dir}: {below + above} voxel(s) fall outside the discretisation "
        f"range [{lower}, {upper}] HU (observed {hu.min():.1f}..{hu.max():.1f}). "
        f"MIRP would resegment or clamp them away. Re-declare "
        f"_DISCRETISATION_ANCHOR_HU below the observed minimum and re-extract "
        f"every series, so one workbook never mixes two grey-level origins."
    )


def read_insert_patch_dicom_volume(
    patch_dir: Path,
    fallback_spacing_xyz: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Read one ROI (insert / star / texture) from cropped DICOM patch files.

    Current dataset layout stores each ROI as small cropped DICOM patches,
    not as full 512x512 masks. Examples:
        insert 5mm  -> 11x11 pixels per slice
        insert 10mm -> 19x19 pixels per slice
        insert 15mm -> 27x27 pixels per slice
        star / texture -> ~46-66 pixels per slice

    Therefore this function does NOT try to place the patch on the full CT
    matrix. It stacks the cropped DICOMs directly as the radiomics image and
    creates an automatic binary mask from the patch itself.

    The exporter writes HU -33792 (stored -32768) outside the ROI, so a
    circular ROI arrives as a disc inside a square patch. Two ROI shapes
    exist in this dataset:

      * squares (the 8 texture patterns) fill their patch completely. They
        are passed through untouched and keep producing identical values.
      * discs (inserts, Star16, Star8) are CROPPED to the largest square
        that contains only real voxels. Nothing is padded, masked around,
        or filled in: the emitted patch simply has no sentinel in it.

    Cropping rather than masking matters because a mask does not protect
    the filter layer. MIRP convolves each filter over the whole image and
    applies the mask afterwards, so a -33792 voxel anywhere in the patch
    leaks into in-mask response values near the border, and resegmentation
    cannot undo it (it trims base HU, not response values). Measured on a
    real Star16 ROI, the sentinel moved every filter response feature, by
    up to ~4000%. With a padding-free patch the kernel meets the declared
    ``*_boundary_condition`` at the patch edge instead, which is IBSI-2
    defined behaviour.

    How much of the disc survives depends on the reconstruction FOV, because a
    small disc quantises badly: an insert keeps 71-80% of its voxels at FOV 300
    but only 43-55% at FOV 500 (a 5mm insert there becomes 3x3 in-plane, 270
    voxels). Stars keep 61-67%. That is a deliberate trade — see
    README for the per-ROI voxel budget and for what changed relative to
    earlier runs.
    """
    files = sorted(patch_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM patch files found in {patch_dir}")

    slices = []
    for f in files:
        ds = pydicom.dcmread(str(f), force=True)
        slices.append((f, ds))
    slices.sort(key=lambda item: _slice_sort_key(item[1]))

    arrays = []
    for f, ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:
            # Defensive: split multi-frame pixel data into slices.
            for k in range(arr.shape[0]):
                arrays.append(_rescale_to_hu(arr[k], ds, f))
            continue

        arrays.append(_rescale_to_hu(arr, ds, f))

    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent patch shapes in {patch_dir}: {shapes}")

    hu = np.stack(arrays, axis=0).astype(np.float32)

    # Real voxels are the ones above the exporter's padding sentinel.
    # _PADDING_HU_FLOOR sits far below the air floor (-1024) yet far above the
    # sentinel (HU ~ -33792), so it separates them cleanly.
    finite = np.isfinite(hu)
    real = finite & (hu > _PADDING_HU_FLOOR)
    if not real.any():
        # Every voxel is the exporter's out-of-ROI sentinel. There is no ROI
        # here. Emitting it as data would publish HU -33792 as a measurement —
        # the one thing this reader exists to prevent — so it is refused and
        # the ROI is reported as skipped.
        raise ValueError(
            f"Every voxel in {patch_dir} is out-of-ROI padding "
            f"(HU <= {_PADDING_HU_FLOOR}); there is no ROI to measure."
        )

    if real.all():
        # Already padding-free (the texture squares) — pass through untouched so
        # those ROIs keep producing byte-identical values.
        hu = np.where(finite, hu, -1024.0).astype(np.float32)
        _assert_within_discretisation_range(hu, patch_dir)
        return hu, real.astype(np.uint8), _patch_spacing(slices[0][1], fallback_spacing_xyz)

    # Circular ROI inside a square patch (inserts, stars). Crop to the largest
    # square that contains ONLY real voxels instead of carrying the sentinel
    # along. This matters beyond the mask: filters convolve over the whole
    # image and the mask is applied afterwards, so a -33792 voxel anywhere in
    # the patch contaminates in-mask response values near the ROI border, and
    # resegmentation cannot undo it (it trims base HU, not response values).
    # With a padding-free patch the filter kernel meets the declared
    # ``*_boundary_condition`` at the edge, which is IBSI-2 defined behaviour.
    footprint = real.all(axis=0)
    r0, c0, size = _largest_inscribed_square(footprint)
    if size <= 0:
        raise ValueError(f"No padding-free square inside patch {patch_dir}")
    hu = hu[:, r0:r0 + size, c0:c0 + size]
    binary = np.ones(hu.shape, dtype=np.uint8)

    # Fill NaNs (if any survived the crop) with air HU before writing NIfTI.
    hu = np.where(np.isfinite(hu), hu, -1024.0).astype(np.float32)
    _assert_within_discretisation_range(hu, patch_dir)
    return hu, binary, _patch_spacing(slices[0][1], fallback_spacing_xyz)


def _patch_spacing(
    first_ds,
    fallback_spacing_xyz: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Patch DICOM spacing when present, otherwise the parent CT spacing."""
    try:
        sy, sx = float(first_ds.PixelSpacing[0]), float(first_ds.PixelSpacing[1])
    except Exception:
        sx, sy = fallback_spacing_xyz[0], fallback_spacing_xyz[1]
    try:
        sz = float(getattr(first_ds, "SliceThickness", fallback_spacing_xyz[2])
                   or fallback_spacing_xyz[2])
    except Exception:
        sz = fallback_spacing_xyz[2]
    return (sx, sy, sz)


def collect_insert_rois_from_dicom(
    series_dir: Path,
    fallback_spacing_xyz: Tuple[float, float, float],
) -> List[Dict]:
    """Create ROI definitions from cropped Insert_XX DICOM patch folders."""
    roi_dirs = discover_insert_roi_dirs(series_dir)
    rois: List[Dict] = []

    for size_label, insert_label, patch_dir in roi_dirs:
        try:
            hu, binary, spacing_xyz = read_insert_patch_dicom_volume(
                patch_dir, fallback_spacing_xyz
            )
        except Exception as e:
            print(f"  WARN: could not read {size_label}/{insert_label}: {e}")
            _note_skipped_roi("Inserts", f"{size_label}/{insert_label}", e)
            continue

        if int(binary.sum()) == 0:
            print(f"  WARN: empty mask for {size_label}/{insert_label}")
            _note_skipped_roi("Inserts", f"{size_label}/{insert_label}",
                              "empty mask")
            continue

        # The patch is already cropped to a padding-free square, so the mask
        # covers all of it and the summary columns below (roi_mean_hu /
        # roi_std_hu / voxel_count) describe exactly the voxels MIRP sees.
        stats_mask = binary.astype(bool)
        voxel_count = int(stats_mask.sum())
        hu_vals = hu[stats_mask]
        z_idx = np.where(stats_mask.any(axis=(1, 2)))[0]
        rois.append({
            "group": "Inserts",
            "section": size_label,
            "label": insert_label,
            "insert_size": size_label,
            "image": hu,
            "mask": binary.astype(np.uint8),
            "spacing": spacing_xyz,
            "patch_shape": f"{hu.shape[1]}x{hu.shape[2]}",
            "n_slices": int(len(z_idx)),
            "roi_mean_hu": float(np.mean(hu_vals)),
            "roi_std_hu": float(np.std(hu_vals)),
            "voxel_count": voxel_count,
        })

    def _insert_no(label: str) -> int:
        m = re.search(r"Insert[_-]?(\d+)", label, re.IGNORECASE)
        return int(m.group(1)) if m else 9999

    rois.sort(key=lambda r: (int(r["insert_size"].replace("mm", "")), _insert_no(r["label"])))
    return rois


# Which star module each texture pattern belongs to. Used only to set the
# ``section`` field so the Excel writer colours texture rows correctly; the
# radiomics computation itself does not depend on it. STAR16 carries the four
# line/dot gratings; STAR8 carries the honeycomb + sinusoidal patterns.
TEXTURE_TO_SECTION: Dict[str, str] = {
    "Diagonal": "STAR16", "Horizontal": "STAR16",
    "Vertical": "STAR16", "Square": "STAR16",
    "Honeylarge": "STAR8", "Honeymedium": "STAR8",
    "Honeysmall": "STAR8", "Sinusoidal": "STAR8",
}


def discover_star_roi_dirs(series_dir: Path) -> List[Tuple[str, str, Path]]:
    """Return (section, label, dir) for each central-star ROI folder.

    Layout written by the labelling pipeline:
        Stars/Star16/*.dcm
        Stars/Star8/*.dcm
    """
    stars_dir = series_dir / "Stars"
    out: List[Tuple[str, str, Path]] = []
    if not stars_dir.exists():
        return out
    for sub in sorted(stars_dir.iterdir()):
        if not sub.is_dir() or not list(sub.glob("*.dcm")):
            continue
        # Folder name is "Star16" / "Star8"; section is the STAR16/STAR8 key.
        section = "STAR16" if "16" in sub.name else "STAR8"
        out.append((section, sub.name, sub))
    return out


def discover_texture_roi_dirs(series_dir: Path) -> List[Tuple[str, str, Path]]:
    """Return (section, label, dir) for each texture-square ROI folder.

    Layout written by the labelling pipeline:
        Textures/<TextureName>/*.dcm   (Diagonal, Horizontal, Vertical,
        Square, Honeylarge, Honeymedium, Honeysmall, Sinusoidal)
    """
    textures_dir = series_dir / "Textures"
    out: List[Tuple[str, str, Path]] = []
    if not textures_dir.exists():
        return out
    for sub in sorted(textures_dir.iterdir()):
        if not sub.is_dir() or not list(sub.glob("*.dcm")):
            continue
        section = TEXTURE_TO_SECTION.get(sub.name, "STAR16")
        out.append((section, sub.name, sub))
    return out


def _collect_patch_group_from_dicom(
    roi_dirs: List[Tuple[str, str, Path]],
    group: str,
    fallback_spacing_xyz: Tuple[float, float, float],
) -> List[Dict]:
    """Build ROI dicts for a star/texture group from cropped DICOM patches.

    Shares the insert patch reader: texture squares pass through unchanged,
    while the Star16/Star8 discs are cropped to a padding-free square (see
    ``read_insert_patch_dicom_volume``). Either way the water-equivalent
    matrix and the air gaps inside the pattern are preserved.
    """
    rois: List[Dict] = []
    for section, label, patch_dir in roi_dirs:
        try:
            hu, binary, spacing_xyz = read_insert_patch_dicom_volume(
                patch_dir, fallback_spacing_xyz
            )
        except Exception as e:
            print(f"  WARN: could not read {group}/{label}: {e}")
            _note_skipped_roi(group, label, e)
            continue

        voxel_count = int(binary.sum())
        if voxel_count == 0:
            print(f"  WARN: empty mask for {group}/{label}")
            _note_skipped_roi(group, label, "empty mask")
            continue

        hu_vals = hu[binary.astype(bool)]
        z_idx = np.where(binary.astype(bool).any(axis=(1, 2)))[0]
        rois.append({
            "group": group,
            "section": section,
            "label": label,
            "insert_size": "",          # not applicable to stars / textures
            "image": hu,
            "mask": binary.astype(np.uint8),
            "spacing": spacing_xyz,
            "patch_shape": f"{hu.shape[1]}x{hu.shape[2]}",
            "n_slices": int(len(z_idx)),
            "roi_mean_hu": float(np.mean(hu_vals)),
            "roi_std_hu": float(np.std(hu_vals)),
            "voxel_count": voxel_count,
        })
    return rois


def collect_star_rois_from_dicom(
    series_dir: Path,
    fallback_spacing_xyz: Tuple[float, float, float],
) -> List[Dict]:
    """Create ROI definitions from cropped Star16/Star8 DICOM patch folders."""
    return _collect_patch_group_from_dicom(
        discover_star_roi_dirs(series_dir), "Stars", fallback_spacing_xyz
    )


def collect_texture_rois_from_dicom(
    series_dir: Path,
    fallback_spacing_xyz: Tuple[float, float, float],
) -> List[Dict]:
    """Create ROI definitions from cropped texture-square DICOM patch folders."""
    return _collect_patch_group_from_dicom(
        discover_texture_roi_dirs(series_dir), "Textures", fallback_spacing_xyz
    )


def write_roi_nifti(
    hu_volume: np.ndarray,
    binary_mask: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    image_path: Path,
    mask_path: Path,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)

    img = sitk.GetImageFromArray(hu_volume.astype(np.float32))
    img.SetSpacing(spacing_xyz)
    sitk.WriteImage(img, str(image_path))

    msk = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
    msk.SetSpacing(spacing_xyz)
    sitk.WriteImage(msk, str(mask_path))


# =========================================================================
# MIRP feature extraction (base families, 3D + 2D)
# =========================================================================

def extract_roi_features(
    image_path: Path,
    mask_path: Path,
    roi_label: str,
    do_3d: bool,
    do_2d: bool,
    feature_families: str = "statistics",
    log: callable = print,
) -> Dict[str, pd.DataFrame]:
    """Run MIRP on one ROI (base features only — see the module docstring for
    why the IBSI-2 filter layer is not computed)."""
    roi_name = {"1": roi_label}
    modality = "CT"
    out: Dict[str, pd.DataFrame] = {}

    if do_3d:
        t0 = time.time()
        params = get_3d_base_params()
        params["base_feature_families"] = feature_families
        base = extract_base_features(image_path, mask_path, roi_name, modality, params)
        n_base = sum(
            1 for c in base.columns
            if not c.startswith("image_") and not c.startswith("sample_")
        )
        log(f"      [3D] base ({feature_families}): {n_base} features "
            f"({time.time() - t0:.1f}s)")

        out["3D"] = build_feature_table(base)

    if do_2d:
        t0 = time.time()
        params = get_2d_base_params()
        params["base_feature_families"] = feature_families
        base = extract_base_features(image_path, mask_path, roi_name, modality, params)
        n_base = sum(
            1 for c in base.columns
            if not c.startswith("image_") and not c.startswith("sample_")
        )
        log(f"      [2D] base ({feature_families}): {n_base} features "
            f"({time.time() - t0:.1f}s)")

        out["2D"] = build_feature_table(base)

    return out


# Per-group feature policy.
#
# Inserts are HOMOGENEOUS solid cylinders — texture-matrix families (GLCM,
# GLRLM, GLSZM, GLDZM, NGTDM, NGLDM) carry no real signal on them and just
# multiply runtime, so inserts are kept to first-order ``statistics``.
# Stars and Textures ARE structured patterns, so they get the full base family
# set. Neither group gets filters — see the module docstring.
GROUP_FEATURE_POLICY: Dict[str, Dict] = {
    "Inserts": {"feature_families": "statistics"},
    "Stars": {"feature_families": "all"},
    "Textures": {"feature_families": "all"},
}
_FULL_POLICY = {"feature_families": "all"}  # fallback for unknown groups


# =========================================================================
# Per-series processing
# =========================================================================

def process_series(
    series: Dict,
    output_dir: Path,
    cache_dir: Path,
    do_3d: bool,
    do_2d: bool,
    skip_existing: bool,
) -> List[Dict]:
    """
    Process ALL ROI groups (Inserts + Stars + Textures) of one series.

    Modified input logic:
      - CT DICOMs are used for spacing/reference metadata.
      - ROI image patches come from:
            <series>/Inserts/<size>/Insert_XX/*.dcm
            <series>/Stars/Star16|Star8/*.dcm
            <series>/Textures/<TextureName>/*.dcm
      - .npy files and metadata.json are no longer used.
    """
    name = series["series_name"]
    series_dir = series["series_dir"]
    log_prefix = f"  [{name[-50:]}]"
    print(f"\n{log_prefix}")

    checkpoint = output_dir / f"_per_series/{name}.xlsx"
    want_modes = ([m for m, on in (("3D", do_3d), ("2D", do_2d)) if on])
    if skip_existing and checkpoint.exists() and checkpoint.stat().st_size > 4096:
        reusable, why = checkpoint_is_reusable(checkpoint, want_modes)
        if reusable:
            print(f"{log_prefix}   SKIP - checkpoint exists ({why})")
            return load_checkpoint_rows(checkpoint, want_modes)
        print(f"{log_prefix}   checkpoint not reusable: {why} — re-extracting")

    # Use the pre-resolved dicoms directory when available (set by
    # discover_series); fall back to the canonical sub-folder name.
    dicoms_dir = series.get("dicoms_dir") or _find_dicoms_dir(series_dir) or (series_dir / "dicoms")
    spacing = read_dicom_spacing(dicoms_dir)
    rois = (
        collect_insert_rois_from_dicom(series_dir, spacing)
        + collect_star_rois_from_dicom(series_dir, spacing)
        + collect_texture_rois_from_dicom(series_dir, spacing)
    )

    n_ins = sum(1 for r in rois if r["group"] == "Inserts")
    n_star = sum(1 for r in rois if r["group"] == "Stars")
    n_tex = sum(1 for r in rois if r["group"] == "Textures")
    print(f"{log_prefix}   {len(rois)} ROIs "
          f"(Inserts={n_ins}, Stars={n_star}, Textures={n_tex}) | "
          f"CT spacing X/Y/Z = {spacing[0]:.4f}/{spacing[1]:.4f}/{spacing[2]:.2f} mm")

    series_cache = cache_dir / name
    rows: List[Dict] = []

    for i, roi in enumerate(rois, 1):
        group = roi["group"]
        label = roi["label"]
        section = roi["section"]
        image = roi["image"]
        mask = roi["mask"]
        roi_spacing = roi["spacing"]
        insert_size = roi.get("insert_size", "")

        # Compact per-ROI progress line; size column only meaningful for inserts.
        size_col = f"{insert_size:4s}" if insert_size else "    "
        print(f"{log_prefix}   [{i:2d}/{len(rois)}] {group:8s} "
              f"{size_col} {label:24s} | "
              f"patch={roi['patch_shape']} | {roi['n_slices']} slices | "
              f"mean_hu={roi['roi_mean_hu']:.1f} | voxels={roi['voxel_count']}")

        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        # Unique cache id per ROI: group + (size for inserts) + section + label.
        id_size = f"{insert_size}_" if insert_size else ""
        roi_id = f"{group}_{id_size}{section}_{safe_label}"
        img_path = series_cache / f"{roi_id}_image.nii.gz"
        msk_path = series_cache / f"{roi_id}_mask.nii.gz"

        # Image is the cropped DICOM patch stack itself; mask is automatic.
        write_roi_nifti(image, mask, roi_spacing, img_path, msk_path)

        policy = GROUP_FEATURE_POLICY.get(group, _FULL_POLICY)
        try:
            tables = extract_roi_features(
                img_path, msk_path, label,
                do_3d=do_3d,
                do_2d=do_2d,
                feature_families=policy["feature_families"],
                log=lambda m: print(f"{log_prefix} {m}"),
            )
        except Exception as e:
            print(f"{log_prefix}     ERROR - {e}")
            _note_skipped_roi(group, label, e)
            continue

        meta_cols = {
            "series_name": name,
            "study_date": series.get("study_date", ""),
            "recon_kernel": series.get("recon_kernel", ""),
            "conv_kernel": series.get("conv_kernel", ""),
            "k_strength": series.get("k_strength", ""),
            "kvp": series.get("kvp", ""),
            "mas": series.get("mas", ""),
            "fov": series.get("fov", ""),
            "slice_thickness_mm": spacing[2],
            "pixel_spacing_mm": spacing[0],
            "repeat": series.get("repeat", ""),
            "group": group,
            "section": section,
            "label": label,
            "insert_size": insert_size,
            "n_slices": roi["n_slices"],
            "roi_mean_hu": roi["roi_mean_hu"],
            "roi_std_hu": roi["roi_std_hu"],
            "voxel_count": roi["voxel_count"],
        }

        for mode, table in tables.items():
            for _, r in table.iterrows():
                rows.append({
                    **meta_cols,
                    "mode": mode,
                    "feature_id": r["Feature ID"],
                    "family": r["Family"],
                    "source": r["Source"],
                    "value": r["Value"],
                })

    save_checkpoint(rows, checkpoint, _SKIPPED_ROIS)
    print(f"{log_prefix}   checkpoint saved: {checkpoint.relative_to(output_dir)}")
    return rows


def _print_failed_series_summary(failed: List[Tuple[str, str]]) -> None:
    """Report series that produced no rows at all.

    A crashed series prints one traceback in the middle of a long log and then
    vanishes: the workbook simply has fewer series, the TOC counts only the
    survivors, and the process still exits 0. Nothing downstream can tell an
    80-series run that lost 3 from a 77-series dataset.
    """
    if not failed:
        return
    print(chr(10) + "!" * 70)
    print(f"FAILED SERIES: {len(failed)} series produced no rows and are ABSENT "
          f"from the workbook.")
    print("!" * 70)
    for name, err in failed:
        print(f"  {name}")
        print(f"      {err}")
    print("!" * 70)


def _print_skipped_roi_summary(skipped: List[Tuple[str, str, str, str]]) -> None:
    """Report ROIs that exist on disk but produced no rows.

    Without this a dropped ROI is undetectable in the workbook: it is one
    missing row among hundreds, and the run still exits 0.
    """
    if not skipped:
        return
    print("\n" + "!" * 70)
    print(f"SKIPPED ROIs: {len(skipped)} ROI(s) were found on disk but produced "
          f"no features.")
    print("Those rows are ABSENT from the workbook.")
    print("!" * 70)
    by_reason: Dict[str, List[Tuple[str, str, str]]] = {}
    for series_name, group, roi, reason in skipped:
        by_reason.setdefault(reason, []).append((series_name, group, roi))
    for reason, items in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        print(f"  x{len(items):4d}  {reason}")
        for series_name, group, roi in items[:3]:
            print(f"          {group}/{roi}  in {series_name[-46:]}")
        if len(items) > 3:
            print(f"          ... and {len(items) - 3} more")
    print("!" * 70)


def _incompleteness_summary(
    failed_series: List[Tuple[str, str]],
    skipped_rois: List[Tuple[str, str, str, str]],
    rejected: List[Tuple[str, str]],
) -> Optional[str]:
    """One line naming everything this run failed to put in the workbook.

    Returns None only for a genuinely clean run, so the caller can turn it
    straight into the exit code.

    Rejected folders are counted alongside crashed series and skipped ROIs
    because all three mean the same thing downstream: rows that should exist and
    do not. They used to be printed during discovery and then dropped from the
    accounting, so a dataset that yielded 77 of 80 series still exited 0 and an
    automated caller could not tell it from a complete extraction. Counting
    distinct folder names matches what _print_rejected_series shows, which
    de-duplicates.
    """
    losses: List[str] = []
    if rejected:
        losses.append(f"{len({name for name, _ in rejected})} rejected folder(s)")
    if failed_series:
        losses.append(f"{len(failed_series)} failed series")
    if skipped_rois:
        losses.append(f"{len(skipped_rois)} skipped ROI(s)")
    return ", ".join(losses) if losses else None


def _process_series_worker(payload):
    """Top-level (picklable) wrapper so ``process_series`` can run in a
    ``ProcessPoolExecutor`` worker. Returns (series_name, rows, error_str).

    Each series is fully independent — its own input patches, its own
    ``_per_series/<name>.xlsx`` checkpoint — so N series can run in N
    processes with no shared state and no locking. This is the single
    biggest speed-up available: MIRP itself does not parallelise a single
    ROI ("jobs processed sequentially"), but the 100+ series fan out across
    all CPU cores.
    """
    (series, output_dir, cache_dir, do_3d, do_2d, skip_existing) = payload
    _SKIPPED_ROIS.clear()
    try:
        rows = process_series(
            series, output_dir, cache_dir, do_3d, do_2d,
            skip_existing=skip_existing,
        )
        skipped = [(series["series_name"], g, r, why)
                   for _, g, r, why in _SKIPPED_ROIS]
        return (series["series_name"], rows, None, skipped)
    except Exception:
        import traceback
        skipped = [(series["series_name"], g, r, why)
                   for _, g, r, why in _SKIPPED_ROIS]
        return (series["series_name"], [], traceback.format_exc(), skipped)


# =========================================================================
# Checkpoint I/O
# =========================================================================

_CHECKPOINT_META_COLS = [
    "series_name", "study_date", "recon_kernel", "conv_kernel", "k_strength",
    "kvp", "mas", "fov", "slice_thickness_mm", "pixel_spacing_mm",
    "repeat", "group", "section", "label", "insert_size", "n_slices",
    "roi_mean_hu", "roi_std_hu", "voxel_count",
    "mode", "feature_id", "family", "source", "value",
]


def _library_versions() -> Dict[str, str]:
    """Versions of the libraries that decide the numbers.

    Recorded in every checkpoint and in the workbook, because a feature value is
    only reproducible together with the code that produced it — and this project
    is run across several sites.
    """
    import importlib.metadata as md

    versions = {}
    for package in ("mirp", "numpy", "scipy", "pandas", "SimpleITK", "pydicom"):
        try:
            versions[package] = md.version(package)
        except Exception:
            versions[package] = "not-installed"
    return versions


def extraction_params_hash() -> str:
    """Fingerprint of everything that changes what a feature value MEANS.

    The schema version above has to be bumped by hand, and a parameter change is
    exactly the kind of edit that gets made without remembering to. Measured
    stakes: moving the discretisation anchor from -2000 to -1000/400 changes 191
    of 218 feature values, so a workbook that mixes two anchors is not merely
    inconsistent, it is uninterpretable. Deriving the stamp from the parameters
    themselves means a checkpoint written under different settings is rejected
    without anyone having to notice.
    """
    payload = json.dumps(
        {
            "3d": {k: v for k, v in sorted(get_3d_base_params().items())},
            "2d": {k: v for k, v in sorted(get_2d_base_params().items())},
            "discretisation_range": list(DISCRETISATION_RANGE_HU),
            "padding_floor": _PADDING_HU_FLOOR,
            "insert_sizes": list(INSERT_SIZES),
            "group_policy": {k: v for k, v in sorted(GROUP_FEATURE_POLICY.items())},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_checkpoint(rows: List[Dict], path: Path,
                    skipped: Optional[List] = None) -> None:
    """Write the per-series checkpoint plus the stamp used to validate reuse.

    The stamp also records how many ROIs were skipped while the checkpoint was
    produced. Without it a resumed run silently reports a complete series: the
    skips happened in an earlier process whose warnings are long gone.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df[[c for c in _CHECKPOINT_META_COLS if c in df.columns]]
    with pd.ExcelWriter(str(path), engine="openpyxl") as w:
        for mode, sub in df.groupby("mode"):
            sub.to_excel(w, sheet_name=f"{mode}_long", index=False)
        pd.DataFrame([{
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            # Derived, not hand-maintained: changing the reported feature set
            # invalidates old checkpoints automatically, without anyone having
            # to remember to bump the integer above.
            "dropped_features": ",".join(sorted(_DROPPED_FEATURES)),
            # Same idea, for the extraction parameters themselves.
            "params_hash": extraction_params_hash(),
            "mirp_version": _library_versions().get("mirp", "?"),
            "skipped_rois": len(skipped or []),
            "skipped_detail": "; ".join(f"{g}/{r}: {why}"
                                        for _, g, r, why in (skipped or []))[:4000],
        }]).to_excel(w, sheet_name=_CHECKPOINT_SCHEMA_SHEET, index=False)


def checkpoint_is_reusable(path: Path, want_modes: List[str]) -> Tuple[bool, str]:
    """Can this checkpoint stand in for a fresh extraction of *want_modes*?

    Checks three things, because each has silently corrupted a workbook before:
      * the schema stamp matches this pipeline version;
      * it actually contains every mode the current run asked for (a --no-2d
        checkpoint must not satisfy a 3D+2D run, nor add a 2D half to a
        3D-only one);
      * the ``*_long`` sheets genuinely parse. Reading only the stamp sheet
        certifies a different zip member than the loader later reads, so a
        half-written file could pass the gate and then load as zero rows.
    """
    try:
        xl = pd.ExcelFile(path)
        sheets = xl.sheet_names
        if _CHECKPOINT_SCHEMA_SHEET not in sheets:
            return False, "no schema stamp (predates versioning)"
        version = int(pd.read_excel(path, sheet_name=_CHECKPOINT_SCHEMA_SHEET)
                      ["schema_version"].iloc[0])
        if version != CHECKPOINT_SCHEMA_VERSION:
            return False, f"schema v{version}, this pipeline writes v{CHECKPOINT_SCHEMA_VERSION}"
        stamp0 = pd.read_excel(path, sheet_name=_CHECKPOINT_SCHEMA_SHEET)
        want_dropped = ",".join(sorted(_DROPPED_FEATURES))
        got_dropped = (str(stamp0["dropped_features"].iloc[0])
                       if "dropped_features" in stamp0 else "")
        if got_dropped != want_dropped:
            return False, "written with a different reported feature set"
        want_params = extraction_params_hash()
        got_params = (str(stamp0["params_hash"].iloc[0])
                      if "params_hash" in stamp0 else "")
        if got_params != want_params:
            return False, (
                f"written with different extraction parameters "
                f"({got_params or 'unstamped'} != {want_params})"
            )
        # MIRP decides the numbers, and save_checkpoint stamps its version
        # expressly so a workbook can support a reproducibility claim — but the
        # stamp was written and never read. If MIRP is upgraded between a partial
        # run and its resume, one workbook silently mixes two implementations
        # while the methods sheet declares only the installed one.
        want_mirp = _library_versions().get("mirp", "?")
        got_mirp = (str(stamp0["mirp_version"].iloc[0])
                    if "mirp_version" in stamp0 else "")
        if got_mirp and got_mirp != want_mirp:
            return False, f"written with MIRP {got_mirp}, this run has {want_mirp}"
        have = {sh[:-len("_long")] for sh in sheets if sh.endswith("_long")}
        missing = [m for m in want_modes if m not in have]
        if missing:
            return False, f"missing {'/'.join(missing)} rows"
        # A superset is fine: the extra modes are simply filtered out on load.
        # Rejecting it would re-extract AND overwrite the file, throwing away a
        # half that may have cost hours to compute.
        rows = 0
        for sh in sheets:
            if sh.endswith("_long"):
                rows += len(pd.read_excel(path, sheet_name=sh))
        if rows == 0:
            return False, "no rows"
        stamp = pd.read_excel(path, sheet_name=_CHECKPOINT_SCHEMA_SHEET)
        n_skipped = int(stamp["skipped_rois"].iloc[0]) if "skipped_rois" in stamp else 0
        if n_skipped:
            # Re-enter them into this run's accounting, otherwise resuming a
            # partial extraction launders it into a clean-looking exit 0.
            detail = str(stamp.get("skipped_detail", pd.Series([""])).iloc[0] or "")
            for item in filter(None, (d.strip() for d in detail.split(";"))):
                _SKIPPED_ROIS.append(("", "(from checkpoint)", item, "recorded earlier"))
            if len(_SKIPPED_ROIS) < n_skipped:
                for _ in range(n_skipped - len(_SKIPPED_ROIS)):
                    _SKIPPED_ROIS.append(("", "(from checkpoint)", "?", "recorded earlier"))
            return True, f"{rows} rows, {n_skipped} ROI(s) were skipped when it was written"
        return True, f"{rows} rows"
    except Exception as e:
        return False, f"unreadable ({type(e).__name__})"


def load_checkpoint_rows(path: Path,
                         modes: Optional[List[str]] = None) -> List[Dict]:
    """Rows from a checkpoint, restricted to *modes* when given.

    A checkpoint may hold more modes than the current run asked for (a 3D+2D
    file being reused by a 3D-only run); returning the extra rows would put a
    mode in the workbook that the user switched off.
    """
    rows: List[Dict] = []
    try:
        for sheet in pd.ExcelFile(path).sheet_names:
            if not sheet.endswith("_long"):
                continue
            if modes is not None and sheet[:-len("_long")] not in modes:
                continue
            df = pd.read_excel(path, sheet_name=sheet)
            rows.extend(df.to_dict(orient="records"))
    except Exception as e:
        print(f"  Could not load checkpoint {path.name}: {e}")
    return rows


# =========================================================================
# Insert ranking — global, by ascending mean HU averaged across series+sizes
# =========================================================================

def compute_insert_ranking(rows: List[Dict]) -> pd.DataFrame:
    """
    Rank the 40 insert labels in LABELING order: ``ranked_index`` is the
    insert's own number (Insert_01 -> 1), so this sheet can never disagree
    with the number drawn in the overlay or carried in the ``label`` column.
    ``mean_hu_global`` (mean HU averaged across all series AND all insert
    sizes) is reported alongside, but no longer drives the order.

    It used to sort by ``mean_hu_global`` ascending. The phantom's rods are
    loaded in decreasing density around the rings, so that ordering was
    exactly ``41 - insert number`` -- two numbering systems for one object,
    which is what this change removes.

    Returns a DataFrame with columns: label, mean_hu_global, ranked_index
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["label", "mean_hu_global", "ranked_index"])

    inserts = df[df["group"] == "Inserts"]
    if inserts.empty:
        return pd.DataFrame(columns=["label", "mean_hu_global", "ranked_index"])

    # The long table repeats each ROI once per feature, so collapse to one
    # row per ROI first. ``insert_size`` MUST be part of the key: without it
    # the larger-size rows of an insert are dropped as duplicates and the
    # ranking reflects the 5mm ROI alone.
    key_cols = [c for c in ("series_name", "insert_size", "label")
                if c in inserts.columns]
    per_roi = (
        inserts.drop_duplicates(subset=key_cols)
        [key_cols + ["roi_mean_hu"]]
    )
    avg = (
        per_roi.groupby("label")["roi_mean_hu"]
        .mean().reset_index()
        .rename(columns={"roi_mean_hu": "mean_hu_global"})
    )
    avg["ranked_index"] = avg["label"].str.split("_").str[1].astype(int)
    avg = avg.sort_values("ranked_index").reset_index(drop=True)
    return avg


# =========================================================================
# Excel colour palette — one fill per ROI category (qualitative)
# =========================================================================

ROI_COLOR_MAP: Dict[Tuple[str, Optional[str]], str] = {
    # All 40 inserts share a single colour — INSERT1/INSERT2 metadata sections
    # are presentationally merged.
    ("Inserts", None):           "D6EAF8",  # light blue
    # STAR16 group
    ("STAR16", "Star"):          "FCF3CF",  # light yellow
    ("STAR16", "Star16"):        "FCF3CF",  # light yellow (per-section star label)
    ("STAR16", "Diagonal"):      "FAD7A0",  # peach
    ("STAR16", "Horizontal"):    "F5B7B1",  # light coral
    ("STAR16", "Square"):        "D7BDE2",  # light purple
    ("STAR16", "Vertical"):      "AED6F1",  # mid blue
    # STAR8 group
    ("STAR8", "Star"):           "F9E79F",  # light gold
    ("STAR8", "Star8"):          "F9E79F",  # light gold (per-section star label)
    ("STAR8", "Honeylarge"):     "F5CBA7",  # light salmon
    ("STAR8", "Honeymedium"):    "EDBB99",  # mid salmon
    ("STAR8", "Honeysmall"):     "F1948A",  # light red
    ("STAR8", "Sinusoidal"):     "BB8FCE",  # light violet
}


_INSERT_SECTION_RE = re.compile(r"^\s*(?:5|10|15)\s*mm\s*$", re.IGNORECASE)


def _color_for_roi(section: Optional[str], label: Optional[str]) -> Optional[str]:
    """Resolve a fill colour for a row given (section, label)."""
    if section is None:
        return None
    if (section, label) in ROI_COLOR_MAP:
        return ROI_COLOR_MAP[(section, label)]
    # Insert rows carry their size (5mm/10mm/15mm) in ``section``; all 40
    # inserts share one fill regardless of size.
    if section in ("INSERT1", "INSERT2", "Inserts") or _INSERT_SECTION_RE.match(str(section)):
        return ROI_COLOR_MAP[("Inserts", None)]
    return None


def _apply_roi_colors(
    ws,
    df: pd.DataFrame,
    section_col: str = "section",
    label_col: str = "label",
    color_full_row: bool = True,
) -> None:
    """Colour each data row by ROI category.

    By default the entire row is filled with the (light) category colour.
    The pastel palette stays readable behind text but gives an immediate
    visual band per ROI when scrolling features.
    """
    from openpyxl.styles import PatternFill

    if section_col not in df.columns or label_col not in df.columns:
        return

    cols = list(df.columns)
    section_idx = cols.index(section_col) + 1
    label_idx = cols.index(label_col) + 1
    last_col = len(cols) if color_full_row else label_idx

    for r in range(len(df)):
        section = df.iat[r, section_idx - 1]
        label = df.iat[r, label_idx - 1]
        color = _color_for_roi(section, label)
        if color is None:
            continue
        fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
        for c in range(1, last_col + 1):
            ws.cell(row=r + 2, column=c).fill = fill


def _attach_dicom_columns(
    wide: pd.DataFrame,
    dicom_by_series: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Add the source DICOM tags to a wide feature table, keyed on series.

    The tags already have their own sheet, but a protocol comparison reads one
    row at a time: having to cross-reference a second sheet for the kernel or
    the acquisition time is exactly the friction that makes people re-derive
    the information from folder names, which this pipeline has measured to be
    wrong on some series.

    Promoted tags join the acquisition block; the rest follow it. Tags that
    duplicate a column the pipeline already parsed (kVp, mAs, FOV, slice
    thickness) are deliberately NOT promoted — they stay available further
    right rather than being shown twice side by side.
    """
    if not dicom_by_series or "series" not in wide.columns:
        return wide

    tags = pd.DataFrame(
        [dicom_by_series.get(str(name), {}) for name in wide["series"]],
        index=wide.index,
    )
    if tags.empty or not len(tags.columns):
        return wide

    tags = tags.fillna("")
    ordered_tags = ([c for c in _DICOM_PROMOTED_COLUMNS if c in tags.columns]
                    + [c for c in tags.columns if c not in _DICOM_PROMOTED_COLUMNS])

    meta = [c for c in _COMPACT_META_ORDER if c in wide.columns]
    trailing = [c for c in _COMPACT_TRAILING if c in wide.columns]
    features = [c for c in wide.columns if c not in meta and c not in trailing]

    combined = pd.concat([wide, tags[ordered_tags]], axis=1)
    return combined[meta + ordered_tags + features + trailing]


# The acquisition parameters that define a PROTOCOL. Series agreeing on all of
# them are the same protocol measured more than once.
_PROTOCOL_KEY_COLUMNS = ("recon_kernel", "k_strength", "kvp", "mas", "fov",
                         "slice_thickness_mm", "me_kev")


def _assign_repeat_index(
    df: pd.DataFrame,
    dicom_by_series: Dict[str, Dict[str, str]],
) -> Dict[str, int]:
    """Number the series of each protocol 1..n in acquisition order.

    This phantom is scanned test-retest, so the same protocol legitimately
    appears several times, and until now nothing in the workbook said so: the
    ``repeat`` column existed but was always empty and then dropped. Measured on
    the NAEOTOM set, 24 of 80 series share an identical (recon_k, k_strength,
    kVp, mAs, FOV, thickness, me_keV) tuple — three repeats of eight protocols,
    acquired minutes apart. Their folder names claim 80/100/120 kVp, but the KVP
    tag reads 120 for all three and CTDIvol agrees to within 0.0003%, so the
    names are wrong and the tag is right. That left the series NAME as the only
    thing separating those rows — a field this pipeline has measured to be
    untrustworthy.

    Ordering is by AcquisitionTime, falling back to the series name when the tag
    is absent, so the index is stable across runs rather than following whatever
    order the workers happened to finish in. Derived here rather than in
    ``process_series`` because a repeat index is a property of the whole SET of
    series, which a per-series worker process cannot see.
    """
    if "series_name" not in df.columns:
        return {}
    key_cols = [c for c in _PROTOCOL_KEY_COLUMNS if c in df.columns]
    per_series = df.drop_duplicates(subset=["series_name"])[
        ["series_name"] + key_cols]

    def acquired_at(name: str) -> str:
        tags = dicom_by_series.get(_compact_series_name(str(name)), {})
        return str(tags.get("dcm_AcquisitionTime", "") or "")

    repeats: Dict[str, int] = {}
    grouped: Dict[Tuple, List[str]] = {}
    for row in per_series.itertuples(index=False):
        protocol = tuple(str(getattr(row, c)) for c in key_cols)
        grouped.setdefault(protocol, []).append(str(row.series_name))
    for names in grouped.values():
        for index, name in enumerate(sorted(names, key=lambda n: (acquired_at(n), n)), 1):
            repeats[name] = index
    return repeats


def _apply_block_colors(ws, df: pd.DataFrame) -> None:
    """Tint the meta columns by what they answer: ROI / bookkeeping / acquisition.

    Runs after _apply_roi_colors, which fills whole rows with the ROI-category
    pastel. Overwriting the meta block is deliberate — on the left the question
    is "which ROI, under what conditions", and on the right the row banding
    still carries the category while scrolling through 200 feature columns.
    """
    from openpyxl.styles import PatternFill

    columns = list(df.columns)
    remaining_dicom = [c for c in columns
                       if c.startswith("dcm_") and c not in _DICOM_PROMOTED_COLUMNS]
    blocks = (
        (_COMPACT_ROI_COLUMNS, _BLOCK_FILL_ROI),
        (_COMPACT_BOOKKEEPING_COLUMNS, _BLOCK_FILL_BOOKKEEPING),
        (_COMPACT_ACQUISITION_COLUMNS, _BLOCK_FILL_ACQUISITION),
        (list(_DICOM_PROMOTED_COLUMNS), _BLOCK_FILL_ACQUISITION),
        (remaining_dicom, _BLOCK_FILL_BOOKKEEPING),
    )
    for names, color in blocks:
        fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
        for name in names:
            if name not in columns:
                continue
            index = columns.index(name) + 1
            for row in range(2, len(df) + 2):
                ws.cell(row=row, column=index).fill = fill


def _hide_columns(ws, df: pd.DataFrame, names: Tuple[str, ...]) -> None:
    """Hide helper columns that are only needed while styling the sheet."""
    from openpyxl.utils import get_column_letter

    cols = list(df.columns)
    for name in names:
        if name not in cols:
            continue
        letter = get_column_letter(cols.index(name) + 1)
        ws.column_dimensions[letter].hidden = True


def _apply_ranking_colors(ws, df: pd.DataFrame) -> None:
    """Colour insert_ranking rows uniformly with the Inserts category fill."""
    from openpyxl.styles import PatternFill

    if "label" not in df.columns:
        return
    label_idx = list(df.columns).index("label") + 1
    color = _color_for_roi("Inserts", None)
    if color is None:
        return
    fill = PatternFill(start_color=color, end_color=color, fill_type="solid")

    for r in range(len(df)):
        label = str(df.iat[r, label_idx - 1])
        if not re.match(r"Insert_\d+", label):
            continue
        for c in range(1, len(df.columns) + 1):
            ws.cell(row=r + 2, column=c).fill = fill


# =========================================================================
# DICOM metadata sheet helpers — column ordering + outline grouping
# =========================================================================

# Clinically and technically high-value DICOM tags. Sets the order the dcm_*
# columns appear in on the feature sheets (in this exact order if the tag is
# present); everything else follows alphabetically, private tags last.
_DICOM_PRIORITY_KEYWORDS: Tuple[str, ...] = (
    "SeriesInstanceUID", "StudyInstanceUID",
    "StudyDate", "StudyTime", "AcquisitionDateTime",
    "Modality", "Manufacturer", "ManufacturerModelName",
    "KVP", "Exposure", "XRayTubeCurrent", "ExposureTime",
    "SliceThickness", "PixelSpacing", "ReconstructionDiameter",
    "ConvolutionKernel", "ImageType", "ImageComments",
    "SeriesNumber", "AcquisitionNumber", "InstanceNumber",
    "FilterType", "BodyPartExamined", "ProtocolName",
)

# Sheet tab colours by sheet category — purely cosmetic, helps users
# navigate at a glance.
_TAB_COLORS: Dict[str, str] = {
    "methods":         "C00000",
    "series":          "1F4E78",
    "insert_ranking":  "548235",
    "2D_base":         "BF8F00",
    "2D_filters":      "BF8F00",
    "3D_base":         "7030A0",
    "3D_filters":      "7030A0",
}


def _order_dicom_columns(all_keys) -> List[str]:
    """Return columns ordered: series_name, then priority block,
    then standard alphabetical, then private/private-tag fallback."""
    keys = set(all_keys)
    ordered: List[str] = []
    if "series_name" in keys:
        ordered.append("series_name")
        keys.discard("series_name")
    for kw in _DICOM_PRIORITY_KEYWORDS:
        if kw in keys:
            ordered.append(kw)
            keys.discard(kw)
    private_re = re.compile(r"^\([0-9A-Fa-f]{4},[0-9A-Fa-f]{4}\)")
    standard = sorted(k for k in keys if not private_re.match(k))
    private = sorted(k for k in keys if private_re.match(k))
    ordered.extend(standard)
    ordered.extend(private)
    return ordered


def _apply_tab_colors(wb) -> None:
    """Stamp tab colours on every known sheet name."""
    for name, color in _TAB_COLORS.items():
        if name in wb.sheetnames:
            wb[name].sheet_properties.tabColor = color


def _methods_declaration_rows(
    df: pd.DataFrame,
    n_series: int = 0,
    dataset_root: str = "",
) -> List[Tuple[str, str, str]]:
    """The IBSI reporting checklist for this run, as (section, item, value).

    Derived from the live constants rather than written out by hand, so it
    cannot drift away from what actually produced the numbers next to it. The
    entries that are prose rather than a parameter are the ones a reader cannot
    reconstruct from the workbook alone, and those are exactly the ones that get
    left out of a methods section.
    """
    versions = _library_versions()
    lower, upper = DISCRETISATION_RANGE_HU
    params_3d = get_3d_base_params()

    def observed(column: str) -> str:
        if column not in df.columns:
            return "n/a"
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            return "n/a"
        low, high = float(values.min()), float(values.max())
        return f"{low:g}" if low == high else f"{low:g} – {high:g}"

    rows: List[Tuple[str, str, str]] = [
        ("Dataset", "Series", str(n_series or df.get("series_name", pd.Series()).nunique())),
        ("Dataset", "Feature rows", str(len(df))),
        # ``insert_size`` MUST be in the key — the same reason it is in
        # compute_insert_ranking's. Each of the 40 insert labels is a SEPARATE
        # ROI at each diameter, with its own crop, voxel count and feature
        # values. Keying on the label alone counted 50 ROIs per series instead
        # of 90 and printed a 44% undercount on the one sheet whose purpose is
        # to be pasted into a methods section — contradicting the row count of
        # the sheet next to it (80 series x 90 = 7200 rows, "4000" declared).
        ("Dataset", "ROIs measured",
         str(df.groupby([c for c in ("series_name", "label", "insert_size")
                         if c in df.columns]).ngroups)
         if {"series_name", "label"} <= set(df.columns) else "n/a"),
        ("Dataset", "Source", dataset_root or "n/a"),
        ("Dataset", "Test-retest repeats",
         (lambda r: f"{sum(1 for v in r if v > 1)} of {len(r)} series repeat a "
                    f"protocol already present; see the 'repeat' column"
          if len(r) else "n/a")(
             pd.to_numeric(df.drop_duplicates(subset=["series_name"])["repeat"],
                           errors="coerce").dropna().tolist()
             if {"series_name", "repeat"} <= set(df.columns) else [])),
        ("Dataset", "Generated", time.strftime("%Y-%m-%d %H:%M:%S")),

        ("Software", "Feature extraction library",
         f"MIRP {versions.get('mirp', '?')} (IBSI-1 compliant feature definitions)"),
        ("Software", "Numerical stack",
         ", ".join(f"{k} {v}" for k, v in versions.items() if k != "mirp")),
        ("Software", "Extraction parameter fingerprint", extraction_params_hash()),

        ("Image", "Modality", "CT (calibrated, Hounsfield units)"),
        ("Image", "In-plane voxel size (mm)", observed("pixel_spacing_mm")),
        ("Image", "Slice thickness (mm)", observed("slice_thickness_mm")),
        ("Image", "Interpolation / resampling",
         "NONE — voxels kept at acquisition geometry, so no intensity value is "
         "interpolated into existence. Voxels are therefore anisotropic and "
         "the anisotropy varies with reconstruction FOV."),
        ("Image", "Intensity normalisation / denoising", "Not applied"),
        ("Image", "Filter-based (IBSI-2) features", "Not computed"),

        ("ROI", "Source",
         "ROI patches exported by the labelling pipeline; circular ROIs arrive "
         "as a disc inside a square, with out-of-ROI pixels carrying a padding "
         "sentinel (HU -33792)."),
        ("ROI", "Mask construction",
         "Circular ROIs are CROPPED to the largest square containing only real "
         "voxels — never masked and never in-filled, so the sentinel does not "
         "enter the image. Square (texture) ROIs pass through unchanged."),
        ("ROI", "Analysis mask shape",
         "Rectangular cuboid (all-ones). Consequence: morphological features "
         "describe the crop box, not the phantom insert."),
        ("ROI", "Voxel retention",
         "43–80% of the segmented voxels depending on ROI size and "
         "reconstruction FOV (measured; smallest inserts at the largest FOV "
         "retain the least)."),
        ("ROI", "Voxels per ROI", observed("voxel_count")),

        ("Re-segmentation", "Range (HU)", f"({lower:g}, {upper:g})"),
        ("Re-segmentation", "Voxels excluded",
         "None. The range spans the whole int16 encoding; it exists only "
         "because MIRP derives the fixed-bin-size grey-level origin from it. "
         "A voxel outside it aborts the run rather than being dropped."),
        ("Re-segmentation", "Outlier (sigma) filtering", "Not applied"),

        ("Discretisation", "Method", "Fixed bin size (= fixed bin width)"),
        ("Discretisation", "Bin width (HU)",
         f"{params_3d['base_discretisation_bin_width']:g}"),
        ("Discretisation", "Grey-level origin (HU)", f"{_DISCRETISATION_ANCHOR_HU:g}"),
        ("Discretisation", "Why this origin",
         "The int16 encoding floor: a fixed constant rather than a value fitted "
         "to one dataset, so it holds across scanners, kernels and sites. Air "
         "reconstructs below -1000 HU (deepest observed -1246 HU with a sharp "
         "kernel) and fixed bin size clamps everything at or below the origin "
         "into bin 1, so a shallower origin would collapse the air population."),
        ("Discretisation", "IVH intensity axis",
         "Derived from the re-segmentation range (MIRP default 'none')"),

        ("Texture", "Aggregation, 3D",
         ", ".join(params_3d["glcm_spatial_method"])),
        ("Texture", "Aggregation, 2D / 2.5D",
         ", ".join(get_2d_base_params()["glcm_spatial_method"])),
        ("Texture", "GLCM / NGLDM distance", "1 voxel"),
        ("Texture", "NGLDM coarseness parameter", "0"),

        ("Features", "Families by ROI group",
         "; ".join(f"{group}: {policy['feature_families']}"
                   for group, policy in sorted(GROUP_FEATURE_POLICY.items()))),
        ("Features", "Excluded as non-reproducible",
         ", ".join(sorted(_DROPPED_FEATURES))),
        ("Features", "Why excluded",
         "Moran's I and Geary's C are Monte-Carlo estimates whose seed folds in "
         "the file path; the IVH volume fractions collapse to 0 on the anchored "
         "intensity axis; the local-intensity peaks convolve a kernel that "
         "fabricates air beyond the patch edge."),

        ("Caveats", "3D texture directionality",
         "Because no isotropic resampling is performed, the 13 GLCM/GLRLM "
         "direction vectors are not of equal physical length, and the anisotropy "
         "differs between reconstruction FOVs AND between slice thicknesses. "
         "Measured across the acquisitions in this project it spans 1.02x "
         "(FOV 500, 1 mm) to 5.12x (FOV 300, 3 mm) — the thickness term dominates, "
         "so a 1 mm vs 3 mm comparison is geometrically far less alike than any "
         "two FOVs at one thickness. Direction-averaged 3D texture features are "
         "therefore not strictly rotationally invariant, and comparisons across "
         "FOV or thickness carry a geometric component. Prefer comparing within "
         "one (FOV, thickness) pair, or use the 2D/2.5D variants, whose in-plane "
         "directions are of equal length by construction."),
        ("Caveats", "Scanner-side truncation of the air population",
         "The grey-level origin is set below any value a CT produces, so this "
         "pipeline never truncates. The SCANNER may already have: a "
         "reconstruction that clips at its own floor delivers an air population "
         "piled up on a single value, and that pile-up is data, not an artefact "
         "this pipeline can undo. Measured on the acquisitions here, one device "
         "clips at exactly -1024 HU with roughly a fifth of a honeycomb texture's "
         "voxels sitting on the clip, while another has a continuous noise tail "
         "to -1203 HU. Any cross-DEVICE comparison of first-order, "
         "intensity-histogram and low-grey-level-emphasis features therefore "
         "includes that difference in truncation. Check the observed minimum per "
         "device before comparing those families across scanners; within one "
         "device the comparison is unaffected."),
        ("Caveats", "Grey-level origin and protocol comparison",
         "The origin is a declared constant, identical for every series, so it "
         "cannot bias a comparison in one direction. It is not, however, a pure "
         "offset: the empty leading bins compress the spread of the "
         "low-grey-level-emphasis families (rlm_lgre, szm_lgze, dzm_sdlge, "
         "ngl_hdlge and relatives). Measured, changing the origin alone moves "
         "58 of 226 features, and a handful of those can reorder protocols that "
         "sit within a few percent of each other. Read those families as "
         "internal comparisons only, and prefer difference-based features "
         "(cm_contrast and relatives) when ranking protocols — they are "
         "origin-invariant."),
        ("Caveats", "Morphology on Stars and Textures",
         "These groups request all families, so the morphology family is "
         "reported although the mask is a cuboid; morph_vol_dens_conv_hull is "
         "1.0 by construction. Morphology is NOT computed for inserts."),
    ]
    return rows


_METHODS_PARAGRAPH = (
    "Radiomics features were extracted with MIRP {mirp} using IBSI-1 compliant "
    "feature definitions. Discretisation used a fixed bin size of {width:g} HU "
    "with a grey-level origin of {anchor:g} HU (the int16 encoding floor, chosen "
    "so it lies below the lowest value any reconstruction produces; the deepest "
    "observed was -1246 HU). A re-segmentation range of ({low:g}, {high:g}) HU "
    "was supplied because MIRP derives the fixed-bin-size origin from it; it "
    "excludes no voxel. No interpolation was performed, so voxels retained the "
    "acquisition geometry and remained anisotropic. Circular ROIs were cropped "
    "to their largest padding-free square, retaining 43-80% of the segmented "
    "voxels depending on ROI size and field of view; the analysis mask is "
    "consequently a rectangular cuboid, so morphological features describe the "
    "crop rather than the object and are not reported for inserts. Texture "
    "matrices were aggregated as {agg3d} in 3D, with 2D and 2.5D variants "
    "reported separately. No filter-based (IBSI-2) features were computed. "
    "{n_dropped} features were excluded as non-reproducible."
)


def _write_methods_sheet(
    wb,
    df: pd.DataFrame,
    n_series: int = 0,
    dataset_root: str = "",
) -> None:
    """A sheet stating everything a paper has to declare about this extraction.

    It travels with the numbers instead of living in a README, because the
    workbook is what gets shared, archived and re-opened a year later. It sits
    first so that opening the file shows how the numbers were made before it
    shows the numbers.
    """
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    if "methods" in wb.sheetnames:
        del wb["methods"]
    ws = wb.create_sheet("methods", 0)

    title_fill = PatternFill(start_color="1F4E78", end_color="1F4E78",
                             fill_type="solid")
    section_fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7",
                               fill_type="solid")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")

    ws["A1"] = "Methods declaration — what to report for this extraction"
    ws["A1"].font = Font(bold=True, color="FFFFFF", size=14)
    ws["A1"].fill = title_fill
    ws.merge_cells("A1:C1")
    ws.row_dimensions[1].height = 26

    ws["A2"] = ("Generated from the live pipeline constants, so it always matches "
                "the numbers in this workbook.")
    ws["A2"].font = Font(italic=True, color="595959", size=10)
    ws.merge_cells("A2:C2")

    header_row = 4
    for col, heading in enumerate(("Section", "Item", "Declaration"), start=1):
        cell = ws.cell(header_row, col, heading)
        cell.font = Font(bold=True, color="FFFFFF", size=10)
        cell.fill = title_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

    row = header_row + 1
    previous_section = None
    for section, item, value in _methods_declaration_rows(df, n_series, dataset_root):
        label = section if section != previous_section else ""
        previous_section = section
        section_cell = ws.cell(row, 1, label)
        section_cell.font = Font(bold=True, color="1F4E78", size=10)
        if label:
            section_cell.fill = section_fill
        ws.cell(row, 2, item).font = Font(bold=True, size=10)
        ws.cell(row, 3, str(value))
        for col in range(1, 4):
            ws.cell(row, col).border = border
            ws.cell(row, col).alignment = wrap
        row += 1

    row += 1
    ws.cell(row, 1, "Ready to paste").font = Font(bold=True, color="1F4E78", size=11)
    row += 1
    params_3d = get_3d_base_params()
    paragraph = _METHODS_PARAGRAPH.format(
        mirp=_library_versions().get("mirp", "?"),
        width=params_3d["base_discretisation_bin_width"],
        anchor=_DISCRETISATION_ANCHOR_HU,
        low=DISCRETISATION_RANGE_HU[0],
        high=DISCRETISATION_RANGE_HU[1],
        agg3d=" and ".join(params_3d["glcm_spatial_method"]),
        n_dropped=len(_DROPPED_FEATURES),
    )
    cell = ws.cell(row, 1, paragraph)
    cell.alignment = wrap
    cell.border = border
    ws.merge_cells(start_row=row, start_column=1, end_row=row + 10, end_column=3)

    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 34
    ws.column_dimensions["C"].width = 96
    ws.freeze_panes = ws.cell(header_row + 1, 1)


def _sheet_meta_count(wide: pd.DataFrame) -> int:
    """How many leading columns of a wide sheet are metadata, not features."""
    # Yalnızca satırı TANIMLAYAN sütunlar donduruluyor: ROI adı, boyutu ve seri.
    # Çekim koşulları da solda ve renkli, ama onları da sabitlemek ekranın
    # yarısını yer ve asıl bakılan öznitelikleri görünmez kılar.
    frozen = _COMPACT_ROI_COLUMNS + _COMPACT_BOOKKEEPING_COLUMNS
    return sum(1 for c in frozen if c in wide.columns)


def _compact_series_name(name: str) -> str:
    """Strip the long Radiologie_USZ_..._NAEOTOM_Alpha_<date>_ prefix."""
    if not isinstance(name, str):
        return name
    m = re.match(
        r"^Radiologie_USZ_Siemens_Healthineers_NAEOTOM_Alpha_\d+_(.+)$", name
    )
    return m.group(1) if m else name


# Columns removed from the user-facing wide sheets. They are either redundant
# with a column that is kept, or a non-MIRP summary this pipeline computed for
# its own bookkeeping (ranking, per-ROI HU statistics) rather than a feature.
_COMPACT_DROP = [
    "study_date", "conv_kernel", "pixel_spacing_mm", "group",
    "n_slices", "roi_mean_hu", "roi_std_hu",
]

# Short, readable headers for the acquisition identifiers that stay.
_COMPACT_RENAME = {
    "series_name": "series",
    "recon_kernel": "recon_k",
    "kvp": "kVp",
    "mas": "mAs",
    "fov": "FOV",
    "slice_thickness_mm": "SliceThickness",
    "voxel_count": "n_vox",
    # Also what puts it in _COMPACT_TRAILING's reach, so the spectral-only
    # column ends up on the far right instead of splitting the feature block.
    "me_kev": "me_keV",
}

# Left-hand identifier block, grouped by what the column answers. Everything not
# listed here and not in _COMPACT_TRAILING is a MIRP feature column and follows.
#
#   WHICH ROI  — the first thing anyone looks for when scanning the sheet, so it
#                leads. ``insert_size`` belongs with the name because without it
#                a row is not uniquely identified: the same insert appears once
#                per diameter.
#   BOOKKEEPING— needed to trace a row back, but not read during analysis.
#   ACQUISITION— the conditions that differ between rows; this is what a
#                protocol comparison is actually comparing.
#   ``repeat`` belongs here too: this phantom is scanned test-retest, so the same
#   protocol appears several times and the repeat index is an acquisition
#   condition like any other. It is derived at write time (see
#   _assign_repeat_index) because it is a property of the SET of series, which no
#   single per-series worker can see.
_COMPACT_ROI_COLUMNS = ["label", "insert_size"]
_COMPACT_BOOKKEEPING_COLUMNS = ["series", "n_vox"]
_COMPACT_ACQUISITION_COLUMNS = [
    "recon_k", "k_strength", "kVp", "mAs", "FOV", "SliceThickness", "me_keV",
    "repeat",
]
_COMPACT_META_ORDER = (_COMPACT_ROI_COLUMNS
                       + _COMPACT_BOOKKEEPING_COLUMNS
                       + _COMPACT_ACQUISITION_COLUMNS)

# ``section`` is a hidden helper that row colouring reads to resolve the ROI
# category; it carries no information a reader needs.
_COMPACT_TRAILING = ["section"]

# Source DICOM tags that join the acquisition (blue) block. Deliberately NOT the
# same list as phantomxpro: kVp, mAs, FOV and slice thickness are already parsed
# into their own columns here, so promoting dcm_KVP and friends would show the
# same number twice. What is genuinely missing is when the scan happened, on what
# machine, and WHICH OF SEVERAL SIMILAR SERIES this row is — those are promoted.
# The remaining ~150 tags follow in the yellow block.
#
# The second group exists because the parsed meta columns alone do not always
# identify a series. Measured on these three datasets:
#
#   * 24 of the 80 NAEOTOM series share an identical (recon_k, k_strength, kVp,
#     mAs, FOV, thickness, me_keV) tuple. They are genuine repeat acquisitions
#     minutes apart — the folder names claim 80/100/120 kVp but the KVP tag reads
#     120 for all three and CTDIvol agrees to 0.0003%, so the names are wrong and
#     the tag is right. Without AcquisitionTime / AcquisitionNumber / CTDIvol in
#     view, the only thing separating those rows is a series name this pipeline
#     has measured to be untrustworthy.
#   * The SOMATOM reconstructs each scan twice — once non-iteratively and once
#     with ADMIRE — and SeriesDescription is what says which ("Scan 3  1.0  Br36"
#     vs "Scan 3  1.0  Br36  3").
#   * On Philips the algorithm behind a bare kernel letter is only in
#     (00E1,1040): kernel 'B' with Image Label IMR_BR_3 is an IMR level 3
#     reconstruction, which recon_k alone cannot express.
#
# Private-tag column names come from _column_for_element, which falls back to
# "(gggg,eeee)_Name" when the element has no keyword. A name that does not match
# is simply not promoted, so a pydicom dictionary change degrades to "stays in
# the yellow block" rather than breaking the sheet.
_DICOM_PROMOTED_COLUMNS = (
    "dcm_StudyDate",
    "dcm_AcquisitionTime",
    "dcm_Modality",
    "dcm_Manufacturer",
    "dcm_ManufacturerModelName",
    "dcm_ConvolutionKernel",
    # --- what distinguishes this series from a near-identical neighbour ---
    "dcm_SeriesDescription",
    "dcm_SeriesNumber",
    "dcm_AcquisitionNumber",
    "dcm_CTDIvol",
    "dcm_(0029,1043)",                  # NAEOTOM IQ level
    "dcm_(00E1,1040)_Image_Label",      # Philips reconstruction label
    "dcm_(01F7,109B)_iDose_Level",      # Philips iDose level
)

# Column-block fills, matching the phantomxpro workbook so the two outputs read
# the same way. Applied only to the meta block: the feature columns keep the
# per-ROI row banding, which is what makes long horizontal scrolling navigable.
_BLOCK_FILL_ROI = "C6E0B4"          # green — which insert / star / texture
_BLOCK_FILL_BOOKKEEPING = "FFF2CC"  # yellow
_BLOCK_FILL_ACQUISITION = "DDEBF7"  # blue

# Columns exempt from the generic meta-column width cap, matched BY HEADER NAME.
#
# The series identifier is the only thing distinguishing some rows: two
# acquisitions can share every other meta value — a folder named _80_ whose KVP
# tag reads 120 sits next to the real 120 kVp series with an identical meta
# block — so a truncated series column makes genuinely different rows look
# identical in Excel.
#
# By name and not by position, because it WAS pinned to column A. When the meta
# block was regrouped (ROI / bookkeeping / acquisition) ``label`` took column A
# and inherited the 60-wide exemption at 9 characters, while ``series`` moved to
# column C and fell back to the 14-character cap at 43 characters — precisely
# the collision this exemption exists to prevent.
_WIDE_DISPLAY_COLUMNS = ("series",)
_WIDE_DISPLAY_WIDTH = 60


def _compact_wide(wide: pd.DataFrame) -> pd.DataFrame:
    """Prepare user-facing wide feature tables.

    Output row order requested by the user:
        Insert_01 -> Series/Protocol A -> 5mm, 10mm
                 -> Series/Protocol B -> 5mm, 10mm
        Insert_02 -> ...

    The visible table keeps only acquisition identifiers, insert label/size,
    explicit voxel count, and MIRP feature columns. Ranking and extra
    non-MIRP ROI summary calculations are removed from the user-facing sheets.
    ``section`` is kept as a hidden helper column so row colouring can
    resolve the ROI category.
    """
    drop_now = [c for c in _COMPACT_DROP if c in wide.columns]
    wide = wide.drop(columns=drop_now)

    if "series_name" in wide.columns:
        wide["series_name"] = wide["series_name"].astype(str).map(_compact_series_name)

    wide = wide.rename(columns=_COMPACT_RENAME)

    def _insert_no(value) -> int:
        m = re.search(r"Insert[_-]?(\d+)", str(value), re.IGNORECASE)
        return int(m.group(1)) if m else 9999

    def _size_order(value) -> int:
        m = re.search(r"(5|10|15)\s*mm", str(value), re.IGNORECASE)
        if not m:
            return 9999
        return {"5": 5, "10": 10, "15": 15}.get(m.group(1), 9999)

    sort_keys = pd.DataFrame({
        "_insert_no": wide["label"].map(_insert_no) if "label" in wide.columns else 9999,
        "_series": wide["series"].astype(str) if "series" in wide.columns else "",
        "_size": wide["insert_size"].map(_size_order) if "insert_size" in wide.columns else 9999,
    })
    wide = (wide.assign(**{k: sort_keys[k] for k in sort_keys.columns})
                .sort_values(["_insert_no", "_series", "_size"])
                .drop(columns=list(sort_keys.columns))
                .reset_index(drop=True))

    # Fixed metadata block, then feature columns, then the parked extras.
    preferred_existing = [c for c in _COMPACT_META_ORDER if c in wide.columns]
    trailing = [c for c in _COMPACT_TRAILING if c in wide.columns]
    features = [c for c in wide.columns
                if c not in preferred_existing and c not in trailing]
    return wide[preferred_existing + features + trailing]

def _style_sheet(
    ws,
    df: pd.DataFrame,
    freeze_meta_count: int,
    n_rows_for_width_sample: int = 8,
    as_table: bool = False,
    table_name: Optional[str] = None,
) -> None:
    """Header style, frozen panes, capped column widths, number format,
    optional autofilter or Excel Table wrapper, and a thicker right border
    on the last meta column for visual separation.

    as_table=True wraps the data range as a real Excel Table (banded rows
    + native filter dropdowns from the Table style). Use it on small
    sheets where there is no custom row colouring (otherwise the Table's
    banded fill overwrites the row colours).

    as_table=False keeps the manual header fill and adds a plain
    AutoFilter so headers still get filter dropdowns without disturbing
    any custom cell fills set elsewhere (e.g. _apply_roi_colors).
    """
    from openpyxl.styles import (
        Font, PatternFill, Alignment, Border, Side
    )
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    if ws.max_row < 1:
        return

    center = Alignment(horizontal="center", vertical="center")
    thin = Side(style="thin", color="BFBFBF")
    medium_dark = Side(style="medium", color="1F4E78")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    if not as_table:
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78",
                                  fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=10)
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center
            cell.border = border
    else:
        for cell in ws[1]:
            cell.alignment = center

    if freeze_meta_count > 0 and ws.max_column > freeze_meta_count:
        ws.freeze_panes = ws.cell(
            row=2, column=freeze_meta_count + 1
        ).coordinate
    else:
        ws.freeze_panes = "A2"

    # Resolved from the header row, so the exemption follows the column when the
    # meta block is reordered instead of staying behind on whatever lands in A.
    wide_display = {
        i for i in range(1, ws.max_column + 1)
        if str(ws.cell(1, i).value or "") in _WIDE_DISPLAY_COLUMNS
    }

    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        max_len = len(str(ws.cell(1, col_idx).value or ""))
        for r in range(2, min(ws.max_row + 1, 2 + n_rows_for_width_sample)):
            v = ws.cell(r, col_idx).value
            if v is not None:
                max_len = max(max_len, len(str(v)[:24]))
        if col_idx in wide_display:
            # See _WIDE_DISPLAY_COLUMNS for why this column must not be capped.
            ws.column_dimensions[letter].width = _WIDE_DISPLAY_WIDTH
        elif col_idx <= freeze_meta_count:
            ws.column_dimensions[letter].width = min(max_len + 2, 14)
        else:
            ws.column_dimensions[letter].width = min(max(max_len + 1, 11), 14)

    # Number format, chosen PER COLUMN from that column's own SPREAD, so that the
    # variation between protocols stays visible on screen.
    #
    # A fixed "0.000" was hiding the differences the workbook exists to show.
    # Measured on the 80-protocol NAEOTOM workbook, for one texture ROI: 102 of
    # 207 feature columns rendered fewer distinct values than they hold, and
    # rlm_lrlge_3d_v_mrg showed 80 genuinely different protocol values as one of
    # only TWO strings while spanning 71% of its own mean. Stored values were
    # always full precision — formulas and pandas were unaffected — but this
    # workbook is built to be read and filtered in Excel, and a feature that
    # looks constant on screen is a feature nobody investigates.
    #
    # The criterion is the SPREAD, not the magnitude. Anchoring on magnitude
    # (four significant figures) fails in the opposite direction on columns whose
    # values are large but tightly clustered: cm_auto_corr spans ~2000..5000, so
    # four significant figures rounds it to whole numbers and neighbouring
    # protocols merge again. Resolving (max - min) into ~10^4 steps keeps the
    # variation legible whatever the magnitude, and a floor of four significant
    # figures keeps a constant column from collapsing to "0".
    #
    # Width is respected: openpyxl cannot know Excel's rendered width, and a
    # format wider than the column shows as "####", which is strictly worse than
    # a rounded number. Decimals are therefore capped to what fits.
    # The binding constraint is the NARROWEST within-ROI spread, not the column's
    # full range. A reader compares one ROI across protocols, and for stat_p10 the
    # column spans 1400 HU while a single insert varies by 3 HU — a format chosen
    # from the full range shows that insert in 30 steps at best. Grouping by
    # ``label`` targets the comparison the sheet exists for. Sheets without a
    # label column (series, insert_ranking) fall back to the whole column.
    columns = list(df.columns)
    labels = df["label"] if "label" in df.columns else None

    def _finest_spread(name) -> Optional[float]:
        """Smallest non-noise within-ROI spread of a column, or None."""
        if labels is None or name not in df.columns:
            return None
        numeric = pd.to_numeric(df[name], errors="coerce")
        finest = None
        for _, group in numeric.groupby(labels, observed=True):
            group = group.dropna()
            if len(group) < 2:
                continue
            spread = float(group.max() - group.min())
            scale = float(group.abs().max()) or 1.0
            if spread <= 0 or spread / scale < 1e-9:
                continue
            finest = spread if finest is None else min(finest, spread)
        return finest

    for col_idx in range(freeze_meta_count + 1, ws.max_column + 1):
        # Ints count towards the DECISION even though only floats receive the
        # format. Many HU-derived features (stat_p10 and relatives) land on whole
        # numbers, so a float-only sample can be three values out of 160 and
        # picks a format from an unrepresentative minority.
        values = []
        for r in range(2, ws.max_row + 1):
            v = ws.cell(row=r, column=col_idx).value
            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and v == v and v not in (float("inf"), float("-inf"))):
                values.append(float(v))

        number_format = None
        if values:
            magnitudes = [abs(v) for v in values if v != 0.0]
            high = max(magnitudes) if magnitudes else 0.0
            low = min(magnitudes) if magnitudes else 0.0
            spread = max(values) - min(values)

            if high and (high >= 1e6 or low < 1e-4):
                # Uniform scientific notation. Six mantissa decimals resolve a
                # spread three orders below the magnitude, which covers every
                # such column here (the local-intensity and IVH-derived ones).
                number_format = "0.000000E+00"
            else:
                wanted = 3
                if spread > 0:
                    wanted = -int(math.floor(math.log10(spread / 1e4)))
                if high > 0:
                    wanted = max(wanted, 3 - int(math.floor(math.log10(high))))
                # Outliers must not coarsen the format for everyone else. A
                # column holds every ROI type, and a single insert whose mean HU
                # sits near zero sends stat_cov into the thousands — one extreme
                # value then chose a format that rounded the 0.16-wide spread of
                # the texture ROIs into two visible steps. The comparison a
                # reader makes is WITHIN an ROI, so the format follows the robust
                # (5th-95th percentile) spread and lets the outlier carry more
                # decimals than it needs.
                ordered = sorted(values)
                if len(ordered) >= 20:
                    robust = (ordered[int(0.95 * (len(ordered) - 1))]
                              - ordered[int(0.05 * (len(ordered) - 1))])
                    if robust > 0:
                        wanted = max(wanted,
                                     -int(math.floor(math.log10(robust / 1e4))))
                # …and finally the narrowest within-ROI spread, resolved into
                # ~100 steps. This is the constraint that actually matters.
                finest = _finest_spread(columns[col_idx - 1]
                                        if col_idx - 1 < len(columns) else None)
                if finest:
                    wanted = max(wanted,
                                 -int(math.floor(math.log10(finest / 1e2))))
                int_digits = max(1, int(math.floor(math.log10(high))) + 1) if high else 1
                affordable = max(0, 13 - int_digits - 1)
                decimals = max(0, min(wanted, affordable, 10))
                number_format = ("0." + "0" * decimals) if decimals else "0"

        for r in range(2, ws.max_row + 1):
            cell = ws.cell(row=r, column=col_idx)
            if number_format and isinstance(cell.value, float):
                cell.number_format = number_format
            cell.alignment = Alignment(horizontal="right")

    # Thicker right border on the last meta column to mark the meta/feature
    # boundary. Skipped when there are no feature columns.
    if 0 < freeze_meta_count < ws.max_column:
        for r in range(1, ws.max_row + 1):
            cell = ws.cell(row=r, column=freeze_meta_count)
            existing = cell.border
            cell.border = Border(
                left=existing.left, top=existing.top,
                bottom=existing.bottom, right=medium_dark,
            )

    if as_table:
        ref = (
            f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
        )
        tbl_name = table_name or f"tbl_{ws.title}".replace(" ", "_")
        tbl_name = re.sub(r"[^A-Za-z0-9_]", "_", tbl_name)
        tbl = Table(displayName=tbl_name, ref=ref)
        tbl.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False,
        )
        ws.add_table(tbl)
    else:
        ws.auto_filter.ref = (
            f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
        )


# =========================================================================
# Combined Excel writer
# =========================================================================

def _add_meta_data_bar(ws, df: pd.DataFrame, col_name: str, color: str) -> None:
    """Add a data-bar conditional format on a numeric meta column."""
    from openpyxl.formatting.rule import DataBarRule
    from openpyxl.utils import get_column_letter

    if col_name not in df.columns or ws.max_row < 2:
        return
    col_idx = list(df.columns).index(col_name) + 1
    letter = get_column_letter(col_idx)
    ref = f"{letter}2:{letter}{ws.max_row}"
    rule = DataBarRule(
        start_type="min", end_type="max",
        color=color, showValue=True,
    )
    ws.conditional_formatting.add(ref, rule)


def _add_meta_color_scale(ws, df: pd.DataFrame, col_name: str) -> None:
    """3-colour scale on a numeric meta column (low=red, mid=yellow, high=green)."""
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter

    if col_name not in df.columns or ws.max_row < 2:
        return
    col_idx = list(df.columns).index(col_name) + 1
    letter = get_column_letter(col_idx)
    ref = f"{letter}2:{letter}{ws.max_row}"
    rule = ColorScaleRule(
        start_type="min", start_color="F8696B",
        mid_type="percentile", mid_value=50, mid_color="FFEB84",
        end_type="max", end_color="63BE7B",
    )
    ws.conditional_formatting.add(ref, rule)


def write_combined_excel(
    rows: List[Dict],
    ranking: Optional[pd.DataFrame],
    output_path: Path,
    series_dirs: Optional[Dict[str, Path]] = None,
    dataset_root: Optional[Path] = None,
) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows to write - combined Excel skipped.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta_id_cols = [
        "series_name", "study_date", "recon_kernel", "conv_kernel", "k_strength",
        "kvp", "mas", "fov", "slice_thickness_mm", "pixel_spacing_mm",
        "repeat", "group", "section", "label", "insert_size",
        "n_slices", "roi_mean_hu", "roi_std_hu", "voxel_count",
    ]
    meta_id_cols = [c for c in meta_id_cols if c in df.columns]

    for c in meta_id_cols:
        df[c] = df[c].astype(object).where(df[c].notna(), other="")

    # Columns whose values are STRINGS when parsed fresh but come back as
    # integers through the Excel checkpoint round-trip. On a resumed run the two
    # populations mix in one column and Excel then treats them as two different
    # values: the AutoFilter dropdown lists "1" twice, a pivot splits one
    # reconstruction level into two items, and =A2=1 / MATCH / SUMPRODUCT
    # silently skip the text half. That lands squarely on k_strength, which is
    # the primary experimental variable of the SOMATOM (ADMIRE 1..5) and Philips
    # (IMR/iDose level) acquisitions. kvp/mas/fov/slice_thickness escape it only
    # because pydicom's DSfloat/IS are float/int subclasses.
    #
    # Normalised to text on BOTH paths rather than to numbers: these are
    # categorical labels, not quantities — nobody averages a kernel strength —
    # and an empty string for "the scanner recorded none" has no numeric
    # equivalent that Excel would not read as zero.
    for c in ("k_strength", "study_date"):
        if c in df.columns:
            df[c] = df[c].map(
                lambda v: "" if v in (None, "") else
                (str(int(v)) if isinstance(v, float) and float(v).is_integer()
                 else str(v).strip()))

    n_series = df["series_name"].nunique() if "series_name" in df.columns else 0

    # ------------------------------------------------------------------ #
    # Per-series DICOM tags, read BEFORE any sheet is written.
    #
    # Order matters and used not to. Two derived values feed the sheets:
    # ``me_kev`` (part of the protocol key) and ``AcquisitionTime`` (which orders
    # test-retest repeats). The repeat index was previously computed 100 lines
    # AFTER the series sheet had already been written from a detached copy of the
    # frame, so that sheet always carried the empty placeholder — while the
    # methods sheet told the reader to "see the 'repeat' column". Gathering the
    # tags first means every sheet is written from the same, complete frame.
    # ------------------------------------------------------------------ #
    series_names = sorted(df["series_name"].unique())
    _SERIES_TAG_COLUMNS = ["recon_type", "me_keV", "acq_dt", "series_no",
                           "image_type", "recon_diam_mm", "q_strength"]
    series_tags: Dict[str, Dict[str, str]] = {}
    dicom_by_series: Dict[str, Dict[str, str]] = {}

    if series_dirs:
        full_rows: List[Dict[str, str]] = []
        all_keys: set = set()
        for sname in series_names:
            sdir = series_dirs.get(sname)
            dcm_dir = (_find_dicoms_dir(sdir) or (sdir / "dicoms")) if sdir else None
            series_tags[sname] = read_dicom_tags(dcm_dir) if dcm_dir else {}
            # All header tags, for the dcm_* columns on the feature sheets.
            tags: Dict[str, str] = {}
            if dcm_dir is not None:
                try:
                    tags = read_dicom_full_metadata(dcm_dir)
                except Exception as e:
                    print(f"  WARN dicom metadata read failed for {sname}: {e}")
            tags["series_name"] = sname
            full_rows.append(tags)
            all_keys.update(tags.keys())

        # These used to have their own sheet. They now travel with the feature
        # rows instead: reading one row no longer means cross-referencing a
        # second sheet, and there is only one place where a tag can be read.
        if full_rows:
            dicom_df = (pd.DataFrame(full_rows)
                        .reindex(columns=_order_dicom_columns(all_keys))
                        .fillna(""))
            dicom_df["series_name"] = (
                dicom_df["series_name"].astype(str).map(_compact_series_name))
            for record in dicom_df.to_dict(orient="records"):
                dicom_by_series[str(record["series_name"])] = {
                    f"dcm_{k}": v for k, v in record.items()
                    if k != "series_name" and str(v) != ""
                }

        # me_keV into the per-ROI rows so the wide sheets can distinguish paired
        # MonoE reconstructions — and so it can take part in the protocol key.
        kev_map = {n: pd.to_numeric(series_tags[n].get("me_keV", ""),
                                    errors="coerce")
                   for n in series_names}
        df["me_kev"] = df["series_name"].map(kev_map)
        df["me_kev"] = df["me_kev"].astype(object).where(
            df["me_kev"].notna(), other="")
        if "me_kev" not in meta_id_cols:
            meta_id_cols.insert(meta_id_cols.index("fov") + 1, "me_kev")

    # ----- test-retest repeat index -----
    # Derived from the whole set of series, so it cannot live in a per-series
    # worker. Overwrites the placeholder the checkpoints carry, which is empty.
    repeats = _assign_repeat_index(df, dicom_by_series)
    if repeats:
        df["repeat"] = df["series_name"].map(repeats).fillna(1).astype(int)
        n_repeated = sum(1 for v in repeats.values() if v > 1)
        if n_repeated:
            print(f"  Test-retest: {n_repeated} of {len(repeats)} series are "
                  f"repeat acquisitions of a protocol already present "
                  f"(numbered in the 'repeat' column).")

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as w:
        # ----- series sheet -----
        series_meta = (
            df.drop_duplicates(subset=["series_name"])
            [["series_name", "study_date", "recon_kernel", "k_strength",
              "kvp", "mas", "fov", "slice_thickness_mm", "pixel_spacing_mm",
              "repeat"]]
            .sort_values("series_name")
            .reset_index(drop=True)
        )
        if series_dirs:
            for col in _SERIES_TAG_COLUMNS:
                series_meta[col] = [
                    series_tags.get(n, {}).get(col, "")
                    for n in series_meta["series_name"]
                ]
            # Numeric me_keV so the colour scale works (blank when absent).
            series_meta["me_keV"] = pd.to_numeric(
                series_meta["me_keV"], errors="coerce")

        series_meta_disp = series_meta.copy()
        series_meta_disp["series_name"] = (
            series_meta_disp["series_name"].astype(str).map(_compact_series_name)
        )
        series_meta_disp = series_meta_disp.rename(columns={
            "series_name": "series",
            "study_date": "date",
            "recon_kernel": "recon_k",
            "k_strength": "k_strength",
            "kvp": "kVp", "mas": "mAs", "fov": "FOV",
            "slice_thickness_mm": "thick_mm",
            "pixel_spacing_mm": "pixel_mm",
        })
        series_meta_disp.to_excel(w, sheet_name="series", index=False)
        ws_series = w.sheets["series"]
        _style_sheet(ws_series, series_meta_disp, freeze_meta_count=1,
                     as_table=True, table_name="SeriesSummary")
        _add_meta_color_scale(ws_series, series_meta_disp, "kVp")
        _add_meta_color_scale(ws_series, series_meta_disp, "mAs")

        # ----- insert ranking sheet -----
        # Global 1..40 ranking of the inserts by their mean HU (averaged
        # across every series), rank 1 = lowest HU. Uses the roi_mean_hu
        # summary column, so it is independent of the feature policy
        # (works whether inserts are 'statistics' or 'all').
        if ranking is not None and not ranking.empty:
            rank_disp = ranking.rename(columns={
                "mean_hu_global": "mean_HU_global",
                "ranked_index": "rank",
            })[["rank", "label", "mean_HU_global"]]
            rank_disp.to_excel(w, sheet_name="insert_ranking", index=False)
            ws_rank = w.sheets["insert_ranking"]
            _style_sheet(ws_rank, rank_disp, freeze_meta_count=2)
            _apply_ranking_colors(ws_rank, rank_disp)
            _add_meta_data_bar(ws_rank, rank_disp, "mean_HU_global", "638EC6")

        # ----- per-mode wide tables -----
        for mode in sorted(df["mode"].unique()):
            sub = df[df["mode"] == mode]
            base = sub[sub["source"] == "Base"]

            if not base.empty:
                wide = base.pivot_table(
                    index=meta_id_cols,
                    columns="feature_id",
                    values="value",
                    aggfunc="first",
                ).reset_index()
                wide = _compact_wide(wide)
                wide = _attach_dicom_columns(wide, dicom_by_series)
                sheet = f"{mode}_base"[:31]
                wide.to_excel(w, sheet_name=sheet, index=False)
                ws = w.sheets[sheet]
                _style_sheet(ws, wide, freeze_meta_count=_sheet_meta_count(wide))
                _apply_roi_colors(ws, wide, color_full_row=True)
                _apply_block_colors(ws, wide)
                _add_meta_color_scale(ws, wide, "kVp")
                _add_meta_color_scale(ws, wide, "mAs")
                _add_meta_color_scale(ws, wide, "me_keV")
                _hide_columns(ws, wide, ("section",))

        # ----- Methods declaration (first sheet) + tab colours -----
        _write_methods_sheet(
            w.book, df,
            n_series=n_series,
            dataset_root=str(dataset_root) if dataset_root else "",
        )
        _apply_tab_colors(w.book)

    print(f"  Combined Excel written: {output_path}")


# =========================================================================
# Main
# =========================================================================

# Where the finished workbook ended up, for the single closing prompt in
# __main__. A module-level handoff rather than a return value because the .exe
# must ask its ONE question after main() has reported everything, whatever exit
# path it took.
_RESULT_PATH: Optional[Path] = None


def _close_frozen_console() -> None:
    """The .exe's single closing interaction.

    Deliberately ONE prompt. There used to be two — "open the result?" from the
    run and "press Enter to close" from the launcher — which is two questions too
    many for a tool whose whole premise is that pointing at a folder is all the
    user does. When a workbook was produced, Enter opens it; otherwise Enter just
    closes, and the console has stayed readable either way.
    """
    if not getattr(sys, "frozen", False):
        return
    result = _RESULT_PATH
    try:
        if result is not None and result.exists():
            answer = input(
                f"\nSonuç: {result}\n"
                f"Açmak için Enter, sadece kapatmak için q + Enter: "
            ).strip().lower()
            if answer not in ("q", "n", "no", "hayir", "hayır"):
                try:
                    os.startfile(str(result))
                except Exception as e:
                    print(f"Açılamadı ({e}). Dosya yukarıdaki yolda.")
        else:
            input("\nKapatmak için Enter...")
    except (EOFError, KeyboardInterrupt):
        pass


def _pick_dataset_folder() -> Optional[str]:
    """Open the native folder chooser, or return None if it is unavailable.

    Pointing at a folder is the ONLY thing this tool asks of the person running
    it, so it should not also require them to know how to paste a Windows path
    into a console. tkinter ships with Python and is bundled into the .exe;
    every failure path here falls back to typing, so a machine without Tcl/Tk
    still works.
    """
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:
        return None
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.update()
        chosen = filedialog.askdirectory(
            title="Etiketlenmiş veri seti klasörünü seçin "
                  "(serileri İÇEREN klasör)",
            mustexist=True,
        )
        root.destroy()
        return chosen or None
    except Exception:
        return None


def _ask_for_dataset_root() -> Optional[str]:
    """Get the dataset folder when none was supplied on the command line.

    A double-click passes no argument at all, and there is no sensible default —
    every site keeps its labelled data somewhere different.
    """
    print("Klasör seçme penceresi açılıyor...")
    chosen = _pick_dataset_folder()
    if chosen:
        print(f"Seçilen: {chosen}")
        return chosen

    print("\nKlasör seçilmedi.")
    print("Etiketlenmiş veri seti klasörünü bu pencereye SÜRÜKLEYIP BIRAKIN")
    print("(ya da yolunu yapıştırın) ve Enter'a basın. Ctrl+C iptal eder.")
    try:
        return input("\nKlasör: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nİptal edildi.")
        return None


def _publish_result(path: Path) -> None:
    """Record the finished workbook for the closing prompt (see __main__)."""
    global _RESULT_PATH
    _RESULT_PATH = path


def _explain_empty_dataset(dataset_root: Path) -> None:
    """Say what a usable folder looks like when discovery found nothing.

    "No series matched filters" is true but useless to somebody who has just
    pointed the tool at the wrong level of their folder tree — by far the most
    likely mistake, since the labelled export nests series two deep at some
    sites and one deep at others.
    """
    print("\n" + "!" * 70)
    print(f"NO SERIES FOUND under {dataset_root}")
    print("!" * 70)
    print("Expected the folder to CONTAIN series folders (searched up to 3 levels),")
    print("each of them shaped like this:")
    print("")
    print("  <series name>/")
    print("    dicoms/              *.dcm or *.IMA   (the raw CT slices)")
    print(f"    Inserts/{'|'.join(INSERT_SIZES)}/Insert_01/*.dcm")
    print("    Stars/Star16|Star8/*.dcm              (optional)")
    print("    Textures/<pattern>/*.dcm              (optional)")
    print("")
    print("Common mistakes:")
    print("  * pointed at ONE series instead of the folder holding the series")
    print("  * pointed at the un-labelled export (no Inserts/ folder)")
    print("  * the insert size folders are named differently — only "
          f"{', '.join(INSERT_SIZES)} are read")
    children = [c.name for c in sorted(dataset_root.iterdir())[:6] if c.is_dir()]
    if children:
        print(f"\nWhat is actually in there: {', '.join(children)}"
              f"{' ...' if len(children) == 6 else ''}")
    print("!" * 70)


def _default_jobs(n_series: int) -> int:
    """Workers to use by default: bounded by cores AND by memory.

    Each worker holds a MIRP extraction of a texture patch and peaks around
    1.2 GB. Choosing purely on core count is fine on the 20-core/31 GB machine
    this was developed on and thrashes an 8 GB laptop, which is what the people
    receiving the .exe actually have.
    """
    by_cores = max(1, min(12, (os.cpu_count() or 2) - 2))
    by_memory = by_cores
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            # Leave 2 GB for the OS and whatever else is open.
            usable_gb = max(0.0, status.ullAvailPhys / 2 ** 30 - 2.0)
            by_memory = max(1, int(usable_gb / 1.2))
    except Exception:
        pass
    return max(1, min(by_cores, by_memory, max(1, n_series)))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract MIRP/IBSI radiomics for DICOM-labeled insert ROIs only."
    )
    p.add_argument("dataset_root_pos", nargs="?", default=None,
                   help="Dataset root directory (positional; supports drag-and-drop "
                        "of a folder onto the .exe).")
    # No default path. A hardcoded one belonged to the machine this was written
    # on, so a colleague double-clicking the .exe scanned a folder that does not
    # exist on their disk and got "0 series" with nothing pointing at the cause.
    p.add_argument("--dataset-root", default=None,
                   help="Dataset root directory. Omit it and the .exe asks.")
    p.add_argument("--output-dir", default=None,
                   help="Output dir (default: {dataset_root}/radiomics_output).")
    p.add_argument("--series", default=None,
                   help="Process only series whose name contains this substring.")
    p.add_argument("--series-file", default=None,
                   help="Path to a text file with one EXACT series name per line. "
                        "Only series whose name matches exactly are processed. "
                        "Takes precedence over --series.")
    p.add_argument("--max-series", type=int, default=None,
                   help="Process only the first N series (after --series filter).")
    p.add_argument("--no-3d", action="store_true",
                   help="Skip 3D extraction.")
    p.add_argument("--no-2d", action="store_true",
                   help="Skip 2D extraction.")
    p.add_argument("--no-skip", action="store_true",
                   help="Reprocess series even if a checkpoint already exists.")
    p.add_argument("--jobs", "-j", type=int, default=None,
                   help="Number of series to process in PARALLEL (separate "
                        "processes). Series are independent, so this scales "
                        "almost linearly with CPU cores. Default: "
                        "min(12, cores-2). Use 1 for the old sequential "
                        "behaviour. Lower it if you run out of RAM.")
    p.add_argument("--dry-run", action="store_true",
                   help="List series/ROIs and exit.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the pre-extraction confirmation prompt "
                        "(only relevant when running as a frozen .exe).")
    args = p.parse_args()

    given = args.dataset_root_pos or args.dataset_root
    if not given:
        given = _ask_for_dataset_root()
        if not given:
            return 1
    # Windows drag-and-drop and copy-as-path both hand over a quoted string, and
    # a trailing backslash before the closing quote survives as part of the name.
    dataset_root = Path(given.strip().strip('"').strip("'").rstrip("\\/") or given)
    if not dataset_root.is_dir():
        print(f"\nERROR: not a folder: {dataset_root}")
        print("Give the LABELLED dataset root — the folder that holds the series "
              "folders, each with dicoms/ and Inserts/.")
        return 1

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else dataset_root / "radiomics_output"
    )
    cache_dir = output_dir / "_nifti_cache"

    series_list = discover_series(dataset_root)
    _print_rejected_series()
    _print_series_meta_conflicts()
    if args.series_file:
        with open(args.series_file, "r", encoding="utf-8") as f:
            target_names = {line.strip() for line in f if line.strip()}
        series_list = [s for s in series_list if s["series_name"] in target_names]
    elif args.series:
        series_list = [s for s in series_list if args.series in s["series_name"]]
    if args.max_series is not None:
        series_list = series_list[:args.max_series]

    do_3d = not args.no_3d
    do_2d = not args.no_2d

    print("=" * 70)
    print("PCCT Phantom Radiomics Extraction (MIRP / IBSI)")
    print("=" * 70)
    print(f"Dataset:    {dataset_root}")
    print(f"Output:     {output_dir}")
    print(f"Series:     {len(series_list)}")
    hata_series = [s for s in series_list if s["series_name"].startswith("HATA")]
    if hata_series:
        print(f"WARNING:    {len(hata_series)} series with 'HATA' prefix included "
              f"(user-flagged as error):")
        for s in hata_series:
            print(f"              - {s['series_name']}")
    print(f"Modes:      "
          f"{'3D' if do_3d else ''} {'2D' if do_2d else ''} | "
          f"base features only")
    print(f"Policy:     Inserts -> first-order (statistics), no filters; "
          f"Stars + Textures -> ALL families"
          f"; "
          f"input masks = DICOM labels under Inserts/, Stars/, Textures/")
    print("=" * 70)

    if args.dry_run:
        # Counts ROI FOLDERS. It used to build every ROI, which meant reading the
        # pixel data of 90 patches x 30 slices per series — 216,000 small file
        # reads on the 80-series set, ~13 minutes on a local disk and far worse
        # over a hospital share. That is not a "quick look at what was found",
        # which is what --dry-run is documented as and what somebody pointing the
        # .exe at a new folder for the first time actually wants.
        #
        # The trade: this no longer proves the patches are READABLE. That is
        # checked by the real run, which reports every unreadable ROI in the
        # SKIPPED ROIs block and exits 2 rather than losing it quietly.
        for s in series_list:
            print(f"  {s['series_name']}")
            try:
                ins = discover_insert_roi_dirs(s["series_dir"])
                stars = discover_star_roi_dirs(s["series_dir"])
                texs = discover_texture_roi_dirs(s["series_dir"])
            except Exception as e:
                print(f"     ROI discovery error: {e}")
                continue

            by_size: Dict[str, int] = {}
            for size_label, _label, _path in ins:
                by_size[size_label] = by_size.get(size_label, 0) + 1
            size_str = ', '.join(f'{k}={v}' for k, v in sorted(by_size.items()))
            print(f"     Inserts: {len(ins)} ({size_str})")
            print(f"     Stars:   {len(stars)} "
                  f"({', '.join(label for _s, label, _p in stars)})")
            print(f"     Textures:{len(texs)} "
                  f"({', '.join(label for _s, label, _p in texs)})")
        print(f"\nTotal series: {len(series_list)}")
        return 0

    if not series_list:
        if args.series or args.series_file or args.max_series:
            print("No series matched the --series / --series-file / --max-series "
                  "filters.")
        else:
            _explain_empty_dataset(dataset_root)
        return 1

    from collections import Counter
    date_counts = Counter(s["study_date"] for s in series_list)
    print()
    print(f"Series breakdown by date ({len(date_counts)} dates):")
    for d in sorted(date_counts):
        print(f"  {d}: {date_counts[d]:3d} series")
    print(f"  -----")
    print(f"  Total: {len(series_list)} series "
          f"(insert ROI count is read from DICOM masks: "
          f"{'/'.join(INSERT_SIZES)} only)")
    print()

    if getattr(sys, "frozen", False) and not args.yes:
        try:
            ans = input(
                "Press Enter to START extraction, "
                "or type 'q' + Enter to CANCEL: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("Cancelled by user.")
            return 0
        if ans in ("q", "quit", "n", "no", "cancel"):
            print("Cancelled by user.")
            return 0
        print()

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve parallelism. Series are independent (own patches, own checkpoint)
    # so we fan them out across processes; the default leaves cores for the OS
    # and refuses to start more workers than RAM can hold (see _default_jobs).
    jobs = args.jobs if args.jobs else _default_jobs(len(series_list))
    jobs = max(1, min(jobs, len(series_list)))
    print(f"Parallel:   {jobs} series at a time "
          f"({os.cpu_count()} CPU cores detected)")
    # Measured on the three reference acquisitions: ~1.9 min per series for
    # 3D+2D with the insert policy, on one worker. Reported as a range because
    # a slower disk or a busy machine moves it, and stated per-worker so it
    # tracks --jobs instead of pretending the run is sequential.
    per_series_min = 1.9 * (2 if do_3d and do_2d else 1) / 2
    eta_min = per_series_min * len(series_list) / jobs
    print(f"Estimate:   ~{eta_min:.0f} min extraction"
          f" + ~{max(2.0, 0.08 * len(series_list)):.0f} min to write the workbook"
          f"  (interruptible; resumes from _per_series checkpoints)")

    all_rows: List[Dict] = []
    skipped_rois: List[Tuple[str, str, str, str]] = []
    failed_series: List[Tuple[str, str]] = []
    t_start = time.time()
    n_total = len(series_list)

    if jobs <= 1:
        # Sequential (original behaviour).
        for i, s in enumerate(series_list, 1):
            print(f"\n[{i}/{n_total}]", end="")
            name, rows, err, skipped = _process_series_worker((
                s, output_dir, cache_dir, do_3d, do_2d, not args.no_skip,
            ))
            skipped_rois.extend(skipped)
            if err:
                print(f"  ERROR processing {name}:\n{err}")
                failed_series.append((name, err.strip().splitlines()[-1][:150]))
            elif not rows:
                # No exception, but nothing came out. Still a failed series: the
                # workbook would simply be missing it.
                failed_series.append((name, "produced no rows"))
            else:
                all_rows.extend(rows)
    else:
        # Parallel: one process per series, up to `jobs` at a time.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        payloads = [
            (s, output_dir, cache_dir, do_3d, do_2d, not args.no_skip)
            for s in series_list
        ]
        done = 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(_process_series_worker, p): p[0]["series_name"]
                for p in payloads
            }
            for fut in as_completed(futures):
                name, rows, err, skipped = fut.result()
                skipped_rois.extend(skipped)
                done += 1
                elapsed = time.time() - t_start
                rate = elapsed / max(done, 1)
                eta = rate * (n_total - done)
                if err:
                    print(f"[{done}/{n_total}] ERROR {name}:\n{err}")
                    failed_series.append((name, err.strip().splitlines()[-1][:150]))
                elif not rows:
                    failed_series.append((name, "produced no rows"))
                    print(f"[{done}/{n_total}] EMPTY {name[-45:]}")
                else:
                    all_rows.extend(rows)
                    print(f"[{done}/{n_total}] OK {name[-45:]} "
                          f"({len(rows)} rows) | elapsed {elapsed/60:.0f} min "
                          f"| ETA {eta/60:.0f} min")

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Total elapsed: {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"Rows collected: {len(all_rows)}")
    _print_failed_series_summary(failed_series)
    _print_skipped_roi_summary(skipped_rois)

    ranking = compute_insert_ranking(all_rows)
    print(f"Insert ranking: {len(ranking)} insert labels ranked by mean HU.")

    combined_path = output_dir / "radiomics_combined.xlsx"
    series_dirs = {s["series_name"]: s["series_dir"] for s in series_list}
    write_combined_excel(
        all_rows, ranking, combined_path,
        series_dirs=series_dirs, dataset_root=dataset_root,
    )
    print("=" * 70)
    # Exit non-zero so an automated caller cannot mistake a partial run for a
    # complete one; the workbook is still written either way. The reason is
    # printed rather than left implicit in the number, because a bare 2 sends
    # whoever sees it back through a long log to work out what was lost.
    lost = _incompleteness_summary(failed_series, skipped_rois, _REJECTED_SERIES)
    if lost:
        print(f"INCOMPLETE RUN (exit 2): {lost}.")
        print("The workbook was written but is missing rows — see the blocks "
              "above for which and why.")
        _publish_result(combined_path)
        return 2
    _publish_result(combined_path)
    return 0


if __name__ == "__main__":
    # A PyInstaller-frozen exe re-executes itself to spawn each worker, so
    # without this the --jobs>1 ProcessPoolExecutor restarts the whole CLI
    # in every child instead of running the worker function.
    multiprocessing.freeze_support()
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    _close_frozen_console()
    sys.exit(rc)