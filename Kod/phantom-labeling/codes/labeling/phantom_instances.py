"""Multi-phantom instance splitting for long acquisitions.

Some sites scan SEVERAL physical CalibraMics phantom assemblies end-to-end
in one long spiral acquisition (observed at Acibadem Kent on a Siemens
SOMATOM Force: three complete phantoms — each with 2 star + 2 insert
modules — separated by air gaps, followed by half a metre of bare couch).

The annotation pipeline assumes exactly ONE phantom per volume (one
STAR16, one STAR8, two INSERT modules), so such scans must be split into
per-instance sub-series BEFORE annotation. This module finds the z-ranges
of the individual phantom instances so the labeler can materialize each
as its own ``<name>_P<k>/dicoms`` folder.

Detection is deliberately image-based and cheap: a subsample of slices is
tested for "a hollow phantom-sized disc is present here". Air gaps have no
content at all; the patient couch produces content whose enclosing circle
is far larger than the 19 cm phantom; solid table cross-sections fail the
hollowness test. Everything runs on ~200 sampled slices per series, so the
cost is a few seconds.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pydicom

from utils.logger import log


# A slice "contains a phantom" when the largest content contour's
# min-enclosing circle has a radius in this physical range (the CalibraMics
# shell is 9.5 cm; allow generous margins for FOV/centering differences).
_PHANTOM_R_MIN_CM = 6.5
_PHANTOM_R_MAX_CM = 12.5

# Minimum >-500 HU pixels for a slice to count as "has content" at all.
_MIN_CONTENT_PX = 1500

# Instance geometry (mm): bridge phantom-negative gaps up to _BRIDGE_MM
# inside one instance (a rare fluke sample), require at least _MIN_SPAN_MM
# of phantom signal for a real instance (one CalibraMics assembly is
# ~200 mm), and pad each instance's kept z-range by up to _EDGE_PAD_MM
# into the neighbouring gap so boundary/PVE slices stay with their
# instance. _BRIDGE_MM must stay clearly BELOW the physical air gap
# between two stacked phantoms (~24 mm on the Acibadem Force scan) or
# adjacent instances would merge back into one.
_BRIDGE_MM = 12.0
_MIN_SPAN_MM = 100.0
_EDGE_PAD_MM = 10.0

# Sampling: test roughly this many slices per series regardless of length.
_TARGET_SAMPLES = 220


def _slice_has_phantom(hu: np.ndarray, px_per_cm: float) -> bool:
    """True when this HU slice contains the phantom's circular shell.

    Criterion: a Hough circle of phantom radius exists in the slice.
    This is robust against everything simpler geometry tests tripped
    over on real Force data: the phantom cradle hanging below the shell
    (which inflates any point-set enclosing circle), the thin shell
    fragmenting into arc contours on star-module slices (which breaks
    largest-contour tests), and solid spacer modules (whose outer edge is
    still the phantom circle). Couch pads and table slabs contain no
    phantom-sized circle; air-gap slices are rejected by the cheap
    content pre-check before Hough runs.
    """
    mask = hu > -500
    if int(mask.sum()) < _MIN_CONTENT_PX:
        return False  # air gap / empty slice

    img8 = np.clip((hu + 1000.0) / 1400.0 * 255.0, 0, 255).astype(np.uint8)
    img8 = cv2.medianBlur(img8, 5)
    circles = cv2.HoughCircles(
        img8, cv2.HOUGH_GRADIENT,
        dp=1.5, minDist=200, param1=80, param2=50,
        minRadius=int(_PHANTOM_R_MIN_CM * px_per_cm),
        maxRadius=int(_PHANTOM_R_MAX_CM * px_per_cm),
    )
    return circles is not None


def find_phantom_instances(
    dcm_paths: Sequence[str],
) -> Optional[List[Tuple[float, float]]]:
    """Return the phantom instances' z-ranges, or ``None`` on failure.

    Reads every file's SliceLocation (header-only) plus the pixel data of
    ~:data:`_TARGET_SAMPLES` evenly spaced slices. Returns a list of
    ``(z_lo, z_hi)`` tuples sorted ascending; a normal single-phantom
    series yields one tuple. ``None`` means the series could not be
    analysed (missing SliceLocations etc.) — callers must fall back to
    the regular no-split behaviour.
    """
    zs: List[Tuple[float, str]] = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(
                p, stop_before_pixels=True,
                specific_tags=["SliceLocation"],
            )
            loc = ds.get("SliceLocation", None)
            if loc is not None:
                zs.append((float(loc), p))
        except Exception:
            continue
    if len(zs) < 20:
        return None
    zs.sort(key=lambda t: t[0])

    step = max(1, len(zs) // _TARGET_SAMPLES)
    samples: List[Tuple[float, bool]] = []
    px_per_cm = None
    for z, p in zs[::step]:
        try:
            ds = pydicom.dcmread(p)
            if px_per_cm is None:
                spacing = ds.get("PixelSpacing", [1.0, 1.0])
                px_per_cm = 10.0 / float(spacing[0])
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            hu = ds.pixel_array.astype(np.float32) * slope + inter
            samples.append((z, _slice_has_phantom(hu, px_per_cm)))
        except Exception:
            continue
    if len(samples) < 10:
        return None

    # Group phantom-positive samples into z-runs, bridging short negative
    # gaps (module joints) and dropping short runs (stray couch hits).
    runs: List[List[float]] = []  # [z_start, z_end]
    for z, ok in samples:
        if not ok:
            continue
        if runs and (z - runs[-1][1]) <= _BRIDGE_MM:
            runs[-1][1] = z
        else:
            runs.append([z, z])
    runs = [r for r in runs if (r[1] - r[0]) >= _MIN_SPAN_MM]
    if not runs:
        return None

    # Pad each run into its neighbouring gaps (half the gap, capped).
    padded: List[Tuple[float, float]] = []
    for i, (lo, hi) in enumerate(runs):
        gap_lo = (runs[i - 1][1] if i > 0 else lo - 2 * _EDGE_PAD_MM)
        gap_hi = (runs[i + 1][0] if i + 1 < len(runs) else hi + 2 * _EDGE_PAD_MM)
        pad_lo = min(_EDGE_PAD_MM, max(0.0, (lo - gap_lo) / 2.0))
        pad_hi = min(_EDGE_PAD_MM, max(0.0, (gap_hi - hi) / 2.0))
        padded.append((lo - pad_lo, hi + pad_hi))
    return padded


def partition_files(
    dcm_paths: Sequence[str],
    instances: Sequence[Tuple[float, float]],
) -> List[List[str]]:
    """Assign each file to the instance whose z-range contains it.

    Files outside every instance (air gaps, bare couch) are dropped —
    they carry no phantom content. Files without a SliceLocation are
    dropped as well (topograms never reach this code path).
    """
    parts: List[List[str]] = [[] for _ in instances]
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(
                p, stop_before_pixels=True, specific_tags=["SliceLocation"],
            )
            loc = ds.get("SliceLocation", None)
            if loc is None:
                continue
            z = float(loc)
        except Exception:
            continue
        for k, (lo, hi) in enumerate(instances):
            if lo <= z <= hi:
                parts[k].append(p)
                break
    return parts


def split_series_into_instances(
    series_path: str,
    dcm_paths: Sequence[str],
) -> Optional[List[List[str]]]:
    """Full splitting pipeline for one series directory.

    Returns per-instance file lists (ascending z-order of instances) when
    MORE than one phantom instance is present; ``None`` when the series is
    a normal single-phantom scan (or could not be analysed) and the caller
    should keep the regular whole-folder behaviour.
    """
    instances = find_phantom_instances(dcm_paths)
    if not instances or len(instances) < 2:
        return None
    parts = partition_files(dcm_paths, instances)
    kept = [p for p in parts if len(p) >= 20]
    if len(kept) < 2:
        return None
    n_dropped = len(dcm_paths) - sum(len(p) for p in kept)
    log.info(
        f"  Multi-phantom series: {len(kept)} phantom instances at "
        f"z-ranges {[(round(a), round(b)) for a, b in instances]} "
        f"({n_dropped} non-phantom slices dropped: air gaps / couch)"
    )
    return kept
