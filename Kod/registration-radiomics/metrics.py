"""Geometric metrics for physical-space label-map comparisons.

Surface metrics follow the Metrics Reloaded / DeepMind ``surface-distance``
conventions:
  * HD95         : robust (area-weighted) Hausdorff distance, 95th percentile
                   of both directed distances (max of the two).
  * ASSD         : average symmetric surface distance (mean of the two
                   area-weighted directed averages).
  * MSSD         : mean symmetric surface distance (pooled, unweighted).
  * NSD @ 1 mm   : normalised surface distance (surface Dice at 1 mm).
  * clDice       : centreline Dice from 3D skeletons (tubular structures).
  * Dice         : standard volumetric Dice (unchanged).
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.ndimage as _ndi
import SimpleITK as sitk

# surface-distance 0.1 still references the old scipy.ndimage.filters /
# ndimage.morphology aliases, removed in recent SciPy releases.
if not hasattr(_ndi, "filters"):
    _ndi.filters = _ndi
if not hasattr(_ndi, "morphology"):
    _ndi.morphology = _ndi

from skimage.morphology import skeletonize  # noqa: E402
from surface_distance import (  # noqa: E402
    compute_average_surface_distance,
    compute_robust_hausdorff,
    compute_surface_dice_at_tolerance,
    compute_surface_distances,
)

NSD_TOLERANCE_MM = 1.0


def _arrays(reference: sitk.Image, candidate: sitk.Image, label: int):
    ref = sitk.GetArrayFromImage(reference) == label
    cand = sitk.GetArrayFromImage(candidate) == label
    if ref.shape != cand.shape:
        raise ValueError(f"Label maps have different array shapes: {ref.shape} != {cand.shape}")
    return ref, cand


def dice(reference: sitk.Image, candidate: sitk.Image, label: int) -> float:
    ref, cand = _arrays(reference, candidate, label)
    ref_n = int(ref.sum())
    cand_n = int(cand.sum())
    if ref_n + cand_n == 0:
        return math.nan
    return 2.0 * float(np.logical_and(ref, cand).sum()) / (ref_n + cand_n)


def centroid(image: sitk.Image, label: int) -> tuple[float, float, float] | None:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(image == label, sitk.sitkUInt8))
    if not stats.HasLabel(1):
        return None
    return tuple(float(v) for v in stats.GetCentroid(1))


def centroid_displacement(
    reference: sitk.Image, candidate: sitk.Image, label: int
) -> float:
    first = centroid(reference, label)
    second = centroid(candidate, label)
    if first is None or second is None:
        return math.nan
    return float(np.linalg.norm(np.subtract(first, second)))


def touches_border(image: sitk.Image, label: int) -> bool:
    """True if the label touches any face of the image (FOV truncation)."""
    array = sitk.GetArrayViewFromImage(image) == label
    if not array.any():
        return False
    return bool(
        array[0].any() or array[-1].any()
        or array[:, 0].any() or array[:, -1].any()
        or array[:, :, 0].any() or array[:, :, -1].any()
    )


def _cldice(ref: np.ndarray, cand: np.ndarray) -> float:
    """Centreline Dice on the union bounding box."""
    if not ref.any() or not cand.any():
        return math.nan
    zs, ys, xs = np.nonzero(ref | cand)
    pad = 2
    z0, z1 = max(0, int(zs.min()) - pad), min(ref.shape[0], int(zs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(ref.shape[1], int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(ref.shape[2], int(xs.max()) + pad + 1)
    sub_ref = ref[z0:z1, y0:y1, x0:x1]
    sub_cand = cand[z0:z1, y0:y1, x0:x1]
    sk_ref = skeletonize(sub_ref)
    sk_cand = skeletonize(sub_cand)
    l_ref = int(sk_ref.sum())
    l_cand = int(sk_cand.sum())
    if l_ref == 0 or l_cand == 0:
        return math.nan
    tprec = float(np.logical_and(sk_ref, sub_cand).sum() / l_ref)
    tsens = float(np.logical_and(sk_cand, sub_ref).sum() / l_cand)
    denom = tprec + tsens
    return float(2.0 * tprec * tsens / denom) if denom else 0.0


def surface_metrics(
    reference: sitk.Image,
    candidate: sitk.Image,
    label: int,
    reference_array: np.ndarray | None = None,
    candidate_array: np.ndarray | None = None,
) -> dict[str, float]:
    """Metrics Reloaded-compatible surface metrics (mm / unitless)."""
    empty = {
        "hd95_mm": math.nan,
        "hd95_pooled_mm": math.nan,
        "assd_mm": math.nan,
        "mssd_mm": math.nan,
        "nsd_1mm": math.nan,
        "cldice": math.nan,
    }
    if reference_array is None or candidate_array is None:
        ref, cand = _arrays(reference, candidate, label)
    else:
        ref = reference_array == label
        cand = candidate_array == label
    if not ref.any() or not cand.any():
        return empty

    # surface-distance expects the spacing in numpy-array order (z, y, x).
    spacing_mm = tuple(reversed(reference.GetSpacing()))
    surface_distances = compute_surface_distances(ref, cand, spacing_mm)
    d_gt = surface_distances["distances_gt_to_pred"]
    d_pred = surface_distances["distances_pred_to_gt"]
    pooled = np.concatenate([d_gt, d_pred]) if (d_gt.size or d_pred.size) else np.array([])

    hd95 = float(compute_robust_hausdorff(surface_distances, 95))
    gt_mean, pred_mean = compute_average_surface_distance(surface_distances)
    assd = 0.5 * (float(gt_mean) + float(pred_mean))
    mssd = float(np.mean(pooled)) if pooled.size else math.nan
    hd95_pooled = float(np.percentile(pooled, 95)) if pooled.size else math.nan
    nsd = float(compute_surface_dice_at_tolerance(surface_distances, NSD_TOLERANCE_MM))

    return {
        "hd95_mm": hd95,
        "hd95_pooled_mm": hd95_pooled,
        "assd_mm": assd,
        "mssd_mm": mssd,
        "nsd_1mm": nsd,
        "cldice": _cldice(ref, cand),
    }


def compare_label(
    reference: sitk.Image,
    candidate: sitk.Image,
    label: int,
    reference_array: np.ndarray | None = None,
    candidate_array: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    if reference_array is None or candidate_array is None:
        ref, cand = _arrays(reference, candidate, label)
    else:
        ref = reference_array == label
        cand = candidate_array == label
    total = int(ref.sum()) + int(cand.sum())
    dice_value = (
        2.0 * float(np.logical_and(ref, cand).sum()) / total if total else math.nan
    )
    metrics = surface_metrics(reference, candidate, label, reference_array, candidate_array)
    return {
        "label": int(label),
        "dice": dice_value,
        "centroid_displacement_mm": centroid_displacement(reference, candidate, label),
        **metrics,
        "reference_touches_border": touches_border(reference, label),
        "candidate_touches_border": touches_border(candidate, label),
    }


def compare_labels(
    reference: sitk.Image, candidate: sitk.Image, labels: list[int]
) -> list[dict[str, float | int | bool]]:
    if not labels:
        return []
    reference_array = sitk.GetArrayFromImage(reference)
    candidate_array = sitk.GetArrayFromImage(candidate)
    workers = min(max(1, os.cpu_count() or 1), len(labels))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(
            lambda label: compare_label(
                reference, candidate, label, reference_array, candidate_array
            ),
            labels,
        ))
