"""Geometric metrics for physical-space label-map comparisons."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt


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


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    # border_value=1 avoids treating the FOV cut plane as a real surface.
    return np.logical_xor(
        mask,
        binary_erosion(mask, structure=np.ones((3, 3, 3)), border_value=1),
    )


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


def surface_metrics(
    reference: sitk.Image, candidate: sitk.Image, label: int
) -> tuple[float, float]:
    """Return HD95 and ASSD in millimetres."""
    ref, cand = _arrays(reference, candidate, label)
    ref_surface = _surface(ref)
    cand_surface = _surface(cand)
    if not ref_surface.any() or not cand_surface.any():
        return math.nan, math.nan

    sampling = tuple(reversed(reference.GetSpacing()))
    ref_distance = distance_transform_edt(~ref_surface, sampling=sampling)
    cand_distance = distance_transform_edt(~cand_surface, sampling=sampling)
    ref_to_cand = cand_distance[ref_surface]
    cand_to_ref = ref_distance[cand_surface]
    distances = np.concatenate([ref_to_cand, cand_to_ref])
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def compare_label(
    reference: sitk.Image,
    candidate: sitk.Image,
    label: int,
    reference_array: np.ndarray | None = None,
    candidate_array: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compare one label using a safe crop for the surface distance transform."""
    if reference_array is None or candidate_array is None:
        ref, cand = _arrays(reference, candidate, label)
    else:
        ref = reference_array == label
        cand = candidate_array == label
    occupied = np.argwhere(np.logical_or(ref, cand))
    if not occupied.size:
        hd95, assd = math.nan, math.nan
    else:
        low = np.maximum(occupied.min(axis=0) - 2, 0)
        high = np.minimum(occupied.max(axis=0) + 3, np.array(ref.shape))
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(low, high))
        ref_small = ref[slices]
        cand_small = cand[slices]
        ref_surface = _surface(ref_small)
        cand_surface = _surface(cand_small)
        if not ref_surface.any() or not cand_surface.any():
            hd95, assd = math.nan, math.nan
        else:
            sampling = tuple(reversed(reference.GetSpacing()))
            ref_distance = distance_transform_edt(~ref_surface, sampling=sampling)
            cand_distance = distance_transform_edt(~cand_surface, sampling=sampling)
            distances = np.concatenate(
                [cand_distance[ref_surface], ref_distance[cand_surface]]
            )
            hd95, assd = float(np.percentile(distances, 95)), float(np.mean(distances))
    total = int(ref.sum()) + int(cand.sum())
    return {
        "label": int(label),
        "dice": math.nan if total == 0 else 2.0 * float(np.logical_and(ref, cand).sum()) / total,
        "centroid_displacement_mm": centroid_displacement(reference, candidate, label),
        "hd95_mm": hd95,
        "assd_mm": assd,
        "reference_touches_border": touches_border(reference, label),
        "candidate_touches_border": touches_border(candidate, label),
    }


def compare_labels(
    reference: sitk.Image, candidate: sitk.Image, labels: list[int]
) -> list[dict[str, float | int]]:
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
