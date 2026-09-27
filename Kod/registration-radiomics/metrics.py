"""Geometric metrics for physical-space label-map comparisons."""

from __future__ import annotations

import math

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
    return np.logical_xor(mask, binary_erosion(mask, structure=np.ones((3, 3, 3))))


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


def compare_labels(
    reference: sitk.Image, candidate: sitk.Image, labels: list[int]
) -> list[dict[str, float | int]]:
    rows = []
    for label in labels:
        hd95, assd = surface_metrics(reference, candidate, label)
        rows.append(
            {
                "label": int(label),
                "dice": dice(reference, candidate, label),
                "centroid_displacement_mm": centroid_displacement(
                    reference, candidate, label
                ),
                "hd95_mm": hd95,
                "assd_mm": assd,
            }
        )
    return rows
