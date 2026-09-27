"""Phase 3: Global rotation angle detection using texture landmarks."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from core.dicom_loader import DicomVolume
from core.phantom_detector import PhantomDetectionResult, SliceStatus
from core.section_classifier import SectionClassificationResult
from analysis.insert_detector import _adaptive_threshold
from utils.logger import log
import config


@dataclass
class RotationResult:
    global_offset_deg: float
    per_slice_offsets: Dict[int, float]
    confidence: float


def detect_rotation(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    sections: SectionClassificationResult,
) -> RotationResult:
    """Detect global rotation offset by locating texture blobs in STAR sections."""
    log.info("Phase 3: Detecting rotation...")

    texture_dist_px = dv.cm_to_pixels(config.TEXTURE_CENTER_DISTANCE_CM)
    texture_half_px = dv.cm_to_pixels(config.TEXTURE_SIDE_CM / 2.0)

    offsets = []
    per_slice = {}

    for sec in sections.sections:
        if sec.name not in ("STAR8", "STAR16"):
            continue

        for i in range(sec.start_idx, sec.end_idx):
            if sections.slice_labels[i] == "TRANSITION":
                continue
            circ = phantom.circles[i]
            if circ.status != SliceStatus.VALID:
                continue

            offset = _measure_offset(
                dv.volume[i], circ.center_x, circ.center_y, circ.radius,
                texture_dist_px, texture_half_px
            )
            if offset is not None:
                offsets.append(offset)
                per_slice[i] = offset

    if not offsets:
        log.warning("No texture landmarks found, assuming 0 deg rotation")
        return RotationResult(0.0, {}, 0.0)

    # Circular mean
    angles_rad = np.deg2rad(offsets)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    global_offset = float(np.degrees(np.arctan2(mean_sin, mean_cos)))
    confidence = float(np.sqrt(mean_sin ** 2 + mean_cos ** 2))

    log.info(f"Rotation offset: {global_offset:.2f} deg "
             f"({len(offsets)} measurements, confidence={confidence:.3f})")

    return RotationResult(global_offset, per_slice, confidence)


def _measure_offset(
    hu_slice: np.ndarray,
    cx: float, cy: float,
    phantom_r_px: float,
    texture_dist_px: float,
    texture_half_px: float,
) -> Optional[float]:
    """Find 4 texture blobs and compute rotation offset from expected angles."""
    h, w = hu_slice.shape

    binary = _adaptive_threshold(hu_slice, cx, cy, phantom_r_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(phantom_r_px - 5), 255, -1)
    cv2.circle(mask, (int(cx), int(cy)), int(texture_dist_px * 0.3), 0, -1)
    binary = cv2.bitwise_and(binary, mask)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    min_area = texture_half_px * texture_half_px
    detected_angles = []

    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        ccx, ccy = centroids[i]
        dist = np.sqrt((ccx - cx) ** 2 + (ccy - cy) ** 2)
        if abs(dist - texture_dist_px) < texture_half_px * 2.5:
            angle = np.degrees(np.arctan2(ccy - cy, ccx - cx)) % 360
            detected_angles.append(angle)

    if len(detected_angles) < 3:
        return None

    detected_angles.sort()
    expected = np.array([0.0, 90.0, 180.0, 270.0])

    # Match each detected angle to its nearest expected angle
    # and compute the average offset
    offsets = []
    for det in detected_angles:
        # Find nearest expected angle (handling wraparound)
        diffs = ((det - expected + 180) % 360) - 180
        best_idx = np.argmin(np.abs(diffs))
        offsets.append(diffs[best_idx])

    if not offsets:
        return None

    # Use circular mean of the individual offsets
    rads = np.deg2rad(offsets)
    mean_offset = np.degrees(np.arctan2(np.mean(np.sin(rads)), np.mean(np.cos(rads))))

    return float(mean_offset)
