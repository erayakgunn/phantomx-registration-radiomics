"""Insert detection using adaptive thresholding, circle fitting, and ring constraints."""

import math

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, replace
from scipy import optimize

from utils.logger import log
import config


@dataclass
class InsertCandidate:
    x: float
    y: float
    radius: float
    mean_hu: float
    angle: float  # Angle from phantom center in degrees
    ring: str  # "outer" or "inner"
    area: int  # Pixel area of the blob
    distance_cm: float  # Distance from phantom center in cm


def _adaptive_threshold(hu_slice: np.ndarray, phantom_cx: float, phantom_cy: float,
                         phantom_radius_px: float) -> np.ndarray:
    """Compute adaptive threshold for insert detection.

    Samples phantom interior background and sets threshold above noise level.
    Uses Otsu's method on phantom interior for robust bimodal separation.
    """
    h, w = hu_slice.shape

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - phantom_cx) ** 2 + (yy - phantom_cy) ** 2)
    mask = (dist > phantom_radius_px * 0.2) & (dist < phantom_radius_px * 0.8)

    bg_values = hu_slice[mask]
    if bg_values.size == 0:
        return (hu_slice > -500).astype(np.uint8) * 255

    # Use Otsu's method on the phantom interior for robust threshold
    # Scale to uint8 for Otsu
    vmin, vmax = float(np.percentile(bg_values, 1)), float(np.percentile(bg_values, 99))
    if vmax <= vmin:
        vmax = vmin + 1
    scaled = np.clip((hu_slice - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    # Apply mask
    masked_img = np.zeros_like(scaled)
    full_mask = dist < phantom_radius_px * 0.85
    masked_img[full_mask] = scaled[full_mask]

    otsu_val, binary = cv2.threshold(masked_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Zero out outside phantom
    binary[~full_mask] = 0

    return binary


def _fit_ring_radius(points_x: np.ndarray, points_y: np.ndarray,
                     center_x: float, center_y: float) -> float:
    """Fit a circle (centered at phantom center) to a set of points.

    Since the ring is concentric with the phantom, we only fit the radius.
    Returns the best-fit radius.
    """
    distances = np.sqrt((points_x - center_x) ** 2 + (points_y - center_y) ** 2)
    return float(np.median(distances))


def _snap_to_ring(candidate: InsertCandidate, ring_radius_px: float,
                  phantom_cx: float, phantom_cy: float) -> InsertCandidate:
    """Project insert center onto the fitted ring circle.

    Keeps the angle, adjusts the distance to exactly match the ring radius.
    """
    angle_rad = np.arctan2(candidate.y - phantom_cy, candidate.x - phantom_cx)
    new_x = phantom_cx + ring_radius_px * np.cos(angle_rad)
    new_y = phantom_cy + ring_radius_px * np.sin(angle_rad)

    return replace(
        candidate,
        x=float(new_x),
        y=float(new_y),
        angle=float(np.degrees(angle_rad)) % 360,
    )


def detect_inserts(
    hu_slice: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    pixel_spacing: Tuple[float, float],
    phantom_radius_px: float = None,
) -> Tuple[List[InsertCandidate], List[InsertCandidate]]:
    """Detect insert rods using HoughCircles + connected components fallback.

    Uses two detection methods for robustness:
    1. Primary: cv2.HoughCircles for geometric circle detection
    2. Fallback: connected components on adaptive threshold

    Then clusters into inner/outer rings and snaps to fitted ring circles.

    Returns:
        (outer_inserts, inner_inserts)
    """
    avg_spacing = (pixel_spacing[0] + pixel_spacing[1]) / 2.0

    outer_ring_px = config.OUTER_RING_DISTANCE_CM * 10.0 / avg_spacing
    inner_ring_px = config.INNER_RING_DISTANCE_CM * 10.0 / avg_spacing
    insert_r_px = config.INSERT_ROI_RADIUS_CM * 10.0 / avg_spacing

    # Tolerance band for ring membership. Must be smaller than the gap
    # between the two ring centers (~42 px) so the clustering step can
    # still separate inner from outer. The old `insert_r_px * 5` formula
    # silently coupled tolerance to insert size and broke when the
    # nominal diameter was corrected from 1.5 cm to 2.0 cm — keep the
    # tolerance bounded and clamp `min_dist_px` so cv2.circle can never
    # receive a negative radius.
    ring_tol_px = min(insert_r_px * 1.5, (outer_ring_px - inner_ring_px) * 0.45)
    min_dist_px = max(insert_r_px, inner_ring_px - ring_tol_px)
    max_dist_px = outer_ring_px + ring_tol_px

    if phantom_radius_px is None:
        phantom_radius_px = max_dist_px + ring_tol_px

    # --- Method 1: HoughCircles ---
    all_candidates = _detect_via_hough(
        hu_slice, phantom_cx, phantom_cy, phantom_radius_px,
        insert_r_px, min_dist_px, max_dist_px, avg_spacing
    )

    # --- Method 2: Connected components fallback ---
    if len(all_candidates) < 15:
        cc_candidates = _detect_via_cc(
            hu_slice, phantom_cx, phantom_cy, phantom_radius_px,
            insert_r_px, min_dist_px, max_dist_px, avg_spacing
        )
        # Merge: add CC candidates that aren't near existing Hough detections
        for cc in cc_candidates:
            is_duplicate = False
            for hc in all_candidates:
                d = np.sqrt((cc.x - hc.x) ** 2 + (cc.y - hc.y) ** 2)
                if d < insert_r_px * 2:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_candidates.append(cc)

    if len(all_candidates) < 5:
        log.warning(f"Insert detection found only {len(all_candidates)} candidates (need >= 5), returning empty.")
        return [], []

    # --- Cluster into 2 rings ---
    distances = np.array([
        np.sqrt((c.x - phantom_cx) ** 2 + (c.y - phantom_cy) ** 2)
        for c in all_candidates
    ])

    inner_center = inner_ring_px
    outer_center = outer_ring_px
    for _ in range(20):
        inner_mask = np.abs(distances - inner_center) < np.abs(distances - outer_center)
        outer_mask = ~inner_mask
        if inner_mask.sum() > 0:
            inner_center = np.mean(distances[inner_mask])
        if outer_mask.sum() > 0:
            outer_center = np.mean(distances[outer_mask])

    # Assign rings and fit radii
    outer_inserts = []
    inner_inserts = []

    for idx, c in enumerate(all_candidates):
        d = distances[idx]
        if abs(d - inner_center) < abs(d - outer_center):
            c.ring = "inner"
            inner_inserts.append(c)
        else:
            c.ring = "outer"
            outer_inserts.append(c)

    # Fit ring radii and snap
    if outer_inserts:
        outer_fit_r = float(np.median([
            np.sqrt((c.x - phantom_cx) ** 2 + (c.y - phantom_cy) ** 2)
            for c in outer_inserts
        ]))
        for idx_c, c in enumerate(outer_inserts):
            snapped = _snap_to_ring(c, outer_fit_r, phantom_cx, phantom_cy)
            outer_inserts[idx_c] = replace(snapped, distance_cm=outer_fit_r * avg_spacing / 10.0)

    if inner_inserts:
        inner_fit_r = float(np.median([
            np.sqrt((c.x - phantom_cx) ** 2 + (c.y - phantom_cy) ** 2)
            for c in inner_inserts
        ]))
        for idx_c, c in enumerate(inner_inserts):
            snapped = _snap_to_ring(c, inner_fit_r, phantom_cx, phantom_cy)
            inner_inserts[idx_c] = replace(snapped, distance_cm=inner_fit_r * avg_spacing / 10.0)

    return outer_inserts, inner_inserts


def _detect_via_hough(
    hu_slice: np.ndarray, phantom_cx: float, phantom_cy: float,
    phantom_radius_px: float, insert_r_px: float,
    min_dist_px: float, max_dist_px: float, avg_spacing: float,
) -> List[InsertCandidate]:
    """Detect insert circles using HoughCircles."""
    h, w = hu_slice.shape

    # Normalize HU to 8-bit for Hough
    # Use phantom interior range
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - phantom_cx) ** 2 + (yy - phantom_cy) ** 2)
    inside = dist < phantom_radius_px

    vals = hu_slice[inside]
    if vals.size == 0:
        return []

    vmin, vmax = np.percentile(vals, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1
    gray = np.clip((hu_slice - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    # Mask outside phantom
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(phantom_cx), int(phantom_cy)), int(phantom_radius_px - 3), 255, -1)
    center_exclude = int(min_dist_px * 0.6)
    cv2.circle(mask, (int(phantom_cx), int(phantom_cy)), center_exclude, 0, -1)
    gray = cv2.bitwise_and(gray, mask)

    # Apply slight blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)

    min_r = max(int(insert_r_px * 0.4), 3)
    max_r = int(insert_r_px * 2.5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=int(insert_r_px * 1.5),
        param1=80,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )

    candidates = []
    if circles is not None:
        for cx, cy, r in circles[0]:
            cx, cy, r = float(cx), float(cy), float(r)
            d = np.sqrt((cx - phantom_cx) ** 2 + (cy - phantom_cy) ** 2)
            if d < min_dist_px or d > max_dist_px:
                continue

            # Sample mean HU in the detected circle
            rr = max(int(r * 0.7), 2)
            y_lo = max(0, int(cy) - rr)
            y_hi = min(h, int(cy) + rr + 1)
            x_lo = max(0, int(cx) - rr)
            x_hi = min(w, int(cx) + rr + 1)
            patch = hu_slice[y_lo:y_hi, x_lo:x_hi]

            yp, xp = np.ogrid[:patch.shape[0], :patch.shape[1]]
            circ_mask = ((xp - (cx - x_lo)) ** 2 + (yp - (cy - y_lo)) ** 2) <= rr ** 2
            if circ_mask.any():
                mean_hu = float(np.mean(patch[circ_mask]))
            else:
                mean_hu = float(hu_slice[int(cy), int(cx)])

            angle = float(np.degrees(np.arctan2(cy - phantom_cy, cx - phantom_cx))) % 360
            area = int(np.pi * r * r)
            dist_cm = d * avg_spacing / 10.0

            # Filter: insert HU should be significantly above background
            bg_mask = (dist > phantom_radius_px * 0.2) & (dist < phantom_radius_px * 0.8)
            bg_median = np.median(hu_slice[bg_mask]) if bg_mask.any() else -100
            if mean_hu < bg_median + 30:
                continue

            candidates.append(InsertCandidate(
                x=cx, y=cy, radius=r, mean_hu=mean_hu,
                angle=angle, ring="", area=area, distance_cm=dist_cm,
            ))

    return candidates


def _detect_via_cc(
    hu_slice: np.ndarray, phantom_cx: float, phantom_cy: float,
    phantom_radius_px: float, insert_r_px: float,
    min_dist_px: float, max_dist_px: float, avg_spacing: float,
) -> List[InsertCandidate]:
    """Detect inserts via adaptive threshold + connected components (fallback)."""
    binary = _adaptive_threshold(hu_slice, phantom_cx, phantom_cy, phantom_radius_px)

    h, w = hu_slice.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(phantom_cx), int(phantom_cy)), int(phantom_radius_px - 5), 255, -1)
    center_exclude_px = int(min_dist_px * 0.7)
    cv2.circle(mask, (int(phantom_cx), int(phantom_cy)), center_exclude_px, 0, -1)
    binary = cv2.bitwise_and(binary, mask)

    expected_area = np.pi * insert_r_px ** 2
    min_area = expected_area * 0.2
    max_area = expected_area * 10.0

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []
    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        ccx, ccy = float(centroids[i][0]), float(centroids[i][1])
        dist_px = np.sqrt((ccx - phantom_cx) ** 2 + (ccy - phantom_cy) ** 2)

        if dist_px < min_dist_px or dist_px > max_dist_px:
            continue

        angle = float(np.degrees(np.arctan2(ccy - phantom_cy, ccx - phantom_cx))) % 360
        comp_mask = labels == i
        mean_hu = float(np.mean(hu_slice[comp_mask]))
        eff_radius = np.sqrt(area / np.pi)
        dist_cm = dist_px * avg_spacing / 10.0

        candidates.append(InsertCandidate(
            x=ccx, y=ccy, radius=eff_radius, mean_hu=mean_hu,
            angle=angle, ring="", area=area, distance_cm=dist_cm,
        ))

    return candidates


def _isolate_insert_mask(
    crop: np.ndarray,
    window_mask: np.ndarray,
    fallback_threshold_hu: float,
    seed_local_cx: Optional[float] = None,
    seed_local_cy: Optional[float] = None,
    insert_radius_px: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Adaptive insert isolation inside a search window.

    Strategy (preference order):
      1. **Area-based** (when ``insert_radius_px`` is given). For each
         Otsu class, find its largest connected component. Pick the
         class whose largest CC is *closest in area to the expected
         insert disk*. This is robust against:
           * the seed being on the background side of the threshold
             (seed-based would pick background, this still picks insert);
           * the insert being a small fraction of the window OR a large
             fraction (minority-class would invert in the second case).
      2. **Seed-based** (when area was inconclusive but seed coords
         given). Pick the class containing the seed's 3×3-median HU.
      3. **Minority-class** as a last resort.
      4. **Fixed >`fallback_threshold_hu`** when Otsu can't produce a
         usable bimodal split (windowed range too narrow).

    Returns a uint8 mask (255 = insert) inside the circular window, or
    ``None`` if no usable mask could be produced.
    """
    values = crop[window_mask]
    if values.size < 10:
        return None

    vmin, vmax = float(np.min(values)), float(np.max(values))
    if (vmax - vmin) < 30.0:
        chosen = (crop > fallback_threshold_hu) & window_mask
        if int(chosen.sum()) == 0:
            return None
        return chosen.astype(np.uint8) * 255

    scaled = np.clip(
        (values - vmin) / (vmax - vmin) * 255.0, 0, 255
    ).astype(np.uint8)
    otsu_val, _ = cv2.threshold(
        scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_hu = vmin + (float(otsu_val) / 255.0) * (vmax - vmin)

    above = (crop > threshold_hu) & window_mask
    below = (crop < threshold_hu) & window_mask
    n_above = int(above.sum())
    n_below = int(below.sum())

    if n_above == 0 or n_below == 0:
        chosen = (crop > fallback_threshold_hu) & window_mask
        if int(chosen.sum()) == 0:
            return None
        return chosen.astype(np.uint8) * 255

    chosen: Optional[np.ndarray] = None

    # Strategy 1: area-based class selection (most robust).
    if insert_radius_px is not None and insert_radius_px > 0:
        expected_area = math.pi * insert_radius_px ** 2

        def _largest_cc_area(mask: np.ndarray) -> int:
            if not mask.any():
                return 0
            nlab, _, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8) * 255, connectivity=8
            )
            if nlab <= 1:
                return 0
            return int(stats[1:, cv2.CC_STAT_AREA].max())

        def _area_score(area: int) -> float:
            if area <= 0:
                return 0.0
            return 1.0 / (1.0 + abs(area - expected_area) / expected_area)

        a_score = _area_score(_largest_cc_area(above))
        b_score = _area_score(_largest_cc_area(below))
        # Require a meaningful margin to commit; otherwise let seed-based
        # break the tie. 0.05 ≈ "the smaller match is at least 10 % less
        # similar to expected area than the larger".
        if abs(a_score - b_score) >= 0.05:
            chosen = above if a_score > b_score else below

    # Strategy 2: seed-based (fallback / tiebreaker).
    if chosen is None and seed_local_cx is not None and seed_local_cy is not None:
        sy = max(0, min(crop.shape[0] - 1, int(round(seed_local_cy))))
        sx = max(0, min(crop.shape[1] - 1, int(round(seed_local_cx))))
        y_lo = max(0, sy - 1); y_hi = min(crop.shape[0], sy + 2)
        x_lo = max(0, sx - 1); x_hi = min(crop.shape[1], sx + 2)
        seed_hu = float(np.median(crop[y_lo:y_hi, x_lo:x_hi]))
        chosen = above if seed_hu > threshold_hu else below

    # Strategy 3: minority class (last resort).
    if chosen is None:
        chosen = above if n_above < n_below else below

    if int(chosen.sum()) == 0:
        return None
    return chosen.astype(np.uint8) * 255


def _gross_localize_one_window(
    hu_slice: np.ndarray,
    expected_x: float,
    expected_y: float,
    pixel_spacing: Tuple[float, float],
    diam_cm: float,
    non_air_threshold_hu: float,
) -> Optional[Tuple[float, float]]:
    """Stage 1: find the approximate insert centre via Otsu + CC + centroid.

    Returns the blob's intensity-weighted centroid in image coordinates,
    or ``None`` if no usable blob is found in this window. Used by
    ``refine_insert_center`` to land near the insert before Stage 2 does
    the precise centring.

    Morphology order is OPEN-then-CLOSE (specks are removed BEFORE holes
    are filled), so a small noise speck near a real insert can't get
    welded onto the insert blob and drag its centroid.
    """
    h, w = hu_slice.shape
    avg_spacing_mm = (pixel_spacing[0] + pixel_spacing[1]) / 2.0
    insert_r_px_nominal = config.INSERT_ROI_RADIUS_CM * 10.0 / avg_spacing_mm
    min_blob_area_px = max(9, int(np.pi * insert_r_px_nominal ** 2 * 0.25))

    search_r_px = (diam_cm * 10.0 / 2.0) / avg_spacing_mm

    x0 = max(0, int(np.floor(expected_x - search_r_px)))
    y0 = max(0, int(np.floor(expected_y - search_r_px)))
    x1 = min(w, int(np.ceil(expected_x + search_r_px)) + 1)
    y1 = min(h, int(np.ceil(expected_y + search_r_px)) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    crop = hu_slice[y0:y1, x0:x1]
    ch, cw = crop.shape

    cx_local = expected_x - x0
    cy_local = expected_y - y0
    yy, xx = np.ogrid[:ch, :cw]
    dist = np.sqrt((xx - cx_local) ** 2 + (yy - cy_local) ** 2)
    window_mask = dist <= search_r_px

    binary = _isolate_insert_mask(
        crop, window_mask, non_air_threshold_hu,
        seed_local_cx=cx_local, seed_local_cy=cy_local,
        insert_radius_px=insert_r_px_nominal,
    )
    if binary is None:
        return None

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if nlabels <= 1:
        return None

    best_idx = None
    best_dist_local = float("inf")
    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] < min_blob_area_px:
            continue
        ccx, ccy = centroids[i]
        d = np.sqrt((ccx - cx_local) ** 2 + (ccy - cy_local) ** 2)
        if d < best_dist_local:
            best_dist_local = d
            best_idx = i
    if best_idx is None:
        return None

    return (
        float(centroids[best_idx][0]) + x0,
        float(centroids[best_idx][1]) + y0,
    )


def _precise_hough_center(
    hu_slice: np.ndarray,
    cx_approx: float,
    cy_approx: float,
    pixel_spacing: Tuple[float, float],
    diam_cm: float = 2.0,
    max_shift_cm: float = 0.5,
) -> Optional[Tuple[float, float]]:
    """Stage 2: precise insert centre via ``HoughCircles`` on the original
    HU slice (rescaled to uint8) inside a tight window around ``(cx_approx,
    cy_approx)``.

    Runs Hough on GRAYSCALE rather than on the binary blob from Stage 1:
    on grayscale, the rim gradient defines the circle directly, so the
    returned centre is the true geometric centre even when the Stage 1
    blob had a biased centroid (e.g. from window clipping or asymmetric
    PVE). Returns ``None`` if no circle is found or the shift from the
    Stage 1 anchor would exceed ``max_shift_cm`` (Hough probably locked
    onto noise — keep the Stage 1 centroid instead).
    """
    h, w = hu_slice.shape
    avg_spacing_mm = (pixel_spacing[0] + pixel_spacing[1]) / 2.0
    insert_r_px_nominal = config.INSERT_ROI_RADIUS_CM * 10.0 / avg_spacing_mm

    search_r_px = (diam_cm * 10.0 / 2.0) / avg_spacing_mm

    x0 = max(0, int(np.floor(cx_approx - search_r_px)))
    y0 = max(0, int(np.floor(cy_approx - search_r_px)))
    x1 = min(w, int(np.ceil(cx_approx + search_r_px)) + 1)
    y1 = min(h, int(np.ceil(cy_approx + search_r_px)) + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    crop = hu_slice[y0:y1, x0:x1]
    vmin, vmax = float(np.min(crop)), float(np.max(crop))
    if vmax - vmin < 30.0:
        return None  # No contrast — Hough will fail or hallucinate.

    gray = np.clip((crop - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    # Lighter Gaussian blur (σ=0.5) — heavier blur smeared the rim by ~1 px
    # and shifted Hough's detected centre for asymmetric PVE rims. 0.5 still
    # tames noise without dragging the centre.
    gray = cv2.GaussianBlur(gray, (5, 5), 0.5)

    min_r = max(2, int(round(insert_r_px_nominal * 0.6)))
    max_r = max(min_r + 1, int(round(insert_r_px_nominal * 1.5)))

    # Lower param1 (Canny upper threshold) from 80→50 so the rim edge is
    # still picked up on Philips low-contrast inserts (~200 HU rim gradient).
    # Lower param2 (accumulator threshold) from 15→12 so partial rims (e.g.
    # PVE-faded boundary slices) can still hit. The max-shift sanity guard
    # below keeps spurious circles from leaking through.
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(2, int(insert_r_px_nominal * 2)),
        param1=50,
        param2=12,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None or len(circles[0]) == 0:
        return None

    cx_local = cx_approx - x0
    cy_local = cy_approx - y0
    best = min(
        circles[0],
        key=lambda c: (c[0] - cx_local) ** 2 + (c[1] - cy_local) ** 2,
    )
    refined_cx = float(best[0]) + x0
    refined_cy = float(best[1]) + y0

    max_shift_px = (max_shift_cm * 10.0) / avg_spacing_mm
    if (refined_cx - cx_approx) ** 2 + (refined_cy - cy_approx) ** 2 \
            > max_shift_px * max_shift_px:
        # Hough locked onto something far from the gross anchor — distrust.
        return None

    return refined_cx, refined_cy


def refine_insert_center(
    hu_slice: np.ndarray,
    expected_x: float,
    expected_y: float,
    pixel_spacing: Tuple[float, float],
    search_diameters_cm: Tuple[float, ...] = None,
    non_air_threshold_hu: float = None,
) -> Optional[Tuple[float, float]]:
    """Three-stage per-slice insert refinement.

    **Stage 1 — gross localization** (``_gross_localize_one_window``):
        Otsu in a large search window → area-based insert-mask selection
        → morphology (OPEN then CLOSE) → connected component closest to
        seed → centroid. Tries each diameter in ``search_diameters_cm``
        in turn; returns the first successful window.

    **Stage 2 — precise centring via grayscale Hough**
    (``_precise_hough_center``):
        ``cv2.HoughCircles`` on the original HU slice (rescaled to uint8)
        inside a tight 2 cm window around the Stage 1 anchor. Hough on
        the grayscale rim gradient corrects the window-clipping bias
        that affects Stage 1's blob centroid when the seed sat off the
        insert.

    **Stage 3 — final centroid polish**:
        Re-run the centroid step in a tight 2 cm window centred on the
        Stage 2 anchor. With the insert now fully and symmetrically
        inside the window, the moment-based centroid is the exact
        geometric centre of the cleaned binary mask — knocks out the
        0.5–1 px residual error from Hough's accumulator quantisation.
        Bounded to a ≤3 mm shift from Stage 2 so a degenerate centroid
        (e.g. blob clipped at a section edge) can't drag the anchor.

    Returns the Stage 3 result when valid, else Stage 2, else Stage 1,
    else ``None``.
    """
    if search_diameters_cm is None:
        search_diameters_cm = config.INSERT_SEARCH_WINDOW_DIAMETERS_CM
    if non_air_threshold_hu is None:
        non_air_threshold_hu = config.INSERT_NON_AIR_THRESHOLD_HU

    approx_xy: Optional[Tuple[float, float]] = None
    for diam_cm in search_diameters_cm:
        approx_xy = _gross_localize_one_window(
            hu_slice, expected_x, expected_y, pixel_spacing,
            diam_cm, non_air_threshold_hu,
        )
        if approx_xy is not None:
            break

    if approx_xy is None:
        return None

    precise_xy = _precise_hough_center(
        hu_slice, approx_xy[0], approx_xy[1], pixel_spacing,
        diam_cm=2.0,
    )
    anchor = precise_xy if precise_xy is not None else approx_xy

    # Stage 3 polish — moment centroid in a tight, well-centred window.
    # Iterated until the centroid stops moving (≤0.05 px between consecutive
    # iterations) or up to 3 passes. Each iteration's centroid must be ≤3 mm
    # from the previous estimate (sanity guard against pathological drift).
    avg_spacing_mm = (pixel_spacing[0] + pixel_spacing[1]) / 2.0
    max_shift_px = (0.3 * 10.0) / avg_spacing_mm  # 3 mm cap per iteration
    convergence_tol_px_sq = 0.05 * 0.05  # 0.05 px convergence threshold
    current = anchor
    for _ in range(3):
        polished_xy = _gross_localize_one_window(
            hu_slice, current[0], current[1], pixel_spacing,
            2.0, non_air_threshold_hu,
        )
        if polished_xy is None:
            return current
        dx = polished_xy[0] - current[0]
        dy = polished_xy[1] - current[1]
        shift_sq = dx * dx + dy * dy
        if shift_sq > max_shift_px * max_shift_px:
            # Reject pathological drift — keep last good estimate.
            return current
        current = polished_xy
        if shift_sq <= convergence_tol_px_sq:
            break
    return current


def count_blobs(
    hu_slice: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    phantom_radius_px: float,
) -> int:
    """Count distinct solid blobs inside the phantom.

    INSERT sections have ~20 blobs, STAR sections have ~5.
    """
    binary = _adaptive_threshold(hu_slice, phantom_cx, phantom_cy, phantom_radius_px)

    h, w = hu_slice.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(phantom_cx), int(phantom_cy)), int(phantom_radius_px - 5), 255, -1)
    binary = cv2.bitwise_and(binary, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Scale min area by phantom size: insert radius ~ 0.08 * phantom radius
    insert_r_est = phantom_radius_px * (config.INSERT_ROI_RADIUS_CM / config.PHANTOM_OUTER_RADIUS_CM)
    min_area = max(20, int(np.pi * insert_r_est ** 2 * 0.1))
    count = 0
    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            count += 1
    return count


def classify_slice_type(
    hu_slice: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    pixel_spacing: Tuple[float, float],
    phantom_radius_px: float,
) -> str:
    """Classify a slice as INSERT, STAR, or UNKNOWN by blob count.

    A real STAR slice carries 4 texture squares + 1 central star = ~5 blobs.
    Slices with 0-2 blobs are empty / uniformity / LCD modules and must
    be labelled UNKNOWN so the section_classifier can keep them OUT of
    STAR runs. Without this floor, the old ``n_blobs <= 8`` branch swept
    Siemens PCCT LCD slices (only 1-2 dark spheres visible) into STAR
    groups, which then straddled multiple unrelated phantom modules.
    """
    n_blobs = count_blobs(hu_slice, phantom_cx, phantom_cy, phantom_radius_px)

    if n_blobs >= 12:
        return "INSERT"
    elif 3 <= n_blobs <= 8:
        return "STAR"
    else:
        # 0-2: empty/LCD/uniformity; 9-11: ambiguous transition
        return "UNKNOWN"
