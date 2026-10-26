"""Phase 1: Phantom localization via robust geometric edge voting."""

import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum

from core.dicom_loader import DicomVolume
from utils.logger import log
import config


class SliceStatus(Enum):
    VALID = "VALID"
    TRANSITION = "TRANSITION"
    INVALID = "INVALID"


@dataclass
class PhantomCircle:
    center_x: float
    center_y: float
    radius: float
    confidence: float
    status: SliceStatus


@dataclass
class PhantomDetectionResult:
    circles: List[PhantomCircle]  # One per slice
    global_center_x: float
    global_center_y: float
    global_radius: float


def _robust_circle_fit(cnt: np.ndarray):
    """Fit (x, y, r) to a contour robustly against a few outlier points.

    ``cv2.minEnclosingCircle`` is dictated entirely by whichever single
    contour point sits farthest from the center — on this pipeline's
    thin-shell phantoms, small bridged artifacts (a mounting-tab "ear", or
    a nearby insert dot fused on by the morphological CLOSE) reliably
    pulled the reported radius ~8-9% too large even though the contour's
    bulk still traced the true wall. Centroid + median distance-to-edge
    uses every contour point and is insensitive to a handful of outliers,
    matching a direct all-angle radial measurement of the true wall to
    within ~1 px (vs. ~14 px over-estimate from minEnclosingCircle).
    """
    moments = cv2.moments(cnt)
    if moments["m00"] == 0:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        return x, y, r
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    pts = cnt.reshape(-1, 2).astype(np.float64)
    r = float(np.median(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)))
    return float(cx), float(cy), r


def _largest_valid_circle(
    binary_mask: np.ndarray, expected_r_px: float, expected_area: float
):
    """Return the (x, y, r) of the best phantom-sized circular contour in
    ``binary_mask``, or ``None`` if no contour clears the area / aspect /
    circularity / radius gates.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circle = None
    best_score = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if expected_area * 0.60 < area < expected_area * 1.40:
            # Aspect Ratio check
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            aspect_ratio = float(w_b) / float(h_b) if h_b > 0 else 0
            if not (0.82 <= aspect_ratio <= 1.18):
                continue

            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.55:
                continue

            x, y, r = _robust_circle_fit(cnt)
            r_diff = abs(r - expected_r_px)
            score = r_diff + (1.0 - circularity) * 40

            if score < best_score and r_diff < (expected_r_px * 0.25):
                best_score = score
                best_circle = (x, y, r)

    return best_circle


def _detect_center_by_feature_cluster(
    hu_slice: np.ndarray, expected_r_px: float, pixel_spacing: Tuple[float, float]
):
    """Wall-independent per-slice phantom centre from the cluster of interior
    calibration features (texture squares / insert dots), or ``None``.

    The thin air-cavity shell vanishes below the air threshold on soft
    kernels (Br36f) and coarse pixels (large FOV), so the contour detector
    finds nothing. But the texture squares and insert dots are large,
    high-contrast features that survive any kernel/FOV. Their concentric
    arrangement pins the centre even with no detectable wall. Radius is NOT
    estimated here — the phantom is physically 19 cm, so the caller uses the
    fixed ``expected_r_px``. Noisy per-slice (that's fine: the z-track
    smoother regularises it); the validation gates below keep it from
    locking onto the body phantom (whose organ blobs are filtered by size
    and don't form a concentric ring).
    """
    avg_sp = (pixel_spacing[0] + pixel_spacing[1]) / 2.0
    insert_dot_area = np.pi * (config.INSERT_ROI_RADIUS_CM * 10.0 / avg_sp) ** 2
    tex_area = (config.TEXTURE_SIDE_CM * 10.0 / avg_sp) ** 2

    smoothed = cv2.GaussianBlur(hu_slice.astype(np.float32), (5, 5), 1.0)
    mat = (smoothed > config.AIR_TISSUE_THRESHOLD_HU).astype(np.uint8)
    n_lab, _lab, stats, cent = cv2.connectedComponentsWithStats(mat, connectivity=8)

    pts = []
    for i in range(1, n_lab):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < insert_dot_area * 0.15:      # noise speck
            continue
        if a > tex_area * 2.5:              # body-organ / table fill
            continue
        pts.append((cent[i][0], cent[i][1]))
    if len(pts) < 5:
        return None

    P = np.asarray(pts, dtype=np.float64)
    cx, cy = float(np.median(P[:, 0])), float(np.median(P[:, 1]))
    # Iterate a trimmed centroid: keep blobs inside the phantom footprint,
    # recompute centre. Converges onto the feature ring's centre.
    for _ in range(5):
        rad = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
        keep = rad <= expected_r_px * 1.05
        if int(keep.sum()) < 5:
            break
        cx, cy = float(P[keep, 0].mean()), float(P[keep, 1].mean())

    rad = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    keep = rad <= expected_r_px * 1.05
    Pk = P[keep]
    if len(Pk) < 5:
        return None
    # Must look like a RING, not a clump: features spread out radially and
    # wrap around the centre (a body's vessels clump on one side).
    if float(np.median(np.hypot(Pk[:, 0] - cx, Pk[:, 1] - cy))) < expected_r_px * 0.25:
        return None
    ang = np.arctan2(Pk[:, 1] - cy, Pk[:, 0] - cx)
    quadrants = set(((ang + np.pi) // (np.pi / 2)).astype(int).tolist())
    if len(quadrants) < 3:
        return None
    return (cx, cy)


def _detect_circle_hough(hu_slice: np.ndarray, expected_r_px: float):
    """Detect the phantom wall as a Hough circle, or ``None``.

    This is the fallback for recons where the thin-shell CONTOUR detector
    fails but the wall is still a clean circular EDGE — notably Philips IMR
    (KSW): its smooth iterative reconstruction renders a continuous, low-
    contrast air→plastic wall whose contour fails the area / circularity /
    aspect gates (the air-cavity interior means the wall never closes into a
    phantom-sized filled blob), yet ``HoughCircles`` (Canny-gradient based)
    locks onto that circular edge reliably. Verified to also AGREE with the
    contour result on PCCT and with the feature-cluster result on soft-kernel
    SEP, so it is a safe fallback to try BEFORE the (centroid-biased) cluster.

    The window maps [-1000, +400] HU → [0, 255] so the air→wall transition is
    the dominant gradient. The radius is gated to ±15 % of the physical
    expected radius, and ``minDist`` is set larger than the image so only the
    single strongest phantom-sized circle is returned.
    """
    norm = np.clip((hu_slice.astype(np.float32) + 1000.0) / 1400.0 * 255.0,
                   0, 255).astype(np.uint8)
    min_r = int(expected_r_px * 0.85)
    max_r = int(expected_r_px * 1.15)
    circles = cv2.HoughCircles(
        norm, cv2.HOUGH_GRADIENT,
        dp=config.HOUGH_DP,
        minDist=float(max(hu_slice.shape)),
        param1=config.HOUGH_PARAM1,
        param2=config.HOUGH_PARAM2,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None or len(circles[0]) == 0:
        return None
    # Of the returned candidates (radius already gated), take the one whose
    # radius is closest to the physical expectation — guards against a rare
    # strong spurious circle near the gate edge.
    best = min(circles[0], key=lambda c: abs(float(c[2]) - expected_r_px))
    return float(best[0]), float(best[1]), float(best[2])


def _interior_low_hu_fraction(
    hu_slice: np.ndarray, cx: float, cy: float, r_px: float
) -> float:
    """Fraction of the inner ``PHANTOM_INTERIOR_SAMPLE_FRACTION`` · r disc
    below ``PHANTOM_INTERIOR_LOW_HU_MAX``. High (~0.8-0.96) for the
    air-cavity calibration phantom, low (~0.05) for a solid body section.
    Used to keep body slices out of the boundary track.
    """
    h, w = hu_slice.shape
    sample_r = r_px * config.PHANTOM_INTERIOR_SAMPLE_FRACTION
    yy, xx = np.ogrid[:h, :w]
    interior = (xx - cx) ** 2 + (yy - cy) ** 2 < sample_r ** 2
    if not interior.any():
        return 0.0
    return float(np.mean(hu_slice[interior] < config.PHANTOM_INTERIOR_LOW_HU_MAX))


def _build_smoothed_track(
    raw_cx: np.ndarray,
    raw_cy: np.ndarray,
    raw_r: np.ndarray,
    raw_is_shell: np.ndarray,
    dv: DicomVolume,
    expected_r_px: float,
):
    """Turn per-slice raw detections (NaN where none) into a smooth,
    outlier-rejected, gap-filled (cx, cy, r) track — one entry per slice.

    The radius is held constant at the series median (the phantom diameter
    is fixed by physics; per-slice radius wobble is pure noise). Returns
    ``(track_cx, track_cy, track_r, n_fit, used_fallback)``.
    """
    n = dv.num_slices
    valid_idx = np.where(~np.isnan(raw_cx))[0]

    min_required = max(
        config.PHANTOM_MIN_VALID_SLICES_FOR_TRUST,
        config.PHANTOM_MIN_VALID_FRACTION_FOR_TRUST * n,
    )
    if len(valid_idx) < min_required:
        track_cx = np.full(n, dv.width / 2.0)
        track_cy = np.full(n, dv.height / 2.0)
        track_r = np.full(n, float(expected_r_px))
        return track_cx, track_cy, track_r, 0, True

    # PREFER shell detections for the fit. The thin-shell contour is geometric
    # and unbiased; the feature-cluster fallback is centroid-based and can
    # carry a small section-dependent bias (asymmetric feature blobs), which
    # on a cluster-only fit shows up as a spurious centre offset. So whenever
    # enough shell points exist (≥3 for a robust line), fit from shell ONLY
    # and ignore the cluster points; fall back to cluster only when the shell
    # is genuinely absent (soft kernel / large FOV).
    shell_idx = valid_idx[raw_is_shell[valid_idx]]
    fit_idx = shell_idx if len(shell_idx) >= 3 else valid_idx

    med_cx = float(np.median(raw_cx[fit_idx]))
    med_cy = float(np.median(raw_cy[fit_idx]))
    med_r = float(np.median(raw_r[fit_idx]))

    # Drop gross outliers before fitting.
    d = np.hypot(raw_cx[fit_idx] - med_cx, raw_cy[fit_idx] - med_cy)
    inliers = fit_idx[d <= config.PHANTOM_TRACK_OUTLIER_TOL_PX]
    if len(inliers) < 3:
        inliers = fit_idx

    # Robust LINEAR drift model (Theil-Sen) per axis: phantom centre drift
    # along z is a slow, near-monotonic couch/gantry effect, so a robust line
    # captures the real drift while rejecting per-slice wobble (a running
    # median would chase the wobble and re-inject the jitter that broke
    # STAR16/STAR8). With no real drift it's a flat line ≈ the median, so
    # well-detected sharp-kernel series behave like the old global anchor.
    #
    # Physical cap: real couch/gantry drift is ≤ ~10 px over a full scan, so
    # ANY fitted line whose realised span across the series exceeds 1.5× that
    # is the detector inventing a trend — discard the slope and hold a flat
    # median. Keeps cluster-dominated series pinned to a stable centre.
    drift_cap = config.MAX_CENTER_DRIFT_PX * 1.5
    idx_all = np.arange(n, dtype=float)

    def _robust_line(zs, vs, med):
        if len(zs) >= 3 and np.ptp(zs) > 0:
            try:
                from scipy.stats import theilslopes
                slope, intercept, _lo, _hi = theilslopes(vs, zs)
                pred = np.clip(intercept + slope * idx_all,
                               med - drift_cap, med + drift_cap)
                if float(pred.max() - pred.min()) <= drift_cap:
                    return pred
            except Exception:
                pass
        return np.full(n, med)

    track_cx = _robust_line(inliers.astype(float), raw_cx[inliers], med_cx)
    track_cy = _robust_line(inliers.astype(float), raw_cy[inliers], med_cy)
    track_r = np.full(n, med_r)
    return track_cx, track_cy, track_r, len(inliers), False


def detect_phantom(dv: DicomVolume) -> PhantomDetectionResult:
    """Localize the phantom boundary PER SLICE and emit a smoothed track.

    Each slice is localized independently — primary: thin-shell contour
    (accurate on sharp kernels / small FOV); fallback: wall-independent
    interior-feature cluster (works on soft kernels / large FOV where the
    shell is sub-pixel). Solid body-phantom slices are kept out of the
    track via the air-cavity check. The raw per-slice detections are then
    smoothed into a (cx, cy, r) track (see ``_build_smoothed_track``) so
    every slice carries a data-driven boundary that follows real drift
    without injecting per-slice detection jitter into the downstream
    classification / extraction.
    """
    log.info("Phase 1: Detecting phantom boundary per slice (shell + feature-cluster)...")

    expected_r_px = dv.cm_to_pixels(config.PHANTOM_OUTER_RADIUS_CM)
    expected_area = np.pi * (expected_r_px ** 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    n = dv.num_slices
    raw_cx = np.full(n, np.nan)
    raw_cy = np.full(n, np.nan)
    raw_r = np.full(n, np.nan)
    raw_is_shell = np.zeros(n, dtype=bool)
    conf = np.zeros(n)
    n_shell = 0
    n_cluster = 0

    for i in range(n):
        hu_slice = dv.volume[i]

        # --- Primary: thin-shell contour ---
        smoothed = cv2.GaussianBlur(hu_slice.astype(np.float32), (5, 5), 1.5)
        raw_mask = (smoothed > config.AIR_TISSUE_THRESHOLD_HU).astype(np.uint8) * 255
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        best = _largest_valid_circle(cleaned, expected_r_px, expected_area)
        if best is None:
            close_only = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
            best = _largest_valid_circle(close_only, expected_r_px, expected_area)

        if best is not None:
            cx, cy, r = best
            c = 1.0
            is_shell = True
        else:
            # --- Fallback 1: Hough circle on the wall edge ---
            # Catches recons where the thin-shell contour fails but the wall is
            # still a clean circular edge (Philips IMR / KSW). It finds the
            # actual geometric wall, so it's trusted like a shell fit (and far
            # more accurate than the centroid-biased cluster below — on KSW the
            # cluster put the centre ~12 px high, throwing every insert label
            # off its rod, whereas Hough lands on the true centre).
            hough = _detect_circle_hough(hu_slice, expected_r_px)
            if hough is not None:
                cx, cy, r = hough
                c = 0.9
                is_shell = True
            else:
                # --- Fallback 2: wall-independent feature cluster ---
                cl = _detect_center_by_feature_cluster(
                    hu_slice, expected_r_px, dv.pixel_spacing
                )
                if cl is None:
                    continue
                cx, cy = cl
                r = float(expected_r_px)
                c = 0.6
                is_shell = False

        # Body-phantom guard: only air-cavity slices feed the boundary track,
        # so the solid body (whose centroid is near the phantom's) can't bias
        # the centre or radius. Body slices are localized later anyway via the
        # interpolated track + rejected by the Phase-2 air-cavity gate.
        if _interior_low_hu_fraction(hu_slice, cx, cy, r) < config.PHANTOM_MIN_LOW_HU_FRACTION:
            continue

        raw_cx[i] = cx
        raw_cy[i] = cy
        raw_r[i] = r
        raw_is_shell[i] = is_shell
        conf[i] = c
        if is_shell:
            n_shell += 1
        else:
            n_cluster += 1

    track_cx, track_cy, track_r, n_inliers, used_fallback = _build_smoothed_track(
        raw_cx, raw_cy, raw_r, raw_is_shell, dv, expected_r_px
    )

    if used_fallback:
        log.warning(
            f"Only {n_shell + n_cluster} slice(s) localized the phantom "
            f"(shell={n_shell}, cluster={n_cluster}); too few to trust a "
            f"per-slice track — using volume-center for all slices."
        )

    circles_list = [
        PhantomCircle(
            float(track_cx[i]), float(track_cy[i]), float(track_r[i]),
            float(conf[i]) if not np.isnan(raw_cx[i]) else 0.3,
            SliceStatus.VALID,
        )
        for i in range(n)
    ]

    global_cx = float(np.median(track_cx))
    global_cy = float(np.median(track_cy))
    global_r = float(np.median(track_r))

    log.info(
        f"Phantom boundary track: global=({global_cx:.1f}, {global_cy:.1f}), "
        f"r={global_r:.1f}px | localized {n_shell} shell + {n_cluster} cluster, "
        f"{n_inliers} track inliers, drift "
        f"x[{track_cx.min():.1f},{track_cx.max():.1f}] "
        f"y[{track_cy.min():.1f},{track_cy.max():.1f}]"
    )

    return PhantomDetectionResult(circles_list, global_cx, global_cy, global_r)