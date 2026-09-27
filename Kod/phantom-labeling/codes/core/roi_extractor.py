"""Phase 4: ROI extraction with zero interpolation (nearest-neighbor only)."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from core.dicom_loader import DicomVolume
from core.phantom_detector import PhantomDetectionResult, SliceStatus
from core.section_classifier import SectionClassificationResult
from core.rotation_detector import RotationResult
from analysis.insert_detector import (
    detect_inserts,
    refine_insert_center,
)
from utils.logger import log
import config


@dataclass
class ROI:
    label: str  # e.g., "Diagonal", "Insert_01_5mm", "Star"
    section: str  # INSERT1, INSERT2, STAR8, STAR16
    slice_idx: int
    center_x: float
    center_y: float
    pixels: np.ndarray  # Raw HU pixel data (no interpolation)
    shape_type: str  # "circle" or "square"
    size_cm: float  # diameter for circle, side for square
    angle_deg: float  # Position angle from phantom center
    accepted: bool = True
    rejection_reason: str = ""
    # In-plane rotation of the ROI shape itself (degrees). Used for
    # square texture ROIs when the phantom is tilted — the texture
    # square is rotated with the phantom by this amount in image
    # coords (positive = CW visually). Circles are rotation-invariant
    # and ignore this field.
    shape_rotation_deg: float = 0.0


@dataclass
class ExtractionResult:
    rois: List[ROI] = field(default_factory=list)
    per_slice: Dict[int, List[ROI]] = field(default_factory=dict)


def extract_rois(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    sections: SectionClassificationResult,
    rotation: RotationResult,
) -> ExtractionResult:
    """Extract all ROIs from the volume."""
    log.info("Phase 4: Extracting ROIs...")

    result = ExtractionResult()

    # Collect all INSERT sections to compute shared canonical positions
    insert_sections = [s for s in sections.sections if s.name in ("INSERT1", "INSERT2")]
    if insert_sections:
        _extract_all_insert_rois(
            dv, phantom, insert_sections, sections, rotation, result
        )

    # The phantom has exactly ONE module per star name. If classification
    # ever emits two sections with the same name (residual-fragment bug),
    # extracting both would silently overwrite overlays/ROIs of the real
    # one with fragment-anchored output — keep only the largest per name.
    star_by_name: Dict[str, object] = {}
    for sec in sections.sections:
        if sec.name in ("STAR8", "STAR16"):
            prev = star_by_name.get(sec.name)
            if prev is None:
                star_by_name[sec.name] = sec
            else:
                log.error(
                    f"Duplicate {sec.name} sections from classification "
                    f"(slices {prev.start_idx}-{prev.end_idx} and "
                    f"{sec.start_idx}-{sec.end_idx}) — keeping the larger "
                    f"one; the smaller is a classification artefact."
                )
                if sec.slice_count > prev.slice_count:
                    star_by_name[sec.name] = sec

    for sec in sections.sections:
        if sec.name in ("STAR8", "STAR16") and star_by_name.get(sec.name) is sec:
            _extract_star_rois(dv, phantom, sec, sections, rotation, result)

    log.info(f"Extracted {len(result.rois)} total ROIs")
    return result


def _warn_if_crop_leaves_image(
    hu_slice: np.ndarray, cx: float, cy: float, half_extent: int, label: str
) -> None:
    """Report a crop box that does not fit inside the slice.

    The cutters below skip writes that fall off the image, leaving NaN. Those
    NaNs are indistinguishable downstream from the circle's own corner padding —
    both become the exporter's -32768 sentinel — so a geometry error would read
    as a normal mask boundary. Saying so out loud is the difference between a
    detectable fault and a silently smaller ROI.
    """
    h, w = hu_slice.shape
    x0, x1 = round(cx) - half_extent, round(cx) + half_extent
    y0, y1 = round(cy) - half_extent, round(cy) + half_extent
    if x0 >= 0 and y0 >= 0 and x1 < w and y1 < h:
        return
    log.warning(
        f"{label} crop box [{x0}..{x1}, {y0}..{y1}] leaves the {w}x{h} slice; "
        f"the outside pixels become out-of-ROI padding and the emitted ROI is "
        f"smaller than requested."
    )


def _extract_nearest_neighbor_circle(
    hu_slice: np.ndarray, cx: float, cy: float, radius_px: float
) -> np.ndarray:
    """Extract circular ROI using nearest-neighbor (NO interpolation).

    Returns a square array where pixels outside the circle are NaN.
    """
    r = int(np.ceil(radius_px))
    size = 2 * r + 1
    roi = np.full((size, size), np.nan, dtype=np.float32)
    h, w = hu_slice.shape
    _warn_if_crop_leaves_image(hu_slice, cx, cy, r, "circle")

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= radius_px * radius_px:
                src_x = round(cx + dx)
                src_y = round(cy + dy)
                if 0 <= src_x < w and 0 <= src_y < h:
                    roi[dy + r, dx + r] = hu_slice[src_y, src_x]

    return roi


def _extract_nearest_neighbor_square(
    hu_slice: np.ndarray, cx: float, cy: float, half_side_px: float
) -> np.ndarray:
    """Extract square ROI using nearest-neighbor (NO interpolation)."""
    hs = int(np.ceil(half_side_px))
    size = 2 * hs + 1
    roi = np.full((size, size), np.nan, dtype=np.float32)
    h, w = hu_slice.shape
    _warn_if_crop_leaves_image(hu_slice, cx, cy, hs, "square")

    for dy in range(-hs, hs + 1):
        for dx in range(-hs, hs + 1):
            src_x = round(cx + dx)
            src_y = round(cy + dy)
            if 0 <= src_x < w and 0 <= src_y < h:
                roi[dy + hs, dx + hs] = hu_slice[src_y, src_x]

    return roi


def _extract_all_insert_rois(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    insert_sections: list,
    sections: SectionClassificationResult,
    rotation: RotationResult,
    result: ExtractionResult,
):
    """Per-slice insert detection + concentric sub-ROI extraction.

    Pipeline:
      A. Initial slot calibration — fit start-angle for each ring from
         ``detect_inserts`` (legacy Hough/CC). Seeds Insert_NN positions
         from the ring formula at those angles.
      B. Iterative refinement (max 2 passes):
            * For each insert in each valid slice, call
              ``refine_insert_center`` (Otsu in window → minority-class
              insert mask → morph clean → Hough circle).
            * Apply radius sanity filter — drop refinements that landed
              farther than ``INSERT_REFINEMENT_RADIUS_TOL_PX`` from the
              ring radius.
            * Refit start angles from successful refinements; if the
              shift is below ``INSERT_SLOT_CALIBRATION_REFIT_DEG``,
              break. Otherwise re-seed and run another pass.
      C. Per-section median anchor — per-insert median of refined
         positions; falls back to ring formula when fewer than
         ``INSERT_REFINEMENT_MIN_SLICES`` succeeded.
      D. Concentric sub-ROI emission — one ROI per diameter in
         ``config.INSERT_SUBROI_DIAMETERS_MM`` (5/10/15 mm) per insert
         per slice. Labels are ``Insert_NN_{D}mm``.
    """
    gcx, gcy = phantom.global_center_x, phantom.global_center_y
    spacing = 360.0 / 10  # 36° between insert slots

    # --- Stage A: initial start-angle calibration + ring-formula seeds ---
    # Returns data-derived ring radii (post radius-scrub of detect_inserts
    # cloud). These are the authoritative ring radii downstream — config
    # nominals are only used as initial guess for the scrub tolerance.
    (initial_positions, outer_start, inner_start,
     outer_ring_px, inner_ring_px) = _compute_initial_slot_positions(
        dv, phantom, insert_sections, sections, gcx, gcy, spacing
    )
    if not initial_positions:
        return

    valid_slices_per_sec: Dict[str, List[int]] = {
        sec.name: [
            i for i in range(sec.start_idx, sec.end_idx)
            if sections.slice_labels[i] != "TRANSITION"
        ]
        for sec in insert_sections
    }

    # Build outer/inner num sets so the radius-sanity filter knows which
    # ring each insert belongs to.
    outer_nums: set = set()
    inner_nums: set = set()
    for sec in insert_sections:
        cfg = config.SECTIONS[sec.name]
        outer_nums.update(cfg["outer_inserts"])
        inner_nums.update(cfg["inner_inserts"])

    # --- Stage B: iterative refinement loop ---
    current_positions = initial_positions
    refined: Dict[int, Dict[int, Tuple[float, float]]] = {}
    refit_threshold = config.INSERT_SLOT_CALIBRATION_REFIT_DEG
    radius_tol = config.INSERT_REFINEMENT_RADIUS_TOL_PX
    MAX_ITERATIONS = 2

    for it in range(MAX_ITERATIONS):
        refined = _run_refinement_pass(
            dv, insert_sections, valid_slices_per_sec, current_positions
        )
        _filter_refinements_by_radius(
            refined, gcx, gcy, outer_nums, inner_nums,
            outer_ring_px, inner_ring_px, radius_tol,
        )

        new_outer, new_inner = _refit_start_angles(
            refined, insert_sections, gcx, gcy, spacing,
            fallback_outer=outer_start, fallback_inner=inner_start,
        )
        out_diff = _angle_diff(new_outer, outer_start, spacing)
        in_diff = _angle_diff(new_inner, inner_start, spacing)

        if max(out_diff, in_diff) <= refit_threshold:
            log.info(
                f"  Slot calibration converged at iter {it + 1}: "
                f"outer={new_outer:.1f}° (Δ{out_diff:.2f}°), "
                f"inner={new_inner:.1f}° (Δ{in_diff:.2f}°)"
            )
            break

        log.info(
            f"  Slot calibration refit at iter {it + 1}: "
            f"outer {outer_start:.1f}° → {new_outer:.1f}° (Δ{out_diff:.2f}°), "
            f"inner {inner_start:.1f}° → {new_inner:.1f}° (Δ{in_diff:.2f}°). "
            f"Re-seeding."
        )
        outer_start, inner_start = new_outer, new_inner
        current_positions = _seed_positions_from_angles(
            insert_sections, dv, gcx, gcy, outer_start, inner_start,
            outer_r_px=outer_ring_px, inner_r_px=inner_ring_px,
            spacing=spacing,
        )

    # --- Stage C: per-section median anchor (with ring-formula fallback) ---
    final_positions: Dict[int, Tuple[float, float, float]] = {}
    fallback_nums: List[int] = []
    min_slices = config.INSERT_REFINEMENT_MIN_SLICES
    for num, init_pos in current_positions.items():
        per_slice = refined.get(num, {})
        if len(per_slice) >= min_slices:
            xs = np.fromiter((p[0] for p in per_slice.values()), dtype=np.float64)
            ys = np.fromiter((p[1] for p in per_slice.values()), dtype=np.float64)
            final_positions[num] = (
                float(np.median(xs)),
                float(np.median(ys)),
                init_pos[2],
            )
        else:
            final_positions[num] = init_pos
            fallback_nums.append(num)

    if fallback_nums:
        log.warning(
            f"  Insert refinement: {len(fallback_nums)}/{len(initial_positions)} "
            f"inserts had <{min_slices} successful per-slice detections; fell "
            f"back to ring formula (Insert_"
            f"{','.join(f'{n:02d}' for n in sorted(fallback_nums))})."
        )
    else:
        log.info(
            f"  Insert refinement: all {len(initial_positions)} inserts anchored "
            f"by per-section median of per-slice refinements."
        )

    # --- Stage C2: mean-image polish ---
    # Single highest-precision refinement: average all valid slices of each
    # section to suppress per-slice noise (~ √N reduction), then run the full
    # three-stage refine_insert_center on the mean image, seeded by the
    # per-section median anchor we just computed. The mean image gives
    # cleaner rims and a more symmetric intensity profile than any single
    # slice, so Hough lands more precisely. A 3 mm shift cap rejects
    # pathological mean-image refinements (e.g. a section with too few
    # slices producing a noisy mean).
    avg_spacing_mm = (dv.pixel_spacing[0] + dv.pixel_spacing[1]) / 2.0
    polish_max_shift_px = config.INSERT_POLISH_MAX_SHIFT_MM / avg_spacing_mm
    # Fallback inserts (ring-formula anchor) are allowed a larger snap onto
    # the real insert that the high-SNR mean image reveals — see config.
    fallback_polish_max_shift_px = config.INSERT_FALLBACK_POLISH_MAX_SHIFT_MM / avg_spacing_mm
    fallback_set = set(fallback_nums)
    for sec in insert_sections:
        sec_valid = valid_slices_per_sec[sec.name]
        if len(sec_valid) < 2:
            continue
        sec_mean = np.mean(dv.volume[sec_valid], axis=0).astype(np.float32)

        sec_cfg = config.SECTIONS[sec.name]
        sec_nums = sec_cfg["outer_inserts"] + sec_cfg["inner_inserts"]
        polished_count = 0
        rescued = 0
        for num in sec_nums:
            cx, cy, angle = final_positions[num]
            polished = refine_insert_center(
                sec_mean, cx, cy, dv.pixel_spacing,
            )
            if polished is None:
                continue
            dx = polished[0] - cx
            dy = polished[1] - cy
            is_fallback = num in fallback_set
            cap = fallback_polish_max_shift_px if is_fallback else polish_max_shift_px
            if dx * dx + dy * dy <= cap * cap:
                final_positions[num] = (polished[0], polished[1], angle)
                polished_count += 1
                if is_fallback:
                    rescued += 1
        log.info(
            f"  {sec.name}: mean-image polish refined "
            f"{polished_count}/{len(sec_nums)} insert centres "
            f"(mean of {len(sec_valid)} slices; {rescued} ring-formula "
            f"fallback inserts snapped onto the real insert)"
        )

    # --- Stage B2: per-slice re-refinement seeded by the polished anchor ---
    # The first pass used ring-formula seeds that could be 5-10 px off true
    # centre. With the polished per-section anchor as seed, every slice's
    # Stage 1 window now crops cleanly onto the insert and Stage 2 Hough has
    # essentially zero correction to make — each slice converges to its own
    # sub-pixel-accurate refined centre. This ALWAYS runs now: each label is
    # centred per slice on that slice's own insert (then denoised in Stage
    # B3) so the ROI sits exactly on the insert it labels, following any real
    # drift. A per-slice ±N mm sanity tolerance drops pathological single-
    # slice refinements before they reach the smoother.
    max_dev_px = config.INSERT_ROI_PER_SLICE_MAX_DEV_MM / avg_spacing_mm
    per_slice_positions: Dict[int, Dict[int, Tuple[float, float]]] = {
        n: {} for n in final_positions
    }
    for sec in insert_sections:
        sec_cfg = config.SECTIONS[sec.name]
        sec_nums = sec_cfg["outer_inserts"] + sec_cfg["inner_inserts"]
        attempted = 0
        kept = 0
        for slice_idx in valid_slices_per_sec[sec.name]:
            for num in sec_nums:
                anchor_cx, anchor_cy, _ = final_positions[num]
                attempted += 1
                pt = refine_insert_center(
                    dv.volume[slice_idx], anchor_cx, anchor_cy,
                    dv.pixel_spacing,
                )
                if pt is None:
                    continue
                dx = pt[0] - anchor_cx
                dy = pt[1] - anchor_cy
                if dx * dx + dy * dy <= max_dev_px * max_dev_px:
                    per_slice_positions[num][slice_idx] = pt
                    kept += 1
        log.info(
            f"  {sec.name}: Stage B2 (per-slice re-refinement) — "
            f"{kept}/{attempted} per-slice positions kept "
            f"(within ±{config.INSERT_ROI_PER_SLICE_MAX_DEV_MM:.1f} mm "
            f"of anchor)."
        )

    # --- Stage B3: denoise each insert's per-slice track ---
    # Robust Theil-Sen smoothing per insert (see _smooth_insert_track):
    # removes the per-slice refinement jitter while preserving real drift, so
    # every slice gets a clean, exactly-centred per-slice position. Unless
    # INSERT_ROI_PER_SLICE_DYNAMIC forces the RAW (un-smoothed) per-slice
    # positions for diagnostics, the smoothed track is what Stage D emits.
    smoothed_positions: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for sec in insert_sections:
        sec_cfg = config.SECTIONS[sec.name]
        sec_nums = sec_cfg["outer_inserts"] + sec_cfg["inner_inserts"]
        sec_slices = valid_slices_per_sec[sec.name]
        for num in sec_nums:
            anchor_cx, anchor_cy, _ = final_positions[num]
            smoothed_positions[num] = _smooth_insert_track(
                per_slice_positions.get(num, {}), sec_slices,
                (anchor_cx, anchor_cy), max_dev_px,
            )

    # --- Stage D: emit concentric sub-ROIs per insert per slice ---
    # Default: each slice's ROI uses its denoised per-slice centre from Stage
    # B3 (exactly centred on that slice's insert, jitter removed). Setting
    # INSERT_ROI_PER_SLICE_DYNAMIC=True swaps in the RAW per-slice centres
    # (un-smoothed) for diagnostics.
    subroi_diameters_mm = config.INSERT_SUBROI_DIAMETERS_MM
    use_raw = config.INSERT_ROI_PER_SLICE_DYNAMIC

    for sec in insert_sections:
        sec_cfg = config.SECTIONS[sec.name]
        sec_nums = sec_cfg["outer_inserts"] + sec_cfg["inner_inserts"]

        for slice_idx in valid_slices_per_sec[sec.name]:
            slice_rois: List[ROI] = []
            for num in sec_nums:
                anchor_cx, anchor_cy, angle_deg = final_positions[num]
                if use_raw:
                    slice_xy = per_slice_positions.get(num, {}).get(slice_idx)
                    cx, cy = slice_xy if slice_xy is not None else (anchor_cx, anchor_cy)
                else:
                    cx, cy = smoothed_positions.get(num, {}).get(
                        slice_idx, (anchor_cx, anchor_cy)
                    )

                for diam_mm in subroi_diameters_mm:
                    radius_px = (diam_mm / 2.0) / avg_spacing_mm
                    label = f"Insert_{num:02d}_{diam_mm}mm"
                    pixels = _extract_nearest_neighbor_circle(
                        dv.volume[slice_idx], cx, cy, radius_px
                    )
                    slice_rois.append(ROI(
                        label=label,
                        section=sec.name,
                        slice_idx=slice_idx,
                        center_x=cx,
                        center_y=cy,
                        pixels=pixels,
                        shape_type="circle",
                        size_cm=diam_mm / 10.0,
                        angle_deg=angle_deg,
                    ))
            result.rois.extend(slice_rois)
            result.per_slice[slice_idx] = slice_rois


def _smooth_insert_track(
    per_slice_pts: Dict[int, Tuple[float, float]],
    section_slices: List[int],
    anchor_xy: Tuple[float, float],
    max_dev_px: float,
) -> Dict[int, Tuple[float, float]]:
    """Denoise one insert's per-slice centre detections into a smooth track.

    Each slice's ROI should sit on THAT slice's own insert centre (so it
    follows any real couch drift and is exactly centred), but the raw
    per-slice refinement carries ~0.3-1 px detection jitter. A robust
    Theil-Sen line per axis removes that jitter while preserving real slow
    drift — the same denoising used for the phantom boundary track. With no
    real drift the line is flat ≈ the median, i.e. ≈ the per-section anchor,
    so a rigid phantom is unaffected; where the insert genuinely shifts
    across z the track follows it. Predictions are clamped to ``anchor ±
    max_dev_px`` and the whole track is rejected (→ flat anchor) if its span
    would exceed that, so a noisy fit can never throw an ROI off its insert.

    Returns a centre for EVERY slice in ``section_slices`` (anchor fallback
    for slices the per-slice refinement missed).
    """
    ax, ay = anchor_xy
    det = sorted(per_slice_pts.keys())
    if len(det) < 3:
        return {s: (ax, ay) for s in section_slices}

    zs = np.asarray(det, dtype=float)
    sec = np.asarray(section_slices, dtype=float)

    def _line(values: np.ndarray, anchor: float) -> np.ndarray:
        try:
            from scipy.stats import theilslopes
            slope, intercept, _lo, _hi = theilslopes(values, zs)
            pred = np.clip(intercept + slope * sec,
                           anchor - max_dev_px, anchor + max_dev_px)
            if float(pred.max() - pred.min()) <= max_dev_px:
                return pred
        except Exception:
            pass
        med = float(np.clip(np.median(values), anchor - max_dev_px, anchor + max_dev_px))
        return np.full(len(section_slices), med)

    px = _line(np.asarray([per_slice_pts[s][0] for s in det]), ax)
    py = _line(np.asarray([per_slice_pts[s][1] for s in det]), ay)
    return {s: (float(px[k]), float(py[k])) for k, s in enumerate(section_slices)}


def _angle_diff(a: float, b: float, period: float) -> float:
    """Smallest signed difference between two angles modulo ``period``."""
    d = abs(a - b) % period
    return min(d, period - d)


def _run_refinement_pass(
    dv: DicomVolume,
    insert_sections: list,
    valid_slices_per_sec: Dict[str, List[int]],
    positions: Dict[int, Tuple[float, float, float]],
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """One sweep of per-slice refinement across all (insert, slice) pairs.

    Returns ``num -> {slice_idx: (refined_x, refined_y)}``. Inserts whose
    refinement failed at every slice will have an empty inner dict.
    """
    refined: Dict[int, Dict[int, Tuple[float, float]]] = {n: {} for n in positions}
    for sec in insert_sections:
        sec_cfg = config.SECTIONS[sec.name]
        sec_nums = sec_cfg["outer_inserts"] + sec_cfg["inner_inserts"]
        for slice_idx in valid_slices_per_sec[sec.name]:
            for num in sec_nums:
                if num not in positions:
                    continue
                ix, iy, _ = positions[num]
                pt = refine_insert_center(
                    dv.volume[slice_idx], ix, iy, dv.pixel_spacing
                )
                if pt is not None:
                    refined[num][slice_idx] = pt
    return refined


def _filter_refinements_by_radius(
    refined: Dict[int, Dict[int, Tuple[float, float]]],
    gcx: float,
    gcy: float,
    outer_nums: set,
    inner_nums: set,
    outer_r_px: float,
    inner_r_px: float,
    tol_px: float,
) -> None:
    """Drop per-slice refinements whose distance from the phantom centre
    deviates from the expected ring radius by more than ``tol_px``.

    Catches two failure modes: (1) a refinement that locked onto a
    neighbouring insert (lands on the wrong ring); (2) a refinement that
    locked onto a phantom-mount artefact at a non-ring radius. Operates
    in-place on ``refined``.
    """
    for num in list(refined.keys()):
        expected_r = outer_r_px if num in outer_nums else inner_r_px
        before = len(refined[num])
        refined[num] = {
            s_idx: (x, y)
            for s_idx, (x, y) in refined[num].items()
            if abs(np.sqrt((x - gcx) ** 2 + (y - gcy) ** 2) - expected_r) <= tol_px
        }
        dropped = before - len(refined[num])
        if dropped > 0:
            log.debug(
                f"  Insert_{num:02d}: dropped {dropped}/{before} refinements "
                f"outside ring radius (±{tol_px:.0f}px)"
            )


def _refit_start_angles(
    refined: Dict[int, Dict[int, Tuple[float, float]]],
    insert_sections: list,
    gcx: float,
    gcy: float,
    spacing: float,
    fallback_outer: float,
    fallback_inner: float,
) -> Tuple[float, float]:
    """Refit per-ring start angle from successful per-slice refinements.

    For each insert with ≥2 surviving refinements, computes the angle of
    its median refined centre. Aggregates these per ring and reuses
    ``_fit_grid_start_angle`` to recover slot 0's angle. Returns the
    fallback values when too few inserts have usable refinements (<4 per
    ring) — not enough to fit a 10-slot grid robustly.
    """
    outer_angles: List[float] = []
    inner_angles: List[float] = []

    for sec in insert_sections:
        cfg = config.SECTIONS[sec.name]
        for num in cfg["outer_inserts"]:
            per_slice = refined.get(num, {})
            if len(per_slice) >= 2:
                xs = np.fromiter((p[0] for p in per_slice.values()), dtype=np.float64)
                ys = np.fromiter((p[1] for p in per_slice.values()), dtype=np.float64)
                mx, my = float(np.median(xs)), float(np.median(ys))
                outer_angles.append(
                    float(np.degrees(np.arctan2(my - gcy, mx - gcx))) % 360
                )
        for num in cfg["inner_inserts"]:
            per_slice = refined.get(num, {})
            if len(per_slice) >= 2:
                xs = np.fromiter((p[0] for p in per_slice.values()), dtype=np.float64)
                ys = np.fromiter((p[1] for p in per_slice.values()), dtype=np.float64)
                mx, my = float(np.median(xs)), float(np.median(ys))
                inner_angles.append(
                    float(np.degrees(np.arctan2(my - gcy, mx - gcx))) % 360
                )

    new_outer = (
        _fit_grid_start_angle(
            outer_angles, spacing, config.INSERT_OUTER_SLOT0_ANGLE_DEG)
        if len(outer_angles) >= 4 else fallback_outer
    )
    new_inner = (
        _fit_grid_start_angle(
            inner_angles, spacing, config.INSERT_INNER_SLOT0_ANGLE_DEG)
        if len(inner_angles) >= 4 else fallback_inner
    )
    return new_outer, new_inner


def _seed_positions_from_angles(
    insert_sections: list,
    dv: DicomVolume,
    gcx: float,
    gcy: float,
    outer_start: float,
    inner_start: float,
    outer_r_px: Optional[float] = None,
    inner_r_px: Optional[float] = None,
    spacing: float = 36.0,
) -> Dict[int, Tuple[float, float, float]]:
    """Build per-insert ring-formula seeds at the given start angles.

    ``outer_r_px`` / ``inner_r_px`` default to config nominals; pass the
    data-derived robust radii from ``_compute_initial_slot_positions``
    when re-seeding so we don't snap back to a wrong nominal.
    """
    if outer_r_px is None:
        outer_r_px = dv.cm_to_pixels(config.OUTER_RING_DISTANCE_CM)
    if inner_r_px is None:
        inner_r_px = dv.cm_to_pixels(config.INNER_RING_DISTANCE_CM)

    positions: Dict[int, Tuple[float, float, float]] = {}
    for sec in insert_sections:
        cfg = config.SECTIONS[sec.name]
        for idx, num in enumerate(cfg["outer_inserts"]):
            angle_deg = (outer_start + idx * spacing) % 360
            ar = np.deg2rad(angle_deg)
            positions[num] = (
                float(gcx + outer_r_px * np.cos(ar)),
                float(gcy + outer_r_px * np.sin(ar)),
                float(angle_deg),
            )
        for idx, num in enumerate(cfg["inner_inserts"]):
            angle_deg = (inner_start + idx * spacing) % 360
            ar = np.deg2rad(angle_deg)
            positions[num] = (
                float(gcx + inner_r_px * np.cos(ar)),
                float(gcy + inner_r_px * np.sin(ar)),
                float(angle_deg),
            )
    return positions


def _filter_cloud_by_radius(
    detections: List[Tuple[float, float, float]],
    gcx: float,
    gcy: float,
    expected_r_px: float,
    tol_px: float,
) -> Tuple[List[Tuple[float, float, float]], float]:
    """Filter a detect_inserts detection cloud to entries within ±tol_px
    of the expected ring radius. Returns ``(filtered, robust_radius_px)``.

    ``robust_radius_px`` is the median radius of the SURVIVING detections
    (which can differ from ``expected_r_px`` when the phantom geometry is
    slightly off-nominal). Falls back to ``expected_r_px`` when too few
    detections survive to compute a robust median.
    """
    if not detections:
        return [], expected_r_px

    filtered: List[Tuple[float, float, float]] = []
    survivor_radii: List[float] = []
    for x, y, ang in detections:
        r = float(np.sqrt((x - gcx) ** 2 + (y - gcy) ** 2))
        if abs(r - expected_r_px) <= tol_px:
            filtered.append((x, y, ang))
            survivor_radii.append(r)

    if len(survivor_radii) < 4:
        # Too few survived to trust a median — keep raw cloud, raw radius.
        return detections, expected_r_px

    return filtered, float(np.median(survivor_radii))


def _compute_initial_slot_positions(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    insert_sections: list,
    sections: SectionClassificationResult,
    gcx: float,
    gcy: float,
    spacing: float,
) -> Tuple[Dict[int, Tuple[float, float, float]], float, float, float, float]:
    """One-time slot calibration via the legacy ``detect_inserts`` cloud,
    with per-ring radius scrubbing.

    Returns ``(positions, outer_start_deg, inner_start_deg, outer_r_px,
    inner_r_px)``. The two ``_r_px`` values are data-derived medians of
    the surviving (radius-filtered) detections — used downstream for
    seeding and for the per-slice radius sanity filter, so that phantoms
    whose ring radii differ slightly from config nominals are handled
    correctly.

    Filtering rationale: the outer ring sits ~2 cm from the phantom outer
    wall, and ``detect_inserts`` k-means assigns wall artefacts to the
    "outer" cluster, polluting the angle cloud and biasing the start-
    angle fit. The inner ring has no such neighbour, which is why it
    typically labels correctly while the outer ring shifts.
    """
    outer_nominal_r_px = dv.cm_to_pixels(config.OUTER_RING_DISTANCE_CM)
    inner_nominal_r_px = dv.cm_to_pixels(config.INNER_RING_DISTANCE_CM)
    filter_tol_px = config.INSERT_RING_RADIUS_FILTER_TOL_PX

    raw_outer: List[Tuple[float, float, float]] = []
    raw_inner: List[Tuple[float, float, float]] = []

    # Primary signal for slot calibration: the MEAN IMAGE across all valid
    # insert-section slices. Averaging suppresses per-slice noise, lets
    # detect_inserts return a clean angle cloud, and is the difference
    # between "inner ring labels stable, outer way off" (per-slice cloud
    # mode) and consistent labelling on both rings.
    all_valid_slices: List[int] = []
    for sec in insert_sections:
        for i in range(sec.start_idx, sec.end_idx):
            if sections.slice_labels[i] != "TRANSITION":
                all_valid_slices.append(i)

    if all_valid_slices:
        mean_img = np.mean(dv.volume[all_valid_slices], axis=0).astype(np.float32)
        m_outer, m_inner = detect_inserts(mean_img, gcx, gcy, dv.pixel_spacing)
        for c in m_outer:
            raw_outer.append((c.x, c.y, c.angle))
        for c in m_inner:
            raw_inner.append((c.x, c.y, c.angle))
        log.info(
            f"  Slot calibration on mean image of {len(all_valid_slices)} "
            f"insert slices: {len(raw_outer)} outer + {len(raw_inner)} inner "
            f"detect_inserts candidates"
        )

    # Per-slice supplement: only run when the mean-image cloud is too sparse
    # to fit a 10-slot grid robustly (≥6 angles per ring is a comfortable
    # minimum). Avoids the noisy per-slice flood that polluted the outer
    # ring in the previous design.
    if len(raw_outer) < 6 or len(raw_inner) < 6:
        log.info(
            "  Mean-image cloud sparse — supplementing with per-slice "
            "detect_inserts (will be aggressively radius-scrubbed)."
        )
        for sec in insert_sections:
            for i in range(sec.start_idx, sec.end_idx):
                if sections.slice_labels[i] == "TRANSITION":
                    continue
                outer, inner = detect_inserts(
                    dv.volume[i], gcx, gcy, dv.pixel_spacing
                )
                for c in outer:
                    raw_outer.append((c.x, c.y, c.angle))
                for c in inner:
                    raw_inner.append((c.x, c.y, c.angle))

    # Scrub wall / mount false positives.
    filt_outer, outer_r_px = _filter_cloud_by_radius(
        raw_outer, gcx, gcy, outer_nominal_r_px, filter_tol_px,
    )
    filt_inner, inner_r_px = _filter_cloud_by_radius(
        raw_inner, gcx, gcy, inner_nominal_r_px, filter_tol_px,
    )

    outer_angles = [a for _, _, a in filt_outer]
    inner_angles = [a for _, _, a in filt_inner]

    outer_start = (
        _fit_grid_start_angle(
            outer_angles, spacing, config.INSERT_OUTER_SLOT0_ANGLE_DEG)
        if outer_angles else config.INSERT_OUTER_SLOT0_ANGLE_DEG
    )
    inner_start = (
        _fit_grid_start_angle(
            inner_angles, spacing, config.INSERT_INNER_SLOT0_ANGLE_DEG)
        if inner_angles else config.INSERT_INNER_SLOT0_ANGLE_DEG
    )

    if not raw_outer and not raw_inner:
        log.warning(
            "  No insert detections found by detect_inserts — seeding both "
            "rings at 0° start angle; iterative refit will try to recover."
        )
    else:
        log.info(
            f"  Initial slot calibration: "
            f"outer_start={outer_start:.1f}° "
            f"({len(filt_outer)}/{len(raw_outer)} samples after radius scrub, "
            f"r={outer_r_px:.1f}px / nominal {outer_nominal_r_px:.1f}px), "
            f"inner_start={inner_start:.1f}° "
            f"({len(filt_inner)}/{len(raw_inner)} samples, "
            f"r={inner_r_px:.1f}px / nominal {inner_nominal_r_px:.1f}px)"
        )

    positions = _seed_positions_from_angles(
        insert_sections, dv, gcx, gcy, outer_start, inner_start,
        outer_r_px=outer_r_px, inner_r_px=inner_r_px, spacing=spacing,
    )
    return positions, outer_start, inner_start, outer_r_px, inner_r_px


def _fit_grid_start_angle(
    detected_angles: List[float], spacing: float = 36.0,
    reference_angle: Optional[float] = None,
) -> float:
    """Find start angle that best aligns all detections to an equal-spacing grid.

    Sweeps candidate start angles in 0.5° steps and picks the one that
    minimizes the sum of squared residuals to the nearest grid point.

    ``reference_angle`` anchors slot 0 to a fixed physical insert. The grid is
    ``spacing``-periodic, so the fitted phase alone (returned in
    ``[0, spacing)``) cannot say WHICH detected insert is slot 0. For a ring
    whose slot 0 sits at the 0/``spacing`` boundary — the OUTER ring, slot 0 at
    0° — noise that nudges the fit from +0.3° to ``spacing``−0.2° (the SAME
    phase) places the slot-0 seed a full ``spacing`` away and rotates the whole
    ring by one slot. Snapping the result to the grid representative nearest
    ``reference_angle`` pins slot 0 to the same physical insert on every
    protocol, eliminating that off-by-one. ``None`` keeps the legacy
    ``[0, spacing)`` behaviour.
    """
    detected = np.asarray(detected_angles)
    best_start = 0.0
    best_cost = float('inf')
    for candidate in np.arange(0, spacing, 0.5):
        offsets = (detected - candidate) % spacing
        residuals = np.minimum(offsets, spacing - offsets)
        cost = float(np.sum(residuals ** 2))
        if cost < best_cost:
            best_cost = cost
            best_start = float(candidate)
    if reference_angle is not None:
        best_start += round((reference_angle - best_start) / spacing) * spacing
    return best_start




def _extract_star_rois(
    dv: DicomVolume,
    phantom: PhantomDetectionResult,
    sec,
    sections: SectionClassificationResult,
    rotation: RotationResult,
    result: ExtractionResult,
):
    """Extract star and texture ROIs for a STAR section with 3D consistency.

    Two-pass approach:
    1. Detect texture blob positions across all slices
    2. Compute canonical (median) distance and angle per texture
    3. Use canonical positions for all slices (star always at phantom center)
    """
    star_r_px = dv.cm_to_pixels(config.STAR_RADIUS_CM) * config.ROI_SCALE
    # Physical (full) star radius — used by the star-centre polish so it
    # sees the WHOLE star material (including outer spoke tips), not the
    # ROI_SCALE-shrunken extraction window. If the polish only sees the
    # inner 70 % of the star, its centroid is biased toward whichever
    # half of the star sits inside the truncated window — exactly the
    # wrong direction when the seed is already offset from the true
    # star centre.
    star_full_r_px = dv.cm_to_pixels(config.STAR_RADIUS_CM)
    texture_half_side_px = dv.cm_to_pixels(config.TEXTURE_SIDE_CM / 2.0) * config.ROI_SCALE
    texture_dist_px = dv.cm_to_pixels(config.TEXTURE_CENTER_DISTANCE_CM)
    # Full texture side in pixels — passed into _detect_texture_blobs so its
    # Stage-2 edge-refinement step can constrain its rectangle search.
    texture_side_px = dv.cm_to_pixels(config.TEXTURE_SIDE_CM)

    section_cfg = config.SECTIONS[sec.name]
    textures = section_cfg["textures"]
    offset = rotation.global_offset_deg

    # Detect z-flipped (inserts_first) scan. When the phantom is physically
    # inverted in z, the x-y plane is also mirrored about the y-axis, which
    # swaps the textures at angles 0° (right) and 180° (left). Top (270°)
    # and bottom (90°) stay put. We apply the mirror here so the texture
    # names stay bound to their physical-pattern identity regardless of
    # scan orientation. Detection: for inserts_first phantoms, INSERT1
    # (= the insert section physically closer to the stars in the CalibraMics
    # convention) appears BEFORE the stars in z; for the normal stars_first
    # convention, INSERT1 appears AFTER the stars.
    insert1_sec = next((s for s in sections.sections if s.name == "INSERT1"), None)
    inserts_first = insert1_sec is not None and insert1_sec.start_idx < sec.start_idx
    if inserts_first:
        textures = {(180 - a) % 360: name for a, name in textures.items()}
        log.debug(f"  {sec.name}: mirroring texture angles for inserts-first scan "
                  f"(0<->180 deg swap)")

    # Per-SECTION consensus centre from the per-slice boundary track: the
    # median of this section's own track circles. For a rigid phantom the
    # physical centre is constant within a section, so the median is the
    # truth and averages out residual noise (the right answer for radiomics
    # harmonization — a stable, per-section anchor that still follows slow
    # inter-section drift, rather than one global value for the whole
    # volume). Falls back to the global median if the section is empty.
    _sec_cx = [phantom.circles[i].center_x for i in range(sec.start_idx, sec.end_idx)]
    _sec_cy = [phantom.circles[i].center_y for i in range(sec.start_idx, sec.end_idx)]
    _sec_r = [phantom.circles[i].radius for i in range(sec.start_idx, sec.end_idx)]
    gcx = float(np.median(_sec_cx)) if _sec_cx else phantom.global_center_x
    gcy = float(np.median(_sec_cy)) if _sec_cy else phantom.global_center_y
    g_phantom_r = float(np.median(_sec_r)) if _sec_r else getattr(phantom, "global_radius", None)

    # Previously this did its own second boundary trim on top of the
    # classifier's trim, effectively double-trimming (e.g. a 40-slice
    # STAR16 lost 3 slices here + 3 in the classifier + 3 "safety" = 9
    # per side). The classifier already converts transition slices to
    # TRANSITION labels, so we rely on that and the TRANSITION-skip in
    # the loop below (line ~418: `if sections.slice_labels[i] == "TRANSITION"`).
    boundary_slices: set = set()

    valid_slices = []
    for i in range(sec.start_idx, sec.end_idx):
        if i in boundary_slices:
            continue
        if sections.slice_labels[i] == "TRANSITION":
            continue
        # Use global phantom center for star sections, so isolated INVALID
        # slices (bad Hough detection) don't break 3D continuity
        valid_slices.append(i)

    # --- Adaptive PVE guard (replaces the old fixed 1-slice trim) ---
    # The fixed 1-slice-each-side trim was too aggressive on clean
    # sections (kept only 28 of 30 slices for STAR). Removing it
    # entirely was too permissive — boundary slices where the section
    # transitions to the next module sometimes slip through the
    # classifier's texture-square gate and end up labelled.
    #
    # The fix: measure each candidate slice's signal weight (sum of
    # gradient magnitude) and compare it to the section's central
    # plateau. The boundary slices of a real STAR section are clean and
    # have signal weight close to the section median. A slice that's
    # actually in a transition region has signal weight noticeably
    # below the median (the radial spokes are smearing out into plastic
    # + air gaps from the adjacent module, reducing gradient activity).
    #
    # CRITICAL: the signal must be measured over the STAR DISC ONLY, not
    # the whole phantom interior. The 4 cardinal texture squares are
    # large, high-contrast, and resolve (sharpen) earlier than the
    # star's thin spokes — if they're included in the mask their gradient
    # dominates the sum and completely hides a still-blurred star (user-
    # observed: slice 166 shows crisp Diagonal/Horizontal/Vertical/Square
    # edges but a visibly soft-edged Star16 wheel, yet the whole-interior
    # signal doesn't dip because the 4 sharp squares carry most of the
    # total gradient). Restricting the mask to the star's own disc (with
    # a small margin so Sobel sees the star-to-background edge) makes the
    # metric sensitive to exactly the slowest-resolving component. The
    # decision itself stays slice-level: if the star disc looks unstable,
    # the WHOLE slice (star + all 4 textures) is peeled, per the pipeline
    # rule that ROI inclusion is decided per-slice, not per-shape.
    if phantom is not None and len(valid_slices) >= 6:
        import cv2 as _cv2
        h_img, w_img = dv.volume.shape[1], dv.volume.shape[2]
        _cx, _cy = gcx, gcy
        # Margin beyond the star's full physical radius: covers (a) up to
        # STAR_ANCHOR_MAX_OFFSET_MM of anchor imprecision (gcx/gcy is the
        # phantom WALL centre here, not yet the polished star centre —
        # that refinement happens later in this function) and (b) a few
        # extra mm of background so Sobel can see the star-to-background
        # edge itself, not just interior spoke boundaries.
        _avg_sp_px = (dv.pixel_spacing[0] + dv.pixel_spacing[1]) / 2.0
        _anchor_slack_px = config.STAR_ANCHOR_MAX_OFFSET_MM / _avg_sp_px
        _r = float(star_full_r_px) + _anchor_slack_px + (3.0 / _avg_sp_px)
        _yy, _xx = np.ogrid[:h_img, :w_img]
        _interior = ((_xx - _cx) ** 2 + (_yy - _cy) ** 2) <= _r * _r

        def _slice_signal(idx: int) -> float:
            slc = dv.volume[idx].astype(np.float32)
            gx_ = _cv2.Sobel(slc, _cv2.CV_32F, 1, 0, ksize=3)
            gy_ = _cv2.Sobel(slc, _cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx_ * gx_ + gy_ * gy_)
            return float(grad[_interior].sum())

        # Reference: median signal over the WHOLE candidate window (robust
        # to the 1-2 genuinely bad boundary slices section_trim may have
        # left at an end — 30 slices with 2 outliers barely move a median).
        n = len(valid_slices)
        central_end = n - n // 4  # floor for the peel loops below
        all_weights = [_slice_signal(i) for i in valid_slices]
        ref_weight = float(np.median(all_weights))
        if ref_weight > 0:
            # DISTANCE-TO-COLLAPSE rule (replaces the old MAD "subtle
            # band"). What actually distinguishes a safe edge slice from
            # one sitting in the transition is NOT how far its own signal
            # sits below the plateau — it's how close it is to the module-
            # edge signal COLLAPSE. Two verified counter-examples killed
            # the band approach:
            #   * KSW STAR8 slice 107 read only -1.5 % (well inside any
            #     reasonable band) yet was visually on the STAR8→INSERT1
            #     transition — because the collapse (ratio 0.62) was just
            #     2 slices away.
            #   * KSW low-dose SharpPlus STAR16 slices 49-51 read -1.1 %
            #     (below the 0.99 band → wrongly peeled, stranding the
            #     section at 27/30) yet were visually clean, fully-formed
            #     STAR16 patterns — the nearest collapse was 6+ slices
            #     away; the dip was a broad shallow noise depression that
            #     even RECOVERED (ratio 0.993 at slice 56) before the
            #     real boundary.
            # So: an edge slice is unsafe iff (a) its own signal has
            # genuinely collapsed (< 0.75 × ref — real transition material,
            # never plateau noise), or (b) a collapse sits within the
            # physical buffer distance just outside it. The buffer is
            # expressed in mm so the guard's physical reach is protocol-
            # independent (2 mm = 2 slices at dz=1, 1 slice at dz≥2).
            aggressive_thr = 0.75 * ref_weight
            dz = float(dv.slice_thickness) if dv.slice_thickness else 1.0
            buffer_n = max(1, round(config.STAR_GUARD_CLIFF_BUFFER_MM / dz))
            # Peel budget in PHYSICAL mm (not a raw slice count) so the
            # guard reaches the same real margin on every protocol — a
            # flat "3 slices" cap would be 3 mm on a 1 mm scan but 9 mm
            # on a 3 mm scan (see config.STAR_ADAPTIVE_GUARD_MAX_MM).
            # Still leaves enough slices via the `> central_end` floor.
            guard_cap = max(1, round(config.STAR_ADAPTIVE_GUARD_MAX_MM / dz))

            # SLIDE, DON'T JUST SHRINK: every peel below tries to reclaim
            # one slice on the OPPOSITE end of the window, so a genuinely
            # unstable boundary slice shifts the window instead of eating
            # into the section's target slice count (the fixed-mm window
            # from section_trim.py, e.g. 30 slices @ 1 mm / 10 @ 3 mm).
            # Without this, peeling one bad lead slice permanently drops
            # the section to N-1 slices even though a perfectly good
            # candidate slice sits just past the old tail (observed: USZ
            # CalibraMics slice 166 peeled, slice 196 — one past the old
            # tail — was already visually clean but sat unclaimed in
            # TRANSITION). The reclaimed slice must not already belong to
            # a different named section (INSERT1/2 or the other STAR
            # module) and must itself pass the same stability check.
            other_section_names = {
                s.name for s in sections.sections
                if s.name not in ("UNKNOWN", "INVALID", "TRANSITION", sec.name)
            }
            n_total_slices = len(sections.slice_labels)

            # A COLLAPSED slice is real transition material: the star
            # pattern has smeared into the inter-module gap (plastic/air/
            # mount hardware), dropping the star-disc gradient far below
            # anything plateau noise can produce. Out-of-volume indices
            # are NOT collapses (a truncated scan ends mid-module with a
            # clean pattern on its last slice — there's no transition
            # there), and neither are slices of another named section
            # (their signal belongs to that module's own pattern).
            def _collapsed(idx: int) -> bool:
                if idx < 0 or idx >= n_total_slices:
                    return False
                if sections.slice_labels[idx] in other_section_names:
                    return False
                return _slice_signal(idx) < aggressive_thr

            def _unsafe_edge(idx: int, direction: int) -> bool:
                """True if the window-edge slice ``idx`` is itself
                collapsed OR sits within ``buffer_n`` slices of a collapse
                in the outward ``direction`` (+1 past the tail, -1 before
                the head) — i.e. it has no physical safety buffer between
                it and the module-edge transition."""
                if _slice_signal(idx) < aggressive_thr:
                    return True
                return any(
                    _collapsed(idx + direction * k)
                    for k in range(1, buffer_n + 1)
                )

            def _reclaimable(idx: int, direction: int) -> bool:
                """direction: +1 when reclaiming past the tail, -1 when
                reclaiming before the head — a candidate must satisfy the
                same edge-safety rule in its own outward direction, or
                the slide would just recreate the zero-buffer problem on
                the opposite end of the window."""
                if idx < 0 or idx >= n_total_slices:
                    return False
                if sections.slice_labels[idx] in other_section_names:
                    return False
                return not _unsafe_edge(idx, direction)

            trimmed_lead = 0
            extended_lead = 0
            while (trimmed_lead < guard_cap
                    and len(valid_slices) > central_end
                    and _unsafe_edge(valid_slices[0], -1)):
                bad = valid_slices.pop(0)
                sections.slice_labels[bad] = "TRANSITION"
                trimmed_lead += 1
                candidate = valid_slices[-1] + 1
                if _reclaimable(candidate, +1):
                    sections.slice_labels[candidate] = sec.name
                    valid_slices.append(candidate)
                    extended_lead += 1
            trimmed_tail = 0
            extended_tail = 0
            while (trimmed_tail < guard_cap
                    and len(valid_slices) > central_end
                    and _unsafe_edge(valid_slices[-1], +1)):
                bad = valid_slices.pop(-1)
                sections.slice_labels[bad] = "TRANSITION"
                trimmed_tail += 1
                candidate = valid_slices[0] - 1
                if _reclaimable(candidate, -1):
                    sections.slice_labels[candidate] = sec.name
                    valid_slices.insert(0, candidate)
                    extended_tail += 1
            if valid_slices:
                sec.start_idx = min(valid_slices)
                sec.end_idx = max(valid_slices) + 1
            if trimmed_lead + trimmed_tail > 0:
                log.info(
                    f"  {sec.name}: adaptive PVE guard trimmed "
                    f"{trimmed_lead} lead + {trimmed_tail} trail "
                    f"slice(s) (cap {guard_cap} = "
                    f"{config.STAR_ADAPTIVE_GUARD_MAX_MM:.1f} mm at "
                    f"dz={dz:.2f} mm) that were collapsed or sat within "
                    f"{config.STAR_GUARD_CLIFF_BUFFER_MM:.1f} mm of the "
                    f"module-edge signal collapse (no safety buffer); slid "
                    f"window by "
                    f"reclaiming {extended_lead + extended_tail} slice(s) "
                    f"from the opposite end to hold the target count "
                    f"(final: {len(valid_slices)} slices)"
                )

    # Removed the extra `valid_slices[1:-1]` safety trim: with the gate
    # filtering boundary slices and the 1-slice PVE pad above, this third
    # symmetric trim only erased clean STAR data the user expected to see
    # annotated (e.g. CalibraMicsOnly slice 48 visually showed STAR16
    # gratings but lost its ROIs to the cumulative trim stack).

    if not valid_slices:
        return

    # --- Pass 1: Detect texture positions across slices using global center ---
    tex_angles_collected = {name: [] for name in textures.values()}
    tex_dists_collected = {name: [] for name in textures.values()}

    for i in valid_slices:
        tex_positions = _detect_texture_blobs(
            dv.volume[i], gcx, gcy, texture_dist_px, textures, offset,
            texture_side_px=texture_side_px, phantom_r=g_phantom_r,
            tilt_deg=offset,
        )
        for tex_name, (tx, ty) in tex_positions.items():
            d = np.sqrt((tx - gcx) ** 2 + (ty - gcy) ** 2)
            a = float(np.degrees(np.arctan2(ty - gcy, tx - gcx))) % 360
            tex_dists_collected[tex_name].append(d)
            tex_angles_collected[tex_name].append(a)

    # --- Compute canonical texture positions (angles per texture, SHARED distance) ---
    # Angles: per-texture median of detected angles (reliable — asymmetric
    #         centroid bias mostly shifts radially, not tangentially).
    # Distance: median across ALL textures of this section. Averaging across
    #           4 textures cancels per-texture centroid bias (e.g. Diagonal
    #           pattern biases outward while Horizontal biases inward) and
    #           auto-calibrates to whichever phantom is being scanned
    #           (CalibraMics ~6.5 cm vs PhantomX Pro ~6.4 cm).
    all_detected_dists = [d for lst in tex_dists_collected.values() for d in lst]
    if all_detected_dists:
        measured_dist_px = float(np.median(all_detected_dists))
        measured_dist_cm = measured_dist_px * (dv.pixel_spacing[0] + dv.pixel_spacing[1]) / 20.0
        # Sanity-clip to +/- 15% of config value — if detection drifted wildly
        # (e.g. on low-contrast slices) we'd rather keep the nominal value.
        lo = texture_dist_px * 0.85
        hi = texture_dist_px * 1.15
        if lo <= measured_dist_px <= hi:
            canonical_dist_px = measured_dist_px
            log.info(f"  {sec.name} auto-calibrated texture distance: "
                     f"{measured_dist_px:.2f}px ({measured_dist_cm:.2f}cm) "
                     f"[config was {texture_dist_px:.2f}px]")
        else:
            canonical_dist_px = texture_dist_px
            log.debug(f"  {sec.name} measured texture distance {measured_dist_px:.2f}px "
                      f"out of [{lo:.2f},{hi:.2f}] — keeping config value")
    else:
        canonical_dist_px = texture_dist_px

    # --- Per-section anchor for the central star ---
    # The phantom is mechanically rigid: its x,y location does not change
    # along z. Re-detecting the centre per slice just chases Hough jitter
    # (±1-3 px). We use the per-section MEDIAN of valid Hough centres as
    # a stable anchor for the STAR ROI placement only. Textures use their
    # OWN detected absolute coordinates (see below) — using the section
    # anchor for textures would shift them by the (gcx, gcy) → (fx, fy)
    # drift, since the texture positions were detected relative to
    # (gcx, gcy) but would then be applied at (fx, fy). On scans with
    # significant phantom centre drift (Siemens NAEOTOM Alpha can show
    # 10+ px of drift across the full scan length) that 3-5 px shift is
    # exactly the "labels are way off" symptom the user reported.
    sec_cx, sec_cy = [], []
    for i in valid_slices:
        circ = phantom.circles[i]
        if circ.status == SliceStatus.VALID:
            sec_cx.append(circ.center_x)
            sec_cy.append(circ.center_y)
    if len(sec_cx) >= 3:
        fx = float(np.median(sec_cx))
        fy = float(np.median(sec_cy))
    else:
        fx, fy = gcx, gcy

    # --- Canonical texture positions: store ABSOLUTE (x, y) world coords ---
    # Previously stored as (angle_deg, dist_px) relative to (gcx, gcy)
    # then converted to (tx, ty) via fx, fy = per-section centre. That
    # was broken when fx ≠ gcx: textures got shifted by (fx-gcx, fy-gcy)
    # away from where the mean-image actually detected them.
    #
    # Now we keep absolute coordinates throughout. Initial seed: angle
    # = per-texture median of detected angles (absorbs rotation-offset
    # error + per-texture pattern bias), distance = shared median;
    # converted to (tx_world, ty_world) using GLOBAL centre. The
    # mean-image polish below then OVERWRITES this with the directly
    # detected world coordinates.
    tex_positions_fixed: Dict[str, Tuple[float, float, float]] = {}
    for nominal_angle, tex_name in textures.items():
        detected = tex_angles_collected.get(tex_name, [])
        if len(detected) >= 3:
            angs = np.deg2rad(detected)
            mean_x = np.median(np.cos(angs))
            mean_y = np.median(np.sin(angs))
            actual_angle = float(np.degrees(np.arctan2(mean_y, mean_x))) % 360
        else:
            actual_angle = (nominal_angle + offset) % 360
        ar = np.deg2rad(actual_angle)
        tx = gcx + canonical_dist_px * np.cos(ar)
        ty = gcy + canonical_dist_px * np.sin(ar)
        tex_positions_fixed[tex_name] = (float(tx), float(ty), actual_angle)

    log.info(f"  {sec.name} canonical positions: "
             f"global=({gcx:.1f},{gcy:.1f}), dist={canonical_dist_px:.2f}px, "
             f"{len(tex_positions_fixed)} textures, from {len(valid_slices)} slices")

    # --- Mean-image polish: detect textures on the section mean image ---
    # Random pixel noise drops by √N (N = number of slices, typically
    # 10-60). Stable phantom features (the outer rectangular boundary,
    # unchanging across z) add coherently. Per-slice detection artefacts
    # average out. Running detection on the mean image gives the cleanest
    # single estimate of each texture's (x, y).
    #
    # We use the polished position DIRECTLY as the world coordinate for
    # every slice in the section — Stage 2 edge refinement (peak-pair-
    # spacing) inside `_detect_texture_blobs` is robust to seed bias and
    # internal-pattern strength, so its output is the most accurate
    # position estimate we can produce for this rigid phantom.
    # Refined STAR centre — the texture-array centre (see config
    # STAR_ANCHOR_MAX_OFFSET_MM). Computed once here on the section mean
    # image and reused both as the texture frame anchor below AND for the
    # star ROI itself further down.
    refined_star: Optional[Tuple[float, float]] = None
    if len(valid_slices) >= 2:
        try:
            _star_mean = np.mean(
                dv.volume[list(valid_slices)], axis=0, dtype=np.float64
            ).astype(np.float32)
            refined_star = _refine_star_center(
                _star_mean, fx, fy, star_full_r_px,
                phantom_cx=gcx, phantom_cy=gcy, phantom_r=g_phantom_r,
            )
        except Exception as e:
            log.debug(f"  {sec.name} early star centre detect skipped: {e}")

    polish_count = 0
    if len(valid_slices) >= 2:
        try:
            mean_img = np.mean(
                dv.volume[list(valid_slices)], axis=0, dtype=np.float64
            ).astype(np.float32)
            mean_positions = _detect_texture_blobs(
                mean_img, gcx, gcy, canonical_dist_px,
                textures, offset, texture_side_px=texture_side_px,
                phantom_r=g_phantom_r, tilt_deg=offset,
            )
            polished_dists = []
            polished_xy: list[Tuple[float, float]] = []
            for tex_name, (mx, my) in mean_positions.items():
                # Verify the polish landed close to the per-slice canonical
                # — within ±10 px (~6 mm) is sane (the mean image and per-
                # slice cloud should never disagree by more than that on a
                # rigid phantom). If polish drifted further, suspect Stage 2
                # locked onto an adjacent feature and KEEP the per-slice
                # canonical for that texture.
                prev_tx, prev_ty, prev_angle = tex_positions_fixed[tex_name]
                if (mx - prev_tx) ** 2 + (my - prev_ty) ** 2 > 100.0:
                    continue
                new_angle = float(np.degrees(np.arctan2(my - gcy, mx - gcx))) % 360
                tex_positions_fixed[tex_name] = (float(mx), float(my), new_angle)
                polished_dists.append(
                    float(np.sqrt((mx - gcx) ** 2 + (my - gcy) ** 2))
                )
                polished_xy.append((float(mx), float(my)))
                polish_count += 1

            # FRAME ANCHOR for the textures — the texture-array centre.
            #
            # Critically, the anchor must NOT be validated against the phantom
            # WALL centre: on soft-kernel / large-FOV recons the thin shell is
            # sub-pixel and the wall centre is mislocalized by ~6 px. The old
            # code gated the (good) star centre against that (bad) wall centre
            # — when the star sat just past STAR_ANCHOR_MAX_OFFSET_MM from the
            # mislocalized wall it was REJECTED and the code fell back to the
            # wall itself, translating ALL FOUR textures by the wall's ~6 px
            # error while the star ROI (which uses refined_star directly) stayed
            # correctly centred. That is exactly the "star centred, textures
            # 6 px low" failure seen on the SEP Br36f 500 mm series.
            #
            # Instead, anchor to the TEXTURE CONSENSUS — the centroid of the 4
            # polished texture positions, which IS the array's own centre and
            # averages out per-pattern centroid bias — and use the refined STAR
            # centre only to VALIDATE it (the two are physically concentric, so
            # they must agree to within the star gate). Priority:
            #   1. texture consensus, if it agrees with the star centre (or the
            #      star is unavailable) — lowest texture error, self-referential.
            #   2. refined star centre, when there is no 4-texture consensus.
            #   3. wall centre — only when star & consensus disagree (one
            #      detection is bad) or nothing else is available.
            frame_cx, frame_cy = gcx, gcy
            _avg_sp = (dv.pixel_spacing[0] + dv.pixel_spacing[1]) / 2.0
            _star_gate_px = config.STAR_ANCHOR_MAX_OFFSET_MM / _avg_sp
            tex_consensus = None
            if len(polished_xy) >= 4:
                tex_consensus = (
                    float(np.mean([p[0] for p in polished_xy])),
                    float(np.mean([p[1] for p in polished_xy])),
                )
            if tex_consensus is not None and refined_star is not None:
                if ((refined_star[0] - tex_consensus[0]) ** 2 +
                        (refined_star[1] - tex_consensus[1]) ** 2
                        <= _star_gate_px ** 2):
                    frame_cx, frame_cy = tex_consensus       # agree → array centre
                # else: disagree → keep wall (conservative; one detection bad)
            elif tex_consensus is not None:
                frame_cx, frame_cy = tex_consensus
            elif refined_star is not None and (
                (refined_star[0] - gcx) ** 2 + (refined_star[1] - gcy) ** 2
                <= _star_gate_px ** 2
            ):
                frame_cx, frame_cy = float(refined_star[0]), float(refined_star[1])

            # Place the 4 textures as a RIGID, GEOMETRIC frame on the star.
            #
            # The 4 texture squares are one rigid module concentric with the
            # central star, so their true centres lie at EQUAL radius and
            # EXACT 90°-spaced angles (the cardinals + the phantom's rotation).
            # Per-texture blob/centroid detection carries a pattern-specific
            # bias — the diagonal grating's intensity centroid, say, sits off
            # the square's geometric centre — which left some textures 2-4 px
            # off even though the star centre itself localises to <0.5 px
            # (measured: Diagonal/Vertical ~3 px, Square ~2 px on a clean PCCT
            # STAR16). So we DISCARD each texture's individually-detected angle
            # and rebuild all four geometrically from the (accurate) star-frame
            # anchor:
            #   * radius = shared median of the polished distances — rigid
            #              module → one radius for all four; the 4-texture
            #              median cancels residual per-pattern radial bias.
            #   * angle  = nominal cardinal + global rotation offset — rigid
            #              module → exact 90° spacing. `offset` is a robust
            #              circular mean over every slice×texture (Phase 3), so
            #              it carries far less variance than any single
            #              texture's detected angle.
            # Net: the texture-array geometry the phantom physically has,
            # centred on the star, with no per-pattern centroid drift.
            name_to_nominal = {name: nom for nom, name in textures.items()}
            if polished_dists:
                shared_dist = float(np.median(polished_dists))
                for tex_name in list(tex_positions_fixed.keys()):
                    nominal = name_to_nominal.get(tex_name)
                    if nominal is None:
                        continue
                    ang_geo = (nominal + offset) % 360.0
                    ar = np.deg2rad(ang_geo)
                    tx_new = frame_cx + shared_dist * np.cos(ar)
                    ty_new = frame_cy + shared_dist * np.sin(ar)
                    tex_positions_fixed[tex_name] = (
                        float(tx_new), float(ty_new), float(ang_geo)
                    )
                anchor_note = (
                    f", anchor=({frame_cx:.2f},{frame_cy:.2f})"
                    f" [shifted ({frame_cx-gcx:+.2f},{frame_cy-gcy:+.2f}) px"
                    f" from wall centre]"
                )
                log.info(
                    f"  {sec.name} mean-image texture polish: "
                    f"refined {polish_count}/{len(textures)} textures, "
                    f"shared distance updated to {shared_dist:.2f}px"
                    f"{anchor_note}"
                )
        except Exception as e:
            log.debug(
                f"  {sec.name} mean-image texture polish skipped: {e}"
            )

    # --- STAR centre for the star ROI ---
    # Reuse the star centre already detected above (it anchors the textures
    # too). The star module isn't guaranteed co-axial with the phantom outer
    # wall, so this refined centre — not the wall centre — is used for the
    # star ROI as well.
    star_world_polished: Optional[Tuple[float, float]] = None
    if refined_star is not None:
        star_world_polished = refined_star
        log.info(
            f"  {sec.name} star centre: "
            f"({refined_star[0]:.2f}, {refined_star[1]:.2f}) "
            f"[shifted ({refined_star[0]-fx:+.2f}, "
            f"{refined_star[1]-fy:+.2f}) px from Hough centre; "
            f"also anchors the 4 textures]"
        )

    log.debug(f"  {sec.name} fixed 3D centre for all slices: ({fx:.2f},{fy:.2f}) "
              f"[from {len(sec_cx)} slice detections]")

    # --- Smoothed per-slice phantom anchor ---
    # The polish placed each texture at an ABSOLUTE world coordinate
    # that corresponds to the MEAN phantom position across the slices
    # that built the mean image. If the phantom drifted significantly
    # within the section, every individual slice's true texture position
    # differs from the mean. We want to track that drift WITHOUT
    # inheriting the per-slice Hough quantization noise (Hough returns
    # centres rounded to ~1.2 px grid steps, so raw per-slice centres
    # wobble visually even when the true phantom position is steady).
    #
    # Strategy:
    #   1. Collect all VALID per-slice Hough centres in the section.
    #   2. Robust linear fit cx(z), cy(z) using slice index as z.
    #   3. Compare the fit slope × section span (= total drift) to the
    #      raw scatter (= noise floor):
    #         * drift < 1.5 × noise → no real drift, use CONSTANT
    #           section median for every slice (zero fluctuation).
    #         * drift ≥ 1.5 × noise → use the LINEAR FIT value at each
    #           slice (smooth tracking, still zero fluctuation slice-
    #           to-slice because the function is differentiable).
    #   4. Polish anchor = smooth function evaluated at the mid-section
    #      slice (= effective phantom position the mean image samples).
    #
    # This eliminates the quantization-noise wobble seen with raw per-
    # slice Hough centres while preserving the ability to track real
    # drift on scans that have it.
    valid_idxs: List[int] = []
    valid_cxs: List[float] = []
    valid_cys: List[float] = []
    for i in valid_slices:
        c = phantom.circles[i]
        if c.status == SliceStatus.VALID:
            valid_idxs.append(i)
            valid_cxs.append(float(c.center_x))
            valid_cys.append(float(c.center_y))

    cx_const = fx
    cy_const = fy
    cx_coef = (0.0, fx)
    cy_coef = (0.0, fy)
    use_linear_x = False
    use_linear_y = False
    mid_idx = (valid_slices[0] + valid_slices[-1]) / 2.0

    if len(valid_idxs) >= 5:
        cx_arr = np.asarray(valid_cxs, dtype=np.float64)
        cy_arr = np.asarray(valid_cys, dtype=np.float64)
        cx_const = float(np.median(cx_arr))
        cy_const = float(np.median(cy_arr))
        # NOTE: linear-fit mode disabled (2026-05-18). On real PCCT
        # data across all tested protocols, real phantom drift across
        # a 30-slice STAR section is sub-pixel, but the linear-mode
        # smooth slide was visible to the operator as "fluctuation"
        # in the overlays (std 0.6-0.9 px across the section). Constant
        # mode placed at the section median is always at worst ~1 px
        # off on the most-drifted edge slice — invisible in practice.
        # Keep `use_linear_x`/`use_linear_y` permanently False; if a
        # future scan has truly large drift (>3 px across section) the
        # ROIs will track the section median, which the operator can
        # see immediately and re-acquire if needed.
    log.debug(
        f"  {sec.name} per-slice anchor: "
        f"x={'linear' if use_linear_x else 'const'} ({cx_const:.2f}), "
        f"y={'linear' if use_linear_y else 'const'} ({cy_const:.2f})"
    )

    def _slice_centre(i: int) -> Tuple[float, float]:
        cx = (float(np.polyval(cx_coef, i)) if use_linear_x else cx_const)
        cy = (float(np.polyval(cy_coef, i)) if use_linear_y else cy_const)
        return (cx, cy)

    # Polish anchor = smooth function at the mid-section slice
    # (= effective phantom position the mean image samples).
    polish_anchor_x, polish_anchor_y = _slice_centre(mid_idx)

    # Texture OFFSETS relative to the polish anchor (rigid geometry
    # that moves with the phantom).
    tex_offsets: Dict[str, Tuple[float, float, float]] = {}
    for tex_name, (tx, ty, angle_deg) in tex_positions_fixed.items():
        tex_offsets[tex_name] = (
            tx - polish_anchor_x,
            ty - polish_anchor_y,
            angle_deg,
        )

    # Star OFFSET — same logic as textures. If star polish succeeded,
    # use the polished position; otherwise fall back to "star sits AT
    # the phantom Hough centre" (offset 0,0 from the anchor).
    if star_world_polished is not None:
        star_offset_x = star_world_polished[0] - polish_anchor_x
        star_offset_y = star_world_polished[1] - polish_anchor_y
    else:
        star_offset_x = 0.0
        star_offset_y = 0.0

    # --- Pass 2: Extract ROIs at per-slice anchored positions ---
    for i in valid_slices:
        slice_rois = []
        slice_cx, slice_cy = _slice_centre(i)
        star_cx = slice_cx + star_offset_x
        star_cy = slice_cy + star_offset_y

        star_pixels = _extract_nearest_neighbor_circle(
            dv.volume[i], star_cx, star_cy, star_r_px
        )
        # Section-specific star label so overlays distinguish "Star8"
        # from "Star16" at a glance. Section name is "STAR8" / "STAR16";
        # convert to "Star8" / "Star16" for the human-readable label.
        star_label = sec.name.replace("STAR", "Star")
        slice_rois.append(ROI(
            label=star_label, section=sec.name, slice_idx=i,
            center_x=star_cx, center_y=star_cy, pixels=star_pixels,
            shape_type="circle", size_cm=config.STAR_DIAMETER_CM, angle_deg=0.0,
        ))

        for tex_name, (dx, dy, angle_deg) in tex_offsets.items():
            tx = slice_cx + dx
            ty = slice_cy + dy
            tex_pixels = _extract_nearest_neighbor_square(
                dv.volume[i], tx, ty, texture_half_side_px
            )
            slice_rois.append(ROI(
                label=tex_name, section=sec.name, slice_idx=i,
                center_x=tx, center_y=ty, pixels=tex_pixels,
                shape_type="square", size_cm=config.TEXTURE_SIDE_CM,
                angle_deg=angle_deg,
                shape_rotation_deg=offset,
            ))

        result.rois.extend(slice_rois)
        result.per_slice[i] = slice_rois


def _detect_texture_blobs(
    hu_slice: np.ndarray,
    phantom_cx: float,
    phantom_cy: float,
    expected_dist_px: float,
    textures: Dict[int, str],
    offset: float,
    texture_side_px: Optional[float] = None,
    phantom_r: Optional[float] = None,
    tilt_deg: float = 0.0,
) -> Dict[str, Tuple[float, float]]:
    """Detect texture centres via two-stage refinement.

    **Stage 1 (gross)**: threshold the annular band around the expected
    texture radius, find connected components, take each CC's BOUNDING-
    BOX centre (not its intensity-weighted centroid — the centroid is
    biased toward whichever side of a texture has more bright pixels,
    e.g. one corner of a diagonal-stripe pattern).

    **Stage 2 (precise)**: for the CC matched to each expected texture
    angle, locate the texture's outer rectangular boundary by finding
    the strongest Sobel-gradient peaks on each side of the seed. The
    midpoint of those four outer edges is the geometric centre of the
    texture square, free of internal-pattern bias. Falls back to the
    Stage 1 bounding-box centre when edges can't be located reliably.

    Returns dict of tex_name -> (x, y) for detected textures.
    """
    import cv2

    h, w = hu_slice.shape
    # Adaptive threshold for texture regions (they are high-contrast)
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - phantom_cx) ** 2 + (yy - phantom_cy) ** 2)

    # Annular mask around expected texture distance
    tol = expected_dist_px * 0.4
    annular = (dist > expected_dist_px - tol) & (dist < expected_dist_px + tol)

    bg_values = hu_slice[annular]
    if bg_values.size == 0:
        return {}

    bg_median = np.median(bg_values)
    bg_std = np.std(bg_values)
    threshold = max(bg_median + 2 * bg_std, bg_median + 80)

    binary = (hu_slice > threshold).astype(np.uint8) * 255
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[annular] = 255
    binary = cv2.bitwise_and(binary, mask)

    # Closing kernel: large enough to bridge gaps in sparse texture
    # patterns (STAR8 Honeylarge has 5-7 mm honeycomb cells with thin
    # walls — a 5x5 kernel left it as dozens of disconnected CCs and
    # Stage 1 couldn't match it to a single blob). The 11x11 kernel
    # merges the honeycomb dots into a clean ~70x70 CC for every
    # texture type (verified on STAR8 Honeylarge, Honeysmall,
    # Honeymedium, Sinusoidal and STAR16 Diagonal, Vertical, Square,
    # Horizontal). Adjacent textures sit ~155 px apart so they don't
    # accidentally merge.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Collect blob BOUNDING-BOX centres (more robust than intensity-weighted
    # centroids for textures with asymmetric internal patterns — diagonal
    # stripes, vertical bars, etc. — because the bbox depends only on
    # which pixels are bright, not on how many are clustered to one side).
    blobs: List[Tuple[float, float]] = []
    min_area = 100
    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        bb_x = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2.0
        bb_y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
        blobs.append((float(bb_x), float(bb_y)))

    if not blobs:
        return {}

    # Match each expected texture angle to nearest blob, then refine.
    result = {}
    for nominal_angle, tex_name in textures.items():
        actual_angle = nominal_angle + offset
        angle_rad = np.deg2rad(actual_angle)
        expected_x = phantom_cx + expected_dist_px * np.cos(angle_rad)
        expected_y = phantom_cy + expected_dist_px * np.sin(angle_rad)

        # Find closest blob
        best_dist = float('inf')
        best_blob = None
        for bx, by in blobs:
            d = np.sqrt((bx - expected_x) ** 2 + (by - expected_y) ** 2)
            if d < best_dist:
                best_dist = d
                best_blob = (bx, by)

        # Accept if within reasonable distance
        if best_blob is None or best_dist >= expected_dist_px * 0.3:
            continue

        # Stage 2: edge-based rectangle refinement (when we know the
        # texture's nominal physical side length). Pass phantom info so
        # the Sobel detector can mask out the phantom wall (a ~1000 HU
        # jump that would otherwise dominate the gradient profile and
        # form spurious frame pairs with interior pattern peaks).
        refined = best_blob
        if texture_side_px is not None and texture_side_px > 0:
            refined_xy = _refine_texture_center_via_edges(
                hu_slice, best_blob[0], best_blob[1], texture_side_px,
                phantom_cx=phantom_cx, phantom_cy=phantom_cy,
                phantom_r=phantom_r,
                tilt_deg=tilt_deg,
            )
            if refined_xy is not None:
                refined = refined_xy
        result[tex_name] = refined

    return result


def _find_local_maxima(
    profile: np.ndarray, threshold_frac: float = 0.20
) -> List[int]:
    """Return indices of local maxima above ``threshold_frac × max(profile)``.

    Plateaus are reduced to their leftmost qualifying index (a peak that
    is equal to its neighbour on one side and strictly greater on the
    other is treated as the peak; pure flats are skipped). Endpoints
    are excluded.
    """
    n = profile.size
    if n < 3:
        return []
    pmax = float(profile.max())
    if pmax <= 0:
        return []
    threshold = pmax * threshold_frac
    peaks: List[int] = []
    for i in range(1, n - 1):
        v = profile[i]
        if v < threshold:
            continue
        l = profile[i - 1]
        r = profile[i + 1]
        # Strict on at least one side prevents plateaus from being
        # counted multiple times. Equal-on-both means flat → skip.
        if v >= l and v >= r and (v > l or v > r):
            peaks.append(i)
    return peaks


def _find_frame_edge_pair(
    profile: np.ndarray,
    expected_separation: float,
    spacing_tol_frac: float = 0.18,
    threshold_frac: float = 0.20,
) -> Optional[Tuple[int, int]]:
    """Find the pair of local maxima whose separation best matches the
    expected outer-frame width of a texture rectangle.

    Why this works
    --------------
    A texture square has exactly ONE pair of edges separated by
    ``texture_side_px`` — its outer frame. Internal patterns produce
    additional gradient peaks, but at SMALLER spacings (the pattern's
    own period: dot diameter, stripe pitch, etc.) — they cannot form a
    pair at exactly the frame width. By scoring all (i, j) peak pairs
    on spacing accuracy, we recover the frame edges regardless of:

      * **Seed bias** — Stage 1's bbox-centroid seed can be 5-10 px off
        toward whichever quadrant of the texture has the densest
        pattern (e.g. STAR8 honeycomb biases inward toward the central
        star). Narrow-band argmax around the biased seed misses the
        true frame edge; the spacing-pair detector doesn't depend on
        the seed at all (only on the geometry of the gradient peaks).
      * **Internal-pattern strength** — even when a sinusoidal or
        dense-honeycomb pattern produces gradient peaks STRONGER than
        the frame edge, those peaks form intra-pattern pairs of small
        spacing (≪ texture_side_px). Only the frame pair has the
        correct spacing, so it wins by spacing accuracy.
      * **Far-edge noise** — random peaks at the crop boundary won't
        form a pair at exactly ``texture_side_px`` separation with any
        other peak unless they happen to align with the true frame
        position, which is extremely improbable.

    Scoring: spacing error (primary, weight 1.0) − combined peak
    strength (secondary, weight 0.3). Lower score wins. Strength
    tie-breaks pairs with equal spacing — the frame is also typically
    the strongest contiguous gradient line, so this prefers it over
    accidental noise pairs.

    Returns ``(left_idx, right_idx)`` or ``None`` if no pair within
    ±``spacing_tol_frac × expected_separation`` exists.
    """
    peaks = _find_local_maxima(profile, threshold_frac=threshold_frac)
    if len(peaks) < 2:
        return None
    pmax = float(profile.max())
    sep_tol = expected_separation * spacing_tol_frac
    best_pair: Optional[Tuple[int, int]] = None
    best_score = float("inf")
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            sep = peaks[j] - peaks[i]
            err = abs(sep - expected_separation)
            if err > sep_tol:
                continue
            spacing_err = err / expected_separation
            strength = (
                float(profile[peaks[i]]) + float(profile[peaks[j]])
            ) / (2.0 * pmax)
            score = spacing_err - 0.3 * strength
            if score < best_score:
                best_score = score
                best_pair = (peaks[i], peaks[j])
    return best_pair


def _refine_star_center(
    mean_img: np.ndarray,
    seed_x: float, seed_y: float,
    star_r_px: float,
    phantom_cx: Optional[float] = None,
    phantom_cy: Optional[float] = None,
    phantom_r: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Refine the central star pattern's centre on the section mean image.

    Mirrors the philosophy of the insert / texture polish: detect on the
    mean image (noise ↓ √N) and use a feature-specific refinement.

    Why earlier approaches failed
    -----------------------------
    Attempt 1 (median-bg contrast centroid): the air gaps make the crop
    median ≈ −750 HU, so plastic at HU 0 has contrast 750 — almost as
    high as the star spokes. Plastic OUTSIDE the star dominated and
    pulled the centroid away from the true centre.
    Attempt 2 (any |HU − 0| > 200): caught both air gaps inside the
    star AND background air OUTSIDE the star. When the seed was offset
    from the true star centre, the outside-air ring captured by the
    seed-centred mask became asymmetric (more outside-air on the side
    where the star edge fell INSIDE the mask, none on the side where
    the star edge fell at/past the mask edge). The centroid was
    therefore pulled AWAY from the true star centre.

    Algorithm (this implementation)
    -------------------------------
    Use BRIGHT SPOKES ONLY. The water-equivalent matrix is at HU ≈ 0
    and surrounding air at HU ≈ −1000 — both well below the spoke
    material. The bright spokes are always the most intense structure
    within ~star_r_px of the seed (textures are at radius ≈ 109 px,
    outside the crop), so an ADAPTIVE p90 threshold cleanly isolates
    them regardless of:
      * scanner HU calibration drift
      * VMI keV (40 / 70 / 120 / 190 keV all produce different spoke
        HU but the spokes remain the brightest pixels in the crop)
      * different reconstruction kernels (Br40f vs Qr40f vs Br68)
      * different phantom designs with different star materials

    A safety floor (HU > 0) ensures we never select pixels that are
    DARKER than water-matrix even when the crop contains no real
    bright spokes (e.g. on a calcium-suppression map where the spokes
    are subtracted out). In that case the polish returns None and the
    Star ROI falls back to the per-section Hough centre.

    Centroid of "bright spokes" = star geometric centre, regardless
    of the seed offset, because:
      * no plastic / outside air can ever be selected (it falls below
        the bright threshold);
      * the spokes are radially symmetric and fill the star disk;
      * a 3-pass iteration re-centres the mask on the new estimate
        and converges within 2-3 iterations.

    Returns ``None`` when too few bright pixels are found (low
    contrast / mis-detected star section) or when the refined centre
    would move more than ½ × star_r_px from the seed (sanity guard).
    """
    h, w = mean_img.shape
    search_half = int(max(24, np.ceil(star_r_px * 1.6)))
    ix, iy = int(round(seed_x)), int(round(seed_y))
    x0 = max(0, ix - search_half)
    y0 = max(0, iy - search_half)
    x1 = min(w, ix + search_half + 1)
    y1 = min(h, iy + search_half + 1)
    crop = mean_img[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return None

    ch, cw = crop.shape
    yy, xx = np.indices((ch, cw))

    # Adaptive bright-only threshold. The 90th percentile of the crop
    # picks roughly the top 10 % of pixels — the bright-spoke material
    # in any reasonable phantom design. This works for any HU
    # calibration and any VMI keV because the spokes are always the
    # densest structure in the small centre crop (textures are at
    # radius ≈ 109 px, well outside the ~55 px search window).
    # A safety floor of 0 HU rejects degenerate cases where the
    # phantom's central region has no bright structure at all (e.g.
    # calcium-suppressed material map).
    adaptive_threshold = float(np.percentile(crop, 90))
    bright_threshold = max(adaptive_threshold, 0.0)
    bright_pixels = crop > bright_threshold

    # 3-pass refinement so the mask can recentre and converge even
    # when the seed was offset by more than half the spoke length.
    cx_local = seed_x - x0
    cy_local = seed_y - y0
    refined_cx = refined_cy = None
    for _ in range(3):
        rr = np.sqrt((xx - cx_local) ** 2 + (yy - cy_local) ** 2)
        # Mask radius slightly larger than full star, so the outermost
        # spoke tips are not truncated on any side even if the current
        # centre estimate is still 2-3 px off. Bright-only thresholding
        # means there's nothing to capture OUTSIDE the star, so a wider
        # mask carries no bias risk.
        within = rr <= star_r_px * 1.3
        selected = bright_pixels & within
        n_pix = int(selected.sum())
        if n_pix < 30:
            return None
        cx_local = float(xx[selected].mean())
        cy_local = float(yy[selected].mean())
        refined_cx = cx_local + x0
        refined_cy = cy_local + y0

    if refined_cx is None:
        return None

    # --- Stage 2: point-symmetry refinement (2026-05-18) ---
    # The bright-spoke centroid is good to ~1 px but biased by spoke
    # intensity asymmetries (some spokes appear brighter than others
    # due to scanner / kernel asymmetries). The star pattern is RADIALLY
    # SYMMETRIC by design: at the true centre, I(cx+dx, cy+dy) =
    # I(cx-dx, cy-dy) for all (dx, dy). Find the (cx, cy) that minimises
    # the total |I(+v) - I(-v)| over a ring of sample points.
    # This converges to sub-pixel accuracy independent of intensity
    # asymmetry, locking onto the true convergence point of the spokes.
    seed_cx = refined_cx
    seed_cy = refined_cy
    h_full, w_full = mean_img.shape
    # Sample a ring of (r, θ) points inside the star disc
    thetas = np.linspace(0.0, np.pi, 24, endpoint=False)
    radii = np.array([0.30, 0.50, 0.70, 0.85]) * star_r_px
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    best_asym = float('inf')
    best_cx, best_cy = seed_cx, seed_cy
    # Coarse search ±3 px @ 0.5 px, then fine ±0.75 px @ 0.1 px
    for coarse_step in (0.5, 0.1):
        coarse_range = 3.0 if coarse_step == 0.5 else 0.75
        steps = np.arange(-coarse_range, coarse_range + coarse_step * 0.5, coarse_step)
        # Snapshot the pass-start centre so iterations within this pass
        # don't drift cumulatively when a new minimum is found.
        pass_cx, pass_cy = best_cx, best_cy
        # Reset the asymmetry score so the fine pass selects the best
        # point in its own neighbourhood (not stuck on the coarse winner).
        best_asym = float('inf')
        for dx in steps:
            for dy in steps:
                cx_try = pass_cx + dx
                cy_try = pass_cy + dy
                # Bounds-check the outermost sample
                rmax = float(radii.max())
                if (cx_try - rmax < 1 or cx_try + rmax >= w_full - 1
                        or cy_try - rmax < 1 or cy_try + rmax >= h_full - 1):
                    continue
                total = 0.0
                for r in radii:
                    xs_plus = cx_try + r * cos_t
                    ys_plus = cy_try + r * sin_t
                    xs_minus = cx_try - r * cos_t
                    ys_minus = cy_try - r * sin_t
                    # Bilinear interpolation
                    def bilin(xs, ys, arr=mean_img):
                        x0i = np.floor(xs).astype(int)
                        y0i = np.floor(ys).astype(int)
                        fx = xs - x0i
                        fy = ys - y0i
                        return (arr[y0i, x0i] * (1 - fx) * (1 - fy)
                                + arr[y0i, x0i + 1] * fx * (1 - fy)
                                + arr[y0i + 1, x0i] * (1 - fx) * fy
                                + arr[y0i + 1, x0i + 1] * fx * fy)
                    vp = bilin(xs_plus, ys_plus)
                    vm = bilin(xs_minus, ys_minus)
                    total += float(np.abs(vp - vm).sum())
                if total < best_asym:
                    best_asym = total
                    best_cx, best_cy = cx_try, cy_try
        # Lock in the coarse result before fine search
    refined_cx, refined_cy = best_cx, best_cy

    # Sanity guard: refuse to drift more than ½ × star_r_px from seed.
    max_shift = 0.5 * star_r_px
    if (abs(refined_cx - seed_x) > max_shift
            or abs(refined_cy - seed_y) > max_shift):
        return None
    return refined_cx, refined_cy


def _refine_texture_center_via_edges(
    hu_slice: np.ndarray,
    cx_approx: float,
    cy_approx: float,
    texture_side_px: float,
    window_scale: float = 1.6,
    phantom_cx: Optional[float] = None,
    phantom_cy: Optional[float] = None,
    phantom_r: Optional[float] = None,
    tilt_deg: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Refine a texture centre via Sobel-gradient rectangle detection
    using a SEED-INDEPENDENT peak-pair-spacing approach with optional
    phantom-wall masking.

    The texture's outer boundary is a sharp HU transition between the
    texture material and the surrounding phantom matrix. The HU jump
    at the texture frame is a few hundred HU; the jump at the phantom
    OUTER WALL (plastic → air) is ~1000 HU — five to ten times stronger.
    When the analysis crop contains the phantom wall, the wall peak
    dominates the Sobel profile (100 % of max) and forms a (false)
    frame pair with some interior pattern peak that happens to lie at
    ~texture_side_px from the wall. This is the dominant failure mode
    on the STAR8 outer textures (Honeysmall, Sinusoidal): they sit
    only ~55 px from the phantom wall, so any crop that extends >55 px
    radially-outward from the texture centre includes the wall.

    Defences (in order of importance):

      1. **Phantom-wall masking** (this commit) — when phantom_cx /
         phantom_cy / phantom_r are provided, zero out col_profile and
         row_profile entries whose pixel position is within 5 px of
         the wall. The wall peak vanishes, and the peak-pair detector
         is left with only legitimate texture features.
      2. **Tight crop** (window_scale = 1.6) — keeps adjacent textures
         (~155 px away, both directions) safely out of the crop while
         still giving the peak-pair detector ±0.3 × texture_side of
         headroom for seed-bias tolerance.
      3. **Peak-pair-spacing scoring** (`_find_frame_edge_pair`) — only
         pairs whose separation matches texture_side_px to within ±18 %
         are even considered; internal pattern pairs (separation ≪
         frame width) are excluded entirely.

    Returns ``None`` when no valid peak pair is found in either
    direction, when the detected box differs from expected by >25 %,
    or when the centre would move >0.7 × half-side from the seed
    (sanity guard against runaway).
    """
    import cv2

    h, w = hu_slice.shape
    half_side = texture_side_px / 2.0
    half_win = int(np.ceil(texture_side_px * window_scale / 2.0))

    ix = int(round(cx_approx))
    iy = int(round(cy_approx))
    x0 = max(0, ix - half_win)
    y0 = max(0, iy - half_win)
    x1 = min(w, ix + half_win + 1)
    y1 = min(h, iy + half_win + 1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None

    crop = hu_slice[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return None

    # --- Tilt correction ---
    # When the phantom is rotated by tilt_deg, the texture squares
    # rotate with it. Their frame edges are then NOT aligned with the
    # image X/Y axes — Sobel gradient sums along axis-aligned columns
    # and rows give SMEARED, WEAKER peaks (a 5° tilt smears a 66 px
    # frame across 6 px horizontally). The peak-pair detector picks
    # whatever ridge happens to be tallest, biasing the centre by
    # several pixels.
    # Fix: rotate the crop by -tilt_deg around the seed so the
    # texture's frame becomes axis-aligned in the rotated crop. Then
    # the existing Sobel + peak-pair detector works at full accuracy.
    # After detection, we rotate the refined offset BACK by +tilt_deg
    # to recover the position in original image coordinates.
    #
    # The seed (cx_approx, cy_approx) maps to the same pixel in both
    # frames (it's the rotation centre). Only the OFFSET from the
    # seed needs to be back-rotated.
    seed_local_x = cx_approx - x0
    seed_local_y = cy_approx - y0
    do_tilt = abs(tilt_deg) > 0.3  # skip for negligible tilt
    if do_tilt:
        # OpenCV's getRotationMatrix2D with positive angle rotates the
        # image counter-clockwise visually (y-down convention). The
        # rotation_detector measures the phantom tilt as positive when
        # textures appear CLOCKWISE-shifted from their nominal angles
        # (because it uses arctan2(dy, dx) which increases CW in
        # image coords). So to UN-rotate (axis-align the texture
        # frame), we apply +tilt_deg to the OpenCV rotation matrix.
        # Verified empirically with diagnostic/tilt_direction_test.py:
        # +tilt gives peak concentration ~0.65, -tilt gives ~0.05.
        rot_M = cv2.getRotationMatrix2D(
            (float(seed_local_x), float(seed_local_y)),
            float(tilt_deg), 1.0,
        )
        crop = cv2.warpAffine(
            crop, rot_M,
            (crop.shape[1], crop.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    crop_blur = cv2.GaussianBlur(crop, (3, 3), 0.6)

    gx = cv2.Sobel(crop_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_x = np.abs(gx)
    grad_y = np.abs(gy)

    # --- Phantom-wall masking ---
    # The wall is a ~1000 HU step (plastic → air) blurred by PVE over
    # 2-6 px inside the nominal wall radius. Those gradient peaks
    # otherwise pair with interior pattern peaks to form spurious
    # frame pairs at the right ~66 px spacing.
    # We compute a 2D pixel-level mask of "outside safe phantom
    # interior" and zero those pixels in grad_x / grad_y BEFORE
    # summing into the col/row profiles. This works correctly both
    # for the un-rotated case and the tilt-corrected case (the
    # rotation maps each rotated-crop pixel to a unique image
    # location, and we test that image location against the wall).
    if (phantom_cx is not None and phantom_cy is not None
            and phantom_r is not None and phantom_r > 0):
        wall_margin = max(10.0, texture_side_px * 0.10)
        safe_r = phantom_r - wall_margin
        ch, cw = crop.shape
        # Per-pixel image coordinates: for each (col, row) in the
        # crop (possibly rotated), find the corresponding image
        # (x, y). When do_tilt is False, this is just (x0+col,
        # y0+row). When do_tilt is True, we back-rotate the offset
        # from the seed by +tilt_deg (undoing the warp that produced
        # the rotated crop) to recover the source image position.
        cols = np.arange(cw, dtype=np.float32)
        rows = np.arange(ch, dtype=np.float32)
        cols_grid, rows_grid = np.meshgrid(cols, rows)
        if do_tilt:
            # We rotated the crop by +tilt_deg CCW around the seed.
            # To go from rotated-crop pixel (col, row) BACK to the
            # original image source pixel, undo by rotating by
            # -tilt_deg CCW = +tilt_deg CW around the seed.
            # cv2's standard rotation matrix R(θ) maps (x, y) →
            # (x·cos θ - y·sin θ, x·sin θ + y·cos θ) and rotates the
            # POINT counter-clockwise in image coords. For inverse:
            dx = cols_grid - seed_local_x
            dy = rows_grid - seed_local_y
            ar = np.deg2rad(tilt_deg)
            # Inverse rotation: R(-tilt) applied to (dx, dy)
            src_dx = dx * np.cos(ar) + dy * np.sin(ar)
            src_dy = -dx * np.sin(ar) + dy * np.cos(ar)
            img_x = x0 + seed_local_x + src_dx
            img_y = y0 + seed_local_y + src_dy
        else:
            img_x = x0 + cols_grid
            img_y = y0 + rows_grid
        pixel_r = np.sqrt(
            (img_x - phantom_cx) ** 2 + (img_y - phantom_cy) ** 2
        )
        outside_mask = pixel_r > safe_r
        if outside_mask.any():
            grad_x = np.where(outside_mask, 0.0, grad_x)
            grad_y = np.where(outside_mask, 0.0, grad_y)

    col_profile = grad_x.sum(axis=0)
    row_profile = grad_y.sum(axis=1)

    col_pair = _find_frame_edge_pair(col_profile, texture_side_px)
    row_pair = _find_frame_edge_pair(row_profile, texture_side_px)
    if col_pair is None or row_pair is None:
        return None
    li, ri = col_pair
    ti, bi = row_pair

    # Sub-pixel refinement: use a WEIGHTED CENTROID over the local
    # gradient region rather than a single-peak parabolic fit. PVE
    # blur frequently splits a single physical edge into two adjacent
    # gradient peaks separated by 2-3 px (e.g. STAR8 Honeylarge slice
    # 446 in series 2 showed peaks at idx 19+22 for the same left
    # frame edge); single-peak parabolic fit picks one of them and
    # the result jumps by 3 px slice-to-slice depending on which peak
    # happened to be tallest. The weighted centroid in a ±4 px window
    # combines both peaks and gives a stable sub-pixel position
    # regardless of which one wins by tiebreaker. The window is
    # narrow enough to not pull from distant peaks.
    left_x = _edge_subpixel_centroid(col_profile, li)
    right_x = _edge_subpixel_centroid(col_profile, ri)
    top_y = _edge_subpixel_centroid(row_profile, ti)
    bot_y = _edge_subpixel_centroid(row_profile, bi)

    detected_w = right_x - left_x
    detected_h = bot_y - top_y
    if abs(detected_w - texture_side_px) > texture_side_px * 0.25:
        return None
    if abs(detected_h - texture_side_px) > texture_side_px * 0.25:
        return None

    # Detected center is in the (possibly rotated) crop coords. Convert
    # to image coords. For the rotated case the offset from the seed
    # must be back-rotated by +tilt_deg to undo the warp.
    cx_in_crop = (left_x + right_x) / 2.0
    cy_in_crop = (top_y + bot_y) / 2.0
    if do_tilt:
        dx_rot = cx_in_crop - seed_local_x
        dy_rot = cy_in_crop - seed_local_y
        ar = np.deg2rad(tilt_deg)
        dx = dx_rot * np.cos(ar) + dy_rot * np.sin(ar)
        dy = -dx_rot * np.sin(ar) + dy_rot * np.cos(ar)
        refined_cx = cx_approx + dx
        refined_cy = cy_approx + dy
    else:
        refined_cx = cx_in_crop + x0
        refined_cy = cy_in_crop + y0

    max_shift = half_side * 0.7
    if (abs(refined_cx - cx_approx) > max_shift
            or abs(refined_cy - cy_approx) > max_shift):
        return None

    return refined_cx, refined_cy


def _edge_subpixel_centroid(
    profile: np.ndarray, peak_idx: int, half_window: int = 4
) -> float:
    """Return a sub-pixel position for an edge by computing the
    background-subtracted weighted centroid of the profile in a small
    window centred on ``peak_idx``.

    Why this beats single-peak parabolic fit
    ---------------------------------------
    PVE blur often splits one physical edge into 2-3 adjacent gradient
    peaks. A parabolic fit at the tallest peak only uses the immediate
    ±1 neighbours and ignores the secondary peak, so the refined
    position jitters slice-to-slice depending on which sub-peak wins
    the tiebreaker. The weighted centroid integrates over the whole
    edge region (~9 samples here), so the secondary peak contributes
    its share to the position and the result is stable to within
    ~0.1 px regardless of which peak is locally tallest.

    The baseline (profile.min within the window) is subtracted so
    flat-noise floors don't pull the centroid toward the window centre.
    Falls back to ``peak_idx`` when the windowed mass is zero
    (degenerate or all-masked region).
    """
    n = profile.size
    if n == 0:
        return float(peak_idx)
    lo = max(0, peak_idx - half_window)
    hi = min(n, peak_idx + half_window + 1)
    sub = profile[lo:hi].astype(np.float64)
    if sub.size == 0:
        return float(peak_idx)
    baseline = float(sub.min())
    weights = sub - baseline
    total = float(weights.sum())
    if total <= 1e-9:
        return float(peak_idx)
    positions = np.arange(lo, hi, dtype=np.float64)
    return float((positions * weights).sum() / total)


def _parabolic_peak_offset(profile: np.ndarray, peak_idx: int) -> float:
    """Return the sub-pixel offset of a 1-D peak via 3-point parabolic fit."""
    if peak_idx <= 0 or peak_idx >= len(profile) - 1:
        return 0.0
    y_lo = float(profile[peak_idx - 1])
    y_pk = float(profile[peak_idx])
    y_hi = float(profile[peak_idx + 1])
    denom = y_lo - 2.0 * y_pk + y_hi
    if denom == 0.0:
        return 0.0
    offset = 0.5 * (y_lo - y_hi) / denom
    # Clip pathological fits.
    if offset < -1.0 or offset > 1.0:
        return 0.0
    return offset
