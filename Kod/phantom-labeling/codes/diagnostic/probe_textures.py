"""Probe texture detection internals on real DICOM data.

Runs Stage 1 (CC bbox) + Stage 2 (Sobel peak-pair) independently, prints
peaks, and visualizes the chosen frame pair vs the true frame for each
texture on the section mean image.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
import cv2
from core.dicom_loader import load_series
from core.phantom_detector import detect_phantom
from core.section_classifier import classify_sections
from core.orientation_corrector import canonicalize_orientation
from core.section_trim import trim_sections_to_inner_window
from core.rotation_detector import detect_rotation
from core.roi_extractor import (
    _refine_texture_center_via_edges,
    _find_frame_edge_pair,
    _find_local_maxima,
)


def stage1_seeds(hu_slice, phantom_cx, phantom_cy, expected_dist_px, textures, offset):
    """Run only Stage 1 (annular threshold + CC bbox centroids) — return
    dict of tex_name -> (seed_x, seed_y)."""
    h, w = hu_slice.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - phantom_cx) ** 2 + (yy - phantom_cy) ** 2)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    nlabels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    blobs = []
    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            continue
        bb_x = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2.0
        bb_y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
        blobs.append((float(bb_x), float(bb_y),
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                      stats[i, cv2.CC_STAT_AREA]))
    seeds = {}
    for nominal_angle, tex_name in textures.items():
        actual_angle = nominal_angle + offset
        ang_rad = np.deg2rad(actual_angle)
        expected_x = phantom_cx + expected_dist_px * np.cos(ang_rad)
        expected_y = phantom_cy + expected_dist_px * np.sin(ang_rad)
        best_d = float('inf')
        best_b = None
        for b in blobs:
            d = ((b[0] - expected_x) ** 2 + (b[1] - expected_y) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_b = b
        if best_b is None or best_d >= expected_dist_px * 0.3:
            continue
        seeds[tex_name] = best_b
    return seeds


def probe_stage2(hu_slice, seed_x, seed_y, texture_side_px, window_scale=1.6,
                 label="?", expected_x=None, expected_y=None,
                 phantom_cx=None, phantom_cy=None, phantom_r=None):
    """Detailed Stage 2 (Sobel peak-pair) trace with both crop view and pair scoring."""
    h, w = hu_slice.shape
    half_win = int(np.ceil(texture_side_px * window_scale / 2.0))
    ix = int(round(seed_x))
    iy = int(round(seed_y))
    x0 = max(0, ix - half_win)
    y0 = max(0, iy - half_win)
    x1 = min(w, ix + half_win + 1)
    y1 = min(h, iy + half_win + 1)
    crop = hu_slice[y0:y1, x0:x1].astype(np.float32)
    crop_blur = cv2.GaussianBlur(crop, (3, 3), 0.6)
    gx = cv2.Sobel(crop_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop_blur, cv2.CV_32F, 0, 1, ksize=3)
    col_profile = np.abs(gx).sum(axis=0)
    row_profile = np.abs(gy).sum(axis=1)

    # Phantom wall masking (matches the production code)
    if phantom_cx is not None and phantom_cy is not None and phantom_r is not None:
        wall_margin = max(10.0, texture_side_px * 0.10)
        safe_r = phantom_r - wall_margin
        ch, cw = crop.shape
        cols = np.arange(cw) + x0
        col_r = np.sqrt((cols - phantom_cx) ** 2 + (seed_y - phantom_cy) ** 2)
        col_profile = np.where(col_r > safe_r, 0.0, col_profile)
        rows = np.arange(ch) + y0
        row_r = np.sqrt((seed_x - phantom_cx) ** 2 + (rows - phantom_cy) ** 2)
        row_profile = np.where(row_r > safe_r, 0.0, row_profile)

    cx_local = seed_x - x0
    cy_local = seed_y - y0
    print(f"\n  [{label}] Stage 2 probe @ seed=({seed_x:.1f},{seed_y:.1f})")
    print(f"    Crop {crop.shape} cx_local={cx_local:.1f} cy_local={cy_local:.1f}")
    if expected_x is not None:
        ex_local = expected_x - x0
        ey_local = expected_y - y0
        print(f"    Expected center in crop: ({ex_local:.1f}, {ey_local:.1f}) "
              f"-> expected frame X at [{ex_local - texture_side_px/2:.1f}, "
              f"{ex_local + texture_side_px/2:.1f}]")

    # Print all peaks above 20% with their heights for col and row profiles
    def show_peaks(profile, name):
        peaks = _find_local_maxima(profile, threshold_frac=0.20)
        pmax = float(profile.max())
        print(f"    {name}: max={pmax:.0f}, {len(peaks)} peaks ≥20%:")
        for p in peaks:
            marker = ""
            print(f"      idx={p:3d}  height={profile[p]:.0f}  ({profile[p]/pmax*100:.0f}%){marker}")
        return peaks

    col_peaks = show_peaks(col_profile, "col_profile")
    row_peaks = show_peaks(row_profile, "row_profile")

    col_pair = _find_frame_edge_pair(col_profile, texture_side_px)
    row_pair = _find_frame_edge_pair(row_profile, texture_side_px)
    print(f"    => chosen col_pair={col_pair}  row_pair={row_pair}")

    if col_pair and row_pair:
        refined_cx = (col_pair[0] + col_pair[1]) / 2.0 + x0
        refined_cy = (row_pair[0] + row_pair[1]) / 2.0 + y0
        print(f"    => refined center: ({refined_cx:.2f}, {refined_cy:.2f})")
        if expected_x is not None:
            shift = ((refined_cx - expected_x) ** 2 + (refined_cy - expected_y) ** 2) ** 0.5
            print(f"       shift from expected ({expected_x:.1f},{expected_y:.1f}): {shift:.1f} px")
        return refined_cx, refined_cy
    return None


def main():
    if len(sys.argv) < 3:
        print("Usage: probe_textures.py SERIES_DIR SECTION_NAME [--slice IDX]")
        sys.exit(1)

    series_dir = sys.argv[1]
    section_name = sys.argv[2]
    use_slice = None
    if "--slice" in sys.argv:
        use_slice = int(sys.argv[sys.argv.index("--slice") + 1])

    print(f"\n=== Loading {series_dir} ===")
    dv = load_series(series_dir)
    phantom = detect_phantom(dv)
    print(f"Global phantom: cx={phantom.global_center_x:.1f} cy={phantom.global_center_y:.1f} r={phantom.global_radius:.1f}")

    sections = classify_sections(dv, phantom)
    canonicalize_orientation(dv, phantom, sections)
    trim_sections_to_inner_window(dv, sections, phantom=phantom)
    rotation = detect_rotation(dv, phantom, sections)
    print(f"Rotation offset: {rotation.global_offset_deg:.2f} deg")

    sec = next((s for s in sections.sections if s.name == section_name), None)
    if sec is None:
        print(f"Section {section_name} not found")
        sys.exit(1)

    valid_slices = [
        i for i in range(sec.start_idx, sec.end_idx)
        if sections.slice_labels[i] != "TRANSITION"
    ]
    print(f"Section {sec.name}: {len(valid_slices)} valid slices")

    gcx, gcy = phantom.global_center_x, phantom.global_center_y
    sec_cfg = config.SECTIONS[section_name]
    textures = sec_cfg["textures"]
    texture_dist_px = dv.cm_to_pixels(config.TEXTURE_CENTER_DISTANCE_CM)
    texture_side_px = dv.cm_to_pixels(config.TEXTURE_SIDE_CM)
    offset = rotation.global_offset_deg
    print(f"texture_dist_px={texture_dist_px:.2f}  texture_side_px={texture_side_px:.2f}")

    if use_slice is not None:
        target_img = dv.volume[use_slice]
        print(f"\n=== Per-slice {use_slice} ===")
    else:
        target_img = np.mean(dv.volume[list(valid_slices)], axis=0, dtype=np.float64).astype(np.float32)
        print(f"\n=== Mean image (N={len(valid_slices)}) ===")

    # Stage 1 seeds
    seeds = stage1_seeds(target_img, gcx, gcy, texture_dist_px, textures, offset)

    for nominal_angle, tex_name in sorted(textures.items()):
        actual_angle = (nominal_angle + offset) % 360
        ang_rad = np.deg2rad(actual_angle)
        expected_x = gcx + texture_dist_px * np.cos(ang_rad)
        expected_y = gcy + texture_dist_px * np.sin(ang_rad)
        print(f"\n========== {tex_name} (angle={actual_angle:.1f}°) ==========")
        print(f"  Expected absolute pos from config: ({expected_x:.1f}, {expected_y:.1f})")
        if tex_name not in seeds:
            print(f"  STAGE 1 FAILED: no CC found near expected")
            continue
        seed_b = seeds[tex_name]
        seed_x, seed_y, bb_w, bb_h, area = seed_b
        print(f"  Stage 1 seed (CC bbox centroid): ({seed_x:.1f}, {seed_y:.1f})")
        print(f"    CC bbox: {bb_w}x{bb_h} px (area {area}) — texture_side_px={texture_side_px:.1f}")
        print(f"    Seed shift from expected: ({seed_x - expected_x:+.1f}, {seed_y - expected_y:+.1f}) px")

        probe_stage2(target_img, seed_x, seed_y, texture_side_px,
                     label=tex_name, expected_x=expected_x, expected_y=expected_y,
                     phantom_cx=gcx, phantom_cy=gcy, phantom_r=phantom.global_radius)


if __name__ == "__main__":
    main()
