"""Debug why STAR8 Honeylarge Stage 1 fails on the mean image."""
import os
import sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from core.dicom_loader import load_series
from core.phantom_detector import detect_phantom
from core.section_classifier import classify_sections
from core.orientation_corrector import canonicalize_orientation
from core.section_trim import trim_sections_to_inner_window
from core.rotation_detector import detect_rotation


def main():
    series_dir = sys.argv[1]
    dv = load_series(series_dir)
    phantom = detect_phantom(dv)
    sections = classify_sections(dv, phantom)
    canonicalize_orientation(dv, phantom, sections)
    trim_sections_to_inner_window(dv, sections, phantom=phantom)
    rotation = detect_rotation(dv, phantom, sections)

    sec = next(s for s in sections.sections if s.name == "STAR8")
    valid_slices = [
        i for i in range(sec.start_idx, sec.end_idx)
        if sections.slice_labels[i] != "TRANSITION"
    ]
    mean_img = np.mean(dv.volume[list(valid_slices)], axis=0, dtype=np.float64).astype(np.float32)

    gcx, gcy = phantom.global_center_x, phantom.global_center_y
    expected_dist_px = dv.cm_to_pixels(config.TEXTURE_CENTER_DISTANCE_CM)
    offset = rotation.global_offset_deg

    # Honeylarge expected at angle 270 + offset
    honeylarge_angle = 270 + offset
    ex = gcx + expected_dist_px * np.cos(np.deg2rad(honeylarge_angle))
    ey = gcy + expected_dist_px * np.sin(np.deg2rad(honeylarge_angle))
    print(f"Honeylarge expected at: ({ex:.1f}, {ey:.1f})")

    # Examine a window around the expected location
    win = 60
    x0 = int(ex - win)
    y0 = int(ey - win)
    crop = mean_img[y0:y0+2*win, x0:x0+2*win]
    print(f"\nCrop {crop.shape} HU stats: min={crop.min():.0f} max={crop.max():.0f} "
          f"mean={crop.mean():.0f} median={np.median(crop):.0f}")
    print(f"  90th percentile: {np.percentile(crop, 90):.0f}")
    print(f"  95th percentile: {np.percentile(crop, 95):.0f}")
    print(f"  99th percentile: {np.percentile(crop, 99):.0f}")

    # Try the actual annular threshold logic from _detect_texture_blobs
    h, w = mean_img.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - gcx) ** 2 + (yy - gcy) ** 2)
    tol = expected_dist_px * 0.4
    annular = (dist > expected_dist_px - tol) & (dist < expected_dist_px + tol)
    bg_values = mean_img[annular]
    bg_median = np.median(bg_values)
    bg_std = np.std(bg_values)
    threshold = max(bg_median + 2 * bg_std, bg_median + 80)
    print(f"\nAnnular: {annular.sum()} pixels, median={bg_median:.0f}, std={bg_std:.0f}")
    print(f"  Threshold = max({bg_median + 2*bg_std:.0f}, {bg_median + 80:.0f}) = {threshold:.0f}")

    binary = (mean_img > threshold).astype(np.uint8) * 255
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[annular] = 255
    binary = cv2.bitwise_and(binary, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    nlabels, _, stats, _ = cv2.connectedComponentsWithStats(binary_closed)
    print(f"\n{nlabels-1} connected components in binary (after closing, 5x5):")
    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        bb_x = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2.0
        bb_y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
        d_from_expected = ((bb_x - ex)**2 + (bb_y - ey)**2)**0.5
        d_from_phantom = ((bb_x - gcx)**2 + (bb_y - gcy)**2)**0.5
        bbw = stats[i, cv2.CC_STAT_WIDTH]
        bbh = stats[i, cv2.CC_STAT_HEIGHT]
        print(f"  CC{i}: bbox={bbw}x{bbh} area={area} center=({bb_x:.0f},{bb_y:.0f}) "
              f"d_from_phantom={d_from_phantom:.0f} d_from_expected_honeylarge={d_from_expected:.0f}")

    # Try with larger morph kernel
    print("\n--- With 11x11 closing kernel ---")
    kernel11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary_closed11 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel11)
    nlabels11, _, stats11, _ = cv2.connectedComponentsWithStats(binary_closed11)
    print(f"{nlabels11-1} CCs:")
    for i in range(1, nlabels11):
        area = stats11[i, cv2.CC_STAT_AREA]
        if area < 50:
            continue
        bb_x = stats11[i, cv2.CC_STAT_LEFT] + stats11[i, cv2.CC_STAT_WIDTH] / 2.0
        bb_y = stats11[i, cv2.CC_STAT_TOP] + stats11[i, cv2.CC_STAT_HEIGHT] / 2.0
        d_from_expected = ((bb_x - ex)**2 + (bb_y - ey)**2)**0.5
        bbw = stats11[i, cv2.CC_STAT_WIDTH]
        bbh = stats11[i, cv2.CC_STAT_HEIGHT]
        print(f"  CC{i}: bbox={bbw}x{bbh} area={area} center=({bb_x:.0f},{bb_y:.0f}) "
              f"d_from_expected={d_from_expected:.0f}")


if __name__ == "__main__":
    main()
