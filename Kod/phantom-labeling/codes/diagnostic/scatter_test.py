"""Quantify per-slice texture-detection scatter across an entire section.

For each slice in a STAR section, runs Stage 1 + Stage 2 and reports the
detected texture (x, y). Computes:
  * mean position per texture
  * std per texture
  * worst-case shift per texture
  * which slices produce the worst outliers
"""
import os
import sys
import numpy as np

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
from core.roi_extractor import _detect_texture_blobs


def main():
    series_dir = sys.argv[1]
    section_name = sys.argv[2]

    dv = load_series(series_dir)
    phantom = detect_phantom(dv)
    sections = classify_sections(dv, phantom)
    canonicalize_orientation(dv, phantom, sections)
    trim_sections_to_inner_window(dv, sections, phantom=phantom)
    rotation = detect_rotation(dv, phantom, sections)

    sec = next(s for s in sections.sections if s.name == section_name)
    valid_slices = [
        i for i in range(sec.start_idx, sec.end_idx)
        if sections.slice_labels[i] != "TRANSITION"
    ]

    gcx, gcy = phantom.global_center_x, phantom.global_center_y
    sec_cfg = config.SECTIONS[section_name]
    textures = sec_cfg["textures"]
    texture_dist_px = dv.cm_to_pixels(config.TEXTURE_CENTER_DISTANCE_CM)
    texture_side_px = dv.cm_to_pixels(config.TEXTURE_SIDE_CM)
    offset = rotation.global_offset_deg

    print(f"\n=== {section_name}, {len(valid_slices)} slices ===")
    print(f"texture_side_px={texture_side_px:.1f}  texture_dist_px={texture_dist_px:.1f}")

    # Mean image polish position (reference)
    mean_img = np.mean(dv.volume[list(valid_slices)], axis=0, dtype=np.float64).astype(np.float32)
    mean_positions = _detect_texture_blobs(
        mean_img, gcx, gcy, texture_dist_px, textures, offset,
        texture_side_px=texture_side_px,
        phantom_r=phantom.global_radius,
    )
    print(f"\nMean-image polish positions (reference):")
    for tex_name, (mx, my) in mean_positions.items():
        d = ((mx - gcx)**2 + (my - gcy)**2)**0.5
        print(f"  {tex_name:15s}: ({mx:.2f}, {my:.2f}) dist={d:.2f}px")

    # Per-slice detections
    per_slice = {name: [] for name in textures.values()}
    miss_count = {name: 0 for name in textures.values()}
    for i in valid_slices:
        positions = _detect_texture_blobs(
            dv.volume[i], gcx, gcy, texture_dist_px, textures, offset,
            texture_side_px=texture_side_px,
            phantom_r=phantom.global_radius,
        )
        for name in textures.values():
            if name in positions:
                per_slice[name].append((i, positions[name][0], positions[name][1]))
            else:
                miss_count[name] += 1

    print(f"\nPer-slice statistics (N={len(valid_slices)}):")
    print(f"{'Texture':<15s}  {'n_detect':>8s}  {'miss':>4s}  {'mean_x':>8s}  {'mean_y':>8s}  "
          f"{'std_x':>6s}  {'std_y':>6s}  {'max_shift':>9s}  {'shift_vs_polish':>15s}")
    for name in textures.values():
        items = per_slice[name]
        if not items:
            print(f"  {name:13s}: NO DETECTIONS")
            continue
        xs = np.array([it[1] for it in items])
        ys = np.array([it[2] for it in items])
        mx_per = float(xs.mean())
        my_per = float(ys.mean())
        sx = float(xs.std())
        sy = float(ys.std())
        # max distance from mean
        dists = np.sqrt((xs - mx_per)**2 + (ys - my_per)**2)
        max_d = float(dists.max())
        worst_idx_in_list = int(np.argmax(dists))
        worst_slice = items[worst_idx_in_list][0]
        # Shift vs polish
        if name in mean_positions:
            px, py = mean_positions[name]
            shift_vs_polish = ((mx_per - px)**2 + (my_per - py)**2)**0.5
        else:
            shift_vs_polish = float('nan')
        print(f"  {name:13s}: {len(items):>8d}  {miss_count[name]:>4d}  "
              f"{mx_per:>8.2f}  {my_per:>8.2f}  {sx:>6.2f}  {sy:>6.2f}  "
              f"{max_d:>5.2f}@s{worst_slice:<3d}  {shift_vs_polish:>15.2f}")


if __name__ == "__main__":
    main()
