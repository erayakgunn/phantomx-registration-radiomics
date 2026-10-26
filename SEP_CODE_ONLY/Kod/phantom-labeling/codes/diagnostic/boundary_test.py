"""Test detector on PVE-affected boundary slices of each STAR section.

The 'kept' window centred on the section's z-centroid drops boundary
slices that are partial-pattern (PVE). This test probes those dropped
slices to see if the detector still works robustly when the texture
pattern is faint.
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
    # NOTE: do NOT call trim_sections_to_inner_window — we want to
    # probe the FULL section, including PVE boundary slices.
    rotation = detect_rotation(dv, phantom, sections)

    sec = next(s for s in sections.sections if s.name == section_name)
    all_slices = list(range(sec.start_idx, sec.end_idx))

    gcx, gcy = phantom.global_center_x, phantom.global_center_y
    sec_cfg = config.SECTIONS[section_name]
    textures = sec_cfg["textures"]
    texture_dist_px = dv.cm_to_pixels(config.TEXTURE_CENTER_DISTANCE_CM)
    texture_side_px = dv.cm_to_pixels(config.TEXTURE_SIDE_CM)
    offset = rotation.global_offset_deg

    print(f"\n=== {section_name}, untrimmed range [{sec.start_idx}:{sec.end_idx}] = {len(all_slices)} slices ===")

    # Reference: polish on the central 50% of slices (clean)
    central_lo = len(all_slices) // 4
    central_hi = 3 * len(all_slices) // 4
    central_slices = all_slices[central_lo:central_hi]
    mean_img = np.mean(dv.volume[central_slices], axis=0, dtype=np.float64).astype(np.float32)
    mean_positions = _detect_texture_blobs(
        mean_img, gcx, gcy, texture_dist_px, textures, offset,
        texture_side_px=texture_side_px,
        phantom_r=phantom.global_radius,
    )
    print(f"Reference (mean of central {len(central_slices)} slices):")
    for tn, (mx, my) in mean_positions.items():
        print(f"  {tn:15s}: ({mx:.2f}, {my:.2f})")

    # Per-slice across ALL slices (incl. PVE-affected boundaries)
    print(f"\nPer-slice scatter on FULL section (incl. boundary PVE slices):")
    rows = []
    for i in all_slices:
        positions = _detect_texture_blobs(
            dv.volume[i], gcx, gcy, texture_dist_px, textures, offset,
            texture_side_px=texture_side_px,
            phantom_r=phantom.global_radius,
        )
        for name in textures.values():
            if name in positions:
                ref = mean_positions.get(name)
                if ref:
                    d = ((positions[name][0] - ref[0])**2 + (positions[name][1] - ref[1])**2)**0.5
                else:
                    d = float('nan')
                rows.append((i, name, positions[name][0], positions[name][1], d))
            else:
                rows.append((i, name, None, None, None))

    # Find slices where any texture deviates >2 px from reference
    print(f"\n{'slice':>6s} {'texture':<14s} {'x':>8s} {'y':>8s} {'dev':>6s}")
    bad_slices = set()
    for i, name, x, y, d in rows:
        if x is None:
            print(f"{i:>6d} {name:<14s}   MISS")
            bad_slices.add(i)
        elif d is not None and d > 2.0:
            print(f"{i:>6d} {name:<14s} {x:>8.2f} {y:>8.2f} {d:>6.2f}")
            bad_slices.add(i)

    if not bad_slices:
        print("  ALL slices: detector within 2 px of reference.")
    else:
        print(f"\n{len(bad_slices)} bad slice(s) out of {len(all_slices)}")
        # Identify if bad slices are at boundaries
        sorted_bad = sorted(bad_slices)
        for bs in sorted_bad:
            pos_in_section = bs - sec.start_idx
            if pos_in_section < 3 or pos_in_section > len(all_slices) - 4:
                print(f"  slice {bs} is at boundary (pos {pos_in_section}/{len(all_slices)})")
            else:
                print(f"  slice {bs} is MID-section (pos {pos_in_section}/{len(all_slices)}) — investigate!")


if __name__ == "__main__":
    main()
