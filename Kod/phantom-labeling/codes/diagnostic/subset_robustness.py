"""Sanity-check that the mean-image polish position is stable across
different random subsets of slices.

If the polish position varies significantly with subset selection, the
detector is sensitive to which slices are kept (bad). Stable subset
results mean the detector returns the same answer regardless of which
slices contribute to the mean image (good).
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
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 30

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

    print(f"\n=== {section_name}, {len(valid_slices)} valid slices, {n_trials} subset trials ===")

    rng = np.random.default_rng(42)
    # For each trial pick a random subset (half-size) and compute polish
    subset_size = max(2, len(valid_slices) // 2)
    print(f"Each trial uses a random {subset_size}-slice subset.")

    per_texture = {name: [] for name in textures.values()}
    for trial in range(n_trials):
        subset = rng.choice(valid_slices, size=subset_size, replace=False)
        mean_img = np.mean(dv.volume[subset], axis=0, dtype=np.float64).astype(np.float32)
        positions = _detect_texture_blobs(
            mean_img, gcx, gcy, texture_dist_px, textures, offset,
            texture_side_px=texture_side_px,
            phantom_r=phantom.global_radius,
        )
        for name in textures.values():
            if name in positions:
                per_texture[name].append(positions[name])

    print(f"\n{'Texture':<14s} {'n_ok':>5s} {'mean_x':>8s} {'mean_y':>8s} "
          f"{'std_x':>6s} {'std_y':>6s} {'max_dev':>7s}")
    for name in textures.values():
        items = per_texture[name]
        if not items:
            print(f"  {name:12s}: NO DETECTIONS")
            continue
        xs = np.array([it[0] for it in items])
        ys = np.array([it[1] for it in items])
        mx = float(xs.mean())
        my = float(ys.mean())
        sx = float(xs.std())
        sy = float(ys.std())
        dists = np.sqrt((xs - mx)**2 + (ys - my)**2)
        max_d = float(dists.max())
        print(f"  {name:12s}: {len(items):>5d} {mx:>8.2f} {my:>8.2f} {sx:>6.2f} {sy:>6.2f} {max_d:>7.2f}")


if __name__ == "__main__":
    main()
