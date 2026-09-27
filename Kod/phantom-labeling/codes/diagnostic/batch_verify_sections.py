"""Batch-verify section windows across every labeled series.

For each series under the given root (any folder containing a ``dicoms``
subfolder), run the pipeline through ROI extraction and report each
section's final slice window, count, and whether the guard had to act.
Used to verify the fixed-30mm / no-transition-labeling invariants across
all protocols at once.

Usage:
    python -m diagnostic.batch_verify_sections <root> [<root2> ...]
"""

import logging
import os
import sys

from utils.logger import log, setup_logger


def find_series_dirs(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "dicoms":
            out.append(os.path.dirname(dirpath))
            dirnames[:] = []
    return sorted(out)


def verify_series(series_dir):
    from core.dicom_loader import load_series
    from core.phantom_detector import detect_phantom
    from core.section_classifier import classify_sections
    from core.orientation_corrector import canonicalize_orientation
    from core.section_trim import trim_sections_to_inner_window
    from core.rotation_detector import detect_rotation
    from core.roi_extractor import extract_rois

    dv = load_series(os.path.join(series_dir, "dicoms"))
    ph = detect_phantom(dv)
    sec = classify_sections(dv, ph)
    canonicalize_orientation(dv, ph, sec)
    trim_sections_to_inner_window(dv, sec, phantom=ph)
    rot = detect_rotation(dv, ph, sec)
    extract_rois(dv, ph, sec, rot)

    labels = sec.slice_labels
    report = {}
    for nm in ("STAR16", "STAR8", "INSERT1", "INSERT2"):
        idx = [i for i, l in enumerate(labels) if l == nm]
        if idx:
            report[nm] = (min(idx), max(idx), len(idx))
        else:
            report[nm] = None
    return report, dv.slice_thickness


def main():
    setup_logger(level=logging.ERROR)
    roots = sys.argv[1:]
    if not roots:
        print("usage: python -m diagnostic.batch_verify_sections <root> ...")
        sys.exit(1)

    series = []
    for root in roots:
        series.extend(find_series_dirs(root))
    print(f"Found {len(series)} series with dicoms/", flush=True)

    bad = []
    for sd in series:
        name = os.path.basename(sd)
        try:
            report, dz = verify_series(sd)
        except Exception as e:
            print(f"FAIL  {name}: {e}", flush=True)
            bad.append((name, str(e)))
            continue
        target = max(1, round(30.0 / dz)) if dz else 30
        flags = []
        for nm, rng in report.items():
            if rng is None:
                flags.append(f"{nm}:MISSING")
            elif rng[2] != target:
                flags.append(f"{nm}:{rng[2]}!={target}")
        status = "OK " if not flags else "BAD"
        rngs = "  ".join(
            f"{nm}={rng[0]}-{rng[1]}({rng[2]})" if rng else f"{nm}=None"
            for nm, rng in report.items()
        )
        print(f"{status}  {name}: {rngs}", flush=True)
        if flags:
            bad.append((name, "; ".join(flags)))

    print(f"\n{len(series) - len(bad)}/{len(series)} OK", flush=True)
    for name, why in bad:
        print(f"  BAD: {name} -> {why}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
