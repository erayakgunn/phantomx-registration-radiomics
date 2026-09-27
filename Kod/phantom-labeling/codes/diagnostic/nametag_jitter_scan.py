"""Scan inserts for per-slice text-anchor jitter (round(centre) flips)."""
import sys
import numpy as np
from collections import defaultdict

from core.dicom_loader import load_series
from core.phantom_detector import detect_phantom
from core.section_classifier import classify_sections
from core.orientation_corrector import canonicalize_orientation
from core.section_trim import trim_sections_to_inner_window
from core.rotation_detector import detect_rotation
from core.roi_extractor import extract_rois


def main():
    dv = load_series(sys.argv[1])
    ph = detect_phantom(dv)
    se = classify_sections(dv, ph)
    canonicalize_orientation(dv, ph, se)
    trim_sections_to_inner_window(dv, se, phantom=ph)
    ro = detect_rotation(dv, ph, se)
    ex = extract_rois(dv, ph, se, ro)

    cx = defaultdict(list)
    cy = defaultdict(list)
    for r in ex.rois:
        if r.label.endswith("_15mm") and r.label.startswith("Insert_"):
            base = r.label.rsplit("_", 1)[0]
            cx[base].append(r.center_x)
            cy[base].append(r.center_y)

    print(f"{'insert':12s} {'x range':>9s} {'y range':>9s}  "
          f"{'round(x) values':>22s}")
    worst = []
    for lbl in sorted(cx):
        xr = max(cx[lbl]) - min(cx[lbl])
        yr = max(cy[lbl]) - min(cy[lbl])
        rxs = sorted(set(int(round(v)) for v in cx[lbl]))
        rys = sorted(set(int(round(v)) for v in cy[lbl]))
        flips = (len(rxs) - 1) + (len(rys) - 1)
        worst.append((flips, xr + yr, lbl, xr, yr, rxs, rys))
    worst.sort(reverse=True)
    for flips, _, lbl, xr, yr, rxs, rys in worst[:12]:
        print(f"{lbl:12s} {xr:9.2f} {yr:9.2f}  x{rxs} y{rys}"
              + ("   <- TEXT JUMPS" if flips > 0 else ""))
    n_jump = sum(1 for w in worst if w[0] > 0)
    print(f"\n{n_jump}/{len(worst)} inserts had a round() flip on >=1 slice "
          f"(old: text jumps; new: pinned to median = no jump).")


if __name__ == "__main__":
    main()
