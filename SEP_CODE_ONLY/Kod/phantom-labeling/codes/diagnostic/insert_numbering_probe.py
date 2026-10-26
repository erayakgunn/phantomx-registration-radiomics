"""Print each Insert_N's angle from the phantom centre, to check the slot
numbering against the reference (outer: Insert_4 at 0deg/right, Insert_1 at
252deg/upper-left-of-top; inner: Insert_15 at ~18deg)."""
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
import config


def main():
    dv = load_series(sys.argv[1])
    ph = detect_phantom(dv)
    se = classify_sections(dv, ph)
    canonicalize_orientation(dv, ph, se)
    trim_sections_to_inner_window(dv, se, phantom=ph)
    ro = detect_rotation(dv, ph, se)
    ex = extract_rois(dv, ph, se, ro)
    gcx, gcy = ph.global_center_x, ph.global_center_y

    pos = {}
    for r in ex.rois:
        if r.label.endswith("_15mm") and r.label.startswith("Insert_"):
            num = int(r.label.split("_")[1])
            pos[num] = (r.center_x, r.center_y)

    outer = set(config.SECTIONS["INSERT1"]["outer_inserts"] +
                config.SECTIONS["INSERT2"]["outer_inserts"])
    print("  N  ring   angle  (expected)")
    # expected canonical angles
    exp_outer = {}  # num -> angle
    for slot, num in enumerate(config.SECTIONS["INSERT1"]["outer_inserts"]):
        exp_outer[num] = (0.0 + slot * 36.0) % 360
    exp_inner = {}
    for slot, num in enumerate(config.SECTIONS["INSERT1"]["inner_inserts"]):
        exp_inner[num] = (18.0 + slot * 36.0) % 360
    for num in range(1, 21):
        if num not in pos:
            continue
        x, y = pos[num]
        ang = np.degrees(np.arctan2(y - gcy, x - gcx)) % 360
        ring = "outer" if num in outer else "inner"
        exp = exp_outer.get(num) if ring == "outer" else exp_inner.get(num)
        d = (ang - exp + 180) % 360 - 180 if exp is not None else 0
        flag = "  <-- OFF" if abs(d) > 18 else ""
        print(f"  {num:2d} {ring:5s} {ang:6.1f}  ({exp:5.1f}, dev {d:+5.1f}){flag}")


if __name__ == "__main__":
    main()
