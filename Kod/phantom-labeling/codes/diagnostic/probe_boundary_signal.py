"""Probe per-slice signal weights for STAR sections to characterise the
clean-vs-PVE boundary contrast.

For each STAR section in a series, computes the gradient-signal weight
for every slice in valid_slices + the surrounding slices in TRANSITION.
Then reports the ratio against the section median so we can choose
the right PVE threshold.
"""
import os, sys, numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.dicom_loader import load_series
from core.phantom_detector import detect_phantom, SliceStatus
from core.section_classifier import classify_sections
from core.orientation_corrector import canonicalize_orientation
from core.section_trim import trim_sections_to_inner_window


def main():
    series_dir = sys.argv[1]
    dv = load_series(series_dir)
    phantom = detect_phantom(dv)
    sections = classify_sections(dv, phantom)
    canonicalize_orientation(dv, phantom, sections)
    trim_sections_to_inner_window(dv, sections, phantom=phantom)

    gcx = float(phantom.global_center_x)
    gcy = float(phantom.global_center_y)
    gr = 0.9 * float(phantom.global_radius)
    h, w = dv.volume.shape[1], dv.volume.shape[2]
    yy, xx = np.ogrid[:h, :w]
    interior = ((xx - gcx) ** 2 + (yy - gcy) ** 2) <= gr * gr

    def sig(i):
        s = dv.volume[i].astype(np.float32)
        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.sqrt(gx*gx+gy*gy)[interior].sum())

    for sec in sections.sections:
        if sec.name not in ("STAR16", "STAR8"):
            continue
        # Build extended window: ±5 slices around the section's labelled
        # boundaries so we can see TRANSITION-edge behaviour too.
        lo = max(0, sec.start_idx - 5)
        hi = min(dv.num_slices, sec.end_idx + 5)
        widx = list(range(lo, hi))
        weights = [sig(i) for i in widx]
        labels = [sections.slice_labels[i] for i in widx]
        # Section median (use the central HALF of valid slices as ref)
        sec_idx = [i for i in widx if sections.slice_labels[i] == sec.name]
        sec_idx.sort()
        if len(sec_idx) < 4:
            continue
        central = sec_idx[len(sec_idx)//4 : -len(sec_idx)//4 or None]
        ref = float(np.median([sig(i) for i in central]))
        print(f"\n=== {sec.name} (idx range {sec.start_idx}..{sec.end_idx-1}, ref signal {ref:.0f}) ===")
        print(f"{'idx':>5s} {'label':<12s} {'signal':>10s} {'ratio':>8s}")
        for i, w, l in zip(widx, weights, labels):
            ratio = w / ref if ref > 0 else 0
            marker = " <-- ABOVE 0.75" if ratio >= 0.75 and l == "TRANSITION" else \
                     " <-- BELOW 0.85" if ratio < 0.85 and l == sec.name else ""
            print(f"  {i:>3d}  {l:<12s} {w:>10.0f}  {ratio:>6.3f}{marker}")


if __name__ == "__main__":
    main()
