"""Analyse per-slice phantom Hough centres across STAR sections to
characterise the drift signal vs the Hough noise.

If real drift is SMOOTH across z (couch/gantry creep) and Hough noise
is HIGH-FREQUENCY random scatter, we can fit a smooth model (line or
low-order polynomial) and evaluate it per-slice. That gives us drift
correction WITHOUT the per-slice Hough wobble.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.dicom_loader import load_series
from core.phantom_detector import detect_phantom, SliceStatus
from core.section_classifier import classify_sections
from core.orientation_corrector import canonicalize_orientation
from core.section_trim import trim_sections_to_inner_window


def analyse_one_series(series_dir):
    dv = load_series(series_dir)
    phantom = detect_phantom(dv)
    sections = classify_sections(dv, phantom)
    canonicalize_orientation(dv, phantom, sections)
    trim_sections_to_inner_window(dv, sections, phantom=phantom)

    label = os.path.basename(series_dir)
    print(f"\n{'='*78}")
    print(f"SERIES: {label[:74]}")
    print(f"{'='*78}")
    print(f"Global phantom centre: ({phantom.global_center_x:.2f}, "
          f"{phantom.global_center_y:.2f})")

    for sec in sections.sections:
        if sec.name not in ("STAR16", "STAR8"):
            continue
        valid = [
            i for i in range(sec.start_idx, sec.end_idx)
            if sections.slice_labels[i] != "TRANSITION"
            and phantom.circles[i].status == SliceStatus.VALID
        ]
        if len(valid) < 5:
            print(f"  {sec.name}: too few VALID slices ({len(valid)})")
            continue
        xs = np.array([phantom.circles[i].center_x for i in valid])
        ys = np.array([phantom.circles[i].center_y for i in valid])
        idx = np.array(valid, dtype=float)

        print(f"\n  {sec.name}: {len(valid)} VALID slices in [{valid[0]}, {valid[-1]}]")
        print(f"    Raw stats:")
        print(f"      x: mean={xs.mean():.2f}, std={xs.std():.2f}, "
              f"range={xs.max()-xs.min():.2f}, median={np.median(xs):.2f}")
        print(f"      y: mean={ys.mean():.2f}, std={ys.std():.2f}, "
              f"range={ys.max()-ys.min():.2f}, median={np.median(ys):.2f}")

        # Linear fit
        coef_x = np.polyfit(idx, xs, 1)
        coef_y = np.polyfit(idx, ys, 1)
        x_fit = np.polyval(coef_x, idx)
        y_fit = np.polyval(coef_y, idx)
        resid_x = xs - x_fit
        resid_y = ys - y_fit
        print(f"    Linear fit slope (px/slice):  x={coef_x[0]:+.4f}  y={coef_y[0]:+.4f}")
        print(f"    Linear fit residual std (px): x={resid_x.std():.2f}  y={resid_y.std():.2f}")

        # Quadratic fit for comparison
        coef_x2 = np.polyfit(idx, xs, 2)
        coef_y2 = np.polyfit(idx, ys, 2)
        resid_x2 = xs - np.polyval(coef_x2, idx)
        resid_y2 = ys - np.polyval(coef_y2, idx)
        print(f"    Quad fit residual std (px):   x={resid_x2.std():.2f}  y={resid_y2.std():.2f}")

        # Median absolute deviation from median (robust noise estimate)
        mad_x = np.median(np.abs(xs - np.median(xs)))
        mad_y = np.median(np.abs(ys - np.median(ys)))
        print(f"    MAD from median (px):  x={mad_x:.2f}  y={mad_y:.2f}")

        # Per-slice deviation list (first/last few)
        # to see if the drift is monotonic or noisy
        print(f"    First 5 (x_raw, x_fit, dx):")
        for k in range(min(5, len(valid))):
            print(f"      slice {valid[k]:>4d}: x={xs[k]:7.2f} fit={x_fit[k]:7.2f} "
                  f"resid={resid_x[k]:+5.2f}    y={ys[k]:7.2f} fit={y_fit[k]:7.2f} "
                  f"resid={resid_y[k]:+5.2f}")
        print(f"    Last 5:")
        for k in range(max(0, len(valid)-5), len(valid)):
            print(f"      slice {valid[k]:>4d}: x={xs[k]:7.2f} fit={x_fit[k]:7.2f} "
                  f"resid={resid_x[k]:+5.2f}    y={ys[k]:7.2f} fit={y_fit[k]:7.2f} "
                  f"resid={resid_y[k]:+5.2f}")

        # Diagnose: is the drift real?
        # Linear-fit residual std much smaller than raw std → drift is real and linear-ish
        # Both stds small → no real drift, just Hough noise
        # Quadratic residual much smaller than linear → drift is curved
        ratio_x = resid_x.std() / max(xs.std(), 1e-9)
        ratio_y = resid_y.std() / max(ys.std(), 1e-9)
        print(f"    Drift-vs-noise diagnosis:")
        print(f"      x: residual/raw = {ratio_x:.2f}  "
              f"({'mostly drift' if ratio_x < 0.5 else 'mostly noise' if ratio_x > 0.8 else 'mixed'})")
        print(f"      y: residual/raw = {ratio_y:.2f}  "
              f"({'mostly drift' if ratio_y < 0.5 else 'mostly noise' if ratio_y > 0.8 else 'mixed'})")


def main():
    import glob
    input_root = sys.argv[1] if len(sys.argv) > 1 else "C:/input_pcct_clean"
    series_dirs = [
        os.path.join(input_root, d) for d in sorted(os.listdir(input_root))
        if os.path.isdir(os.path.join(input_root, d))
        and glob.glob(os.path.join(input_root, d, "**", "*.dcm"), recursive=True)
    ]
    for sd in series_dirs:
        analyse_one_series(sd)


if __name__ == "__main__":
    main()
