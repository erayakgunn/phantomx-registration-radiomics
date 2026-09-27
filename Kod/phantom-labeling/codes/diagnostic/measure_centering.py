"""Objective per-slice centering error measurement.

For a given series, runs the pipeline through extraction, then independently
re-detects each INSERT disc centroid and each TEXTURE-square pattern centre in
the ORIGINAL slice within a window around the ROI centre the pipeline chose.
Reports the offset (pipeline centre - independent feature centroid) so we can
see, quantitatively and per-slice, how well-centred the labels actually are.

Usage:
    python -m diagnostic.measure_centering <series_dir>
"""
import sys
import numpy as np

from core.dicom_loader import load_series
from core.phantom_detector import detect_phantom
from core.section_classifier import classify_sections
from core.orientation_corrector import canonicalize_orientation
from core.section_trim import trim_sections_to_inner_window
from core.rotation_detector import detect_rotation
from core.roi_extractor import extract_rois


def _disc_centroid(win: np.ndarray):
    """Centroid of the bright insert disc in a window (HU). Returns (dx,dy)
    relative to window centre, or None."""
    if win.size == 0:
        return None
    v = win.astype(np.float32)
    lo, hi = float(np.percentile(v, 5)), float(np.percentile(v, 95))
    if hi - lo < 20:
        return None
    thr = (lo + hi) / 2.0
    mask = v >= thr
    if mask.sum() < 4:
        return None
    ys, xs = np.nonzero(mask)
    # weight by intensity above threshold
    w = (v[mask] - thr)
    cx = float(np.sum(xs * w) / np.sum(w))
    cy = float(np.sum(ys * w) / np.sum(w))
    h, wd = win.shape
    return (cx - (wd - 1) / 2.0, cy - (h - 1) / 2.0)


def main():
    series = sys.argv[1]
    dv = load_series(series)
    phantom = detect_phantom(dv)
    sections = classify_sections(dv, phantom)
    canonicalize_orientation(dv, phantom, sections)
    trim_sections_to_inner_window(dv, sections, phantom=phantom)
    rotation = detect_rotation(dv, phantom, sections)
    extraction = extract_rois(dv, phantom, sections, rotation)

    avg_sp = (dv.pixel_spacing[0] + dv.pixel_spacing[1]) / 2.0
    # Half-window 9 mm: bigger than the 7.5 mm insert radius (so a centred
    # insert fills it) but smaller than the inner-ring neighbour clearance
    # (neighbours sit ~28 mm centre-to-centre, near edge ~20 mm away) so no
    # neighbour leaks into the centroid.
    win_px = int(round(9.0 / avg_sp))

    # Group insert offsets by section
    from collections import defaultdict
    ins_off = defaultdict(list)
    ins_off_per_label = defaultdict(list)
    for roi in extraction.rois:
        if not roi.label.startswith("Insert_") or not roi.label.endswith("_15mm"):
            continue
        i = roi.slice_idx
        cx, cy = roi.center_x, roi.center_y
        r = win_px
        x0 = int(round(cx)) - r
        y0 = int(round(cy)) - r
        x1 = x0 + 2 * r + 1
        y1 = y0 + 2 * r + 1
        if x0 < 0 or y0 < 0 or x1 > dv.width or y1 > dv.height:
            continue
        win = dv.volume[i, y0:y1, x0:x1]
        off = _disc_centroid(win)
        if off is None:
            continue
        d = float(np.hypot(off[0], off[1]))
        ins_off[roi.section].append(d)
        ins_off_per_label[roi.label].append(d)

    print("\n==== INSERT centering offset (px) [pipeline centre vs disc centroid] ====")
    print(f"   pixel spacing = {avg_sp:.3f} mm/px  (1 px = {avg_sp:.3f} mm)")
    for sec in sorted(ins_off):
        arr = np.asarray(ins_off[sec])
        print(f"  {sec}: n={len(arr):4d}  mean={arr.mean():.2f}  "
              f"median={np.median(arr):.2f}  p95={np.percentile(arr,95):.2f}  "
              f"max={arr.max():.2f}  (mm: mean={arr.mean()*avg_sp:.2f} "
              f"max={arr.max()*avg_sp:.2f})")

    # Worst 8 inserts by mean offset
    worst = sorted(ins_off_per_label.items(),
                   key=lambda kv: -np.mean(kv[1]))[:8]
    print("\n  Worst inserts by mean offset:")
    for label, ds in worst:
        a = np.asarray(ds)
        print(f"    {label}: mean={a.mean():.2f}px max={a.max():.2f}px n={len(a)}")

    # Per-slice fluctuation: for each insert label, std of its centre across slices
    print("\n==== INSERT per-slice centre STD (label position stability across z) ====")
    cxs = defaultdict(list)
    cys = defaultdict(list)
    for roi in extraction.rois:
        if roi.label.endswith("_15mm") and roi.label.startswith("Insert_"):
            cxs[roi.label].append(roi.center_x)
            cys[roi.label].append(roi.center_y)
    stds = []
    for label in cxs:
        sx = np.std(cxs[label])
        sy = np.std(cys[label])
        stds.append(np.hypot(sx, sy))
    stds = np.asarray(stds)
    print(f"  inserts: mean std={stds.mean():.3f}px  max std={stds.max():.3f}px "
          f"(std>1px would be visible fluctuation)")

    # ---- STAR + TEXTURE centering via gradient-energy centroid ----
    import cv2

    def _grad_centroid(win: np.ndarray):
        """Centroid of gradient magnitude in a window. The patterned square /
        star fills its window with high-gradient texture on a smooth (air /
        plastic) background, so the |grad| centroid is an independent estimate
        of the pattern's geometric centre. Returns (dx,dy) from window centre."""
        if win.size == 0:
            return None
        v = win.astype(np.float32)
        gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx * gx + gy * gy)
        # keep only the strong-gradient pattern; drop smooth background
        thr = float(np.percentile(g, 75))
        g = np.where(g >= thr, g - thr, 0.0)
        s = float(g.sum())
        if s < 1e-6:
            return None
        ys, xs = np.mgrid[0:win.shape[0], 0:win.shape[1]]
        cx = float((xs * g).sum() / s)
        cy = float((ys * g).sum() / s)
        h, wd = win.shape
        return (cx - (wd - 1) / 2.0, cy - (h - 1) / 2.0)

    tex_win = int(round(22.0 / avg_sp))   # ±22 mm: holds the 39 mm square core
    star_win = int(round(14.0 / avg_sp))  # ±14 mm: star radius is 20 mm; tight central
    tex_off = defaultdict(list)
    star_off = defaultdict(list)
    for roi in extraction.rois:
        if roi.shape_type == "square":   # texture
            w = tex_win
        elif roi.label.startswith("Star"):  # star
            w = star_win
        else:
            continue
        cx, cy = roi.center_x, roi.center_y
        x0 = int(round(cx)) - w
        y0 = int(round(cy)) - w
        x1 = x0 + 2 * w + 1
        y1 = y0 + 2 * w + 1
        if x0 < 0 or y0 < 0 or x1 > dv.width or y1 > dv.height:
            continue
        off = _grad_centroid(dv.volume[roi.slice_idx, y0:y1, x0:x1])
        if off is None:
            continue
        d = float(np.hypot(off[0], off[1]))
        if roi.shape_type == "square":
            tex_off[f"{roi.section}/{roi.label}"].append(d)
        else:
            star_off[roi.section].append(d)

    print("\n==== STAR centering offset (px) [ROI centre vs gradient centroid] ====")
    for sec in sorted(star_off):
        a = np.asarray(star_off[sec])
        print(f"  {sec}: n={len(a):3d}  mean={a.mean():.2f}  median={np.median(a):.2f}  "
              f"p95={np.percentile(a,95):.2f}  max={a.max():.2f}  "
              f"(mm: mean={a.mean()*avg_sp:.2f} max={a.max()*avg_sp:.2f})")

    print("\n==== TEXTURE centering offset (px) [ROI centre vs gradient centroid] ====")
    for name in sorted(tex_off):
        a = np.asarray(tex_off[name])
        print(f"  {name}: n={len(a):3d}  mean={a.mean():.2f}  "
              f"p95={np.percentile(a,95):.2f}  max={a.max():.2f}  "
              f"(mm mean={a.mean()*avg_sp:.2f})")


if __name__ == "__main__":
    main()
