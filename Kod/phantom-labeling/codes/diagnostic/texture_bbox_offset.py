"""Pattern-INDEPENDENT texture centering error via square bounding-box centre.

The 4 texture inserts are physical squares (bright plastic frame + internal
pattern) sitting in air. Otsu-threshold the window, morphologically CLOSE to
fill the internal pattern gaps so the square becomes one solid blob, take the
blob nearest the window centre, and use its BOUNDING-BOX centre (not centroid:
the bbox depends only on the square's extent, so it is immune to the internal
pattern's intensity asymmetry that biases a centroid). Offset = pipeline ROI
centre - bbox centre. Run with LEGACY_TEXTURE_ANGLE=1 vs unset to A/B the fix.
"""
import sys
import numpy as np
import cv2
from collections import defaultdict

from core.dicom_loader import load_series
from core.phantom_detector import detect_phantom
from core.section_classifier import classify_sections
from core.orientation_corrector import canonicalize_orientation
from core.section_trim import trim_sections_to_inner_window
from core.rotation_detector import detect_rotation
from core.roi_extractor import extract_rois
import config


def _bbox_center_offset(win, sq_px):
    v = win.astype(np.float32)
    lo, hi = np.percentile(v, 5), np.percentile(v, 97)
    if hi - lo < 15:
        return None
    norm = np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    _t, b = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # close gaps between internal pattern bars (~ up to 1/6 of square)
    ks = max(3, int(round(sq_px / 6)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(b, 8)
    if n < 2:
        return None
    h, w = win.shape
    cxw, cyw = (w - 1) / 2.0, (h - 1) / 2.0
    # pick component whose centroid is nearest window centre AND whose size is
    # in a plausible square range (reject merges with neighbours / star).
    best, best_d = None, 1e9
    for i in range(1, n):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw < 0.5 * sq_px or bh < 0.5 * sq_px:
            continue
        if bw > 1.8 * sq_px or bh > 1.8 * sq_px:
            continue
        d = np.hypot(cent[i][0] - cxw, cent[i][1] - cyw)
        if d < best_d:
            best_d, best = d, i
    if best is None:
        return None
    x = stats[best, cv2.CC_STAT_LEFT]
    y = stats[best, cv2.CC_STAT_TOP]
    bw = stats[best, cv2.CC_STAT_WIDTH]
    bh = stats[best, cv2.CC_STAT_HEIGHT]
    bcx = x + (bw - 1) / 2.0
    bcy = y + (bh - 1) / 2.0
    return (bcx - cxw, bcy - cyw)


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
    sq_px = config.TEXTURE_SIDE_CM * 10.0 / avg_sp
    half = int(round(sq_px / 2)) + int(round(8 / avg_sp))

    rows = defaultdict(list)
    sec_all = defaultdict(list)
    for r in extraction.rois:
        if r.shape_type != "square":
            continue
        x0, y0 = int(round(r.center_x)) - half, int(round(r.center_y)) - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half >= dv.width or y0 + 2 * half >= dv.height:
            continue
        off = _bbox_center_offset(dv.volume[r.slice_idx, y0:y0 + 2 * half,
                                            x0:x0 + 2 * half], sq_px)
        if off is None:
            continue
        d = float(np.hypot(off[0], off[1]))
        rows[f"{r.section}/{r.label}"].append(d)
        sec_all[r.section].append(d)

    mode = "LEGACY (detected angle)" if __import__("os").environ.get(
        "LEGACY_TEXTURE_ANGLE") == "1" else "GEOMETRIC (rigid frame)"
    print(f"\n==== TEXTURE bbox-centre offset [{mode}] px (1px={avg_sp:.3f}mm) ====")
    for name in sorted(rows):
        a = np.asarray(rows[name])
        print(f"  {name:22s} n={len(a):3d}  mean={a.mean():.2f}  "
              f"p95={np.percentile(a,95):.2f}  max={a.max():.2f}")
    print("  --- section mean ---")
    for s in sorted(sec_all):
        a = np.asarray(sec_all[s])
        print(f"  {s}: mean={a.mean():.2f}px ({a.mean()*avg_sp:.2f}mm)  n={len(a)}")
    alld = np.concatenate([np.asarray(v) for v in sec_all.values()])
    print(f"  ALL textures: mean={alld.mean():.2f}px  p95={np.percentile(alld,95):.2f}px")


if __name__ == "__main__":
    main()
