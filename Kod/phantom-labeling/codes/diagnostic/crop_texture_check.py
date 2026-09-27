"""Zoom crops around each texture ROI to visually confirm centering error.

Draws the pipeline ROI box (green) and the independent gradient-centroid
(red dot) on an upscaled crop, so any off-centering is obvious.
"""
import sys
import numpy as np
import cv2

from core.dicom_loader import load_series
from core.phantom_detector import detect_phantom
from core.section_classifier import classify_sections
from core.orientation_corrector import canonicalize_orientation
from core.section_trim import trim_sections_to_inner_window
from core.rotation_detector import detect_rotation
from core.roi_extractor import extract_rois
import config


def _grad_centroid(win):
    """Reliable pattern-INDEPENDENT square centre via Otsu+close bounding box."""
    v = win.astype(np.float32)
    lo, hi = np.percentile(v, 5), np.percentile(v, 97)
    if hi - lo < 15:
        return None
    sq_px = win.shape[0] - 2 * 6  # window was square side + ~12 px margin
    norm = np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    _t, b = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ks = max(3, int(round(sq_px / 6)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(b, 8)
    if n < 2:
        return None
    h, w = win.shape
    cxw, cyw = (w - 1) / 2.0, (h - 1) / 2.0
    best, bestd = None, 1e9
    for i in range(1, n):
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if bw < 0.5 * sq_px or bh < 0.5 * sq_px or bw > 1.8 * sq_px or bh > 1.8 * sq_px:
            continue
        d = np.hypot(cent[i][0] - cxw, cent[i][1] - cyw)
        if d < bestd:
            bestd, best = d, i
    if best is None:
        return None
    x, y = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP]
    bw, bh = stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    return (x + (bw - 1) / 2.0, y + (bh - 1) / 2.0)


def main():
    series = sys.argv[1]
    target_slice = int(sys.argv[2])
    out_png = sys.argv[3]
    dv = load_series(series)
    phantom = detect_phantom(dv)
    sections = classify_sections(dv, phantom)
    canonicalize_orientation(dv, phantom, sections)
    trim_sections_to_inner_window(dv, sections, phantom=phantom)
    rotation = detect_rotation(dv, phantom, sections)
    extraction = extract_rois(dv, phantom, sections, rotation)

    avg_sp = (dv.pixel_spacing[0] + dv.pixel_spacing[1]) / 2.0
    half = int(round(config.TEXTURE_SIDE_CM * 10.0 / 2.0 / avg_sp)) + 6
    rois = [r for r in extraction.per_slice.get(target_slice, [])
            if r.shape_type == "square"]
    if not rois:
        print(f"no texture ROIs at slice {target_slice}; "
              f"available: {sorted(extraction.per_slice)[:5]}...")
        return

    tiles = []
    for r in rois:
        cx, cy = r.center_x, r.center_y
        x0, y0 = int(round(cx)) - half, int(round(cy)) - half
        win = dv.volume[target_slice, y0:y0 + 2 * half, x0:x0 + 2 * half]
        # window/level to 8-bit
        img = np.clip((win + 200) / 800 * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        z = 6
        img = cv2.resize(img, (img.shape[1] * z, img.shape[0] * z),
                         interpolation=cv2.INTER_NEAREST)
        # ROI box (green) — centred at window centre (cx,cy maps to (half,half))
        roi_half = config.TEXTURE_SIDE_CM * 10.0 / 2.0 / avg_sp * config.ROI_SCALE
        bx = int(round((half - roi_half) * z))
        by = int(round((half - roi_half) * z))
        bx2 = int(round((half + roi_half) * z))
        by2 = int(round((half + roi_half) * z))
        cv2.rectangle(img, (bx, by), (bx2, by2), (0, 255, 0), 2)
        cv2.drawMarker(img, (half * z, half * z), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2)  # ROI centre
        # gradient centroid (red)
        gc = _grad_centroid(win)
        if gc is not None:
            gx_, gy_ = gc
            cv2.drawMarker(img, (int(gx_ * z), int(gy_ * z)), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 20, 2)
            d = np.hypot(gx_ - half, gy_ - half)
            cv2.putText(img, f"{r.label} off={d:.1f}px", (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        tiles.append(img)

    # arrange in a row
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 8,
                                cv2.BORDER_CONSTANT, value=(40, 40, 40))
             for t in tiles]
    grid = np.hstack(tiles)
    cv2.imwrite(out_png, grid)
    print(f"wrote {out_png}  ({grid.shape[1]}x{grid.shape[0]})  "
          f"green=ROI centre, red=gradient centroid")


if __name__ == "__main__":
    main()
