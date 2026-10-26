"""Pinpoint the STAR-section centering geometry: wall centre, star centre,
texture placed vs true centres — on the SECTION MEAN image (high SNR)."""
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
from core.roi_extractor import extract_rois, _refine_star_center
import config


def _bbox_center(win, sq_px):
    v = win.astype(np.float32)
    lo, hi = np.percentile(v, 5), np.percentile(v, 97)
    if hi - lo < 15:
        return None
    norm = np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    _t, b = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ks = max(3, int(round(sq_px / 6)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(b, 8)
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
    return (x + (bw - 1) / 2.0 - cxw, y + (bh - 1) / 2.0 - cyw)


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

    # placed texture/star centres (median across slices), per section
    placed = defaultdict(dict)
    star_placed = {}
    sec_slices = defaultdict(list)
    for r in extraction.rois:
        if r.shape_type == "square":
            placed[r.section].setdefault(r.label, []).append((r.center_x, r.center_y))
        elif r.label.startswith("Star"):
            star_placed.setdefault(r.section, []).append((r.center_x, r.center_y))
        sec_slices[r.section].append(r.slice_idx)

    for sec_name in ("STAR16", "STAR8"):
        if sec_name not in placed:
            continue
        sl = sorted(set(sec_slices[sec_name]))
        mean_img = np.mean(dv.volume[sl], axis=0).astype(np.float32)
        # wall centre (section median of phantom track)
        wcx = float(np.median([phantom.circles[i].center_x for i in sl]))
        wcy = float(np.median([phantom.circles[i].center_y for i in sl]))
        wr = float(np.median([phantom.circles[i].radius for i in sl]))
        # star centre on mean image
        rs = _refine_star_center(mean_img, wcx, wcy,
                                 dv.cm_to_pixels(config.STAR_RADIUS_CM),
                                 phantom_cx=wcx, phantom_cy=wcy, phantom_r=wr)
        spx = np.median([p[0] for p in star_placed[sec_name]])
        spy = np.median([p[1] for p in star_placed[sec_name]])
        print(f"\n===== {sec_name} (mean of {len(sl)} slices) =====")
        print(f"  wall centre      = ({wcx:7.2f},{wcy:7.2f})")
        print(f"  star placed      = ({spx:7.2f},{spy:7.2f})")
        if rs is not None:
            print(f"  star TRUE (mean) = ({rs[0]:7.2f},{rs[1]:7.2f})  "
                  f"placed-true=({spx-rs[0]:+.2f},{spy-rs[1]:+.2f}) "
                  f"|{np.hypot(spx-rs[0],spy-rs[1]):.2f}px|")
        print(f"  {'texture':12s} {'placed(x,y)':>16s} {'true(x,y)':>16s} "
              f"{'placed-true':>14s}  |off|")
        for label in sorted(placed[sec_name]):
            pts = placed[sec_name][label]
            px = float(np.median([p[0] for p in pts]))
            py = float(np.median([p[1] for p in pts]))
            x0, y0 = int(round(px)) - half, int(round(py)) - half
            off = _bbox_center(mean_img[y0:y0 + 2 * half, x0:x0 + 2 * half], sq_px)
            if off is None:
                print(f"  {label:12s} ({px:7.1f},{py:7.1f})   <true n/a>")
                continue
            tx, ty = px + off[0], py + off[1]
            print(f"  {label:12s} ({px:7.1f},{py:7.1f}) ({tx:7.1f},{ty:7.1f}) "
                  f"({px-tx:+6.2f},{py-ty:+6.2f}) {np.hypot(px-tx,py-ty):5.2f}")


if __name__ == "__main__":
    main()
