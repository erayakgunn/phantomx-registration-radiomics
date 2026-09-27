"""Debug star-center polish: compare what my contrast-centroid finds vs
what HoughCircles finds vs visual inspection."""
import os, sys, numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.dicom_loader import load_series
from core.phantom_detector import detect_phantom, SliceStatus
from core.section_classifier import classify_sections
from core.orientation_corrector import canonicalize_orientation
from core.section_trim import trim_sections_to_inner_window
from core.roi_extractor import _refine_star_center
import config


def hough_star_center(mean_img, seed_x, seed_y, star_r_px):
    """Locate the star centre by Hough Circle on the mean image."""
    search_half = int(star_r_px * 2.0)
    h, w = mean_img.shape
    ix, iy = int(round(seed_x)), int(round(seed_y))
    x0 = max(0, ix - search_half); y0 = max(0, iy - search_half)
    x1 = min(w, ix + search_half + 1); y1 = min(h, iy + search_half + 1)
    crop = mean_img[y0:y1, x0:x1].astype(np.float32)
    p1, p99 = np.percentile(crop, [1, 99])
    if p99 - p1 < 10:
        return None
    u8 = np.clip((crop - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (5, 5), 1.0)
    # Try a couple of param2 settings
    for p2 in (30, 20, 15, 10):
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1,
            minDist=int(star_r_px * 1.5),
            param1=80, param2=p2,
            minRadius=int(star_r_px * 0.7),
            maxRadius=int(star_r_px * 1.3),
        )
        if circles is not None and len(circles[0]) > 0:
            cx_l, cy_l, r = circles[0][0]
            return cx_l + x0, cy_l + y0, r, p2
    return None


def main():
    input_root = "C:/input_pcct_clean"
    for d in sorted(os.listdir(input_root)):
        full = os.path.join(input_root, d)
        if not os.path.isdir(full): continue
        print(f"\n{'='*78}\n{d}\n{'='*78}")
        dv = load_series(full)
        phantom = detect_phantom(dv)
        sections = classify_sections(dv, phantom)
        canonicalize_orientation(dv, phantom, sections)
        trim_sections_to_inner_window(dv, sections, phantom=phantom)
        gcx, gcy = phantom.global_center_x, phantom.global_center_y
        star_r_px = dv.cm_to_pixels(config.STAR_RADIUS_CM) * config.ROI_SCALE
        print(f"global Hough centre: ({gcx:.2f}, {gcy:.2f})  star_r_px={star_r_px:.1f}")

        for sec in sections.sections:
            if sec.name not in ("STAR16", "STAR8"): continue
            valid = [
                i for i in range(sec.start_idx, sec.end_idx)
                if sections.slice_labels[i] != "TRANSITION"
                and phantom.circles[i].status == SliceStatus.VALID
            ]
            if not valid: continue
            # Section median Hough centre = fx, fy in roi_extractor
            xs = [phantom.circles[i].center_x for i in valid]
            ys = [phantom.circles[i].center_y for i in valid]
            fx, fy = float(np.median(xs)), float(np.median(ys))
            mean_img = np.mean(dv.volume[list(valid)], axis=0, dtype=np.float64).astype(np.float32)
            # Inspect a small crop around seed
            crop_half = int(star_r_px * 1.5)
            ix, iy = int(round(fx)), int(round(fy))
            x0, y0 = max(0, ix - crop_half), max(0, iy - crop_half)
            x1, y1 = ix + crop_half + 1, iy + crop_half + 1
            crop = mean_img[y0:y1, x0:x1]
            print(f"\n  {sec.name}: {len(valid)} VALID slices")
            print(f"    Seed (section-median Hough): ({fx:.2f}, {fy:.2f})")
            print(f"    Crop HU stats: min={crop.min():.0f} max={crop.max():.0f} "
                  f"median={np.median(crop):.0f} p1={np.percentile(crop,1):.0f} "
                  f"p99={np.percentile(crop,99):.0f}")

            # My contrast-centroid polish
            r1 = _refine_star_center(mean_img, fx, fy, star_r_px,
                                       phantom_cx=gcx, phantom_cy=gcy,
                                       phantom_r=phantom.global_radius)
            if r1:
                print(f"    My polish: ({r1[0]:.2f}, {r1[1]:.2f})  "
                      f"shift=({r1[0]-fx:+.2f}, {r1[1]-fy:+.2f})")
            else:
                print(f"    My polish returned None")

            # Hough circle
            h = hough_star_center(mean_img, fx, fy, star_r_px)
            if h:
                cx, cy, r, p2 = h
                print(f"    Hough circle: ({cx:.2f}, {cy:.2f}) r={r:.1f}  p2={p2}  "
                      f"shift=({cx-fx:+.2f}, {cy-fy:+.2f})")
            else:
                print(f"    Hough circle: not found")


if __name__ == "__main__":
    main()
