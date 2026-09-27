"""Save annotated star-centre crops for visual debugging.

For each series and section, builds the mean image, crops a window around
the seed phantom centre, and saves a high-resolution annotated PNG that
shows:
  * the actual star pattern (raw pixels)
  * the seed (Hough phantom centre) as a YELLOW crosshair
  * the polished star centre as a CYAN crosshair
  * the threshold-based star pixel mask as a faint magenta overlay

This makes it easy to eyeball whether the polish hit the geometric centre.
"""
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


def main():
    out_dir = "C:/output_pcct/star_debug"
    os.makedirs(out_dir, exist_ok=True)
    input_root = "C:/input_pcct_clean"
    for d in sorted(os.listdir(input_root)):
        full = os.path.join(input_root, d)
        if not os.path.isdir(full): continue
        dv = load_series(full)
        phantom = detect_phantom(dv)
        sections = classify_sections(dv, phantom)
        canonicalize_orientation(dv, phantom, sections)
        trim_sections_to_inner_window(dv, sections, phantom=phantom)
        gcx, gcy = phantom.global_center_x, phantom.global_center_y
        star_r_px = dv.cm_to_pixels(config.STAR_RADIUS_CM) * config.ROI_SCALE
        star_full_r_px = dv.cm_to_pixels(config.STAR_RADIUS_CM)

        for sec in sections.sections:
            if sec.name not in ("STAR16", "STAR8"): continue
            valid = [
                i for i in range(sec.start_idx, sec.end_idx)
                if sections.slice_labels[i] != "TRANSITION"
                and phantom.circles[i].status == SliceStatus.VALID
            ]
            if not valid: continue
            xs = [phantom.circles[i].center_x for i in valid]
            ys = [phantom.circles[i].center_y for i in valid]
            fx, fy = float(np.median(xs)), float(np.median(ys))
            mean_img = np.mean(dv.volume[list(valid)], axis=0, dtype=np.float64).astype(np.float32)
            crop_half = int(star_r_px * 2.5)
            ix, iy = int(round(fx)), int(round(fy))
            x0, y0 = max(0, ix - crop_half), max(0, iy - crop_half)
            x1, y1 = min(mean_img.shape[1], ix + crop_half + 1), min(mean_img.shape[0], iy + crop_half + 1)
            crop = mean_img[y0:y1, x0:x1]

            # Normalize to 0-255 for visualisation (HU −1000..+500 → 0..255)
            disp = np.clip((crop + 1000) / 1500.0 * 255.0, 0, 255).astype(np.uint8)
            # Upscale 6× for clear viewing
            scale = 6
            big = cv2.resize(disp, (disp.shape[1] * scale, disp.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
            big_bgr = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

            # Draw seed crosshair (yellow)
            seed_lx = (fx - x0) * scale
            seed_ly = (fy - y0) * scale
            cv2.drawMarker(big_bgr, (int(seed_lx), int(seed_ly)),
                            (0, 255, 255), cv2.MARKER_CROSS, markerSize=40, thickness=2)
            # Draw star ROI circle at seed
            cv2.circle(big_bgr, (int(seed_lx), int(seed_ly)), int(star_r_px * scale), (0, 255, 255), 1)

            # Run polish (with FULL star radius — see roi_extractor for why)
            r = _refine_star_center(mean_img, fx, fy, star_full_r_px,
                                      phantom_cx=gcx, phantom_cy=gcy,
                                      phantom_r=phantom.global_radius)
            if r is not None:
                pcx, pcy = r
                p_lx = (pcx - x0) * scale
                p_ly = (pcy - y0) * scale
                cv2.drawMarker(big_bgr, (int(p_lx), int(p_ly)),
                                (255, 0, 255), cv2.MARKER_CROSS, markerSize=40, thickness=2)
                cv2.circle(big_bgr, (int(p_lx), int(p_ly)), int(star_r_px * scale),
                            (255, 0, 255), 2)
                label = f"seed=({fx:.2f},{fy:.2f}) polish=({pcx:.2f},{pcy:.2f}) " \
                        f"shift=({pcx-fx:+.2f},{pcy-fy:+.2f})"
            else:
                label = f"seed=({fx:.2f},{fy:.2f}) polish=None"

            cv2.putText(big_bgr, label, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(big_bgr, "YELLOW=seed   MAGENTA=polished",
                        (10, big_bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            fname = f"{d[40:80]}_{sec.name}.png"
            outp = os.path.join(out_dir, fname)
            cv2.imwrite(outp, big_bgr)
            print(f"saved {outp}")


if __name__ == "__main__":
    main()
