"""Test tilt-aware texture detection on a synthetic rotated phantom."""
import os
import sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.roi_extractor import _refine_texture_center_via_edges


def make_phantom_with_4_textures(tilt_deg=0.0, h=400, w=400, phantom_r=180,
                                  tex_dist=113, tex_side=66):
    """Create a synthetic phantom with 4 textures at top/right/bottom/left,
    optionally rotated by tilt_deg about the phantom centre."""
    img = np.full((h, w), -1000.0, dtype=np.float32)
    cx_p, cy_p = w // 2, h // 2
    yy, xx = np.mgrid[:h, :w]
    inside = (xx - cx_p) ** 2 + (yy - cy_p) ** 2 < phantom_r ** 2
    img[inside] = 0.0

    # Draw 4 textures at 0, 90, 180, 270 degrees from phantom centre,
    # each rotated by tilt_deg about the phantom centre. The texture
    # itself also rotates with the phantom (rigid).
    rng = np.random.default_rng(42)
    true_centers = {}
    for nominal in [0, 90, 180, 270]:
        ang_rad = np.deg2rad(nominal + tilt_deg)
        cx_t = cx_p + tex_dist * np.cos(ang_rad)
        cy_t = cy_p + tex_dist * np.sin(ang_rad)
        true_centers[nominal] = (cx_t, cy_t)

    # Draw the textures by composing them on a working canvas then
    # rotating the canvas by tilt_deg about the phantom centre.
    # Simpler: directly draw rotated rectangles using cv2.fillPoly.
    for nominal in [0, 90, 180, 270]:
        cx_t, cy_t = true_centers[nominal]
        # Texture corners in TEXTURE frame (axis-aligned)
        corners = np.array([
            [-tex_side/2, -tex_side/2],
            [tex_side/2, -tex_side/2],
            [tex_side/2, tex_side/2],
            [-tex_side/2, tex_side/2],
        ])
        ar = np.deg2rad(tilt_deg)
        # Rotate each corner by tilt_deg
        rot_corners = np.zeros_like(corners)
        for i, (dx, dy) in enumerate(corners):
            rot_corners[i, 0] = dx * np.cos(ar) - dy * np.sin(ar) + cx_t
            rot_corners[i, 1] = dx * np.sin(ar) + dy * np.cos(ar) + cy_t
        pts = np.round(rot_corners).astype(np.int32)
        cv2.fillPoly(img, [pts], 100.0)

        # Add a frame to make the outer edge crisp (simulate texture material)
        # Already covered by the fillPoly + sharp transition to plastic.

        # Add a sinusoidal pattern oriented along the texture's axis
        for k in range(-tex_side//2 + 3, tex_side//2 - 3, 2):
            # Make a thin bright strip along the texture's local x axis
            pass  # skip interior for now to focus on frame detection

        # Add scattered honeycomb-like dots
        for _ in range(30):
            ddx = rng.integers(-tex_side // 2 + 4, tex_side // 2 - 4)
            ddy = rng.integers(-tex_side // 2 + 4, tex_side // 2 - 4)
            # Rotate dot offset
            dx_w = ddx * np.cos(ar) - ddy * np.sin(ar)
            dy_w = ddx * np.sin(ar) + ddy * np.cos(ar)
            px = int(round(cx_t + dx_w))
            py = int(round(cy_t + dy_w))
            img[max(0,py-2):min(h,py+3), max(0,px-2):min(w,px+3)] = 600.0

    # Apply PVE blur to the wall
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    # Add noise
    img += rng.normal(0, 15, img.shape).astype(np.float32)
    return img.astype(np.float32), (cx_p, cy_p, phantom_r), true_centers


def main():
    print("=== Tilt-aware texture detection test ===\n")
    print(f"{'tilt':>5s} {'angle':>5s} {'true_pos':>15s} {'detected':>15s} "
          f"{'err_no_tilt':>11s} {'err_with_tilt':>13s}")

    for tilt in [0, 1, 2, 5, 10, 15, 20, 30]:
        img, (cx_p, cy_p, phantom_r), true_centers = make_phantom_with_4_textures(
            tilt_deg=tilt
        )
        for nominal in [0, 90, 180, 270]:
            tx, ty = true_centers[nominal]
            seed_x, seed_y = tx, ty  # use perfect seed

            # WITHOUT tilt correction
            r_no = _refine_texture_center_via_edges(
                img, seed_x, seed_y, texture_side_px=66.0,
                phantom_cx=cx_p, phantom_cy=cy_p, phantom_r=phantom_r,
                tilt_deg=0.0,
            )
            err_no = (float('nan') if r_no is None else
                      ((r_no[0]-tx)**2 + (r_no[1]-ty)**2)**0.5)

            # WITH tilt correction
            r_with = _refine_texture_center_via_edges(
                img, seed_x, seed_y, texture_side_px=66.0,
                phantom_cx=cx_p, phantom_cy=cy_p, phantom_r=phantom_r,
                tilt_deg=tilt,
            )
            err_with = (float('nan') if r_with is None else
                        ((r_with[0]-tx)**2 + (r_with[1]-ty)**2)**0.5)

            print(f"{tilt:>5d} {nominal:>5d} ({tx:>5.1f},{ty:>5.1f}) "
                  f"{'(skip)' if r_no is None else f'({r_no[0]:>5.1f},{r_no[1]:>5.1f})':>15s} "
                  f"{err_no:>11.2f} {err_with:>13.2f}")


if __name__ == "__main__":
    main()
