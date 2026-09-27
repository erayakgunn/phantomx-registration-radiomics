"""Test which rotation direction is correct for tilt correction."""
import os
import sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_test_image(tilt_deg, h=300, w=300, tex_cx=150, tex_cy=70, tex_side=66):
    """Make a single rotated texture rectangle."""
    img = np.zeros((h, w), dtype=np.float32)

    # Corners of axis-aligned rectangle
    corners = np.array([
        [-tex_side/2, -tex_side/2],
        [tex_side/2, -tex_side/2],
        [tex_side/2, tex_side/2],
        [-tex_side/2, tex_side/2],
    ])
    ar = np.deg2rad(tilt_deg)
    rot_corners = np.zeros_like(corners)
    for i, (dx, dy) in enumerate(corners):
        rot_corners[i, 0] = dx * np.cos(ar) - dy * np.sin(ar) + tex_cx
        rot_corners[i, 1] = dx * np.sin(ar) + dy * np.cos(ar) + tex_cy

    pts = np.round(rot_corners).astype(np.int32)
    cv2.fillPoly(img, [pts], 100.0)
    return img, (tex_cx, tex_cy)


def test_rotation(img, seed_xy, tex_side, opencv_angle, label):
    """Rotate the crop by opencv_angle and check if the texture becomes axis-aligned."""
    h, w = img.shape
    half_win = int(tex_side * 1.0)
    sx, sy = seed_xy
    x0 = sx - half_win
    y0 = sy - half_win
    crop = img[y0:y0+2*half_win, x0:x0+2*half_win]

    # Apply rotation
    seed_local = (half_win, half_win)
    M = cv2.getRotationMatrix2D((float(seed_local[0]), float(seed_local[1])),
                                  float(opencv_angle), 1.0)
    rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

    # Check how axis-aligned the rotated texture is
    blurred = cv2.GaussianBlur(rotated, (3, 3), 0.6)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    col_profile = np.abs(gx).sum(axis=0)
    row_profile = np.abs(gy).sum(axis=1)

    # If axis-aligned: col_profile and row_profile each have 2 strong peaks
    # ~tex_side apart. Otherwise: smeared peaks.
    cmax = col_profile.max()
    rmax = row_profile.max()
    # Count how "peaky" the profile is — concentrate of energy in top 4 indices
    csorted = np.sort(col_profile)[::-1]
    rsorted = np.sort(row_profile)[::-1]
    c_top4 = csorted[:4].sum() / col_profile.sum()
    r_top4 = rsorted[:4].sum() / row_profile.sum()
    print(f"  {label} (OpenCV angle={opencv_angle:+.0f}°): "
          f"col_top4={c_top4:.3f} row_top4={r_top4:.3f} "
          f"max_c={cmax:.0f} max_r={rmax:.0f}")
    return c_top4 + r_top4


def main():
    for tilt in [5, 10, 15, 20]:
        img, (tx, ty) = make_test_image(tilt_deg=tilt)
        print(f"\nTexture at tilt={tilt}°:")
        score_pos = test_rotation(img, (tx, ty), 66, +tilt, "use +tilt")
        score_neg = test_rotation(img, (tx, ty), 66, -tilt, "use -tilt")
        score_zero = test_rotation(img, (tx, ty), 66, 0, "no rotation")
        if score_pos > max(score_neg, score_zero):
            print(f"  WINNER: +tilt (un-rotated best)")
        elif score_neg > score_zero:
            print(f"  WINNER: -tilt")
        else:
            print(f"  WINNER: no rotation")


if __name__ == "__main__":
    main()
