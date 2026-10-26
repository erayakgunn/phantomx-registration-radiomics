"""Test the texture detector against deliberately adversarial synthetic
cases that mimic the failure modes observed on real data."""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.roi_extractor import _refine_texture_center_via_edges


def make_phantom_with_texture(
    h=400, w=400,
    phantom_cx=200, phantom_cy=200, phantom_r=180,
    tex_cx=200, tex_cy=87,  # ~113 px above centre, near phantom wall
    tex_side=66,
    interior_pattern="honeycomb",
    pattern_density=1.0,
    asymmetric_bias=0.0,
):
    """Make a synthetic CT-like image of a phantom with one texture."""
    img = np.full((h, w), -1000.0, dtype=np.float32)  # outside phantom = air
    yy, xx = np.mgrid[:h, :w]
    inside = (xx - phantom_cx) ** 2 + (yy - phantom_cy) ** 2 < phantom_r ** 2
    img[inside] = 0.0  # plastic matrix
    # Add PVE blur to the wall
    import cv2
    img = cv2.GaussianBlur(img, (5, 5), 1.0)

    # Draw texture rectangle
    half = tex_side // 2
    x0, x1 = tex_cx - half, tex_cx + half
    y0, y1 = tex_cy - half, tex_cy + half
    img[y0:y1, x0:x1] = 100.0  # texture matrix

    # Add interior pattern
    rng = np.random.default_rng(42)
    if interior_pattern == "honeycomb":
        # Scattered dots
        n_dots = int(80 * pattern_density)
        for _ in range(n_dots):
            dx = rng.integers(-half + 3, half - 3)
            dy = rng.integers(-half + 3, half - 3)
            # Asymmetric bias: shift dots toward one side
            if rng.random() < asymmetric_bias and dx > 0:
                continue
            cx, cy = tex_cx + dx, tex_cy + dy
            img[max(0,cy-2):min(h,cy+3), max(0,cx-2):min(w,cx+3)] = 600.0
    elif interior_pattern == "sinusoidal":
        for y in range(y0, y1):
            for x in range(x0, x1):
                phase = (x - x0) * 0.4
                img[y, x] = 100 + 500 * np.sin(phase)
    elif interior_pattern == "vertical_bars":
        bar_pitch = 5
        for x in range(x0, x1, bar_pitch):
            img[y0:y1, x:x+2] = 800.0

    # Add Gaussian noise
    img += rng.normal(0, 20, img.shape).astype(np.float32)
    return img.astype(np.float32), (tex_cx, tex_cy), (phantom_cx, phantom_cy, phantom_r)


def main():
    print("=== Synthetic adversarial tests ===\n")

    scenarios = [
        ("Symmetric honeycomb, mid-phantom", dict(
            tex_cx=200, tex_cy=87, interior_pattern="honeycomb",
            pattern_density=1.0, asymmetric_bias=0.0)),
        ("Asymmetric honeycomb (biased seed)", dict(
            tex_cx=200, tex_cy=87, interior_pattern="honeycomb",
            pattern_density=1.5, asymmetric_bias=0.7)),
        ("Sinusoidal, close to wall", dict(
            tex_cx=200, tex_cy=75, interior_pattern="sinusoidal",
            pattern_density=1.0, asymmetric_bias=0.0)),
        ("Vertical bars (strong gradients)", dict(
            tex_cx=200, tex_cy=87, interior_pattern="vertical_bars",
            pattern_density=1.0, asymmetric_bias=0.0)),
        ("Honeycomb near wall (test mask)", dict(
            tex_cx=200, tex_cy=50, interior_pattern="honeycomb",
            pattern_density=1.0, asymmetric_bias=0.0)),
    ]

    bad_seeds = [
        ("perfect seed", (0, 0)),
        ("seed +5 px", (5, 0)),
        ("seed -5 px", (-5, 0)),
        ("seed +10 px inward", (0, +10)),
        ("seed +15 px inward", (0, +15)),
    ]

    for sname, params in scenarios:
        img, (tx, ty), (px, py, pr) = make_phantom_with_texture(**params)
        print(f"\n[{sname}]  True center: ({tx}, {ty})")
        for seed_label, (dx, dy) in bad_seeds:
            seed_x = tx + dx
            seed_y = ty + dy
            result = _refine_texture_center_via_edges(
                img, seed_x, seed_y, texture_side_px=66.0,
                phantom_cx=px, phantom_cy=py, phantom_r=pr,
            )
            if result is None:
                print(f"  {seed_label:25s} seed=({seed_x},{seed_y})  REJECTED")
            else:
                err = ((result[0] - tx)**2 + (result[1] - ty)**2)**0.5
                marker = "  <- LARGE ERROR" if err > 3 else ""
                print(f"  {seed_label:25s} seed=({seed_x},{seed_y})  refined=({result[0]:6.2f},{result[1]:6.2f})  err={err:.2f} px{marker}")


if __name__ == "__main__":
    main()
