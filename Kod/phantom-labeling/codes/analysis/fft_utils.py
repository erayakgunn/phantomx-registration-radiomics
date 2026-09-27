"""FFT and polar transform utilities."""

import numpy as np
from scipy import ndimage
from typing import Tuple


# Minimum pixels per arm a sampling ring must offer before its arm count is
# believed. Two pixels per cycle is the hard sampling limit; 3 leaves room
# for the reconstruction's own blur.
_MIN_PX_PER_ARM = 3.0


def angular_symmetry(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
    order_lo: int = 4,
    order_hi: int = 24,
    n_angles: int = 720,
) -> Tuple[int, float]:
    """Rotational symmetry of the disc at ``(cx, cy)``, read in POLAR space.

    Returns ``(order, modulation_depth)``:

      * ``order`` — the dominant rotational order, i.e. the arm count.
      * ``modulation_depth`` — the HU swing at that order as a fraction of
        the disc's own 5-95 percentile contrast. This is what says whether
        there is a star THERE AT ALL: a genuine star module measures
        0.38-0.58, material with no angular structure measures below 0.05.

    Why this rather than the angular profile of the FFT magnitude
    -------------------------------------------------------------
    In FFT-of-FFT space an 8-arm star's order-16 harmonic outruns its own
    order-8 fundamental (measured p16/p8 = 1.26 on real STAR8 slices vs
    2.70 on STAR16), so a rule that compares those two powers claims
    "STAR16" for a genuine STAR8. Counting the arms directly in the image
    has no such ambiguity — on slices whose 8-fold identity is visually
    unambiguous this returns 8 with a dead-flat 0.57 modulation, while the
    old discriminator answered STAR16 or UNKNOWN on 18 of 22 of them.

    Sampling is restricted to rings that can actually RESOLVE ``order_hi``
    arms: arm width shrinks toward the hub, where neighbouring arms merge
    into the star's solid centre, so an inner ring reports a degraded count
    no matter how good the data is. The per-ring angular profiles are
    averaged BEFORE the FFT — the arms are radial wedges, so every ring
    sees the same angular phase and the true order adds coherently while
    per-ring noise averages down.
    """
    h, w = image.shape
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)

    usable = [radius_px * f for f in np.arange(0.35, 0.86, 0.05)
              if 2.0 * np.pi * radius_px * f / order_hi >= _MIN_PX_PER_ARM]
    if not usable:
        usable = [radius_px * f for f in (0.75, 0.80, 0.85)]

    profiles = []
    samples = []
    for r in usable:
        ys = np.clip(np.rint(cy + r * np.sin(angles)).astype(int), 0, h - 1)
        xs = np.clip(np.rint(cx + r * np.cos(angles)).astype(int), 0, w - 1)
        p = image[ys, xs].astype(np.float64)
        samples.append(p)
        profiles.append(p - p.mean())
    if not profiles:
        return 0, 0.0

    mean_profile = np.mean(profiles, axis=0)
    spectrum = np.abs(np.fft.rfft(mean_profile))
    if len(spectrum) <= order_hi:
        return 0, 0.0
    band = spectrum[order_lo:order_hi + 1]
    k = int(np.argmax(band))

    amplitude = 2.0 * float(band[k]) / len(mean_profile)
    pool = np.concatenate(samples)
    contrast = float(np.percentile(pool, 95) - np.percentile(pool, 5))
    return order_lo + k, float(amplitude / (contrast + 1e-9))


def extract_circular_patch(
    image: np.ndarray, cx: float, cy: float, radius_px: float
) -> np.ndarray:
    """Extract a square patch centered at (cx, cy) containing the circle of given radius.
    Pixels outside the circle are set to the mean of pixels inside."""
    r = int(np.ceil(radius_px))
    x0 = int(round(cx)) - r
    y0 = int(round(cy)) - r
    x1 = x0 + 2 * r
    y1 = y0 + 2 * r

    h, w = image.shape
    # Clamp to image bounds
    px0 = max(0, x0)
    py0 = max(0, y0)
    px1 = min(w, x1)
    py1 = min(h, y1)

    patch = np.full((2 * r, 2 * r), np.nan, dtype=np.float64)
    patch[py0 - y0 : py1 - y0, px0 - x0 : px1 - x0] = image[py0:py1, px0:px1]

    # Create circular mask
    yy, xx = np.ogrid[:2 * r, :2 * r]
    dist = np.sqrt((xx - r) ** 2 + (yy - r) ** 2)
    mask = dist <= radius_px

    # Fill outside circle and NaN with mean
    valid = mask & ~np.isnan(patch)
    if valid.any():
        mean_val = np.nanmean(patch[valid])
        patch[~valid] = mean_val
        patch[np.isnan(patch)] = mean_val
    else:
        patch[:] = 0

    return patch


def compute_fft_magnitude(patch: np.ndarray) -> np.ndarray:
    """Compute centered 2D FFT magnitude spectrum."""
    f = np.fft.fft2(patch)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)
    # Log scale for better visualization
    magnitude = np.log1p(magnitude)
    return magnitude


def cart_to_polar(
    image: np.ndarray, center: Tuple[float, float] = None, num_angles: int = 360, num_radii: int = None
) -> np.ndarray:
    """Convert image from Cartesian to polar coordinates.

    Returns array of shape (num_angles, num_radii) where each row is
    one angular slice from center outward.
    """
    h, w = image.shape
    if center is None:
        center = (h / 2.0, w / 2.0)
    if num_radii is None:
        num_radii = min(h, w) // 2

    max_radius = min(center[0], center[1], h - center[0], w - center[1])

    polar = np.zeros((num_angles, num_radii), dtype=np.float64)
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    radii = np.linspace(0, max_radius, num_radii)

    for ai, angle in enumerate(angles):
        for ri, radius in enumerate(radii):
            y = center[0] + radius * np.sin(angle)
            x = center[1] + radius * np.cos(angle)
            iy, ix = int(round(y)), int(round(x))
            if 0 <= iy < h and 0 <= ix < w:
                polar[ai, ri] = image[iy, ix]

    return polar


def angular_profile(polar_image: np.ndarray) -> np.ndarray:
    """Compute mean radial profile for each angle."""
    return np.mean(polar_image, axis=1)


def count_peaks(profile: np.ndarray, min_prominence: float = None) -> int:
    """Count peaks in a 1D profile using scipy."""
    from scipy.signal import find_peaks

    if min_prominence is None:
        min_prominence = (np.max(profile) - np.min(profile)) * 0.15

    peaks, _ = find_peaks(profile, prominence=min_prominence)
    return len(peaks)
