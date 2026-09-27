"""Texture validation using Radon transform, FFT, and sine fitting."""

import numpy as np
from scipy import optimize
from typing import Tuple

from utils.logger import log
import config


def validate_texture(pixels: np.ndarray, texture_name: str) -> Tuple[bool, str, float]:
    """Validate a texture ROI against its expected pattern.

    Args:
        pixels: 2D ROI array (may contain NaN for out-of-bounds)
        texture_name: Expected texture type

    Returns:
        (is_valid, method, score)
    """
    # Replace NaN with mean for analysis
    clean = pixels.copy()
    mask = ~np.isnan(clean)
    if mask.any():
        clean[~mask] = np.nanmean(clean)
    else:
        return False, "empty", 0.0

    if texture_name == "Diagonal":
        return _validate_diagonal(clean)
    elif texture_name == "Vertical":
        return _validate_vertical(clean)
    elif texture_name == "Horizontal":
        return _validate_horizontal(clean)
    elif texture_name == "Square":
        return _validate_square(clean)
    elif texture_name in ("Honeylarge", "Honeymedium", "Honeysmall"):
        return _validate_honeycomb(clean)
    elif texture_name == "Sinusoidal":
        return _validate_sinusoidal(clean)
    else:
        return False, "unknown_type", 0.0


def _radon_transform_simple(image: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Simple Radon transform: project image along given angles.

    Returns array of shape (len(angles), image_width).
    """
    from scipy import ndimage

    h, w = image.shape
    center = (h / 2.0, w / 2.0)
    sinogram = np.zeros((len(angles_deg), max(h, w)))

    for i, angle in enumerate(angles_deg):
        rotated = ndimage.rotate(image, -angle, reshape=False, order=0)
        projection = np.sum(rotated, axis=0)
        # Center the projection in the sinogram row
        proj_len = len(projection)
        sino_len = sinogram.shape[1]
        if proj_len >= sino_len:
            start = (proj_len - sino_len) // 2
            sinogram[i, :] = projection[start:start + sino_len]
        else:
            start = (sino_len - proj_len) // 2
            sinogram[i, start:start + proj_len] = projection

    return sinogram


def _radon_peak_strength(image: np.ndarray, target_angles: list) -> float:
    """Measure Radon projection strength at target angles relative to others."""
    angles = np.arange(0, 180, 1)
    sinogram = _radon_transform_simple(image, angles)

    # Variance of each projection
    variances = np.var(sinogram, axis=1)
    if np.max(variances) == 0:
        return 0.0

    # Strength at target angles vs overall
    target_vars = []
    for a in target_angles:
        idx = int(a) % 180
        target_vars.append(variances[idx])

    return float(np.mean(target_vars) / (np.mean(variances) + 1e-10))


def _validate_diagonal(pixels: np.ndarray) -> Tuple[bool, str, float]:
    """Diagonal texture: strong Radon peaks at 45° and 135°."""
    score = _radon_peak_strength(pixels, [45, 135])
    return score > 1.2, "radon_diagonal", score


def _validate_vertical(pixels: np.ndarray) -> Tuple[bool, str, float]:
    """Vertical stripes (running along y-axis, alternating along x).

    Radon projection axes: the transform here rotates by -θ then sums
    along axis=0 (integrates each column). Vertical stripes therefore
    produce their strongest projection variance at θ=0: columns in the
    un-rotated image alternate bright/dark, giving a high-variance row
    in the sinogram. At θ=90 the image is rotated so the stripes are
    horizontal, each column has an identical profile, and variance
    collapses.
    """
    score = _radon_peak_strength(pixels, [0])
    return score > 1.2, "radon_vertical", score


def _validate_horizontal(pixels: np.ndarray) -> Tuple[bool, str, float]:
    """Horizontal stripes (running along x-axis, alternating along y).

    Counterpart of :func:`_validate_vertical`: peak variance is at
    θ=90, because rotating horizontal stripes by -90° makes them
    vertical, which is the high-variance case for a column-sum
    projection.
    """
    score = _radon_peak_strength(pixels, [90])
    return score > 1.2, "radon_horizontal", score


def _validate_square(pixels: np.ndarray) -> Tuple[bool, str, float]:
    """Square/grid texture: two orthogonal Radon peaks at 0° and 90°.

    A true grid shows strong projection variance at BOTH 0° and 90°
    with comparatively low variance at off-axis angles. We quantify
    this as the smaller of (V₀, V₉₀) divided by the median variance
    across all angles — taking the weaker peak keeps a single-direction
    stripe (where only one orthogonal is strong) from scoring as a grid.
    """
    angles = np.arange(0, 180, 1)
    sinogram = _radon_transform_simple(pixels, angles)
    variances = np.var(sinogram, axis=1)
    vmed = float(np.median(variances))
    if vmed < 1e-9:
        return False, "radon_square", 0.0

    v0 = float(variances[0])
    v90 = float(variances[90])
    score = min(v0, v90) / vmed
    return score > 2.0, "radon_square", score


def _angular_peak_count(pixels: np.ndarray) -> int:
    """Count peaks in the mid-frequency angular FFT profile of a patch.

    The peak count tracks the patch's rotational-symmetry order: a
    honeycomb lattice (6-fold) yields ~6-12 peaks, a single-direction
    line grating 2, an orthogonal grid 4.
    """
    from analysis.fft_utils import compute_fft_magnitude, cart_to_polar, count_peaks

    fft_mag = compute_fft_magnitude(pixels)
    center = (fft_mag.shape[0] / 2.0, fft_mag.shape[1] / 2.0)
    polar = cart_to_polar(fft_mag, center=center, num_angles=360)

    # Look at mid-frequency range
    h = polar.shape[1]
    profile = np.mean(polar[:, int(h * 0.15):int(h * 0.5)], axis=1)

    return count_peaks(profile)


def texture_family(pixels: np.ndarray) -> str:
    """Classify one texture-card patch as ``"hex"`` or ``"grid"``.

    ``"hex"``: angular symmetry closer to 6/12-fold — the honeycomb
    cards of the STAR8 module. ``"grid"``: closer to 2/4-fold — the
    bar/checker gratings of the STAR16 module. Unlike
    ``_validate_honeycomb`` (a lenient absolute gate where a 4-peak
    grid still scores 0.67 and passes), this is a differential verdict,
    so it can arbitrate WHICH module a star slice belongs to.
    """
    n = _angular_peak_count(pixels)
    hex_dist = min(abs(n - 6), abs(n - 12))
    grid_dist = min(abs(n - 2), abs(n - 4))
    return "hex" if hex_dist < grid_dist else "grid"


def _lattice_ring_order(pixels: np.ndarray) -> Tuple[int, float]:
    """Return (dominant angular order of the lattice, dominance ratio).

    Two-step, and the FIRST step is the one that matters:

      1. Locate the lattice's own RADIAL frequency — the ring in FFT space
         where its repeat pattern lives. A coarse honeycomb puts that ring
         near the centre, a fine one puts it far out.
      2. Read how many spots sit on THAT ring. Hex lattice -> 6,
         orthogonal grid -> 4, single-direction grating -> 2.

    Deriving the ring from the data is what makes this pitch-independent.
    ``_angular_peak_count`` (still used by ``texture_family`` for section
    classification, deliberately untouched) instead reads a FIXED band at
    0.15-0.5 of the spectrum radius. The finer honeycomb cards
    (Honeysmall/Honeymedium) have their fundamental outside that band, so
    only off-lattice leakage was being counted: it returned 2, and
    ``_validate_honeycomb``'s old ``1 - min(|n-6|,|n-12|)/6`` scoring
    turned that into exactly 0.333 — the score that appears on 229 slices
    of the 336-series PCCT run while the ROIs themselves were provably
    on-card.

    The dominance ratio is the winning order's angular power over the mean
    of the other candidate orders. It is what keeps a featureless patch
    (no lattice at all) from "winning" with a meaningless order.
    """
    clean = pixels.astype(np.float64, copy=True)
    bad = np.isnan(clean)
    if bad.all():
        return 0, 0.0
    if bad.any():
        clean[bad] = np.nanmean(clean)

    h, w = clean.shape
    if min(h, w) < 16:
        return 0, 0.0

    # Hann window: a hard patch edge leaks a bright cross along the axes,
    # which would read as spurious order-2/order-4 power.
    clean = clean - clean.mean()
    win = np.outer(np.hanning(h), np.hanning(w))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(clean * win)))

    cy, cx = h / 2.0, w / 2.0
    r_max = min(h, w) / 2.0
    n_bins = int(r_max)
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    radial = np.zeros(n_bins)
    for i in range(n_bins):
        sel = (rr >= i) & (rr < i + 1)
        if sel.any():
            radial[i] = float(mag[sel].mean())

    # Skip the DC core: beam-hardening shading and the patch's own mean
    # gradient live there and would always outrank the lattice.
    lo = max(2, int(0.06 * n_bins))
    if lo >= n_bins - 1:
        return 0, 0.0
    r_star = float(np.argmax(radial[lo:]) + lo)

    # How far the lattice ring stands above the rest of the radial profile.
    # This — not the angular sharpness — is what answers "is there any
    # periodic structure here at all", i.e. the failure that matters: an
    # ROI that slid off its card onto featureless plastic. Measured on the
    # synthetic cards this separates by three orders of magnitude (real
    # lattices 777-19979, featureless plastic 1.29-1.36), because noise
    # has no preferred spatial frequency and its radial profile is flat.
    ring_prominence = float(
        radial[int(r_star)] / (float(np.median(radial[lo:])) + 1e-9)
    )

    n_ang = 360
    angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    radii = np.linspace(0.75 * r_star, 1.25 * r_star, 12)
    profile = np.zeros(n_ang)
    for k, t in enumerate(angles):
        ys = np.clip(np.rint(cy + radii * np.sin(t)).astype(int), 0, h - 1)
        xs = np.clip(np.rint(cx + radii * np.cos(t)).astype(int), 0, w - 1)
        profile[k] = float(mag[ys, xs].mean())

    spectrum = np.abs(np.fft.rfft(profile - profile.mean()))
    # A real image has a centrosymmetric spectrum, so the angular profile
    # has period 180 deg and only EVEN orders can carry power.
    candidates = [o for o in range(2, 13, 2) if o < len(spectrum)]
    if not candidates:
        return 0, 0.0
    powers = {o: float(spectrum[o]) for o in candidates}
    peak = max(powers.values())
    if peak <= 1e-9:
        return 0, 0.0

    # Take the SMALLEST strongly-present order, not the largest.
    #
    # An n-fold spot pattern puts power at n AND at every multiple of n
    # (a set of narrow spots is a comb, and a comb's Fourier series is
    # flat). Picking the argmax therefore reads a 2-fold line grating as
    # whichever of 2/4/6/8 noise happened to favour. The symmetry order is
    # the FUNDAMENTAL — the smallest order the rest are multiples of.
    order = min(o for o in candidates if powers[o] >= 0.5 * peak)
    return order, ring_prominence


def _validate_honeycomb(pixels: np.ndarray) -> Tuple[bool, str, float]:
    """Honeycomb cards: the lattice ring must carry 6-fold symmetry.

    See :func:`_lattice_ring_order` for why the ring is located from the
    data instead of assumed. The reported method carries the ORDER that
    was actually measured, so a future warning says what it saw
    ("hex_ring_order2") instead of only how unhappy it was.
    """
    order, ratio = _lattice_ring_order(pixels)
    is_valid = order == 6 and ratio >= config.LATTICE_MIN_RING_PROMINENCE
    return is_valid, f"hex_ring_order{order}", float(ratio)


def _fit_sine_r2(profile: np.ndarray) -> float:
    """Return the R² of a single-sine fit to ``profile``. 0.0 on failure."""
    if len(profile) < 4:
        return 0.0
    x = np.arange(len(profile), dtype=np.float64)
    profile_norm = profile - np.mean(profile)

    fft = np.abs(np.fft.rfft(profile_norm))
    if len(fft) < 3:
        return 0.0
    freq_idx = np.argmax(fft[1:]) + 1
    freq_est = freq_idx * 2 * np.pi / len(profile)

    def sine_model(x, A, freq, phase, offset):
        return A * np.sin(freq * x + phase) + offset

    try:
        popt, _ = optimize.curve_fit(
            sine_model,
            x,
            profile,
            p0=[np.std(profile), freq_est, 0, np.mean(profile)],
            maxfev=5000,
        )
        fitted = sine_model(x, *popt)
        ss_res = np.sum((profile - fitted) ** 2)
        ss_tot = np.sum((profile - np.mean(profile)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-10))
    except Exception as e:
        log.debug(f"sine fit failed: {type(e).__name__}: {e}")
        return 0.0


def _harmonic_purity(pixels: np.ndarray) -> float:
    """How sine-like the dominant 1D wave is: 1.0 pure, lower = squarer.

    Measured as ``1 - |3f| / |f|`` on whichever axis carries the signal.
    A sine has no third harmonic; an ideal square-wave bar pattern has one
    at 1/3 the fundamental's amplitude. This is what separates the
    phantom's sinusoidal card from the line-grating cards, and it does so
    WITHOUT assuming a constant period — which a global sine fit does.
    """
    prof_cols = np.nanmean(pixels, axis=0)
    prof_rows = np.nanmean(pixels, axis=1)
    profile = prof_cols if np.nanstd(prof_cols) >= np.nanstd(prof_rows) else prof_rows
    profile = np.nan_to_num(profile, nan=float(np.nanmean(profile)))
    if len(profile) < 8:
        return 0.0

    spec = np.abs(np.fft.rfft(profile - profile.mean()))
    if len(spec) < 4:
        return 0.0
    f = int(np.argmax(spec[1:]) + 1)

    def _band(centre: int) -> float:
        # Widen with harmonic index: a chirp's fundamental smears over a
        # few bins, and its harmonics smear proportionally more.
        half = max(1, int(round(0.25 * centre)))
        lo, hi = max(1, centre - half), min(len(spec), centre + half + 1)
        return float(spec[lo:hi].max()) if hi > lo else 0.0

    fund = _band(f)
    if fund <= 1e-9:
        return 0.0
    third = _band(3 * f) if 3 * f < len(spec) else 0.0
    return float(max(0.0, 1.0 - third / fund))


def _validate_sinusoidal(pixels: np.ndarray) -> Tuple[bool, str, float]:
    """Sinusoidal card: one dominant wave direction, and a SINE wave.

    The old implementation collapsed the patch to a 1D profile and fit a
    single sine with ``scipy.optimize.curve_fit``, gating on R². That fit
    is where the 21 "Sinusoidal sine_fit low" warnings of the 336-series
    run came from, and they were an optimiser artefact rather than a
    property of the data: a warned patch and a passing patch had the same
    2D contrast (std 321 vs 326 HU), the same profile energy, and even the
    same FFT starting frequency (bin 5 / bin 7) — but ``curve_fit``
    converged off that seed to a between-bin frequency on the warned one
    (f 0.898 -> 0.761), collapsing R² from 0.998 to 0.055. A brittle
    non-linear fit cannot certify a quantitative pipeline's ROIs.

    Both replacement properties are read straight off the spectrum, with
    no iterative fit and no assumption of a constant period:
      * the lattice ring must be 2-fold (a single wave direction, not a
        grid or a hex lattice), and
      * the wave must be sine-like rather than a square bar pattern
        (see :func:`_harmonic_purity`).
    """
    order, ratio = _lattice_ring_order(pixels)
    purity = _harmonic_purity(pixels)
    is_valid = (
        order == 2
        and ratio >= config.LATTICE_MIN_RING_PROMINENCE
        and purity >= config.SINE_HARMONIC_PURITY_MIN
    )
    return is_valid, f"sine_ring_order{order}_purity", float(purity)
