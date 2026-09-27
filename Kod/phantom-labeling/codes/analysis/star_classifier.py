"""Star pattern classification (8 vs 16 arms) by direct arm counting."""

import numpy as np
from typing import Tuple

from analysis.fft_utils import angular_symmetry
from utils.logger import log
import config


def classify_star(
    hu_slice: np.ndarray, cx: float, cy: float, radius_px: float
) -> Tuple[str, int, float]:
    """Classify star pattern as STAR8 or STAR16.

    Uses the angular-profile FFT to pick out the dominant rotational
    symmetry order, then falls back to peak-counting (legacy Philips
    path) only when the spectrum is ambiguous.

    On Siemens PCCT Qr40 reconstructions, plain peak counting on the
    angular profile yields ~27–32 peaks for BOTH STAR8 and STAR16
    sections (harmonic aliasing on sharper kernels). The angular-FFT
    power at orders 8 vs. 16 separates them cleanly and is cheap to
    compute.

    Args:
        hu_slice: 2D HU image
        cx, cy: phantom center in pixels
        radius_px: star pattern radius in pixels

    Returns:
        (classification, peak_count, confidence)
        classification: "STAR8", "STAR16", or "UNKNOWN"
    """
    # Count the arms directly in the image. ``angular_symmetry`` samples
    # rings around (cx, cy) itself, so there is no patch-extraction step and
    # no half-pixel centring error to leak power into order N±1.
    order, modulation = angular_symmetry(hu_slice, cx, cy, radius_px)

    # ``n_peaks`` is retained only for the return signature and logs; no
    # caller acts on it (``core/section_classifier.py`` assigns it and uses
    # only the class and the confidence).
    n_peaks = 2 * order if order else 0

    if modulation < config.STAR_CLASSIFY_MIN_MODULATION:
        # No angular structure worth the name: an insert slice, a
        # transition, or plain plastic. Saying UNKNOWN lets the caller's
        # own gates decide, which is what it already does with this value.
        return "UNKNOWN", n_peaks, 0.0

    if order not in (8, 16):
        # A genuine star reads exactly 8 or 16 here. Anything else means the
        # measurement is not trustworthy on this slice (severe noise, a
        # centre that is badly off), and inventing a star from it is worse
        # than declining: a wrong star label moves a section boundary.
        return "UNKNOWN", n_peaks, 0.0

    # Confidence tracks how deep the modulation is. ``_find_star_split``
    # treats >= 0.75 as a high-confidence STAR16 when it decides where one
    # star module ends and the next begins, so this has to be well above
    # that gate for genuine stars: measured modulation is 0.38-0.43 for
    # STAR16 (16 narrower arms are blurred more by the reconstruction) and
    # ~0.57 for STAR8, giving 0.95-1.0 here.
    confidence = float(min(1.0, modulation / 0.40))
    return ("STAR16" if order == 16 else "STAR8"), n_peaks, confidence


def _symmetry_powers(profile: np.ndarray) -> Tuple[float, float, float]:
    """Return (p8, p16, bg): peak angular-FFT magnitudes at order-8 and
    order-16 bins, and the mean background magnitude (excluding DC and
    the two target bands).

    With the polar sampling of 720 angles per revolution, order-N
    rotational symmetry lives at FFT bin k = 2N (one bin per cycle).
    A ±1 neighbour span absorbs small sampling offsets.
    """
    n = len(profile)
    if n < 64:
        return 0.0, 0.0, 0.0
    spectrum = np.abs(np.fft.rfft(profile - np.mean(profile)))
    bins_per_order = 720 // 360  # = 2 with num_angles=720 upstream
    k8 = 8 * bins_per_order
    k16 = 16 * bins_per_order
    if k16 + 1 >= len(spectrum):
        return 0.0, 0.0, 0.0

    p8 = float(spectrum[k8 - 1 : k8 + 2].max())
    p16 = float(spectrum[k16 - 1 : k16 + 2].max())

    mask = np.ones_like(spectrum, dtype=bool)
    mask[0] = False
    mask[k8 - 1 : k8 + 2] = False
    mask[k16 - 1 : k16 + 2] = False
    bg = float(np.mean(spectrum[mask])) if mask.any() else 0.0
    return p8, p16, max(bg, 1e-9)
