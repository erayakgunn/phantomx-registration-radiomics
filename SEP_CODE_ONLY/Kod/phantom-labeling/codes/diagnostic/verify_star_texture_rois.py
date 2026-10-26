"""Verify every exported Star / Texture ROI against its expected pattern.

Standalone audit tool: reads the ``.dcm`` ROI patches the pipeline already
wrote to disk and checks each one against the geometry it is supposed to
contain. Nothing is re-derived from the source volume, so this measures the
PRODUCT, not the pipeline's opinion of the product.

Why a second opinion is needed
------------------------------
``core/validator.py`` re-cuts a star patch from the volume at the WALL
centre and asks ``analysis.star_classifier.classify_star`` what it sees.
Two things make that fragile:

  * The wall centre is a few px off the star's true centre on most recons
    (the extractor refines it separately, see ``_refine_star_center``).
    An off-centre angular sample of an N-fold pattern leaks power into
    N±1, so a 16-arm star can read as 15-fold.
  * ``classify_star`` works on the angular profile of the FFT magnitude,
    where an 8-arm star's order-16 harmonic outruns its own order-8
    fundamental (measured p16/p8 ≈ 1.26 on STAR8 vs ≈ 2.7 on STAR16).

The exported patch has none of those problems: it is already centred on
the refined centre, so the arm count can be read straight off the image in
POLAR space, with no FFT-of-FFT step. That is what this tool does.

Metrics
-------
Star patches
    Sample a ring at several fractions of the patch radius, take the
    angular intensity profile, and report the dominant rotational order.
    ``Star16`` must read 16, ``Star8`` must read 8.

Texture patches
    Take the 2D FFT magnitude, locate the lattice's dominant RADIAL
    frequency, then read the dominant ANGULAR order in a band around that
    ring. The radial step is what makes this pitch-independent:

        honeycomb (hex lattice) -> 6      grid  -> 4      stripes -> 2

    ``analysis.texture_validator._validate_honeycomb`` skips that step and
    always looks in a fixed 0.15-0.5 band, which misses the fundamental of
    the finer honeycomb cards — the origin of the 0.333 scores in the run
    logs.

Usage
-----
    python diagnostic/verify_star_texture_rois.py --roots <labeled_dir> [...]
        [--out report.json] [--montage-dir DIR] [--limit N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom

# Allow running as `python diagnostic/verify_star_texture_rois.py` from codes/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Stored value the exporter writes for out-of-mask pixels (see
# output/exporter.py::_DICOM_PADDING_STORED). Reloaded HU would be -33792.
_PADDING_STORED = -32768

# Expected dominant angular order per ROI label.
_STAR_EXPECTED = {"Star16": 16, "Star8": 8}
_TEXTURE_EXPECTED = {
    "Honeylarge": 6, "Honeymedium": 6, "Honeysmall": 6,   # hex lattice
    "Square": 4,                                          # orthogonal grid
    "Diagonal": 2, "Vertical": 2, "Horizontal": 2,        # line gratings
    "Sinusoidal": 2,                                      # single-direction wave
}


# --------------------------------------------------------------------------- #
# patch loading
# --------------------------------------------------------------------------- #

def load_patch(path: str) -> Tuple[np.ndarray, float]:
    """Return (HU array with NaN outside the mask, valid-pixel fraction)."""
    ds = pydicom.dcmread(path)
    stored = ds.pixel_array.astype(np.int32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = stored.astype(np.float64) * slope + intercept
    mask = stored == _PADDING_STORED
    hu[mask] = np.nan
    return hu, float((~mask).mean())


def _filled(patch: np.ndarray) -> np.ndarray:
    """Mean-fill NaNs so FFT/interp stages see a finite array."""
    out = patch.astype(np.float64, copy=True)
    bad = np.isnan(out)
    if bad.all():
        return np.zeros_like(out)
    if bad.any():
        out[bad] = np.nanmean(out)
    return out


# --------------------------------------------------------------------------- #
# star: dominant rotational order, read directly in polar space
# --------------------------------------------------------------------------- #

def star_order(
    patch: np.ndarray,
    fracs: Tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75),
    n_ang: int = 720,
    order_lo: int = 4,
    order_hi: int = 24,
) -> Tuple[int, List[int]]:
    """Dominant rotational order of a centred star patch.

    Sampling several radii and taking the median makes the answer immune to
    one bad ring (the hub is featureless, the rim clips the mask).
    """
    a = _filled(patch)
    h, w = a.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = min(h, w) / 2.0
    ang = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)

    orders: List[int] = []
    for frac in fracs:
        r = radius * frac
        ys = np.clip(np.rint(cy + r * np.sin(ang)).astype(int), 0, h - 1)
        xs = np.clip(np.rint(cx + r * np.cos(ang)).astype(int), 0, w - 1)
        prof = a[ys, xs]
        spec = np.abs(np.fft.rfft(prof - prof.mean()))
        if len(spec) <= order_hi:
            continue
        orders.append(int(np.argmax(spec[order_lo:order_hi + 1]) + order_lo))
    if not orders:
        return 0, []
    return int(np.median(orders)), orders


# --------------------------------------------------------------------------- #
# texture: pitch-independent lattice order
# --------------------------------------------------------------------------- #

def _fft_magnitude(patch: np.ndarray) -> np.ndarray:
    """Centred FFT magnitude of a Hann-windowed, mean-removed patch.

    The window matters: a hard-edged square patch leaks a bright cross
    along the axes, which reads as spurious order-2/4 power.
    """
    a = _filled(patch)
    a = a - a.mean()
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    return np.abs(np.fft.fftshift(np.fft.fft2(a * win)))


def texture_order(
    patch: np.ndarray, n_ang: int = 360
) -> Tuple[int, float, float]:
    """Return (dominant angular order, ring radius fraction, peak/bg ratio).

    Step 1 locates the lattice ring (dominant radial frequency); step 2
    reads how many spots sit on it. Because step 1 is data-driven, the
    same code handles a coarse and a fine honeycomb identically.
    """
    mag = _fft_magnitude(patch)
    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    r_max = min(h, w) / 2.0
    if r_max < 8:
        return 0, 0.0, 0.0

    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Radial profile, skipping the DC core (low-order shading, not lattice).
    n_bins = int(r_max)
    radial = np.zeros(n_bins)
    for i in range(n_bins):
        sel = (rr >= i) & (rr < i + 1)
        if sel.any():
            radial[i] = float(mag[sel].mean())
    lo = max(2, int(0.06 * n_bins))
    if lo >= n_bins - 1:
        return 0, 0.0, 0.0
    r_star = int(np.argmax(radial[lo:]) + lo)

    # Angular profile on a band around that ring.
    r_in, r_out = 0.75 * r_star, 1.25 * r_star
    ang = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    prof = np.zeros(n_ang)
    for k, t in enumerate(ang):
        rs = np.linspace(r_in, r_out, 12)
        ys = np.clip(np.rint(cy + rs * np.sin(t)).astype(int), 0, h - 1)
        xs = np.clip(np.rint(cx + rs * np.cos(t)).astype(int), 0, w - 1)
        prof[k] = mag[ys, xs].mean()

    spec = np.abs(np.fft.rfft(prof - prof.mean()))
    # A real-valued image has a centrosymmetric spectrum, so the angular
    # profile has period 180 deg and only EVEN orders can carry power.
    cand = [o for o in range(2, 13, 2) if o < len(spec)]
    if not cand:
        return 0, 0.0, 0.0
    powers = {o: float(spec[o]) for o in cand}
    best = max(powers, key=powers.get)
    others = [v for o, v in powers.items() if o != best]
    ratio = powers[best] / (float(np.mean(others)) + 1e-9) if others else 0.0
    return best, r_star / float(n_bins), ratio


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #

_SLICE_RE = re.compile(r"slice_(\d+)_")


def _slice_idx(path: str) -> int:
    m = _SLICE_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def collect_warned(root: str) -> Dict[str, dict]:
    """Parse each series' metadata.json into {series: {star:[], tex:set()}}."""
    out: Dict[str, dict] = {}
    for mp in glob.glob(os.path.join(root, "*", "metadata.json")):
        series = os.path.basename(os.path.dirname(mp))
        try:
            meta = json.load(open(mp, encoding="utf-8"))
        except Exception:
            continue
        star, tex = [], set()
        for w in meta.get("validation", {}).get("warnings", []):
            ms = re.match(r"(STAR\d+): (\d+)/(\d+) slices failed FFT", w)
            if ms:
                star.append(ms.group(1))
                continue
            mt = re.match(
                r"(STAR\d+) slice (\d+): (\w+) texture score low \((\w+), ([\d.]+)\)", w
            )
            if mt:
                tex.add((int(mt.group(2)), mt.group(3), float(mt.group(5))))
        out[series] = {"star_sections": star, "textures": tex}
    return out


def scan(roots: List[str], limit: Optional[int] = None) -> dict:
    """Measure every Star and Texture ROI under the given labeled roots."""
    stars: List[dict] = []
    textures: List[dict] = []

    for root in roots:
        warned = collect_warned(root)
        series_dirs = sorted(
            d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
        )
        if limit:
            series_dirs = series_dirs[:limit]
        print(f"[{os.path.basename(root)}] {len(series_dirs)} seri taranıyor...")

        for n, sd in enumerate(series_dirs, 1):
            series = os.path.basename(sd)
            w = warned.get(series, {"star_sections": [], "textures": set()})
            warned_tex = {(s, t) for s, t, _ in w["textures"]}
            tex_scores = {(s, t): sc for s, t, sc in w["textures"]}

            for f in glob.glob(os.path.join(sd, "Stars", "*", "*.dcm")):
                label = os.path.basename(os.path.dirname(f))
                exp = _STAR_EXPECTED.get(label)
                if exp is None:
                    continue
                patch, frac = load_patch(f)
                order, per_r = star_order(patch)
                stars.append({
                    "root": os.path.basename(root), "series": series,
                    "label": label, "slice": _slice_idx(f), "path": f,
                    "order": order, "expected": exp, "per_radius": per_r,
                    "valid_frac": round(frac, 4),
                    "warned": label.replace("Star", "STAR") in w["star_sections"],
                    "ok": order == exp,
                })

            for f in glob.glob(os.path.join(sd, "Textures", "*", "*.dcm")):
                label = os.path.basename(os.path.dirname(f))
                exp = _TEXTURE_EXPECTED.get(label)
                if exp is None:
                    continue
                sl = _slice_idx(f)
                patch, frac = load_patch(f)
                order, ring, ratio = texture_order(patch)
                textures.append({
                    "root": os.path.basename(root), "series": series,
                    "label": label, "slice": sl, "path": f,
                    "order": order, "expected": exp,
                    "ring": round(ring, 4), "ratio": round(ratio, 3),
                    "valid_frac": round(frac, 4),
                    "warned": (sl, label) in warned_tex,
                    "old_score": tex_scores.get((sl, label)),
                    "ok": order == exp,
                })

            if n % 20 == 0 or n == len(series_dirs):
                print(f"   {n}/{len(series_dirs)}  "
                      f"star={len(stars)} texture={len(textures)}")

    return {"stars": stars, "textures": textures}


def summarize(res: dict) -> None:
    stars, textures = res["stars"], res["textures"]

    print("\n" + "=" * 74)
    print("STAR ROI — dışa aktarılan patch'ten doğrudan kol sayımı")
    print("=" * 74)
    for label in ("Star16", "Star8"):
        sub = [s for s in stars if s["label"] == label]
        if not sub:
            continue
        ok = sum(s["ok"] for s in sub)
        wr = [s for s in sub if s["warned"]]
        wr_ok = sum(s["ok"] for s in wr)
        dist = defaultdict(int)
        for s in sub:
            dist[s["order"]] += 1
        print(f"  {label}: {ok}/{len(sub)} doğru ({_STAR_EXPECTED[label]} kol)   "
              f"ölçülen dereceler={dict(sorted(dist.items()))}")
        print(f"     bunlardan UYARI ALAN dilim: {len(wr)} -> {wr_ok} tanesi doğru")

    print("\n" + "=" * 74)
    print("TEXTURE ROI — örgü adımından bağımsız açısal derece")
    print("=" * 74)
    by = defaultdict(list)
    for t in textures:
        by[t["label"]].append(t)
    print(f"  {'kart':<13}{'bekl':>5}{'doğru':>12}{'uyarılı':>9}"
          f"{'uyarılı doğru':>15}   ölçülen dereceler")
    for label in sorted(by):
        sub = by[label]
        ok = sum(t["ok"] for t in sub)
        wr = [t for t in sub if t["warned"]]
        wr_ok = sum(t["ok"] for t in wr)
        dist = defaultdict(int)
        for t in sub:
            dist[t["order"]] += 1
        print(f"  {label:<13}{_TEXTURE_EXPECTED[label]:>5}{ok:>7}/{len(sub):<5}"
              f"{len(wr):>8}{wr_ok:>10}/{len(wr):<5}   {dict(sorted(dist.items()))}")

    bad_s = [s for s in stars if not s["ok"]]
    bad_t = [t for t in textures if not t["ok"]]
    print("\n" + "=" * 74)
    print(f"GERÇEK KUSUR: star={len(bad_s)}  texture={len(bad_t)}")
    for r in (bad_s + bad_t)[:20]:
        print(f"   {r['label']} slice {r['slice']} order={r['order']} "
              f"(bekl {r['expected']})  {r['series'][:60]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Labeled output directories to audit")
    ap.add_argument("--out", default=None, help="Write full JSON report here")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only scan the first N series per root (smoke test)")
    args = ap.parse_args()

    res = scan(args.roots, limit=args.limit)
    summarize(res)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        print(f"\nJSON rapor: {args.out}")


if __name__ == "__main__":
    main()
