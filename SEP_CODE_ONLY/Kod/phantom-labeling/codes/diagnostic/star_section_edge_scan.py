"""Find star-section windows that reach past the physical star module.

The section trim keeps a fixed +/-15 mm slab and an adaptive guard then peels
leading/trailing slices whose gradient signal says the pattern is still
forming or already fading (``config.STAR_ADAPTIVE_GUARD_MAX_MM``). When the
guard under-peels, the outermost kept slices sample module-transition
material instead of the star — the ROI is still written, and downstream
radiomics would treat it as star data.

This scan measures every exported star patch's rotational-symmetry strength
and asks, per section, whether the EDGE slices have collapsed relative to
their own section's plateau. Working per section (rather than against one
global threshold) is what makes it protocol-independent: a 500 mm FOV recon
has a weaker absolute signal everywhere, but its plateau is still flat.

Usage
-----
    python diagnostic/star_section_edge_scan.py --roots <labeled_dir> [...]
        [--out edges.json] [--montage-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                               # noqa: E402
from core.validator import STAR_MIN_MODULATION, _dominant_rotational_order       # noqa: E402
from diagnostic.verify_star_texture_rois import load_patch   # noqa: E402


_SLICE_RE = re.compile(r"slice_(\d+)_")

# An edge slice is called COLLAPSED when its angular MODULATION DEPTH falls
# below this fraction of the section's own plateau (the median over the
# middle 60 % of the section, which is always genuine star material).
#
# Judging each section against its OWN plateau rather than one global number
# is what makes this protocol-independent: a 500 mm-FOV or low-dose recon has
# a weaker absolute signal everywhere, but its plateau is still flat, and a
# module edge still collapses relative to it. Measured on the one known real
# case, the runaway slices sat at 0.03-0.05 x plateau while ordinary
# plateau-to-plateau variation stays within a few percent.
EDGE_COLLAPSE_FRAC = 0.60


def _slice_idx(path: str) -> int:
    m = _SLICE_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def scan_section(files: List[str], expected: int) -> dict:
    """Measure symmetry order/strength per slice and locate edge collapses."""
    files = sorted(files, key=_slice_idx)
    orders, mods, slices = [], [], []
    for f in files:
        patch, _ = load_patch(f)
        order, modulation = _dominant_rotational_order(patch)
        orders.append(order)
        mods.append(modulation)
        slices.append(_slice_idx(f))

    n = len(mods)
    if n < 5:
        return {"n": n, "bad_edges": [], "plateau": 0.0}

    core = mods[int(n * 0.2):int(n * 0.8)] or mods
    plateau = float(np.median(core))
    floor = plateau * EDGE_COLLAPSE_FRAC

    bad: List[dict] = []
    # Walk inward from each end; stop at the first slice that is healthy, so
    # only a CONTIGUOUS run at the edge counts (an isolated dip in the middle
    # is noise, not a module boundary).
    for rng in (range(n), range(n - 1, -1, -1)):
        for i in rng:
            if mods[i] >= floor and orders[i] == expected:
                break
            bad.append({
                "slice": slices[i], "order": orders[i], "expected": expected,
                "modulation": round(mods[i], 4), "plateau": round(plateau, 4),
                "frac_of_plateau": round(mods[i] / (plateau + 1e-9), 3),
                "edge_pos": i + 1 if rng.start == 0 else n - i,
                "path": files[i],
            })
    return {"n": n, "plateau": round(plateau, 4), "bad_edges": bad,
            "modulations": [round(m, 4) for m in mods], "orders": orders,
            "slices": slices}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report: Dict[str, dict] = {}
    n_sec = n_bad_sec = n_bad_slice = 0

    for root in args.roots:
        dirs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                      if os.path.isdir(os.path.join(d, "Stars")))
        print(f"\n[{os.path.basename(root)}] {len(dirs)} seri", flush=True)
        for k, sd in enumerate(dirs, 1):
            series = os.path.basename(sd)
            for sub, section in (("Star16", "STAR16"), ("Star8", "STAR8")):
                files = glob.glob(os.path.join(sd, "Stars", sub, "*.dcm"))
                if not files:
                    continue
                n_sec += 1
                res = scan_section(files, config.SECTIONS[section]["star_arms"])
                if res["bad_edges"]:
                    n_bad_sec += 1
                    n_bad_slice += len(res["bad_edges"])
                    report[f"{series}|{section}"] = {
                        "root": os.path.basename(root), **res}
                    worst = min(res["bad_edges"], key=lambda b: b["frac_of_plateau"])
                    print(f"   !! {section} {series[-52:]}", flush=True)
                    print(f"      {len(res['bad_edges'])} kenar dilimi cokmus; "
                          f"en kotu slice {worst['slice']} "
                          f"({worst['frac_of_plateau']:.2f} x plato)", flush=True)
            if k % 25 == 0 or k == len(dirs):
                print(f"   {k}/{len(dirs)}  bolum={n_sec} kusurlu_bolum={n_bad_sec}",
                      flush=True)

    print("\n" + "=" * 70)
    print(f"taranan yildiz bolumu       : {n_sec}")
    print(f"kenari cokmus bolum         : {n_bad_sec}")
    print(f"etkilenen dilim (ROI)       : {n_bad_slice}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"JSON: {args.out}")


if __name__ == "__main__":
    main()
