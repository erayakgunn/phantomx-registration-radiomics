"""Audit every exported insert ROI against what a cylinder rod must look like.

The insert ROIs are 95 % of this pipeline's output (40 rods x 3 concentric
sub-ROIs x every kept slice x 2 sections). Unlike the star and texture cards
they carry no pattern to recognise — a rod is just a disc of uniform
material — so "is this ROI on its rod" has to be answered from physics
instead:

  NESTING     The 5, 10 and 15 mm sub-ROIs are concentric on the SAME centre
              and sample the SAME rod, so their mean HU must agree. An ROI
              that has drifted off the rod shows it here first: the widest
              sub-ROI starts including surrounding material, so its mean
              pulls away from the 5 mm mean and its spread jumps.
  Z-STABILITY A rod is uniform along z inside the kept window, so one
              insert's mean HU must be flat across its slices. A window that
              ran off the module end, or a centre that drifted, breaks this.
  GEOMETRY    The circular mask must be intact and the patch size must match
              the sub-ROI diameter at the series' pixel spacing. A crop that
              ran off the image edge shows up as a deficient valid fraction.
  DISTINCTNESS The 40 rods are different materials. If a series' 40 mean-HU
              values collapse onto one value, the ROIs are not on the rods.

Everything is measured from the exported ``.dcm`` files, i.e. the product
itself, never re-derived from the source volume.

Usage
-----
    python diagnostic/verify_insert_rois.py --roots <labeled_dir> [...]
        [--out inserts.json] [--workers N]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostic.verify_star_texture_rois import load_patch   # noqa: E402


_INSERT_RE = re.compile(r"slice_(\d+)_Insert_(\d+)_(\d+)mm\.dcm$")

# A rod is one material, so the widest sub-ROI must not disagree with the
# narrowest by more than this. Generous on purpose: the 15 mm ROI genuinely
# samples more of the rod's blurred rim than the 5 mm one, and low-dose
# recons are noisy. What this is built to catch is an ROI sitting off the
# rod, where the disagreement is in the hundreds of HU.
NESTING_MAX_DELTA_HU = 60.0

# Per-insert mean-HU spread across the slices of one section. Same logic:
# sized to catch a window or centre failure, not ordinary noise.
Z_STABILITY_MAX_STD_HU = 40.0

# Truncation is detected from the SHAPE of the mask, not from how much of
# the square it fills.
#
# A fill-fraction threshold cannot work here: the circle covers pi*r^2 of a
# (2r+1)^2 square, which is 0.62 for the 5 mm sub-ROI (r=4 px at 0.586 mm/px)
# and 0.73 for the 15 mm one — so any single cutoff either passes truncated
# large ROIs or fails every intact small one. (It flagged all 3600 ROIs of a
# healthy series before this was fixed.)
#
# What actually distinguishes an intact circular mask is its SYMMETRY: a
# circle is unchanged by a 180-degree rotation, while a crop clipped by the
# image edge loses a straight-edged chunk on one side only. Comparing the
# mask against its own point reflection is size-independent and needs no
# knowledge of the radius the cutter used.
MASK_ASYMMETRY_MAX = 0.02


def audit_series(series_dir: str) -> dict:
    logging.disable(logging.CRITICAL)
    name = os.path.basename(series_dir)
    try:
        # by_insert[(section_label, insert_no, size_mm)][slice] = (mean, std, frac, shape)
        recs: Dict[tuple, Dict[int, tuple]] = defaultdict(dict)
        for f in glob.glob(os.path.join(series_dir, "Inserts", "*", "*", "*.dcm")):
            m = _INSERT_RE.search(os.path.basename(f))
            if not m:
                continue
            sl, num, size = int(m.group(1)), int(m.group(2)), int(m.group(3))
            patch, _frac = load_patch(f)
            mask = ~np.isnan(patch)
            # Point-reflection asymmetry: 0 for an intact circle, large for a
            # circle clipped by the image edge.
            asym = float(np.mean(mask != mask[::-1, ::-1]))
            vals = patch[mask]
            if vals.size == 0:
                recs[(num, size)][sl] = (np.nan, np.nan, asym, patch.shape[0])
                continue
            recs[(num, size)][sl] = (float(vals.mean()), float(vals.std()),
                                     asym, patch.shape[0])
        if not recs:
            return {"series": name, "error": "no insert ROIs"}

        n_roi = sum(len(v) for v in recs.values())
        findings: List[dict] = []

        # --- nesting: 5 vs 15 mm on the same insert+slice --------------
        nest_deltas, spread_ratios = [], []
        for num in {k[0] for k in recs}:
            small, large = recs.get((num, 5), {}), recs.get((num, 15), {})
            for sl in set(small) & set(large):
                m5, s5 = small[sl][0], small[sl][1]
                m15, s15 = large[sl][0], large[sl][1]
                if np.isnan(m5) or np.isnan(m15):
                    continue
                d = abs(m15 - m5)
                nest_deltas.append(d)
                if s5 > 1e-6:
                    spread_ratios.append(s15 / s5)
                if d > NESTING_MAX_DELTA_HU:
                    findings.append({
                        "kind": "nesting", "insert": num, "slice": sl,
                        "mean_5mm": round(m5, 1), "mean_15mm": round(m15, 1),
                        "delta": round(d, 1),
                    })

        # --- z-stability per insert / size ------------------------------
        z_stds = []
        for (num, size), per_slice in recs.items():
            means = [v[0] for v in per_slice.values() if not np.isnan(v[0])]
            if len(means) < 4:
                continue
            sd = float(np.std(means))
            z_stds.append(sd)
            if sd > Z_STABILITY_MAX_STD_HU:
                findings.append({
                    "kind": "z_stability", "insert": num, "size_mm": size,
                    "std_across_slices": round(sd, 1), "n_slices": len(means),
                })

        # --- mask geometry ----------------------------------------------
        shapes_by_size = defaultdict(set)
        for (num, size), per_slice in recs.items():
            for sl, v in per_slice.items():
                shapes_by_size[size].add(v[3])
                if v[2] > MASK_ASYMMETRY_MAX:
                    findings.append({
                        "kind": "truncated_mask", "insert": num, "slice": sl,
                        "size_mm": size, "asymmetry": round(v[2], 4),
                    })

        # --- distinctness of the 40 rods --------------------------------
        per_insert_mean = {}
        for (num, size), per_slice in recs.items():
            if size != 10:
                continue
            means = [v[0] for v in per_slice.values() if not np.isnan(v[0])]
            if means:
                per_insert_mean[num] = float(np.mean(means))
        distinct = len({round(v / 20.0) for v in per_insert_mean.values()})
        if per_insert_mean and distinct < 3:
            findings.append({"kind": "not_distinct",
                             "distinct_levels": distinct,
                             "n_inserts": len(per_insert_mean)})

        return {
            "series": name,
            "n_roi": n_roi,
            "n_inserts": len(per_insert_mean),
            "nesting_delta_max": round(max(nest_deltas), 1) if nest_deltas else None,
            "nesting_delta_p99": round(float(np.percentile(nest_deltas, 99)), 1) if nest_deltas else None,
            "spread_ratio_max": round(max(spread_ratios), 2) if spread_ratios else None,
            "z_std_max": round(max(z_stds), 1) if z_stds else None,
            "shapes_by_size": {k: sorted(v) for k, v in shapes_by_size.items()},
            "distinct_levels": distinct,
            "insert_hu": {str(k): round(v, 1) for k, v in sorted(per_insert_mean.items())},
            "findings": findings,
        }
    except Exception as e:
        return {"series": name, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dirs: List[str] = []
    for root in args.roots:
        dirs += sorted(d for d in glob.glob(os.path.join(root, "*"))
                       if os.path.isdir(os.path.join(d, "Inserts")))
    if args.limit:
        dirs = dirs[:args.limit]

    import multiprocessing as mp
    workers = args.workers or max(1, (mp.cpu_count() or 4) - 2)
    print(f"{len(dirs)} seri, {workers} paralel islem", flush=True)

    results = []
    with mp.Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(audit_series, dirs), 1):
            results.append(res)
            if res.get("error"):
                print(f"   HATA {res['series'][:60]}: {res['error']}", flush=True)
            elif res["findings"]:
                kinds = defaultdict(int)
                for f in res["findings"]:
                    kinds[f["kind"]] += 1
                print(f"   BULGU {res['series'][-52:]}: {dict(kinds)}", flush=True)
            if k % 25 == 0 or k == len(dirs):
                nf = sum(len(r.get("findings", [])) for r in results)
                print(f"   {k}/{len(dirs)}  toplam_bulgu={nf}", flush=True)

    ok = [r for r in results if not r.get("error")]
    total_roi = sum(r["n_roi"] for r in ok)
    all_find = [(r["series"], f) for r in ok for f in r["findings"]]
    kinds = defaultdict(int)
    for _, f in all_find:
        kinds[f["kind"]] += 1

    print("\n" + "=" * 70)
    print(f"seri            : {len(ok)}  (hata {len(results) - len(ok)})")
    print(f"denetlenen insert ROI: {total_roi:,}")
    print(f"BULGU           : {len(all_find)}  {dict(kinds)}")
    if ok:
        print(f"nesting |15mm-5mm| en kotu: "
              f"{max((r['nesting_delta_max'] or 0) for r in ok):.1f} HU "
              f"(esik {NESTING_MAX_DELTA_HU})")
        print(f"z-stabilite std en kotu   : "
              f"{max((r['z_std_max'] or 0) for r in ok):.1f} HU "
              f"(esik {Z_STABILITY_MAX_STD_HU})")
    for s, f in all_find[:20]:
        print(f"   {s[-48:]}  {f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print(f"JSON: {args.out}")


if __name__ == "__main__":
    main()
