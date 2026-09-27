"""Re-run the PRODUCTION validators over ROIs already written to disk.

``verify_star_texture_rois.py`` is a deliberately independent second
opinion — its own measurement code, so it can disagree with the pipeline.
This script is the opposite: it imports the very functions
``core/validator.py`` and ``analysis/texture_validator.py`` use, and applies
them to the exported patches. It answers one question:

    "If the fixed validators had run on this dataset, what would they have
     reported?"

Reading the patches back is what makes that cheap. The pipeline would have
to redo phases 0-4 (~30 s/series) to reach its own validation step; the
patches are already on disk, carry the refined centre they were cut at, and
are the actual product being certified.

The star rule mirrors ``_validate_star_sections`` (a section is reported
when more than 30 % of its star ROIs miss the expected arm count); the
texture rule mirrors ``_validate_textures`` (one report per failing slice).

Usage
-----
    python diagnostic/revalidate_exported_rois.py --roots <labeled_dir> [...]
        [--out revalidation.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                             # noqa: E402
from analysis.texture_validator import validate_texture   # noqa: E402
from core.validator import (                              # noqa: E402
    STAR_MIN_MODULATION, _dominant_rotational_order,
)
from diagnostic.verify_star_texture_rois import load_patch  # noqa: E402


_SLICE_RE = re.compile(r"slice_(\d+)_")
_STAR_SECTION = {"Star16": "STAR16", "Star8": "STAR8"}


def _slice_idx(path: str) -> int:
    m = _SLICE_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def revalidate_series(series_dir: str) -> Dict[str, list]:
    """Return {'warnings': [...], 'star_detail': [...], 'tex_detail': [...]}."""
    warnings: List[str] = []
    star_detail: List[dict] = []
    tex_detail: List[dict] = []

    # ---- stars: per-section arm-count agreement ---------------------------
    by_section: Dict[str, List[str]] = defaultdict(list)
    for f in glob.glob(os.path.join(series_dir, "Stars", "*", "*.dcm")):
        label = os.path.basename(os.path.dirname(f))
        if label in _STAR_SECTION:
            by_section[_STAR_SECTION[label]].append(f)

    for section, files in sorted(by_section.items()):
        expected = config.SECTIONS[section]["star_arms"]
        mismatches = 0
        seen: Dict[int, int] = defaultdict(int)
        for f in files:
            patch, _ = load_patch(f)
            order, modulation = _dominant_rotational_order(patch)
            seen[order] += 1
            bad = order != expected or modulation < STAR_MIN_MODULATION
            if bad:
                mismatches += 1
            star_detail.append({
                "section": section, "slice": _slice_idx(f),
                "order": order, "expected": expected,
                "modulation": round(modulation, 4), "ok": not bad, "path": f,
            })
        if files and mismatches / len(files) > 0.3:
            warnings.append(
                f"{section}: {mismatches}/{len(files)} star ROIs are not "
                f"{expected}-fold (measured orders: {dict(sorted(seen.items()))})"
            )

    # ---- textures: per-slice pattern check --------------------------------
    for f in sorted(glob.glob(os.path.join(series_dir, "Textures", "*", "*.dcm"))):
        label = os.path.basename(os.path.dirname(f))
        patch, _ = load_patch(f)
        ok, method, score = validate_texture(patch, label)
        tex_detail.append({
            "label": label, "slice": _slice_idx(f), "ok": bool(ok),
            "method": method, "score": round(float(score), 3), "path": f,
        })
        if not ok:
            warnings.append(
                f"{label} slice {_slice_idx(f)}: texture score low "
                f"({method}, {score:.3f})"
            )

    return {"warnings": warnings, "star_detail": star_detail,
            "tex_detail": tex_detail}


def _one(series_dir: str):
    """Worker entry: (series, root, old_warnings, revalidation result)."""
    import logging
    logging.disable(logging.CRITICAL)
    meta_path = os.path.join(series_dir, "metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        old = json.load(f).get("validation", {}).get("warnings", [])
    return (os.path.basename(series_dir),
            os.path.basename(os.path.dirname(series_dir)),
            old, revalidate_series(series_dir))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel worker processes (0 = cpu_count-2). The "
                         "texture check Radon-transforms every line/grid card "
                         "(180 rotations per patch), so this is CPU-bound and "
                         "scales nearly linearly.")
    args = ap.parse_args()

    dirs: List[str] = []
    for root in args.roots:
        found = sorted(d for d in glob.glob(os.path.join(root, "*"))
                       if os.path.isdir(d) and os.path.isfile(
                           os.path.join(d, "metadata.json")))
        dirs += found[:args.limit] if args.limit else found

    import multiprocessing as mp
    workers = args.workers or max(1, (mp.cpu_count() or 4) - 2)
    print(f"{len(dirs)} seri, {workers} paralel islem", flush=True)

    report: Dict[str, dict] = {}
    old_total = new_total = 0

    with mp.Pool(workers) as pool:
        for n, (series, root_name, old, res) in enumerate(
                pool.imap_unordered(_one, dirs), 1):
            old_total += len(old)
            new_total += len(res["warnings"])
            report[series] = {
                "root": root_name,
                "old_warnings": old,
                "new_warnings": res["warnings"],
                "star_bad": [d for d in res["star_detail"] if not d["ok"]],
                "tex_bad": [d for d in res["tex_detail"] if not d["ok"]],
                "n_star": len(res["star_detail"]),
                "n_tex": len(res["tex_detail"]),
            }
            if res["warnings"]:
                print(f"   !! {series[:70]}", flush=True)
                for w in res["warnings"][:4]:
                    print(f"      {w}", flush=True)
            if n % 20 == 0 or n == len(dirs):
                print(f"   {n}/{len(dirs)}  eski={old_total} yeni={new_total}",
                      flush=True)

    n_series = len(report)
    n_star = sum(r["n_star"] for r in report.values())
    n_tex = sum(r["n_tex"] for r in report.values())
    still = {s: r["new_warnings"] for s, r in report.items() if r["new_warnings"]}

    print("\n" + "=" * 70)
    print(f"{n_series} seri | {n_star:,} star ROI | {n_tex:,} texture ROI")
    print(f"ESKİ doğrulayıcıların ürettiği uyarı : {old_total}")
    print(f"YENİ doğrulayıcıların ürettiği uyarı : {new_total}")
    print(f"Hâlâ uyarı veren seri                : {len(still)}")
    for s, ws in list(still.items())[:15]:
        print(f"   {s[:66]}")
        for w in ws[:3]:
            print(f"      {w}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
