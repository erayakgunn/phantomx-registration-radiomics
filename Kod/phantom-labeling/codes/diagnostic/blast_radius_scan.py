"""Measure how many series a classifier change would actually move.

Before re-running an export over a whole dataset, this answers the only
question that decides whether the re-run is worth it: for each series, would
the CURRENT code keep the same slices the exported ROIs were cut from?

It runs phases 0-2c (load, detect, classify, canonicalise, trim) — everything
that decides WHICH slices become ROIs — and compares the resulting kept-slice
sets against the slice indices recorded in each series' existing
``metadata.json``. Nothing is written; the labeled output is only read.

The comparison is per section, so the report separates "this series is
untouched" from "only STAR8 moved" from "everything shifted".

Usage
-----
    python diagnostic/blast_radius_scan.py --roots <labeled_dir> [...]
        [--out blast.json] [--workers N]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SECTIONS = ("STAR16", "STAR8", "INSERT1", "INSERT2")


def _current_kept(series_dir: str) -> Dict[str, List[int]]:
    """Kept slice indices per section, as recorded in the exported metadata."""
    meta_path = os.path.join(series_dir, "metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    out: Dict[str, List[int]] = {}
    for name, block in meta.get("sections", {}).items():
        idx = sorted({r["slice_idx"] for r in block.get("rois", [])})
        if idx:
            out[name] = idx
    return out


def _recomputed_kept(series_dir: str) -> Dict[str, List[int]]:
    """Kept slice indices per section under the CURRENT code."""
    # Deferred so the import cost is paid inside the worker process.
    from core.dicom_loader import load_series
    from core.phantom_detector import detect_phantom
    from core.section_classifier import classify_sections
    from core.orientation_corrector import canonicalize_orientation
    from core.section_trim import trim_sections_to_inner_window

    dv = load_series(os.path.join(series_dir, "dicoms"))
    phantom = detect_phantom(dv)
    sections = classify_sections(dv, phantom)
    canonicalize_orientation(dv, phantom, sections)
    trim_sections_to_inner_window(dv, sections, phantom=phantom)

    out: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(sections.slice_labels):
        if label in _SECTIONS:
            out[label].append(i)
    return dict(out)


def compare_series(series_dir: str) -> dict:
    logging.disable(logging.CRITICAL)
    name = os.path.basename(series_dir)
    try:
        old = _current_kept(series_dir)
        new = _recomputed_kept(series_dir)
    except Exception as e:
        return {"series": name, "error": f"{type(e).__name__}: {e}"}

    per_section = {}
    changed = []
    for sec in _SECTIONS:
        o, n = old.get(sec, []), new.get(sec, [])
        same = o == n
        per_section[sec] = {
            "same": same,
            "old_n": len(o), "new_n": len(n),
            "old_range": [min(o), max(o)] if o else None,
            "new_range": [min(n), max(n)] if n else None,
            "kept_both": len(set(o) & set(n)),
        }
        if not same:
            changed.append(sec)
    return {"series": name, "changed_sections": changed,
            "identical": not changed, "sections": per_section}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel worker processes (0 = cpu_count-2)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dirs: List[str] = []
    for root in args.roots:
        dirs += sorted(d for d in glob.glob(os.path.join(root, "*"))
                       if os.path.isfile(os.path.join(d, "metadata.json")))
    if args.limit:
        dirs = dirs[:args.limit]

    import multiprocessing as mp
    workers = args.workers or max(1, (mp.cpu_count() or 4) - 2)
    print(f"{len(dirs)} seri, {workers} paralel işlem", flush=True)

    results = []
    with mp.Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(compare_series, dirs), 1):
            results.append(res)
            if res.get("error"):
                print(f"   HATA {res['series'][:60]}: {res['error']}", flush=True)
            elif not res["identical"]:
                bits = []
                for sec in res["changed_sections"]:
                    s = res["sections"][sec]
                    bits.append(f"{sec} {s['old_n']}->{s['new_n']} dilim "
                                f"(ortak {s['kept_both']})")
                print(f"   DEGISTI {res['series'][-52:]}: {'; '.join(bits)}",
                      flush=True)
            if k % 25 == 0 or k == len(dirs):
                nch = sum(1 for r in results if not r.get("identical", False))
                print(f"   {k}/{len(dirs)}  degisen={nch}", flush=True)

    ok = [r for r in results if not r.get("error")]
    identical = [r for r in ok if r["identical"]]
    changed = [r for r in ok if not r["identical"]]
    errs = [r for r in results if r.get("error")]

    per_sec = defaultdict(int)
    for r in changed:
        for sec in r["changed_sections"]:
            per_sec[sec] += 1

    print("\n" + "=" * 70)
    print(f"seri                 : {len(results)}  (hata: {len(errs)})")
    print(f"BIREBIR AYNI kalan   : {len(identical)}")
    print(f"kept dilimleri DEGISEN: {len(changed)}")
    print(f"bolum bazinda degisim : {dict(per_sec)}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"JSON: {args.out}")


if __name__ == "__main__":
    main()
