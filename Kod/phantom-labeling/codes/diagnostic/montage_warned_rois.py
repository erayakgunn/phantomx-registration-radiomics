"""Render contact sheets of every warned ROI patch for visual review.

Numbers can only reject the failure modes they were written to look for.
This tool puts the actual pixels in front of a human: every patch that the
336-series PCCT run reported a warning for, rendered next to CONTROL
patches — same card, same study, but NOT warned.

That pairing is the whole point. If a warning meant "the ROI slid off the
card", the warned tile would look plainly different from its controls. If
warned and control tiles are indistinguishable, the warning was about the
validator, not the placement.

Each tile is captioned with BOTH verdicts: the score the old validator gave
(the reason the warning exists) and what the fixed validator measures now.

Reads warnings straight out of each series' ``metadata.json``, so it needs
no other tool to have run first.

Usage
-----
    python diagnostic/montage_warned_rois.py --roots <labeled_dir> [...]
        --out-dir DIR [--max-per-sheet 48] [--star-sheets 4]
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

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                              # noqa: E402
from analysis.texture_validator import validate_texture    # noqa: E402
from core.validator import (                               # noqa: E402
    STAR_MIN_MODULATION, _dominant_rotational_order,
)
from diagnostic.verify_star_texture_rois import load_patch  # noqa: E402


_TILE = 130
_PAD = 4
_CAPTION_H = 34
_COLS = 8
_WARN_C = (0, 165, 255)     # amber
_CTRL_C = (150, 150, 150)   # grey
_OK_C = (90, 220, 90)
_BAD_C = (60, 60, 255)

_TEX_WARN_RE = re.compile(
    r"(STAR\d+) slice (\d+): (\w+) texture score low \((\w+), ([\d.]+)\)")
_STAR_WARN_RE = re.compile(r"(STAR\d+): (\d+)/(\d+) slices failed FFT")
_SLICE_RE = re.compile(r"slice_(\d+)_")


def _slice_idx(path: str) -> int:
    m = _SLICE_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _render(patch: np.ndarray, size: int = _TILE) -> np.ndarray:
    """HU patch -> 8-bit BGR tile, windowed on its own 1-99 percentile.

    Per-patch windowing rather than a fixed CT window: these cards span
    very different HU ranges across 80-120 kVp and MonoE settings, and one
    fixed window would render half of them as flat grey — hiding exactly
    the structure the sheet exists to show.
    """
    a = np.asarray(patch, dtype=np.float64)
    valid = ~np.isnan(a)
    if not valid.any():
        return np.zeros((size, size, 3), np.uint8)
    lo, hi = np.percentile(a[valid], (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    img[~valid] = 0.0
    img8 = cv2.resize((img * 255).astype(np.uint8), (size, size),
                      interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)


def _tile(path: str, kind: str, is_warned: bool,
          old_score: Optional[float], label: str) -> np.ndarray:
    patch, _ = load_patch(path)
    canvas = np.zeros((_TILE + _CAPTION_H, _TILE, 3), np.uint8)
    canvas[:_TILE] = _render(patch)

    if kind == "star":
        order, modulation = _dominant_rotational_order(patch)
        expected = 16 if label == "Star16" else 8
        ok = order == expected and modulation >= STAR_MIN_MODULATION
        new_txt = f"YENI: {order}/{expected} kol m={modulation:.2f}"
    else:
        ok, method, score = validate_texture(patch, label)
        order = method.replace("hex_ring_order", "").replace(
            "sine_ring_order", "").replace("_purity", "")
        new_txt = f"YENI: {'GECTI' if ok else 'KALDI'} ord={order}"

    cv2.rectangle(canvas, (0, 0), (_TILE - 1, _TILE - 1),
                  _WARN_C if is_warned else _CTRL_C, 2)
    head = f"{'UYARI' if is_warned else 'kontrol'} s{_slice_idx(path)}"
    if old_score is not None:
        head += f" eski={old_score:.3f}"
    cv2.putText(canvas, head, (3, _TILE + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                _WARN_C if is_warned else _CTRL_C, 1, cv2.LINE_AA)
    cv2.putText(canvas, new_txt, (3, _TILE + 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.33, _OK_C if ok else _BAD_C, 1, cv2.LINE_AA)
    return canvas, bool(ok)


def _sheet(entries: List[tuple], title: str, cols: int = _COLS) -> Tuple[np.ndarray, int]:
    n = len(entries)
    rows = (n + cols - 1) // cols
    tw, th = _TILE + _PAD, _TILE + _CAPTION_H + _PAD
    header = 40
    sheet = np.zeros((header + rows * th, cols * tw, 3), np.uint8)
    cv2.putText(sheet, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(sheet,
                "turuncu cerceve = orijinal kosuda UYARI alan patch | "
                "gri = ayni karttan kontrol | yesil YENI = duzeltilmis dogrulayici geciyor",
                (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (185, 185, 185), 1, cv2.LINE_AA)

    n_bad = 0
    for i, (path, kind, warned, old, label) in enumerate(entries):
        t, ok = _tile(path, kind, warned, old, label)
        n_bad += (not ok)
        r, c = divmod(i, cols)
        y, x = header + r * th, c * tw
        sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
    return sheet, n_bad


# --------------------------------------------------------------------------- #
# warning collection
# --------------------------------------------------------------------------- #

def collect(roots: List[str]) -> Tuple[Dict[str, list], list, dict]:
    """Return (warned textures by card, warned star sections, control pool)."""
    tex_warn: Dict[str, list] = defaultdict(list)
    star_warn: list = []
    controls: Dict[str, list] = defaultdict(list)

    for root in roots:
        for md in sorted(glob.glob(os.path.join(root, "*", "metadata.json"))):
            sd = os.path.dirname(md)
            series = os.path.basename(sd)
            try:
                meta = json.load(open(md, encoding="utf-8"))
            except Exception:
                continue
            warnings = meta.get("validation", {}).get("warnings", [])

            warned_pairs = {}
            for w in warnings:
                mt = _TEX_WARN_RE.match(w)
                if mt:
                    warned_pairs[(int(mt.group(2)), mt.group(3))] = float(mt.group(5))
                    continue
                ms = _STAR_WARN_RE.match(w)
                if ms:
                    sub = "Star16" if ms.group(1) == "STAR16" else "Star8"
                    files = sorted(glob.glob(
                        os.path.join(sd, "Stars", sub, "*.dcm")))
                    if files:
                        star_warn.append({
                            "series": series, "label": sub,
                            "files": files,
                            "mid": files[len(files) // 2],
                        })

            for f in glob.glob(os.path.join(sd, "Textures", "*", "*.dcm")):
                card = os.path.basename(os.path.dirname(f))
                key = (_slice_idx(f), card)
                if key in warned_pairs:
                    tex_warn[card].append((f, warned_pairs[key]))
                elif len(controls[card]) < 4000:
                    controls[card].append(f)

    return tex_warn, star_warn, controls


def build(roots: List[str], out_dir: str, max_per_sheet: int,
          star_sheets: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    tex_warn, star_warn, controls = collect(roots)
    written: List[str] = []
    summary: Dict[str, dict] = {}

    n_tex = sum(len(v) for v in tex_warn.values())
    print(f"uyarili doku patch: {n_tex} | uyarili star bolumu: {len(star_warn)}")

    # ---- textures: EVERY warned patch, interleaved with controls ----------
    for card, recs in sorted(tex_warn.items()):
        pool = controls.get(card, [])
        step = max(1, len(pool) // max(1, len(recs)))
        ctrl = pool[::step][:len(recs)]

        entries = []
        for i, (path, old) in enumerate(recs):
            entries.append((path, "tex", True, old, card))
            if i < len(ctrl):
                entries.append((ctrl[i], "tex", False, None, card))

        bad_total = 0
        for k in range(0, len(entries), max_per_sheet):
            chunk = entries[k:k + max_per_sheet]
            n_w = sum(1 for e in chunk if e[2])
            img, bad = _sheet(
                chunk,
                f"{card}  |  bu sayfada {n_w} uyarili patch  "
                f"(kartin toplam uyarisi: {len(recs)})")
            bad_total += bad
            p = os.path.join(out_dir, f"texture_{card}_{k // max_per_sheet + 1}.png")
            cv2.imwrite(p, img)
            written.append(p)
        summary[card] = {"warned": len(recs), "new_fail": bad_total}
        print(f"  {card:<13} {len(recs):>4} uyarili -> yeni dogrulayicida "
              f"{bad_total} basarisiz")

    # ---- stars: one representative per warned section ---------------------
    per_sheet = max_per_sheet
    reps = [(s["mid"], "star", True, None, s["label"]) for s in star_warn]
    for k in range(0, min(len(reps), star_sheets * per_sheet), per_sheet):
        chunk = reps[k:k + per_sheet]
        img, bad = _sheet(chunk,
                          f"UYARI alan STAR bolumleri - her bolumden 1 temsilci "
                          f"({k + 1}-{k + len(chunk)} / {len(reps)})")
        p = os.path.join(out_dir, f"star_warned_{k // per_sheet + 1}.png")
        cv2.imwrite(p, img)
        written.append(p)
        print(f"  star sayfa {k // per_sheet + 1}: {len(chunk)} bolum -> "
              f"{bad} basarisiz")

    # ---- one warned section, every slice (intra-section consistency) ------
    if star_warn:
        s = star_warn[0]
        entries = [(f, "star", True, None, s["label"]) for f in s["files"]]
        img, bad = _sheet(entries,
                          f"Tek bolumun TUM dilimleri: {s['label']} | "
                          f"{s['series'][:66]}", cols=10)
        p = os.path.join(out_dir, "star_one_section_all_slices.png")
        cv2.imwrite(p, img)
        written.append(p)
        print(f"  tek bolum tum dilimler: {len(entries)} dilim -> {bad} basarisiz")

    with open(os.path.join(out_dir, "montage_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"textures": summary, "star_sections": len(star_warn)}, f, indent=1)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-per-sheet", type=int, default=48)
    ap.add_argument("--star-sheets", type=int, default=4)
    args = ap.parse_args()

    files = build(args.roots, args.out_dir, args.max_per_sheet, args.star_sheets)
    print(f"\n{len(files)} kontakt sayfa -> {args.out_dir}")


if __name__ == "__main__":
    main()
