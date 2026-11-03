"""Recompute metrics.csv for existing registration outputs with the new
Metrics Reloaded-compatible metrics (no re-registration).

Batches: longitudinal PCCT v1.0.0, SEP v6_skip2, Zurich REGISTRATION and
REGISTRATION_skip5.
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "registration-radiomics"))
import label_map  # noqa: E402
from pilot import _metric_row  # noqa: E402
from registration import read_image, resample_mask  # noqa: E402

PCCT = Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX")
SEP = next(Path(r"C:/Users/eray/Desktop").glob("SEP -*")) / "PhantomX"
ZURICH = [Path(r"C:/Users/eray/Desktop/zurich_pcct_work"), Path(r"C:/Users/eray/Desktop/zurich_sep_work")]


def process_pair(base: Path, lab: Path, manifest: dict, pair: dict) -> int:
    fixed_path = lab / manifest["fixed_session"] / "segmentation" / manifest["fixed_series"] / "seg.nii.gz"
    moving_path = lab / pair["session"] / "segmentation" / pair["series"] / "seg.nii.gz"
    pair_dir = base / f"{pair['session']}__{pair['series']}"
    reg_path = pair_dir / "seg_registered.nii.gz"
    if not (fixed_path.exists() and moving_path.exists() and reg_path.exists()):
        return 0
    fixed = read_image(fixed_path, sitk.sitkUInt16)
    moving = read_image(moving_path, sitk.sitkUInt16)
    registered = read_image(reg_path, sitk.sitkUInt16)
    moving_on_fixed = resample_mask(moving, fixed, sitk.Transform(3, sitk.sitkIdentity))
    fixed_arr = sitk.GetArrayFromImage(fixed)
    moving_arr = sitk.GetArrayFromImage(moving_on_fixed)
    registered_arr = sitk.GetArrayFromImage(registered)
    mapping = label_map.load_label_map(fixed_path)
    labels = [label for label in (mapping.get("spleen"), mapping.get("aorta")) if label is not None]
    rows = [
        _metric_row(fixed, moving_on_fixed, registered, int(label),
                    fixed_arr, moving_arr, registered_arr)
        for label in labels
    ]
    with (pair_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 1


def recompute(base: Path, lab: Path) -> None:
    manifest_path = base / "batch_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs", [])
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process_pair, base, lab, manifest, pair) for pair in pairs]
        for future in as_completed(futures):
            done += future.result()
    print(f"[{base.parent.name}/{base.name}] {done}/{len(pairs)} pair")


def main() -> None:
    batches = [
        (PCCT / "REGISTRATION" / "batch_to_20260805_v1.0.0", PCCT / "LABELED"),
        (SEP / "REGISTRATION" / "batch_to_20260805_v6_skip2", SEP / "LABELED"),
    ]
    for work in ZURICH:
        for name in ("REGISTRATION", "REGISTRATION_skip5"):
            root = work / name
            if not root.exists():
                continue
            for group in sorted(p for p in root.iterdir() if p.is_dir()):
                batches.append((group, work / "LABELED"))
    for base, lab in batches:
        recompute(base, lab)


if __name__ == "__main__":
    main()
