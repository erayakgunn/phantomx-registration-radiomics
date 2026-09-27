"""Registration-free physical-coordinate baseline for one PhantomX session."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import SimpleITK as sitk

from metrics import compare_label


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _metric_row(fixed_seg: sitk.Image, moving_seg: sitk.Image,
                fixed_array, moving_array, label: int) -> dict:
    row = compare_label(
        fixed_seg,
        moving_seg,
        label,
        reference_array=fixed_array,
        candidate_array=moving_array,
    )
    return {
        "label": label,
        "dice": row["dice"],
        "centroid_displacement_mm": row["centroid_displacement_mm"],
        "hd95_mm": row["hd95_mm"],
        "assd_mm": row["assd_mm"],
    }


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _find_series(root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in root.glob("nifti/*/ct.nii.gz")
        if "_take" in path.parent.name
    )


def run(root: Path, output: Path, labels: list[int]) -> None:
    series = _find_series(root)
    if len(series) < 2:
        raise RuntimeError("Dataset içinde en az iki take bulunamadı")
    fixed = next((path for path in series if "_take1" in path.name), series[0])
    moving_series = [path for path in series if path != fixed]
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "baseline_state.json"
    metadata = {"root": str(root.resolve()), "fixed": fixed.name, "labels": labels}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if {key: state.get(key) for key in metadata} != metadata:
            raise RuntimeError("Baseline output farklı bir dataset ile oluşturulmuş")
    else:
        state = {**metadata, "completed": []}
        _atomic_json(state_path, state)

    fixed_seg_path = root / "segmentation" / fixed.name / "seg.nii.gz"
    fixed_seg = sitk.ReadImage(str(fixed_seg_path), sitk.sitkUInt16)
    fixed_array = sitk.GetArrayFromImage(fixed_seg)
    partial = output / "baseline_metrics.partial.csv"
    rows: list[dict] = []
    if partial.exists():
        with partial.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    completed = {row["moving"] for row in rows}

    for moving in moving_series:
        if moving.name in completed:
            print(f"[baseline] atlandı: {moving.name}", flush=True)
            continue
        print(f"[baseline] {fixed.name} -> {moving.name}", flush=True)
        moving_seg = sitk.ReadImage(
            str(root / "segmentation" / moving.name / "seg.nii.gz"),
            sitk.sitkUInt16,
        )
        moving_array = sitk.GetArrayFromImage(moving_seg)
        pair_rows: list[dict] = []
        workers = min(max(1, os.cpu_count() or 1), len(labels))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _metric_row, fixed_seg, moving_seg,
                    fixed_array, moving_array, label
                ): label
                for label in labels
            }
            for future in as_completed(futures):
                row = future.result()
                pair_rows.append(row)
                print(f"  label {row['label']} tamamlandı", flush=True)
        for row in sorted(pair_rows, key=lambda item: item["label"]):
            rows.append({"moving": moving.name, **row})
        with partial.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        state["completed"] = sorted({row["moving"] for row in rows})
        _atomic_json(state_path, state)

    rows.sort(key=lambda row: (row["moving"], int(row["label"])))
    with (output / "baseline_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[baseline] tamamlandı: {output / 'baseline_metrics.csv'}", flush=True)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute registration-free PhantomX baseline")
    ap.add_argument("root", type=Path, help="PhantomX session folder containing nifti/ and segmentation/")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--labels", type=int, nargs="+", default=[1, 52])
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.root, args.output, args.labels)
