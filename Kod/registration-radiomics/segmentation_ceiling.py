"""Estimate the observed TotalSegmentator Dice ceiling within sessions."""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from pathlib import Path

import SimpleITK as sitk

from metrics import dice
from registration import resample_mask


def series_in_session(session: Path) -> list[tuple[str, Path]]:
    return sorted(
        (p.parent.name, session / "segmentation" / p.parent.name / "seg.nii.gz")
        for p in session.glob("nifti/*/ct.nii.gz")
        if (session / "segmentation" / p.parent.name / "seg.nii.gz").exists()
    )


def main(args: argparse.Namespace) -> None:
    rows = []
    for session in sorted(args.dataset_root.iterdir()):
        if not session.is_dir():
            continue
        series = series_in_session(session)
        if len(series) < 2:
            continue
        for (name_a, path_a), (name_b, path_b) in itertools.combinations(series, 2):
            first = sitk.ReadImage(str(path_a), sitk.sitkUInt16)
            second = sitk.ReadImage(str(path_b), sitk.sitkUInt16)
            second_on_first = resample_mask(
                second, first, sitk.Transform(3, sitk.sitkIdentity)
            )
            for label, organ in ((1, "spleen"), (52, "aorta")):
                rows.append({
                    "session": session.name,
                    "series_a": name_a,
                    "series_b": name_b,
                    "organ": organ,
                    "dice": dice(first, second_on_first, label),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"pairwise rows: {len(rows)}")
    for organ in ("spleen", "aorta"):
        values = [float(r["dice"]) for r in rows if r["organ"] == organ]
        print(
            organ,
            "n=", len(values),
            "median=", round(statistics.median(values), 4),
            "p10=", round(sorted(values)[max(0, int(len(values) * 0.1))], 4),
            "max=", round(max(values), 4),
        )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Estimate within-session segmentation Dice ceiling")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
