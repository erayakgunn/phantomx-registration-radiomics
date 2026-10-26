"""Estimate the observed TotalSegmentator Dice ceiling within sessions.

Primary use: quantify within-session (world-space / identity) organ Dice, i.e.
the reproducibility ceiling that any registration result is compared against.

With ``--landmark-dice`` the script additionally reports, in memory, the organ
Dice after a bone-landmark rigid (Kabsch) alignment of the moving series onto the
fixed one. This isolates whether a within-session Dice drop is explained by a
rigid pose difference (landmark dice improves) or by a non-rigid / organ-shape /
reconstruction difference (landmark dice does not improve). No images are
written to disk.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import statistics
import sys
from pathlib import Path

import SimpleITK as sitk

from metrics import centroid_displacement, dice
from registration import landmark_rigid_output_transform, resample_mask


WORKFLOW_VERSION = "1.0.0"

REGISTRATION_LABELS = (*range(27, 44), *range(92, 116), 116)
VERTEBRA_LABELS = tuple(range(27, 44))
ORGAN_LABELS = ((1, "spleen"), (52, "aorta"))
SERIES_SUFFIX = re.compile(r"_s(\d+)$")


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def take_order(series_name: str) -> tuple[int, str]:
    """Sort key: the unsuffixed take1 first, then ascending ``s`` value."""
    match = SERIES_SUFFIX.search(series_name)
    if match is None:
        return (-1, series_name)
    return (int(match.group(1)), series_name)


def take_label(series_name: str) -> str:
    match = SERIES_SUFFIX.search(series_name)
    return f"s{match.group(1)}" if match else "take1"


def series_in_session(session: Path) -> list[tuple[str, Path]]:
    entries = [
        (path.parent.name, session / "segmentation" / path.parent.name / "seg.nii.gz")
        for path in session.glob("nifti/*/ct.nii.gz")
        if (session / "segmentation" / path.parent.name / "seg.nii.gz").exists()
    ]
    return sorted(entries, key=lambda item: take_order(item[0]))


def pair_resamples(
    first: sitk.Image,
    second: sitk.Image,
    want_landmark: bool,
) -> dict[str, sitk.Image]:
    """Resample the moving label map once per transform (shared across organs)."""
    resampled = {
        "identity": resample_mask(second, first, sitk.Transform(3, sitk.sitkIdentity)),
    }
    if want_landmark:
        resampled["landmark_all"] = resample_mask(
            second, first, landmark_rigid_output_transform(
                first, second, list(REGISTRATION_LABELS)
            )
        )
        resampled["landmark_vertebrae"] = resample_mask(
            second, first, landmark_rigid_output_transform(
                first, second, list(VERTEBRA_LABELS)
            )
        )
    return resampled


def write_summary(rows: list[dict], path: Path) -> None:
    by_session: dict[tuple[str, str], list[dict]] = {}
    by_take: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        by_session.setdefault((row["session"], row["organ"]), []).append(row)
        by_take.setdefault((row["session"], row["take_a"], row["organ"]), []).append(row)
        by_take.setdefault((row["session"], row["take_b"], row["organ"]), []).append(row)

    def aggregates(items: list[dict]) -> dict[str, float]:
        values = [float(item["dice"]) for item in items]
        result = {
            "n": len(values),
            "mean_dice": statistics.mean(values),
            "min_dice": min(values),
            "max_dice": max(values),
        }
        for column in ("dice_landmark_all", "dice_landmark_vertebrae"):
            if column in items[0]:
                landmark_values = [float(item[column]) for item in items]
                result["mean_" + column.replace("dice_", "")] = statistics.mean(
                    landmark_values
                )
        return result

    fields = [
        "level", "session", "take", "organ", "n", "mean_dice", "min_dice",
        "max_dice", "mean_landmark_all", "mean_landmark_vertebrae",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (session, organ), items in sorted(by_session.items()):
            writer.writerow({"level": "session", "session": session, "take": "",
                             "organ": organ, **aggregates(items)})
        for (session, take, organ), items in sorted(by_take.items()):
            writer.writerow({"level": "take", "session": session, "take": take,
                             "organ": organ, **aggregates(items)})


def main(args: argparse.Namespace) -> None:
    sessions = sorted(path for path in args.dataset_root.iterdir() if path.is_dir())
    if args.sessions:
        wanted = {name.strip() for name in args.sessions.split(",") if name.strip()}
        sessions = [session for session in sessions if session.name in wanted]

    rows: list[dict] = []
    for session in sessions:
        series = series_in_session(session)
        if len(series) < 2:
            continue
        loaded = {
            name: sitk.ReadImage(str(path), sitk.sitkUInt16) for name, path in series
        }
        pairs = list(itertools.combinations(series, 2))
        for index, ((name_a, _), (name_b, _)) in enumerate(pairs, start=1):
            first, second = loaded[name_a], loaded[name_b]
            resampled = pair_resamples(first, second, args.landmark_dice)
            for label, organ in ORGAN_LABELS:
                row = {
                    "session": session.name,
                    "series_a": name_a,
                    "series_b": name_b,
                    "take_a": take_label(name_a),
                    "take_b": take_label(name_b),
                    "organ": organ,
                    "dice": dice(first, resampled["identity"], label),
                    "centroid_displacement_mm": centroid_displacement(first, second, label),
                }
                for key in ("landmark_all", "landmark_vertebrae"):
                    if key in resampled:
                        row["dice_" + key] = dice(first, resampled[key], label)
                rows.append(row)
            print(
                f"[ceiling] {session.name} pair {index}/{len(pairs)} ({name_b})",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"workflow_version={WORKFLOW_VERSION}")
    print(f"pairwise rows: {len(rows)}")
    for organ in ("spleen", "aorta"):
        values = [float(r["dice"]) for r in rows if r["organ"] == organ]
        if not values:
            continue
        print(
            organ,
            "n=", len(values),
            "median=", round(statistics.median(values), 4),
            "min=", round(min(values), 4),
            "max=", round(max(values), 4),
        )
        if args.landmark_dice:
            for column in ("dice_landmark_all", "dice_landmark_vertebrae"):
                landmark = [float(r[column]) for r in rows if r["organ"] == organ]
                print(
                    f"  {column} median=", round(statistics.median(landmark), 4),
                    "min=", round(min(landmark), 4),
                )

    summary = args.summary or args.output.with_name(
        args.output.stem + "_summary.csv"
    )
    write_summary(rows, summary)
    print(f"summary: {summary}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Estimate within-session segmentation Dice ceiling")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None,
                    help="per-session/per-take summary CSV (default: <output>_summary.csv)")
    ap.add_argument("--sessions", default=None,
                    help="comma-separated session folder names to restrict to")
    ap.add_argument("--landmark-dice", action="store_true",
                    help="also report in-memory bone-landmark (Kabsch) Dice")
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
