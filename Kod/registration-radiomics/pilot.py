"""Run one fixed/moving PhantomX registration pair.

Example:
    python pilot.py --fixed-ct ... --fixed-seg ... --moving-ct ... \
        --moving-seg ... --output ... --labels 1 52
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import SimpleITK as sitk

from metrics import compare_labels
from registration import read_image, resample_image, resample_mask, rigid_register


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    fixed_ct = read_image(Path(args.fixed_ct), sitk.sitkFloat32)
    moving_ct = read_image(Path(args.moving_ct), sitk.sitkFloat32)
    fixed_seg = read_image(Path(args.fixed_seg), sitk.sitkUInt16)
    moving_seg = read_image(Path(args.moving_seg), sitk.sitkUInt16)

    if fixed_ct.GetDimension() != 3 or moving_ct.GetDimension() != 3:
        raise ValueError("Only 3D CT volumes are supported")
    if fixed_ct.GetSize() != fixed_seg.GetSize():
        raise ValueError("Fixed CT and fixed segmentation are not on the same grid")
    if moving_ct.GetSize() != moving_seg.GetSize():
        raise ValueError("Moving CT and moving segmentation are not on the same grid")

    transform = rigid_register(fixed_ct, moving_ct)
    registered_ct = resample_image(
        moving_ct, fixed_ct, transform, sitk.sitkLinear, default_value=-1024.0
    )
    registered_seg = resample_mask(moving_seg, fixed_seg, transform)

    sitk.WriteImage(registered_ct, str(output / "ct_registered.nii.gz"))
    sitk.WriteImage(registered_seg, str(output / "seg_registered.nii.gz"))
    sitk.WriteTransform(transform, str(output / "moving_to_fixed.tfm"))

    labels = args.labels or sorted(
        set(sitk.GetArrayViewFromImage(fixed_seg).ravel())
        & set(sitk.GetArrayViewFromImage(moving_seg).ravel())
        - {0}
    )
    before = compare_labels(fixed_seg, moving_seg, labels)
    after = compare_labels(fixed_seg, registered_seg, labels)
    rows = []
    for pre, post in zip(before, after):
        rows.append(
            {
                "label": pre["label"],
                "before_dice": pre["dice"],
                "after_dice": post["dice"],
                "before_centroid_displacement_mm": pre["centroid_displacement_mm"],
                "after_centroid_displacement_mm": post["centroid_displacement_mm"],
                "before_hd95_mm": pre["hd95_mm"],
                "after_hd95_mm": post["hd95_mm"],
                "before_assd_mm": pre["assd_mm"],
                "after_assd_mm": post["assd_mm"],
            }
        )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run one rigid PhantomX registration pair")
    ap.add_argument("--fixed-ct", required=True)
    ap.add_argument("--fixed-seg", required=True)
    ap.add_argument("--moving-ct", required=True)
    ap.add_argument("--moving-seg", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--labels", type=int, nargs="*")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
