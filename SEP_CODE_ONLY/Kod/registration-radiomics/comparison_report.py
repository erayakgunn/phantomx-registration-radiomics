"""Create a human-readable before/after spatial comparison report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from metrics import compare_label
from registration import read_image, resample_mask


def draw_contours(axis, image: np.ndarray, fixed: np.ndarray, moving: np.ndarray, z: int) -> None:
    low, high = np.percentile(image[z], [1, 99])
    axis.imshow(image[z], cmap="gray", vmin=low, vmax=high)
    axis.contour(fixed[z], levels=[0.5], colors="lime", linewidths=1.2)
    axis.contour(moving[z], levels=[0.5], colors="red", linewidths=1.2)
    axis.axis("off")


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    fixed_ct = read_image(args.fixed_ct, sitk.sitkFloat32)
    fixed_seg = read_image(args.fixed_seg, sitk.sitkUInt16)
    moving_seg = read_image(args.moving_seg, sitk.sitkUInt16)
    registered_seg = read_image(args.registered_seg, sitk.sitkUInt16)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    identity_seg = resample_mask(moving_seg, fixed_seg, identity)
    image = sitk.GetArrayFromImage(fixed_ct)
    rows = []
    figure, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, label, name in ((0, 1, "spleen"), (1, 52, "aorta")):
        fixed_array = sitk.GetArrayFromImage(fixed_seg) == label
        identity_array = sitk.GetArrayFromImage(identity_seg) == label
        registered_array = sitk.GetArrayFromImage(registered_seg) == label
        occupied = np.argwhere(fixed_array | identity_array | registered_array)
        z = int(np.median(occupied[:, 0])) if occupied.size else image.shape[0] // 2
        draw_contours(axes[row, 0], image, fixed_array, identity_array, z)
        draw_contours(axes[row, 1], image, fixed_array, registered_array, z)
        axes[row, 2].imshow(fixed_array[z].astype(int) - registered_array[z].astype(int), cmap="bwr", vmin=-1, vmax=1)
        axes[row, 2].axis("off")
        axes[row, 0].set_title(f"{name}: identity/world-space")
        axes[row, 1].set_title(f"{name}: registered")
        axes[row, 2].set_title(f"{name}: mask difference")
        before = compare_label(fixed_seg, identity_seg, label)
        after = compare_label(fixed_seg, registered_seg, label)
        rows.append({
            "organ": name,
            "before_dice": before["dice"],
            "after_dice": after["dice"],
            "before_hd95_mm": before["hd95_mm"],
            "after_hd95_mm": after["hd95_mm"],
            "before_assd_mm": before["assd_mm"],
            "after_assd_mm": after["assd_mm"],
        })
    figure.suptitle("PhantomX spatial standardization: green=fixed, red=moving")
    figure.tight_layout()
    figure.savefig(args.output / "comparison_report.png", dpi=180)
    plt.close(figure)
    with (args.output / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Create before/after registration figure")
    ap.add_argument("--fixed-ct", type=Path, required=True)
    ap.add_argument("--fixed-seg", type=Path, required=True)
    ap.add_argument("--moving-seg", type=Path, required=True)
    ap.add_argument("--registered-seg", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
