"""Generate before/after overlay figures for every registered pair."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from metrics import compare_label
from registration import read_image, resample_mask
import label_map


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def overlay(axis, image, fixed, moving, z, title):
    low, high = np.percentile(image[z], [1, 99])
    axis.imshow(image[z], cmap="gray", vmin=low, vmax=high)
    axis.contour(fixed[z], levels=[0.5], colors="lime", linewidths=1.0)
    axis.contour(moving[z], levels=[0.5], colors="red", linewidths=1.0)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def main(args: argparse.Namespace) -> None:
    fixed_ct = read_image(args.fixed_ct, sitk.sitkFloat32)
    fixed_seg = read_image(args.fixed_seg, sitk.sitkUInt16)
    mapping = label_map.load_label_map(args.fixed_seg)
    organ_labels = [
        (row, mapping[name], name)
        for row, name in enumerate(("spleen", "aorta"))
        if name in mapping
    ]
    image = sitk.GetArrayFromImage(fixed_ct)
    fixed_array = sitk.GetArrayFromImage(fixed_seg)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    output = args.registration_root / "visual_qc"
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    pair_dirs = sorted(
        path for path in args.registration_root.iterdir()
        if path.is_dir() and "__" in path.name
        and (path / "seg_registered.nii.gz").exists()
    )
    for pair in pair_dirs:
        if args.session:
            prefix = f"{args.session}__"
            if not pair.name.startswith(prefix):
                continue
            session = args.session
            series = pair.name[len(prefix):]
        else:
            session, series = pair.name.split("__", 1)
        moving_path = args.dataset_root / session / "segmentation" / series / "seg.nii.gz"
        if not moving_path.exists():
            continue
        moving = read_image(moving_path, sitk.sitkUInt16)
        identity_seg = resample_mask(moving, fixed_seg, identity)
        registered_seg = read_image(pair / "seg_registered.nii.gz", sitk.sitkUInt16)
        figure, axes = plt.subplots(2, 3, figsize=(13, 8))
        for row, label, organ in organ_labels:
            fixed_mask = fixed_array == label
            identity_mask = sitk.GetArrayFromImage(identity_seg) == label
            registered_mask = sitk.GetArrayFromImage(registered_seg) == label
            occupied = np.argwhere(fixed_mask | identity_mask | registered_mask)
            z = int(np.median(occupied[:, 0])) if occupied.size else image.shape[0] // 2
            overlay(axes[row, 0], image, fixed_mask, identity_mask, z, f"{organ} identity z={z}")
            overlay(axes[row, 1], image, fixed_mask, registered_mask, z, f"{organ} registered z={z}")
            axes[row, 2].imshow(
                fixed_mask[z].astype(int) - registered_mask[z].astype(int),
                cmap="bwr", vmin=-1, vmax=1,
            )
            axes[row, 2].set_title(f"{organ} mask difference", fontsize=8)
            axes[row, 2].axis("off")
            before = compare_label(fixed_seg, identity_seg, label, fixed_array, sitk.GetArrayFromImage(identity_seg))
            after = compare_label(fixed_seg, registered_seg, label, fixed_array, sitk.GetArrayFromImage(registered_seg))
            rows.append({
                "pair": pair.name,
                "organ": organ,
                "before_dice": before["dice"],
                "after_dice": after["dice"],
                "before_hd95_mm": before["hd95_mm"],
                "after_hd95_mm": after["hd95_mm"],
            })
        figure.suptitle(pair.name, fontsize=8)
        figure.tight_layout()
        figure.savefig(output / f"{pair.name}.png", dpi=120)
        plt.close(figure)

    with (output / "visual_qc_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[visual-qc] {len(pair_dirs)} pair için çıktı: {output}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate all registered pair overlay reports")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("registration_root", type=Path)
    ap.add_argument("--fixed-ct", type=Path, required=True)
    ap.add_argument("--fixed-seg", type=Path, required=True)
    ap.add_argument("--session")
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
