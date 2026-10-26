"""Separate translation and rotation effects and create overlay figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from metrics import compare_label
from registration import read_image, resample_image, resample_mask, translation_centroid_output_transform


def rotation_only(transform_path: Path, fixed: sitk.Image) -> sitk.Transform:
    source = sitk.ReadTransform(str(transform_path))
    if isinstance(source, sitk.CompositeTransform):
        source = source.GetNthTransform(0)
    parameters = source.GetParameters()
    transform = sitk.Euler3DTransform()
    center_index = [(size - 1) / 2.0 for size in fixed.GetSize()]
    transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(center_index))
    transform.SetParameters((parameters[0], parameters[1], parameters[2], 0.0, 0.0, 0.0))
    return transform


def save_overlay(ct: sitk.Image, fixed_mask: sitk.Image, moving_mask: sitk.Image,
                 output: Path, label: int, title: str) -> None:
    ct_array = sitk.GetArrayFromImage(ct)
    fixed = sitk.GetArrayFromImage(fixed_mask) == label
    moving = sitk.GetArrayFromImage(moving_mask) == label
    occupied = np.argwhere(fixed | moving)
    if not occupied.size:
        return
    z_values = np.linspace(occupied[:, 0].min(), occupied[:, 0].max(), 3).astype(int)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, z in zip(axes, z_values):
        low, high = np.percentile(ct_array[z], [1, 99])
        axis.imshow(ct_array[z], cmap="gray", vmin=low, vmax=high)
        axis.contour(fixed[z], levels=[0.5], colors="lime", linewidths=1.2)
        axis.contour(moving[z], levels=[0.5], colors="red", linewidths=1.2)
        axis.set_title(f"z={z}")
        axis.axis("off")
    figure.suptitle(f"{title} | label {label} | green=fixed red=moving")
    figure.tight_layout()
    figure.savefig(output / f"{title}_label{label}.png", dpi=150)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    fixed_ct = read_image(args.fixed_ct, sitk.sitkFloat32)
    moving_ct = read_image(args.moving_ct, sitk.sitkFloat32)
    fixed_seg = read_image(args.fixed_seg, sitk.sitkUInt16)
    moving_seg = read_image(args.moving_seg, sitk.sitkUInt16)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    translation = translation_centroid_output_transform(fixed_seg, moving_seg)
    transforms = {
        "identity": identity,
        "translation": translation,
        "rotation_only": rotation_only(args.rigid_transform, fixed_ct),
        "rigid": sitk.ReadTransform(str(args.rigid_transform)),
    }
    rows = []
    for name, transform in transforms.items():
        transformed = resample_mask(moving_seg, fixed_seg, transform)
        transformed_ct = resample_image(moving_ct, fixed_ct, transform, sitk.sitkLinear, -1024.0)
        for label in (1, 52):
            fixed_array = sitk.GetArrayFromImage(fixed_seg)
            moving_array = sitk.GetArrayFromImage(transformed)
            metric = compare_label(fixed_seg, transformed, label, fixed_array, moving_array)
            rows.append({"transform": name, **metric})
            save_overlay(fixed_ct, fixed_seg, transformed, output, label, name)
        sitk.WriteImage(transformed_ct, str(output / f"ct_{name}.nii.gz"))
        sitk.WriteImage(transformed, str(output / f"seg_{name}.nii.gz"))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare translation and rotation effects")
    ap.add_argument("--fixed-ct", type=Path, required=True)
    ap.add_argument("--fixed-seg", type=Path, required=True)
    ap.add_argument("--moving-ct", type=Path, required=True)
    ap.add_argument("--moving-seg", type=Path, required=True)
    ap.add_argument("--rigid-transform", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
