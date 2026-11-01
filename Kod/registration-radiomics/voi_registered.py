"""Extract fixed-size centered VOIs from registered (cross-date) pairs.

The VOI is defined on the fixed reference grid from the fixed organ centroid.
For every registered pair, ct_registered/seg_registered are resampled onto the
same physical VOI grid, and coverage/HU/centroid-shift statistics are written.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from registration import read_image

import label_map


def centroid_phys(seg: sitk.Image, label: int):
    arr = sitk.GetArrayViewFromImage(seg)
    idx = np.where(arr == label)
    if len(idx[0]) == 0:
        return None
    com = np.array([idx[0].mean(), idx[1].mean(), idx[2].mean()])
    return np.array(seg.TransformContinuousIndexToPhysicalPoint(
        (float(com[2]), float(com[1]), float(com[0]))
    ))


def make_voi(center, size_mm, spacing):
    size_vox = [max(1, int(round(size_mm / spacing[i]))) for i in range(3)]
    origin = [center[i] - (size_vox[i] - 1) / 2.0 * spacing[i] for i in range(3)]
    return size_vox, origin


def voi_image(reference, size_vox, origin, pixel_type):
    image = sitk.Image(size_vox, pixel_type)
    image.SetSpacing(reference.GetSpacing())
    image.SetOrigin(origin)
    image.SetDirection(reference.GetDirection())
    return image


def parse_organs(values):
    organs = []
    for value in values:
        parts = value.split(":")
        if len(parts) == 2:
            organs.append((None, parts[0], float(parts[1])))
        else:
            organs.append((int(parts[0]), parts[1], float(parts[2])))
    return organs


def main(args: argparse.Namespace) -> None:
    labeled = args.labeled
    registration = args.registration
    fixed_dir = labeled / args.fixed_session
    fixed_ct = read_image(fixed_dir / "nifti" / args.fixed_series / "ct.nii.gz", sitk.sitkFloat32)
    fixed_seg_path = fixed_dir / "segmentation" / args.fixed_series / "seg.nii.gz"
    fixed_seg = read_image(fixed_seg_path, sitk.sitkUInt16)
    spacing = np.array(fixed_ct.GetSpacing(), dtype=float)
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    pairs = sorted(p for p in registration.iterdir()
                   if p.is_dir() and (p / "ct_registered.nii.gz").exists())
    mapping = label_map.load_label_map(fixed_seg_path)
    for label, organ, size_mm in parse_organs(args.organs):
        if label is None:
            label = mapping.get(organ)
        if label is None:
            continue
        center = centroid_phys(fixed_seg, label)
        if center is None:
            print(f"[skip] {organ}: label {label} fixed seg'de yok")
            continue
        size_vox, origin = make_voi(center, size_mm, spacing)
        organ_dir = output_root / organ
        organ_dir.mkdir(parents=True, exist_ok=True)
        (organ_dir / "voi_metadata.json").write_text(json.dumps({
            "fixed_session": args.fixed_session,
            "fixed_series": args.fixed_series,
            "organ": organ, "label": label,
            "center_phys": [float(x) for x in center],
            "size_mm": size_mm, "size_vox": size_vox,
            "spacing": [float(x) for x in spacing],
            "origin": [float(x) for x in origin],
        }, indent=2), encoding="utf-8")

        for pair in pairs:
            ct = read_image(pair / "ct_registered.nii.gz", sitk.sitkFloat32)
            seg = read_image(pair / "seg_registered.nii.gz", sitk.sitkUInt16)
            ct_ref = voi_image(fixed_ct, size_vox, origin, sitk.sitkFloat32)
            seg_ref = voi_image(fixed_ct, size_vox, origin, sitk.sitkUInt16)
            ct_voi = sitk.Resample(ct, ct_ref, sitk.Transform(3, sitk.sitkIdentity),
                                   sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
            seg_voi = sitk.Resample(seg, seg_ref, sitk.Transform(3, sitk.sitkIdentity),
                                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
            pair_dir = organ_dir / pair.name
            pair_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(ct_voi, str(pair_dir / "ct_voi.nii.gz"))
            sitk.WriteImage(seg_voi, str(pair_dir / "seg_voi.nii.gz"))

            mask = sitk.GetArrayViewFromImage(seg_voi) == label
            values = sitk.GetArrayViewFromImage(ct_voi)
            frac = float(mask.mean())
            mean_hu = float(values[mask].mean()) if mask.any() else float("nan")
            sd_hu = float(values[mask].std()) if mask.any() else float("nan")
            native_center = centroid_phys(seg, label)
            shift = float(np.linalg.norm(native_center - center)) if native_center is not None else float("nan")
            rows.append({
                "dataset": registration.parent.name,
                "pair": pair.name,
                "organ": organ,
                "size_mm": size_mm,
                "size_vox": "x".join(str(v) for v in size_vox),
                "organ_fraction": round(frac, 5),
                "mean_hu": round(mean_hu, 3),
                "sd_hu": round(sd_hu, 3),
                "centroid_shift_mm": round(shift, 3),
            })
        print(f"[voi-reg] {organ}: {len(pairs)} pair, VOI {size_vox}")

    with Path(args.summary).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[voi-reg] summary: {args.summary} ({len(rows)} satir)")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract centered VOIs from registered cross-date pairs")
    ap.add_argument("--labeled", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--fixed-session", required=True)
    ap.add_argument("--fixed-series", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--organs", nargs="+", default=["liver:30", "aorta:10"])
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
