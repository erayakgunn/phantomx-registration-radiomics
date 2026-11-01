"""Extract fixed-size centered VOIs for target organs across a LABELED tree.

The VOI is defined in physical space from the reference (base) series:
- center = organ centroid of the reference series
- size   = fixed physical cube (mm), same for every take

Each take's CT and segmentation are resampled onto the VOI grid and saved,
together with per-take coverage/HU statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import label_map
from registration import read_image


def centroid_phys(seg: sitk.Image, label: int):
    arr = sitk.GetArrayViewFromImage(seg)
    idx = np.where(arr == label)
    if len(idx[0]) == 0:
        return None
    com = np.array([idx[0].mean(), idx[1].mean(), idx[2].mean()])
    return np.array(seg.TransformContinuousIndexToPhysicalPoint(
        (float(com[2]), float(com[1]), float(com[0]))
    ))


def make_voi(center: np.ndarray, size_mm: float, spacing: np.ndarray):
    size_vox = [max(1, int(round(size_mm / spacing[i]))) for i in range(3)]
    origin = [center[i] - (size_vox[i] - 1) / 2.0 * spacing[i] for i in range(3)]
    return size_vox, origin


def voi_image(reference: sitk.Image, size_vox, origin, pixel_type):
    image = sitk.Image(size_vox, pixel_type)
    image.SetSpacing(reference.GetSpacing())
    image.SetOrigin(origin)
    image.SetDirection(reference.GetDirection())
    return image


def parse_organs(values: list[str]):
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
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    organs = parse_organs(args.organs)
    rows = []

    for group in sorted(p for p in labeled.iterdir() if p.is_dir()):
        nifti = group / "nifti"
        segmentation = group / "segmentation"
        if not nifti.is_dir() or not segmentation.is_dir():
            continue
        series = sorted(p.name for p in nifti.iterdir() if p.is_dir())
        base = next((s for s in series if "_s" not in s.split("recon")[-1]), series[0])
        base_seg = read_image(segmentation / base / "seg.nii.gz", sitk.sitkUInt16)
        base_ct = read_image(nifti / base / "ct.nii.gz", sitk.sitkFloat32)
        spacing = np.array(base_ct.GetSpacing(), dtype=float)

        organ_specs = []
        for label, organ, size_mm in organs:
            if label is None:
                label = label_map.load_label_map(
                    segmentation / base / "seg.nii.gz"
                ).get(organ)
            if label is not None:
                organ_specs.append((label, organ, size_mm))
        for label, organ, size_mm in organ_specs:
            center = centroid_phys(base_seg, label)
            if center is None:
                print(f"[skip] {group.name} {organ}: label {label} yok")
                continue
            size_vox, origin = make_voi(center, size_mm, spacing)
            group_dir = output_root / group.name
            meta = {
                "group": group.name,
                "organ": organ,
                "label": label,
                "reference_series": base,
                "center_phys": [float(x) for x in center],
                "size_mm": size_mm,
                "size_vox": size_vox,
                "spacing": [float(x) for x in spacing],
                "origin": [float(x) for x in origin],
            }
            organ_dir = group_dir / organ
            organ_dir.mkdir(parents=True, exist_ok=True)
            (organ_dir / "voi_metadata.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

            for s in series:
                seg = read_image(segmentation / s / "seg.nii.gz", sitk.sitkUInt16)
                ct = read_image(nifti / s / "ct.nii.gz", sitk.sitkFloat32)
                ct_ref = voi_image(base_ct, size_vox, origin, sitk.sitkFloat32)
                seg_ref = voi_image(base_ct, size_vox, origin, sitk.sitkUInt16)
                ct_voi = sitk.Resample(ct, ct_ref, sitk.Transform(3, sitk.sitkIdentity),
                                       sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
                seg_voi = sitk.Resample(seg, seg_ref, sitk.Transform(3, sitk.sitkIdentity),
                                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)

                series_dir = organ_dir / s
                series_dir.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(ct_voi, str(series_dir / "ct_voi.nii.gz"))
                sitk.WriteImage(seg_voi, str(series_dir / "seg_voi.nii.gz"))

                mask = sitk.GetArrayViewFromImage(seg_voi) == label
                values = sitk.GetArrayViewFromImage(ct_voi)
                frac = float(mask.mean())
                mean_hu = float(values[mask].mean()) if mask.any() else float("nan")
                sd_hu = float(values[mask].std()) if mask.any() else float("nan")
                native_center = centroid_phys(seg, label)
                shift = float(np.linalg.norm(native_center - center)) if native_center is not None else float("nan")
                rows.append({
                    "dataset": labeled.parent.name,
                    "group": group.name,
                    "organ": organ,
                    "series": s,
                    "size_mm": size_mm,
                    "size_vox": "x".join(str(v) for v in size_vox),
                    "organ_fraction": round(frac, 5),
                    "mean_hu": round(mean_hu, 3),
                    "sd_hu": round(sd_hu, 3),
                    "centroid_shift_mm": round(shift, 3),
                })
            print(f"[voi] {group.name} {organ}: {len(series)} seri, VOI {size_vox}")

    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[voi] summary: {summary} ({len(rows)} satir)")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract fixed-size centered organ VOIs")
    ap.add_argument("--labeled", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--organs", nargs="+", default=["liver:30", "aorta:10"])
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
