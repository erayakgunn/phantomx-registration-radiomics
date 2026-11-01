"""Compute organ Dice restricted to the fixed-size centered VOI.

Processes Zurich test-retest groups and the longitudinal PCCT/SEP batches,
so VOI-restricted Dice can be compared between test-retest and cross-date.

Supports cube and sphere VOIs (``--shapes cube sphere``) and writes one CSV
with a ``shape`` column.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\eray\Desktop\PCCT - Q3_Qr40f_120_100_300_1_PolyE\Kod\registration-radiomics")

import numpy as np
import SimpleITK as sitk
from registration import read_image

import label_map

ORGANS = [("liver", 30.0), ("aorta", 10.0)]


def centroid_phys(seg, label):
    arr = sitk.GetArrayViewFromImage(seg)
    idx = np.where(arr == label)
    if len(idx[0]) == 0:
        return None
    com = np.array([idx[0].mean(), idx[1].mean(), idx[2].mean()])
    return np.array(seg.TransformContinuousIndexToPhysicalPoint(
        (float(com[2]), float(com[1]), float(com[0]))
    ))


def voi_slices(fixed_seg, center, size_mm, spacing):
    size_vox = [max(1, int(round(size_mm / spacing[i]))) for i in range(3)]
    origin = [center[i] - (size_vox[i] - 1) / 2.0 * spacing[i] for i in range(3)]
    p0 = fixed_seg.TransformPhysicalPointToIndex(tuple(origin))
    p1 = fixed_seg.TransformPhysicalPointToIndex(
        tuple(origin[i] + (size_vox[i] - 1) * spacing[i] for i in range(3))
    )
    lo = [max(0, min(p0[i], p1[i])) for i in range(3)]
    hi = [min(fixed_seg.GetSize()[i], max(p0[i], p1[i]) + 1) for i in range(3)]
    # sitk index is x,y,z; numpy array is z,y,x
    return (slice(lo[2], hi[2]), slice(lo[1], hi[1]), slice(lo[0], hi[0])), lo, hi


def sphere_mask_for_slice(fixed_seg, lo, hi, center, radius_mm):
    spacing = np.array(fixed_seg.GetSpacing(), dtype=float)
    origin = np.array(fixed_seg.GetOrigin(), dtype=float)
    direction = np.array(fixed_seg.GetDirection(), dtype=float).reshape(3, 3)
    zs = np.arange(lo[2], hi[2])
    ys = np.arange(lo[1], hi[1])
    xs = np.arange(lo[0], hi[0])
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    idx = np.stack([xx, yy, zz], axis=-1).astype(float)
    phys = origin + (idx * spacing) @ direction.T
    return ((phys - np.asarray(center, dtype=float)) ** 2).sum(-1) <= radius_mm ** 2


def dice(a, b):
    denom = int(a.sum() + b.sum())
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 0.0


def process(fixed_seg_path: Path, reg_root: Path, dataset: str, rows: list, shape: str):
    if not fixed_seg_path.exists() or not reg_root.exists():
        print(f"[skip] {dataset}: eksik yol")
        return
    fixed_seg = read_image(fixed_seg_path, sitk.sitkUInt16)
    spacing = np.array(fixed_seg.GetSpacing(), dtype=float)
    fixed_arr = sitk.GetArrayFromImage(fixed_seg)
    pairs = sorted(p for p in reg_root.iterdir()
                   if p.is_dir() and (p / "seg_registered.nii.gz").exists())
    mapping = label_map.load_label_map(fixed_seg_path)
    for organ, size_mm in ORGANS:
        label = mapping.get(organ)
        if label is None:
            continue
        center = centroid_phys(fixed_seg, label)
        if center is None:
            continue
        sl, lo, hi = voi_slices(fixed_seg, center, size_mm, spacing)
        smask = sphere_mask_for_slice(fixed_seg, lo, hi, center, size_mm / 2.0) if shape == "sphere" else None
        fixed_voi = fixed_arr[sl] == label
        if smask is not None:
            fixed_voi = fixed_voi & smask
        fixed_full = fixed_arr == label
        for pair in pairs:
            moved_arr = sitk.GetArrayFromImage(read_image(pair / "seg_registered.nii.gz", sitk.sitkUInt16))
            moved_voi = moved_arr[sl] == label
            if smask is not None:
                moved_voi = moved_voi & smask
            moved_full = moved_arr == label
            rows.append({
                "dataset": dataset,
                "pair": pair.name,
                "organ": organ,
                "shape": shape,
                "voi_dice": round(dice(fixed_voi, moved_voi), 5),
                "full_dice": round(dice(fixed_full, moved_full), 5),
            })
    print(f"[voi-dice] {dataset} [{shape}]: {len(pairs)} pair")


def main() -> None:
    ap = argparse.ArgumentParser(description="VOI-restricted Dice (cube/sphere)")
    ap.add_argument("--shapes", nargs="+", choices=("cube", "sphere"), default=["cube", "sphere"])
    ap.add_argument("--registration-name", default="REGISTRATION",
                    help="Zurich REGISTRATION alt klasörü (ör. REGISTRATION_skip5)")
    ap.add_argument("--output", type=Path,
                    default=Path(r"C:/Users/eray/Desktop/zurich_pcct_work/VOI_DICE_SUMMARY_shapes.csv"))
    args = ap.parse_args()

    rows = []
    pcct_work = Path(r"C:/Users/eray/Desktop/zurich_pcct_work")
    sep_work = Path(r"C:/Users/eray/Desktop/zurich_sep_work")
    for work in (pcct_work, sep_work):
        ds = "zurich_pcct" if work == pcct_work else "zurich_sep"
        for group in sorted(p for p in (work / "LABELED").iterdir() if p.is_dir()):
            series = sorted(p.name for p in (group / "nifti").iterdir() if p.is_dir())
            base = next(s for s in series if "_s" not in s.split("recon")[-1])
            for shape in args.shapes:
                process(group / "segmentation" / base / "seg.nii.gz",
                        work / args.registration_name / group.name,
                        f"{ds}:{group.name}", rows, shape)

    pcct = Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX")
    sep = Path(r"C:/Users/eray/Desktop/SEP - ['Br40f',_'3']_120_100_300_1/PhantomX")
    for shape in args.shapes:
        process(pcct / "LABELED/20260805_PCCT_Aorta_Spleen/segmentation/USZ_Siemens_Healthineers_NAEOTOM_Alpha_20260805_Q3_Qr40f_120_100_300_1_PolyE_take1/seg.nii.gz",
                pcct / "REGISTRATION/batch_to_20260805_v1.0.0", "pcct_longitudinal", rows, shape)
        process(sep / "LABELED/20260805_SEP_Aorta_Spleen/segmentation/USZ_SIEMENS_SOMATOM_Edge_Plus_20260805_S3_Br40f_120_100_300_1_PolyE_take1_recon1mm/seg.nii.gz",
                sep / "REGISTRATION/batch_to_20260805_v6", "sep_longitudinal", rows, shape)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[voi-dice] summary: {args.output} ({len(rows)} satir)")


if __name__ == "__main__":
    main()
