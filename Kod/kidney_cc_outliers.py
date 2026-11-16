"""Diagnose kidney ASSD/MSSD/HD95 outliers caused by isolated components.

The source masks are not modified. For each selected pair the registered kidney
mask is split into connected components; metrics are recomputed with only the
largest component to show how much of the surface-distance error comes from a
small distant island.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "registration-radiomics"))
from metrics import compare_label  # noqa: E402
from registration import resample_mask  # noqa: E402

CSV_IN = Path(r"C:/Users/eray/Desktop/KIDNEY_METRICS_PCCT_SEP.csv")
CSV_OUT = Path(r"C:/Users/eray/Desktop/KIDNEY_CC_OUTLIERS.csv")
BATCH = {
    "PCCT cross-date": Path(
        r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/"
        r"PhantomX/REGISTRATION/batch_to_20260805_v1.0.0"
    ),
    "SEP cross-date": Path(
        r"C:/Users/eray/Desktop/SEP - ['Br40f',_'3']_120_100_300_1/"
        r"PhantomX/REGISTRATION/batch_to_20260805_v6_skip2"
    ),
}
LABEL = {"kidney_right": 2, "kidney_left": 3}
STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def component_info(image: sitk.Image, label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    array = sitk.GetArrayFromImage(image)
    mask = array == label
    idx = np.where(mask)
    if not len(idx[0]):
        return array, mask, np.array([]), 0
    slices = tuple(
        slice(max(0, int(i.min()) - 2), int(i.max()) + 3) for i in idx
    )
    sub = mask[slices]
    labeled, count = ndimage.label(sub, structure=STRUCTURE)
    sizes = ndimage.sum(sub, labeled, range(1, count + 1))
    keep = int(np.argmax(sizes)) + 1
    return array, slices, sizes, keep


def largest_only(image: sitk.Image, label: int) -> tuple[sitk.Image, int, float]:
    array, slices, sizes, keep = component_info(image, label)
    if not len(sizes):
        return image, 0, 0.0
    mask = array == label
    sub = mask[slices]
    labeled, count = ndimage.label(sub, structure=STRUCTURE)
    out = np.zeros_like(mask)
    sub_out = out[slices]
    sub_out[labeled == keep] = True
    out[slices] = sub_out
    result = sitk.GetImageFromArray(out.astype(np.uint16) * label)
    result.CopyInformation(image)
    return result, int(count), float(sizes.max() / mask.sum())


def main() -> None:
    frame = pd.read_csv(CSV_IN)
    cross = frame[frame["Scope"] == "cross-date"].copy()
    selected = cross[
        (cross["MSSD_after"] > 1.0) | (cross["ASSD_after"] > 1.0)
    ].sort_values("MSSD_after", ascending=False)
    if selected.empty:
        print("Aykiri satir yok.")
        return

    rows = []
    for _, row in selected.iterrows():
        dataset = row["Dataset"]
        base = BATCH[dataset]
        matches = sorted(base.glob(f"*__{row['Pair']}"))
        if not matches:
            print(f"[skip] pair bulunamadi: {row['Pair']}")
            continue
        pair_dir = matches[0]
        state_path = pair_dir / "pilot_state.json"
        tfm_path = pair_dir / "moving_to_fixed.tfm"
        if not state_path.exists() or not tfm_path.exists():
            print(f"[skip] {pair_dir.name}")
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        fixed = sitk.ReadImage(state["fixed_seg"])
        moving = sitk.ReadImage(state["moving_seg"])
        after = resample_mask(
            moving, fixed, sitk.ReadTransform(str(tfm_path))
        )
        label = LABEL[row["Organ"]]
        fixed_largest, _, _ = largest_only(fixed, label)
        after_largest, components, fraction = largest_only(after, label)
        metric = compare_label(fixed_largest, after_largest, label)
        rows.append({
            "Dataset": dataset,
            "Pair": row["Pair"],
            "Organ": row["Organ"],
            "Raw_Dice": round(float(row["Dice_after"]), 4),
            "Raw_ASSD_mm": round(float(row["ASSD_after"]), 4),
            "Raw_MSSD_mm": round(float(row["MSSD_after"]), 4),
            "Raw_HD95_mm": round(float(row["HD95_after"]), 3),
            "Components_after": components,
            "Largest_fraction": round(fraction, 4),
            "Largest_Dice": round(float(metric["dice"]), 4),
            "Largest_ASSD_mm": round(float(metric["assd_mm"]), 4),
            "Largest_MSSD_mm": round(float(metric["mssd_mm"]), 4),
            "Largest_HD95_mm": round(float(metric["hd95_mm"]), 3),
        })
        print(
            f"{dataset} {row['Pair'][-24:]} {row['Organ']}: "
            f"ASSD {row['ASSD_after']:.2f}->{metric['assd_mm']:.2f}, "
            f"MSSD {row['MSSD_after']:.2f}->{metric['mssd_mm']:.2f}, "
            f"HD95 {row['HD95_after']:.1f}->{metric['hd95_mm']:.1f}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"[OK] {CSV_OUT} ({len(out)} satir)")


if __name__ == "__main__":
    main()
