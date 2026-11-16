"""Scan every kidney mask for isolated connected components.

For every row in KIDNEY_METRICS_PCCT_SEP.csv the registered moving mask is
split into 3D connected components. The largest component is kept for a
diagnostic metric pass; the source masks are never modified.

Output: KIDNEY_CC_SCAN.csv
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
CSV_OUT = Path(r"C:/Users/eray/Desktop/KIDNEY_CC_SCAN.csv")
PCCT_BATCH = Path(
    r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/"
    r"PhantomX/REGISTRATION/batch_to_20260805_v1.0.0"
)
SEP_BATCH = Path(
    r"C:/Users/eray/Desktop/SEP - ['Br40f',_'3']_120_100_300_1/"
    r"PhantomX/REGISTRATION/batch_to_20260805_v6_skip2"
)
WORK = {
    "PCCT test-retest": Path(r"C:/Users/eray/Desktop/zurich_pcct_work/REGISTRATION"),
    "SEP test-retest": Path(r"C:/Users/eray/Desktop/zurich_sep_work/REGISTRATION"),
}
LABEL = {"kidney_right": 2, "kidney_left": 3}
STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def pair_directory(dataset: str, group: str, series: str) -> Path | None:
    if dataset == "PCCT cross-date":
        matches = sorted(PCCT_BATCH.glob(f"*__{series}"))
        return matches[0] if matches else None
    if dataset == "SEP cross-date":
        matches = sorted(SEP_BATCH.glob(f"*__{series}"))
        return matches[0] if matches else None
    base = WORK.get(dataset)
    if base is None:
        return None
    path = base / group / f"{group}__{series}"
    return path if path.exists() else None


def component_analysis(image: sitk.Image, label: int) -> dict:
    array = sitk.GetArrayFromImage(image)
    mask = array == label
    if not mask.any():
        return {
            "components": 0, "largest_fraction": 0.0,
            "small_components": 0, "small_volume_mm3": 0.0,
            "largest_image": image,
        }
    idx = np.where(mask)
    slices = tuple(
        slice(max(0, int(i.min()) - 2), int(i.max()) + 3) for i in idx
    )
    sub = mask[slices]
    labeled, count = ndimage.label(sub, structure=STRUCTURE)
    sizes = ndimage.sum(sub, labeled, range(1, count + 1))
    keep = int(np.argmax(sizes)) + 1
    small = sizes[np.arange(len(sizes)) + 1 != keep]
    spacing = image.GetSpacing()
    voxel_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    out = np.zeros_like(mask)
    sub_out = out[slices]
    sub_out[labeled == keep] = True
    out[slices] = sub_out
    largest = sitk.GetImageFromArray(out.astype(np.uint16) * label)
    largest.CopyInformation(image)

    return {
        "components": int(count),
        "largest_fraction": float(sizes.max() / mask.sum()),
        "small_components": int((small >= 50).sum()),
        "small_volume_mm3": float(small.sum() * voxel_mm3),
        "largest_image": largest,
    }


def main() -> None:
    frame = pd.read_csv(CSV_IN)
    cache: dict[Path, tuple[sitk.Image, sitk.Image, sitk.Transform]] = {}
    rows = []
    for index, row in frame.iterrows():
        dataset = row["Dataset"]
        pair_dir = pair_directory(dataset, row["Group"], row["Pair"])
        if pair_dir is None:
            print(f"[skip] {dataset} {row['Pair']}")
            continue
        if pair_dir not in cache:
            state_path = pair_dir / "pilot_state.json"
            tfm_path = pair_dir / "moving_to_fixed.tfm"
            if not state_path.exists() or not tfm_path.exists():
                cache[pair_dir] = None  # type: ignore[assignment]
            else:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                fixed = sitk.ReadImage(state["fixed_seg"])
                moving = sitk.ReadImage(state["moving_seg"])
                after = resample_mask(
                    moving, fixed, sitk.ReadTransform(str(tfm_path))
                )
                cache[pair_dir] = (fixed, after, None)
        cached = cache.get(pair_dir)
        if not cached:
            continue
        fixed, after, _ = cached
        label = LABEL[row["Organ"]]
        fixed_info = component_analysis(fixed, label)
        after_info = component_analysis(after, label)
        metric = compare_label(
            fixed_info["largest_image"], after_info["largest_image"], label
        )
        rows.append({
            "Scope": row["Scope"],
            "Dataset": dataset,
            "Group": row["Group"],
            "Pair": row["Pair"],
            "Organ": row["Organ"],
            "Raw_Dice": round(float(row["Dice_after"]), 4),
            "Raw_ASSD_mm": round(float(row["ASSD_after"]), 4),
            "Raw_MSSD_mm": round(float(row["MSSD_after"]), 4),
            "Raw_HD95_mm": round(float(row["HD95_after"]), 3),
            "Fixed_components": fixed_info["components"],
            "After_components": after_info["components"],
            "Largest_fraction": round(after_info["largest_fraction"], 4),
            "Small_components": after_info["small_components"],
            "Small_volume_mm3": round(after_info["small_volume_mm3"], 1),
            "Largest_ASSD_mm": round(float(metric["assd_mm"]), 4),
            "Largest_MSSD_mm": round(float(metric["mssd_mm"]), 4),
            "Largest_HD95_mm": round(float(metric["hd95_mm"]), 3),
        })
        if (index + 1) % 50 == 0:
            print(f"[{index + 1}/{len(frame)}]")

    out = pd.DataFrame(rows)
    out.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")

    print("\n=== ozet ===")
    print("satir:", len(out))
    island = out[out["After_components"] > 1]
    print("adacikli (components>1):", len(island),
          "| kucuk bilesen >=50 vox:", int((out["Small_components"] > 0).sum()))
    if not island.empty:
        print(island.groupby(["Scope", "Dataset", "Organ"]).size().to_string())
        print("\n=== en buyuk adaciklar (small_volume_mm3) ===")
        print(
            island.sort_values("Small_volume_mm3", ascending=False)[
                ["Dataset", "Pair", "Organ", "Raw_ASSD_mm", "Raw_MSSD_mm",
                 "After_components", "Largest_fraction", "Small_volume_mm3",
                 "Largest_ASSD_mm", "Largest_MSSD_mm"]
            ].head(20).to_string(index=False)
        )
    clean_bad = out[(out["After_components"] == 1) & (out["Raw_ASSD_mm"] > 1.0)]
    print("\n=== tek bilesen ama ASSD>1 mm (gercek segmentasyon) ===")
    print(clean_bad[
        ["Dataset", "Pair", "Organ", "Raw_Dice", "Raw_ASSD_mm",
         "Raw_MSSD_mm", "Raw_HD95_mm"]
    ].to_string(index=False))
    print(f"\n[OK] {CSV_OUT}")


if __name__ == "__main__":
    main()
