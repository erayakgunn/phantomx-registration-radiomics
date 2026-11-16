"""Generate per-view raw images for the detail viewer.

For each existing detail view (3 axial + coronal per organ), render the raw
moving CT at the anatomically corresponding slice, with the moving contour
(red) and the fixed contour (green, when inside the moving FOV). Adds a
``raw`` path to every view in data.js so the detail viewer can show
Ham | Centroid | Rigid side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "registration-radiomics"))
import generate_site_raw_gallery as g  # noqa: E402

base = g.base
SITE = g.SITE
DATA_JS = g.DATA_JS


def union_coronal(fixed, raw, organ, spacing, margin_mm):
    z, y, x = np.where((fixed == organ) | (raw == organ))
    if not len(z):
        return None
    mz = int(round(margin_mm / spacing[2]))
    mx = int(round(margin_mm / spacing[0]))
    z0, z1 = max(0, int(z.min()) - mz), min(fixed.shape[0], int(z.max()) + mz + 1)
    x0, x1 = max(0, int(x.min()) - mx), min(fixed.shape[2], int(x.max()) + mx + 1)
    return z0, z1, x0, x1


def frac_index(fixed_idx, moving_idx, value):
    lo, hi = int(fixed_idx.min()), int(fixed_idx.max())
    frac = 0.5 if hi <= lo else (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    mlo, mhi = int(moving_idx.min()), int(moving_idx.max())
    return int(round(mlo + frac * (mhi - mlo)))


def process_case(root: Path, dataset: str, case: dict) -> bool:
    day = str(case["day"])
    series_full = case["seriesFull"]
    case_dir = SITE / Path(case["gallery"]).parent
    if not case_dir.exists():
        return False
    fixed_seg_path, fixed_ct_path = g.fixed_paths(root, dataset)
    moving_path = g.moving_seg(root, dataset, day, series_full)
    moving_ct_path = g.moving_ct(root, dataset, day, series_full)
    if not (fixed_seg_path.exists() and moving_path.exists() and moving_ct_path.exists()):
        return False

    fixed_seg = sitk.ReadImage(str(fixed_seg_path))
    fixed_ct = sitk.ReadImage(str(fixed_ct_path))
    moving = sitk.ReadImage(str(moving_path))
    moving_ct_img = sitk.ReadImage(str(moving_ct_path))
    fixed_arr = sitk.GetArrayFromImage(fixed_seg)
    fixed_ct_arr = sitk.GetArrayFromImage(fixed_ct)
    moving_arr = sitk.GetArrayFromImage(moving)
    moving_ct_arr = sitk.GetArrayFromImage(moving_ct_img)
    spacing_m = np.array(moving_ct_img.GetSpacing(), dtype=float)
    spacing_f = np.array(fixed_seg.GetSpacing(), dtype=float)

    fixed_on_moving = sitk.Resample(fixed_seg, moving_ct_img, sitk.Transform(3, sitk.sitkIdentity),
                                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    fixed_on_moving_arr = sitk.GetArrayFromImage(fixed_on_moving)
    identity = sitk.Resample(moving, fixed_seg, sitk.Transform(3, sitk.sitkIdentity),
                             sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    identity_arr = sitk.GetArrayFromImage(identity)

    changed = False
    for organ, label in g.ORGANS:
        fixed_idx = np.where(fixed_arr == label)
        moving_idx = np.where(moving_arr == label)
        if not len(fixed_idx[0]) or not len(moving_idx[0]):
            continue
        fy, fz, fx = fixed_idx[1], fixed_idx[0], fixed_idx[2]
        my, mz, mx = moving_idx[1], moving_idx[0], moving_idx[2]

        # axial views
        for i, view in enumerate(case["views"][organ]["axial"]):
            z_f = int(view["label"].split()[1])
            z_m = frac_index(fz, mz, z_f)
            crop = g.union_crop(fixed_on_moving_arr.shape, [fixed_on_moving_arr, moving_arr],
                                label, spacing_m, g.MARGIN_MM[organ])
            if crop is None:
                continue
            y0, y1, x0, x1 = crop
            out = case_dir / f"{organ}_axial_{i}_raw.jpg"
            base.render(moving_ct_arr[z_m, y0:y1, x0:x1],
                        fixed_on_moving_arr[z_m, y0:y1, x0:x1] == label,
                        moving_arr[z_m, y0:y1, x0:x1] == label,
                        out, size=560, quality=82)
            view["raw"] = f"{Path(case['gallery']).parent.as_posix()}/{out.name}"
            changed = True

            crop_f = g.union_crop(fixed_arr.shape, [fixed_arr, identity_arr],
                                  label, spacing_f, g.MARGIN_MM[organ])
            if crop_f is None:
                continue
            yf0, yf1, xf0, xf1 = crop_f
            out_i = case_dir / f"{organ}_axial_{i}_identity.jpg"
            base.render(fixed_ct_arr[z_f, yf0:yf1, xf0:xf1],
                        fixed_arr[z_f, yf0:yf1, xf0:xf1] == label,
                        identity_arr[z_f, yf0:yf1, xf0:xf1] == label,
                        out_i, size=560, quality=82)
            view["identity"] = f"{Path(case['gallery']).parent.as_posix()}/{out_i.name}"
            changed = True

        # coronal view
        for view in case["views"][organ]["coronal"]:
            y_f = int(view["label"].split()[1])
            y_m = frac_index(fy, my, y_f)
            crop = union_coronal(fixed_on_moving_arr, moving_arr, label, spacing_m, g.MARGIN_MM[organ])
            if crop is None:
                continue
            z0, z1, x0, x1 = crop
            out = case_dir / f"{organ}_coronal_raw.jpg"
            base.render(moving_ct_arr[z0:z1, y_m, x0:x1],
                        fixed_on_moving_arr[z0:z1, y_m, x0:x1] == label,
                        moving_arr[z0:z1, y_m, x0:x1] == label,
                        out, size=560, quality=82)
            view["raw"] = f"{Path(case['gallery']).parent.as_posix()}/{out.name}"
            changed = True

            crop_cf = union_coronal(fixed_arr, identity_arr, label, spacing_f, g.MARGIN_MM[organ])
            if crop_cf is None:
                continue
            zc0, zc1, xc0, xc1 = crop_cf
            out_i = case_dir / f"{organ}_coronal_identity.jpg"
            base.render(fixed_ct_arr[zc0:zc1, y_f, xc0:xc1],
                        fixed_arr[zc0:zc1, y_f, xc0:xc1] == label,
                        identity_arr[zc0:zc1, y_f, xc0:xc1] == label,
                        out_i, size=560, quality=82)
            view["identity"] = f"{Path(case['gallery']).parent.as_posix()}/{out_i.name}"
            changed = True
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    data = g.load_dataset(DATA_JS)
    done = failed = 0
    for dataset, root in (("sep", g.SEP_ROOT), ("pcct", g.PCCT_ROOT)):
        for case in data[dataset]["cases"]:
            if args.only and args.only not in case["id"]:
                continue
            if args.limit and done >= args.limit:
                break
            try:
                ok = process_case(root, dataset, case)
            except Exception as exc:
                print(f"[FAIL] {dataset}/{case['id']}: {exc}")
                ok = False
            if ok:
                done += 1
                print(f"[ok] {dataset}/{case['id']}")
            else:
                failed += 1
        if args.limit and done >= args.limit:
            break
    new = "window.DATASETS = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n"
    new += 'const DATASET_KEY = new URLSearchParams(location.search).get("dataset") || "sep";\n'
    new += "window.CASES = window.DATASETS[DATASET_KEY] || window.DATASETS.sep;\n"
    DATA_JS.write_text(new, encoding="utf-8")
    print(f"[done] ok={done} failed={failed}")


if __name__ == "__main__":
    main()
