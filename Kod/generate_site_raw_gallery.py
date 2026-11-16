"""Regenerate site gallery thumbnails as 4-column (ham | identity | centroid | rigid).

The raw state is the moving mask with the identity transform (no registration).
It is rendered on the fixed CT with a crop that includes the raw organ, so the
gross offset is visible next to the centroid and rigid states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "registration-radiomics"))
import generate_assets as base  # noqa: E402

SITE = Path(r"C:/Users/eray/Desktop/phantomx-overlay-site")
DATA_JS = SITE / "data.js"

SEP_ROOT = Path(r"C:/Users/eray/Desktop/SEP - ['Br40f',_'3']_120_100_300_1/PhantomX")
PCCT_ROOT = Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX")

SEP_FIXED = "USZ_SIEMENS_SOMATOM_Edge_Plus_20260805_S3_Br40f_120_100_300_1_PolyE_take1_recon1mm"
PCCT_FIXED = "USZ_Siemens_Healthineers_NAEOTOM_Alpha_20260805_Q3_Qr40f_120_100_300_1_PolyE_take1"
MARGIN_MM = {"aorta": 35.0, "spleen": 30.0, "kidney_right": 28.0, "kidney_left": 28.0}
ORGANS = [("aorta", 52), ("spleen", 1), ("kidney_right", 2), ("kidney_left", 3)]

try:
    FONT = ImageFont.truetype("arialbd.ttf", 16)
except Exception:
    FONT = ImageFont.load_default()


def load_dataset(path: Path) -> dict:
    s = path.read_text(encoding="utf-8")
    return json.loads(s[s.index("{"):s.rindex("}") + 1])


def fixed_paths(root: Path, dataset: str):
    if dataset == "sep":
        lab = root / "LABELED" / "20260805_SEP_Aorta_Spleen"
        return lab / "segmentation" / SEP_FIXED / "seg.nii.gz", lab / "nifti" / SEP_FIXED / "ct.nii.gz"
    lab = root / "LABELED" / "20260805_PCCT_Aorta_Spleen"
    return lab / "segmentation" / PCCT_FIXED / "seg.nii.gz", lab / "nifti" / PCCT_FIXED / "ct.nii.gz"


def moving_seg(root: Path, dataset: str, day: str, series_full: str) -> Path:
    session = f"{day}_SEP_Aorta_Spleen" if dataset == "sep" else f"{day}_PCCT_Aorta_Spleen"
    return root / "LABELED" / session / "segmentation" / series_full / "seg.nii.gz"


def moving_ct(root: Path, dataset: str, day: str, series_full: str) -> Path:
    session = f"{day}_SEP_Aorta_Spleen" if dataset == "sep" else f"{day}_PCCT_Aorta_Spleen"
    return root / "LABELED" / session / "nifti" / series_full / "ct.nii.gz"


def batch_dir(root: Path, dataset: str, day: str, series_full: str) -> Path:
    if dataset == "sep":
        base_dir = root / "REGISTRATION" / "batch_to_20260805_v6_skip2"
        session = f"{day}_SEP_Aorta_Spleen"
    else:
        base_dir = root / "REGISTRATION" / "batch_to_20260805_v1.0.0"
        session = f"{day}_PCCT_Aorta_Spleen"
    matches = list(base_dir.glob(f"{session}__{series_full}"))
    if not matches:
        raise FileNotFoundError(f"batch dir yok: {session} / {series_full}")
    return matches[0]


def union_crop(shape, arrays, organ, spacing, margin_mm):
    masks = [array == organ for array in arrays]
    z, y, x = np.where(np.logical_or.reduce(masks))
    if not len(y):
        return None
    my = int(round(margin_mm / spacing[1]))
    mx = int(round(margin_mm / spacing[0]))
    return (max(0, int(y.min()) - my), min(shape[1], int(y.max()) + my + 1),
            max(0, int(x.min()) - mx), min(shape[2], int(x.max()) + mx + 1))


def make_gallery(case_dir: Path, organ_tiles: dict):
    tile = 320
    labels = [("HAM", "ham"), ("IDENTITY", "identity"),
              ("CENTROID", "centroid"), ("RIGID", "rigid")]
    canvas = Image.new("RGB", (tile * len(labels), tile * len(ORGANS)), (2, 7, 12))
    draw = ImageDraw.Draw(canvas)
    for row, (organ, _) in enumerate(ORGANS):
        for col, (label, key) in enumerate(labels):
            x, y = col * tile, row * tile
            path = organ_tiles.get((organ, key))
            if path and Path(path).exists():
                image = Image.open(path).convert("RGB").resize(
                    (tile, tile), Image.Resampling.LANCZOS)
                canvas.paste(image, (x, y))
            else:
                draw.rectangle([x, y, x + tile - 1, y + tile - 1], fill=(8, 16, 26))
            text = f"{organ.upper().replace('_', ' ')} · {label}"
            bbox = draw.textbbox((0, 0), text, font=FONT)
            pad_x, pad_y = 8, 5
            box = (x + 8, y + 8, x + 8 + (bbox[2] - bbox[0]) + pad_x * 2,
                   y + 8 + (bbox[3] - bbox[1]) + pad_y * 2)
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text((box[0] + pad_x - bbox[0], box[1] + pad_y - bbox[1]), text,
                      font=FONT, fill=(235, 245, 255))
    out = case_dir / "gallery.jpg"
    canvas.save(out, quality=82, optimize=True)
    return out


def process_case(root: Path, dataset: str, case: dict):
    day = str(case["day"])
    series_full = case["seriesFull"]
    case_dir = SITE / Path(case["gallery"]).parent
    if not case_dir.exists():
        return False

    fixed_seg_path, fixed_ct_path = fixed_paths(root, dataset)
    if not fixed_seg_path.exists():
        return False
    moving_path = moving_seg(root, dataset, day, series_full)
    moving_ct_path = moving_ct(root, dataset, day, series_full)
    if not moving_path.exists() or not moving_ct_path.exists():
        return False
    bdir = batch_dir(root, dataset, day, series_full)
    tfm_path = bdir / "moving_to_fixed.tfm"
    if not tfm_path.exists():
        return False

    fixed_seg = sitk.ReadImage(str(fixed_seg_path))
    fixed_ct = sitk.ReadImage(str(fixed_ct_path))
    moving = sitk.ReadImage(str(moving_path))
    moving_ct_img = sitk.ReadImage(str(moving_ct_path))
    fixed_arr = sitk.GetArrayFromImage(fixed_seg)
    ct_arr = sitk.GetArrayFromImage(fixed_ct)
    moving_arr = sitk.GetArrayFromImage(moving)
    moving_ct_arr = sitk.GetArrayFromImage(moving_ct_img)
    spacing = np.array(fixed_seg.GetSpacing(), dtype=float)
    spacing_m = np.array(moving_ct_img.GetSpacing(), dtype=float)

    fixed_on_moving = sitk.Resample(fixed_seg, moving_ct_img, sitk.Transform(3, sitk.sitkIdentity),
                                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    fixed_on_moving_arr = sitk.GetArrayFromImage(fixed_on_moving)

    centroid_t = base.translation_centroid_output_transform(
        base.stable_union(fixed_seg), base.stable_union(moving))
    centroid = sitk.Resample(moving, fixed_seg, centroid_t,
                             sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    rigid = sitk.Resample(moving, fixed_seg, sitk.ReadTransform(str(tfm_path)),
                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    identity = sitk.Resample(moving, fixed_seg, sitk.Transform(3, sitk.sitkIdentity),
                             sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    centroid_arr = sitk.GetArrayFromImage(centroid)
    rigid_arr = sitk.GetArrayFromImage(rigid)
    identity_arr = sitk.GetArrayFromImage(identity)

    tiles = {}
    made = False
    for organ, label in ORGANS:
        views = case.get("views", {}).get(organ, {})
        axial = views.get("axial") or []
        if not axial:
            continue
        z = int((axial[1] if len(axial) > 1 else axial[0])["label"].split()[1])
        # HAM: raw moving CT at the moving organ's own centre slice
        zs_m = np.where(moving_arr == label)[0]
        z_m = int(round(float(zs_m.mean()))) if len(zs_m) else moving_arr.shape[0] // 2
        z_m = max(0, min(moving_ct_arr.shape[0] - 1, z_m))
        crop_m = union_crop(
            fixed_on_moving_arr.shape, [fixed_on_moving_arr, moving_arr],
            label, spacing_m, MARGIN_MM[organ],
        )
        if crop_m is not None:
            ym0, ym1, xm0, xm1 = crop_m
            base.render(
                moving_ct_arr[z_m, ym0:ym1, xm0:xm1],
                fixed_on_moving_arr[z_m, ym0:ym1, xm0:xm1] == label,
                moving_arr[z_m, ym0:ym1, xm0:xm1] == label,
                case_dir / f"{organ}_gal_ham.jpg", size=560, quality=84)
            tiles[(organ, "ham")] = case_dir / f"{organ}_gal_ham.jpg"
            made = True

        # IDENTITY / CENTROID / RIGID: fixed CT, moving mask in each state
        crop = union_crop(
            fixed_arr.shape, [fixed_arr, identity_arr, centroid_arr, rigid_arr],
            label, spacing, MARGIN_MM[organ],
        )
        if crop is None:
            continue
        y0, y1, x0, x1 = crop
        ct = ct_arr[z, y0:y1, x0:x1]
        fixed_slice = fixed_arr[z, y0:y1, x0:x1] == label
        for state, arr in (("identity", identity_arr), ("centroid", centroid_arr),
                           ("rigid", rigid_arr)):
            path = case_dir / f"{organ}_gal_{state}.jpg"
            base.render(ct, fixed_slice, arr[z, y0:y1, x0:x1] == label, path, size=560, quality=84)
            tiles[(organ, state)] = path
            made = True

    if made:
        make_gallery(case_dir, tiles)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="case id substring (test)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    data = load_dataset(DATA_JS)
    done = 0
    failed = 0
    for dataset, root in (("sep", SEP_ROOT), ("pcct", PCCT_ROOT)):
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
    print(f"[done] ok={done} failed={failed}")


if __name__ == "__main__":
    main()

