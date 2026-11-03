"""Generate site assets + data entries for the Zurich test-retest sets.

For each registered pair it renders before (identity) / after (registered)
overlays for spleen and aorta, builds a 2x2 gallery tile, and appends a
``zurich`` dataset to the site's data.js.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "registration-radiomics"))
import generate_assets as base  # noqa: E402

SITE = Path(r"C:/Users/eray/Desktop/phantomx-overlay-site")
ASSETS = SITE / "assets" / "zurich"
DATA_JS = SITE / "data.js"
WORKS = {
    "zpcct": Path(r"C:/Users/eray/Desktop/zurich_pcct_work"),
    "zsep": Path(r"C:/Users/eray/Desktop/zurich_sep_work"),
}
MARGIN_AXIAL_MM = {"aorta": 50.0, "spleen": 30.0}
MARGIN_CORONAL_MM = {"aorta": 30.0, "spleen": 20.0}

try:
    FONT = ImageFont.truetype("arialbd.ttf", 16)
except Exception:
    FONT = ImageFont.load_default()


def short_group(name: str, prefix: str) -> str:
    tail = name.split("__")[-1]
    if prefix == "zpcct":
        m = re.search(r"(\d+)kVp_(\d+)mAs_(\d+(?:\.\d+)?)mm_(MonoE\d+keV|PolyE)", tail)
        if m:
            return f"PCCT {m.group(1)}kV {m.group(2)}mAs {m.group(3)}mm {m.group(4)}"
    m = re.search(r"(S2_Br40f|S3_Br40f|Br40f)_(\d+)_(\d+)_\d+_(\w+)", tail)
    if m:
        return f"SEP {m.group(1)} {m.group(2)}kV {m.group(3)}mAs"
    return tail[:42]


def organ_ids(label_map_path: Path) -> dict:
    data = json.loads(label_map_path.read_text(encoding="utf-8"))
    out = {}
    for key, value in data.items():
        if isinstance(value, str):
            if value.lower() == "spleen":
                out["spleen"] = int(key)
            elif value.lower() == "aorta":
                out["aorta"] = int(key)
    return out


def parse_s_no(item: dict) -> int:
    match = re.search(r"_s(\d+)$", item.get("series", ""))
    return int(match.group(1)) if match else 999


def bounds_axial(fixed, before, after, organ, spacing, margin_mm):
    z, y, x = np.where((fixed == organ) | (before == organ) | (after == organ))
    if not len(y):
        raise ValueError(f"empty mask for organ {organ}")
    my = int(round(margin_mm / spacing[1]))
    mx = int(round(margin_mm / spacing[0]))
    return (
        max(0, int(y.min()) - my), min(fixed.shape[1], int(y.max()) + my + 1),
        max(0, int(x.min()) - mx), min(fixed.shape[2], int(x.max()) + mx + 1),
    )


def bounds_coronal(fixed, before, after, organ, spacing, margin_mm):
    z, y, x = np.where((fixed == organ) | (before == organ) | (after == organ))
    if not len(z):
        raise ValueError(f"empty mask for organ {organ}")
    mz = int(round(margin_mm / spacing[2]))
    mx = int(round(margin_mm / spacing[0]))
    z0, z1 = max(0, int(z.min()) - mz), min(fixed.shape[0], int(z.max()) + mz + 1)
    x0, x1 = max(0, int(x.min()) - mx), min(fixed.shape[2], int(x.max()) + mx + 1)
    fz, fy, _ = np.where(fixed == organ)
    y_slice = int(round(float(np.median(fy)))) if len(fy) else int(round(float(np.median(y))))
    return z0, z1, x0, x1, max(0, min(fixed.shape[1] - 1, y_slice))


def make_gallery(case_out: Path, views: dict) -> str:
    tiles = [
        (views["aorta"]["axial"][0]["before"], "AORTA · ÖNCE"),
        (views["aorta"]["axial"][0]["after"], "AORTA · SONRA"),
        (views["spleen"]["axial"][0]["before"], "SPLEEN · ÖNCE"),
        (views["spleen"]["axial"][0]["after"], "SPLEEN · SONRA"),
    ]
    tile = 320
    canvas = Image.new("RGB", (tile * 2, tile * 2), (2, 7, 12))
    draw = ImageDraw.Draw(canvas)
    for index, (rel_path, label) in enumerate(tiles):
        image = Image.open(SITE / rel_path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        x, y = (index % 2) * tile, (index // 2) * tile
        canvas.paste(image, (x, y))
        bbox = draw.textbbox((0, 0), label, font=FONT)
        pad_x, pad_y = 8, 5
        box = (x + 8, y + 8, x + 8 + (bbox[2] - bbox[0]) + pad_x * 2,
               y + 8 + (bbox[3] - bbox[1]) + pad_y * 2)
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((box[0] + pad_x - bbox[0], box[1] + pad_y - bbox[1]), label, font=FONT,
                  fill=(235, 245, 255))
    out = case_out / "gallery.jpg"
    canvas.save(out, quality=86, optimize=True)
    return f"assets/zurich/{case_out.name}/{out.name}"


def method_label(raw: str | None) -> str:
    if not raw:
        return "Unknown"
    if raw.startswith("identity"):
        return "Identity (no registration)"
    if raw.startswith("landmark"):
        return "Landmark rigid"
    if raw.startswith("centroid"):
        return "Centroid translation"
    return raw


def build_dataset(prefix: str, work: Path, start_order: int):
    lab = work / "LABELED"
    reg = work / "REGISTRATION"
    cases = []
    groups = sorted(p for p in lab.iterdir() if p.is_dir())
    for g_idx, group in enumerate(groups, 1):
        seg_dir = group / "segmentation"
        nii_dir = group / "nifti"
        series = sorted(p.name for p in nii_dir.iterdir() if p.is_dir())
        base_series = next((s for s in series if "_s" not in s.split("recon")[-1]), series[0])
        fixed_ct = sitk.ReadImage(str(nii_dir / base_series / "ct.nii.gz"))
        fixed_seg = sitk.ReadImage(str(seg_dir / base_series / "seg.nii.gz"))
        fixed_ct_arr = sitk.GetArrayFromImage(fixed_ct)
        fixed_arr = sitk.GetArrayFromImage(fixed_seg)
        spacing = np.array(fixed_seg.GetSpacing(), dtype=float)
        ids = organ_ids(seg_dir / base_series / "label_map.json")

        manifest_path = reg / group.name / "batch_manifest.json"
        if not manifest_path.exists():
            print(f"  [skip] manifest yok: {group.name}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pairs = sorted(manifest.get("pairs", []), key=parse_s_no)
        short = short_group(group.name, prefix)
        print(f"  {short}: {len(pairs)} pair")

        for k, item in enumerate(pairs, 2):
            moving_name = item["series"]
            pair_dir = reg / group.name / f"{group.name}__{moving_name}"
            reg_seg_path = pair_dir / "seg_registered.nii.gz"
            if not reg_seg_path.exists():
                print(f"    [skip] seg yok: {moving_name}")
                continue
            moving_seg = sitk.ReadImage(str(seg_dir / moving_name / "seg.nii.gz"))
            before_seg = sitk.Resample(moving_seg, fixed_seg, sitk.Transform(3, sitk.sitkIdentity),
                                       sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
            after_seg = sitk.ReadImage(str(reg_seg_path))
            before_arr = sitk.GetArrayFromImage(before_seg)
            after_arr = sitk.GetArrayFromImage(after_seg)

            metrics = {}
            metrics_path = pair_dir / "metrics.csv"
            if metrics_path.exists():
                for row in metrics_path.read_text(encoding="utf-8").splitlines()[1:]:
                    parts = row.split(",")
                    if len(parts) >= 3:
                        metrics[parts[0]] = parts
            qc = {}
            qc_path = pair_dir / "registration_qc.json"
            if qc_path.exists():
                qc = json.loads(qc_path.read_text(encoding="utf-8"))

            case_id = f"{prefix}_{g_idx}_{k}"
            case_out = ASSETS / case_id
            case_out.mkdir(parents=True, exist_ok=True)
            views = {}
            scores = {}
            residual = {}
            for organ, label in (("aorta", ids["aorta"]), ("spleen", ids["spleen"])):
                y0, y1, x0, x1 = bounds_axial(fixed_arr, before_arr, after_arr, label, spacing,
                                              MARGIN_AXIAL_MM[organ])
                z0, z1, cx0, cx1, y_slice = bounds_coronal(
                    fixed_arr, before_arr, after_arr, label, spacing, MARGIN_CORONAL_MM[organ])
                zs = np.where(fixed_arr == label)[0]
                picks = sorted({int(round(zs.min() + (zs.max() - zs.min()) * f))
                                for f in (0.35, 0.65)})
                axial = []
                for idx, z in enumerate(picks):
                    before_path = case_out / f"{organ}_axial_{idx}_before.jpg"
                    after_path = case_out / f"{organ}_axial_{idx}_after.jpg"
                    ct = fixed_ct_arr[z, y0:y1, x0:x1]
                    fixed_slice = fixed_arr[z, y0:y1, x0:x1] == label
                    base.render(ct, fixed_slice, before_arr[z, y0:y1, x0:x1] == label,
                                before_path, size=640, quality=86)
                    base.render(ct, fixed_slice, after_arr[z, y0:y1, x0:x1] == label,
                                after_path, size=640, quality=86)
                    axial.append({
                        "label": f"z {z}",
                        "before": f"assets/zurich/{case_id}/{before_path.name}",
                        "after": f"assets/zurich/{case_id}/{after_path.name}",
                    })
                coronal_before = case_out / f"{organ}_coronal_before.jpg"
                coronal_after = case_out / f"{organ}_coronal_after.jpg"
                ct_c = fixed_ct_arr[z0:z1, y_slice, cx0:cx1]
                fixed_c = fixed_arr[z0:z1, y_slice, cx0:cx1] == label
                base.render(ct_c, fixed_c, before_arr[z0:z1, y_slice, cx0:cx1] == label,
                            coronal_before, size=640, quality=86)
                base.render(ct_c, fixed_c, after_arr[z0:z1, y_slice, cx0:cx1] == label,
                            coronal_after, size=640, quality=86)
                views[organ] = {
                    "axial": axial,
                    "coronal": [{
                        "label": f"y {y_slice}",
                        "before": f"assets/zurich/{case_id}/{coronal_before.name}",
                        "after": f"assets/zurich/{case_id}/{coronal_after.name}",
                    }],
                }
                row = metrics.get(str(label))
                if row and len(row) >= 9:
                    scores[organ] = {"before": round(float(row[1]), 5), "after": round(float(row[2]), 5)}
                    residual[organ] = {
                        "centroid": round(float(row[4]), 4),
                        "hd95": round(float(row[6]), 4),
                        "assd": round(float(row[8]), 4),
                    }
                else:
                    scores[organ] = {
                        "before": round(base.dice(fixed_arr, before_arr, label), 5),
                        "after": round(base.dice(fixed_arr, after_arr, label), 5),
                    }

            gain = ((scores["aorta"]["after"] - scores["aorta"]["before"])
                    + (scores["spleen"]["after"] - scores["spleen"]["before"])) / 2
            tone = "mint" if gain > 0.01 else "amber" if gain < -0.01 else "sky"
            note = (f"Aorta {scores['aorta']['before']:.3f}→{scores['aorta']['after']:.3f}; "
                    f"Spleen {scores['spleen']['before']:.3f}→{scores['spleen']['after']:.3f}.")
            cases.append({
                "id": case_id,
                "day": "20260929",
                "take": k,
                "series": f"{short} t{k}",
                "seriesFull": moving_name,
                "title": f"{short} t{k}",
                "fixed": "take1",
                "tone": tone,
                "tag": "Test-retest",
                "note": note,
                "featured": False,
                "qc": False,
                "qcReason": None,
                "inWorkbook": True,
                "order": start_order + g_idx * 100 + k,
                "summary": f"{short} t{k} → take1",
                "method": method_label(qc.get("registration_method")),
                "scores": scores,
                "views": views,
                "residual": residual,
                "registration": {
                    "method": qc.get("registration_method"),
                    "attempted": qc.get("registration_attempted"),
                    "accepted": qc.get("registration_accepted"),
                },
                "gallery": make_gallery(case_out, views),
            })
    return cases


def update_data_js(dataset: dict):
    text = DATA_JS.read_text(encoding="utf-8")
    start = text.index("{")
    end = text.rindex("}") + 1
    obj = json.loads(text[start:end])
    obj["zurich"] = dataset
    new = "window.DATASETS = " + json.dumps(obj, ensure_ascii=False, indent=2) + ";\n"
    new += 'const DATASET_KEY = new URLSearchParams(location.search).get("dataset") || "sep";\n'
    new += "window.CASES = window.DATASETS[DATASET_KEY] || window.DATASETS.sep;\n"
    DATA_JS.write_text(new, encoding="utf-8")
    print(f"[data.js] zurich eklendi: {len(dataset['cases'])} case")


def main():
    all_cases = []
    for prefix, work in WORKS.items():
        if not work.exists():
            print(f"[skip] {work}")
            continue
        print(f"=== {prefix}: {work.name}")
        all_cases.extend(build_dataset(prefix, work, start_order=0 if prefix == "zpcct" else 50000))
    dataset = {
        "fixed": "take1",
        "beforeLabel": "Identity (no registration)",
        "afterLabel": "Registered (rigid/landmark)",
        "label": "Zürich",
        "cases": all_cases,
    }
    update_data_js(dataset)


if __name__ == "__main__":
    main()
