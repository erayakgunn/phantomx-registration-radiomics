"""Generate overlays for every registered take.

Every moving series in batch_to_20260805 gets:
  - 3 representative axial slices (before centroid / after rigid)
  - 1 coronal slice
Featured takes additionally get the complete axial stack.

The "before" state is a centroid-translation baseline (stable bone mask).
The "after" state is the production rigid transform used for the workbook.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont

import generate_assets as base

ROOT = base.ROOT
LAB = base.LAB
BATCH = base.BATCH
OUT = base.OUT
ASSETS = base.ASSETS
FIXED_DAY = base.FIXED_DAY
FIXED_BASE = base.FIXED_BASE
SESSION_SUFFIX = "_SEP_Aorta_Spleen"
FIXED_LABEL = base.FIXED_LABEL
ORGANS = base.ORGANS
ORGAN_LABELS = base.ORGAN_LABELS

MARGIN_AXIAL = {52: 95, 1: 55, 2: 55, 3: 55}
MARGIN_CORONAL = {52: 55, 1: 38, 2: 40, 3: 40}

FEATURED = {
    "20260730_t1": {
        "id": "high",
        "title": "Çok iyi iyileşen",
        "tone": "mint",
        "tag": "En yüksek kazanç",
        "note": "Aorta +0.072, Spleen +0.149. Belirgin rigid düzeltmesi; sonuç iki organda da yüksek.",
    },
    "20260128_t1": {
        "id": "medium",
        "title": "İyi iyileşen",
        "tone": "sky",
        "tag": "Dengeli kazanç",
        "note": "Rigid, 0128’de Aorta’yı toparlıyor; Spleen’deki kazanç daha büyük.",
    },
    "20260327_t1": {
        "id": "good",
        "title": "Yüksek final",
        "tone": "violet",
        "tag": "Final 0.932",
        "note": "Final Dice yüksek; önce/sonra farkı orta düzeyde ve rigid katkısı net.",
    },
    "20260731_t10": {
        "id": "low",
        "title": "Az iyileşen / düşük",
        "tone": "amber",
        "tag": "Sınırlı kazanç",
        "note": "Kazanç sınırlı; kalan fark boundary/shape. 0731 cohort’u düşük profil.",
    },
}

QC_FLAGS = json.loads((BATCH / "qc_flags.json").read_text(encoding="utf-8"))
EXCLUDED_FROM_WORKBOOK = {"20260622_t3", "20260731_t20"}


def read_annotations(batch: Path) -> dict:
    metrics_path = batch / "metrics.csv"
    if not metrics_path.exists():
        return {}
    rows = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["label"]] = row
    residual = {}
    for organ, label in {"aorta": "52", "spleen": "1"}.items():
        row = rows.get(label)
        if not row:
            continue
        residual[organ] = {
            "hd95": round(float(row["after_hd95_mm"]), 4),
            "assd": round(float(row["after_assd_mm"]), 4),
            "centroid": round(float(row["after_centroid_displacement_mm"]), 4),
        }
    qc_path = batch / "registration_qc.json"
    registration = {}
    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        registration = {
            "method": qc.get("registration_method"),
            "attempted": qc.get("registration_attempted"),
            "accepted": qc.get("registration_accepted"),
        }
    return {"residual": residual, "registration": registration}


def discover_cases() -> list[dict]:
    by_day: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    pattern = re.compile(rf"^\d{{8}}{re.escape(SESSION_SUFFIX)}")
    for path in BATCH.iterdir():
        if not path.is_dir() or not pattern.match(path.name):
            continue
        day = path.name.split(SESSION_SUFFIX)[0]
        match = re.search(r"_s(\d+)$", path.name)
        sort_key = int(match.group(1)) if match else 0
        by_day[day].append((sort_key, path))

    cases = []
    for day, entries in sorted(by_day.items()):
        offset = 1 if day == FIXED_DAY else 0
        for position, (_, batch_path) in enumerate(sorted(entries, key=lambda item: item[0])):
            take_number = position + 1 + offset
            base_name = batch_path.name.replace(f"{day}{SESSION_SUFFIX}__", "")
            series = f"{day}_t{take_number}"
            info = FEATURED.get(series, {})
            qc_reason = QC_FLAGS.get(batch_path.name)
            cases.append({
                "day": day,
                "take": take_number,
                "series": series,
                "series_full": base_name,
                "batch": batch_path,
                "order": int(day) * 100 + take_number,
                "featured": bool(info),
                "id": info.get("id", series),
                "title": info.get("title", series),
                "tone": info.get("tone", "sky"),
                "tag": info.get("tag", "Take"),
                "note": info.get("note", ""),
                "qc": bool(qc_reason),
                "qc_reason": qc_reason,
                "in_workbook": series not in EXCLUDED_FROM_WORKBOOK,
            })
    return sorted(cases, key=lambda item: item["order"])


def fixed_crop(fixed: np.ndarray, organ: int, margin: int) -> tuple[int, int, int, int]:
    _, y, x = np.where(fixed == organ)
    if not len(y):
        raise ValueError(f"empty fixed mask for organ {organ}")
    y0, y1 = max(0, int(y.min()) - margin), min(fixed.shape[1], int(y.max()) + margin + 1)
    x0, x1 = max(0, int(x.min()) - margin), min(fixed.shape[2], int(x.max()) + margin + 1)
    return y0, y1, x0, x1


def fixed_coronal(fixed: np.ndarray, organ: int, margin: int) -> tuple[int, int, int, int, int]:
    z, y, x = np.where(fixed == organ)
    if not len(z):
        raise ValueError(f"empty fixed mask for organ {organ}")
    z0, z1 = max(0, int(z.min()) - margin), min(fixed.shape[0], int(z.max()) + margin + 1)
    x0, x1 = max(0, int(x.min()) - margin), min(fixed.shape[2], int(x.max()) + margin + 1)
    y_slice = int(round(float(np.mean(y))))
    return z0, z1, x0, x1, y_slice


def gallery_font() -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, 16)
    return ImageFont.load_default()


GALLERY_FONT = gallery_font()


def make_gallery(case_out: Path, views: dict) -> str:
    display = {
        "aorta": "AORTA",
        "spleen": "SPLEEN",
        "kidney_right": "KIDNEY R",
        "kidney_left": "KIDNEY L",
    }
    tiles = []
    for organ, organ_views in views.items():
        axial = organ_views.get("axial") or []
        if not axial:
            continue
        view = axial[1] if len(axial) > 1 else axial[0]
        label = display.get(organ, organ.upper())
        tiles.append((view["before"], f"{label} · ÖNCE"))
        tiles.append((view["after"], f"{label} · SONRA"))
    if not tiles:
        return ""
    tile_size = 320
    rows = (len(tiles) + 1) // 2
    canvas = Image.new("RGB", (tile_size * 2, tile_size * rows), (2, 7, 12))
    draw = ImageDraw.Draw(canvas)
    for index, (asset_path, label) in enumerate(tiles):
        image = Image.open(OUT / asset_path).convert("RGB").resize(
            (tile_size, tile_size), Image.Resampling.LANCZOS
        )
        x = (index % 2) * tile_size
        y = (index // 2) * tile_size
        canvas.paste(image, (x, y))
        bbox = draw.textbbox((0, 0), label, font=GALLERY_FONT)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        pad_x, pad_y = 8, 5
        box = (
            x + 8,
            y + 8,
            x + 8 + text_width + pad_x * 2,
            y + 8 + text_height + pad_y * 2,
        )
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text(
            (box[0] + pad_x - bbox[0], box[1] + pad_y - bbox[1]),
            label,
            font=GALLERY_FONT,
            fill=(235, 245, 255),
        )
    out_path = case_out / "gallery.jpg"
    canvas.save(out_path, quality=86, optimize=True)
    return f"assets/{case_out.name}/{out_path.name}"


def write_case(
    case: dict,
    fixed_ct_array: np.ndarray,
    fixed_array: np.ndarray,
    fixed_seg: sitk.Image,
    crops: dict,
    coronal_crops: dict,
    z_indices: dict,
    picks: dict,
) -> dict:
    moving_seg = sitk.ReadImage(
        str(LAB / f"{case['day']}{SESSION_SUFFIX}" / "segmentation" / case["series_full"] / "seg.nii.gz")
    )
    before_transform = base.translation_centroid_output_transform(
        base.stable_union(fixed_seg), base.stable_union(moving_seg)
    )
    after_transform = sitk.ReadTransform(str(case["batch"] / "moving_to_fixed.tfm"))

    before_seg = sitk.Resample(
        moving_seg, fixed_seg, before_transform,
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
    )
    after_seg = sitk.Resample(
        moving_seg, fixed_seg, after_transform,
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
    )
    before_array = sitk.GetArrayFromImage(before_seg)
    after_array = sitk.GetArrayFromImage(after_seg)

    case_out = ASSETS / case["id"]
    case_out.mkdir(parents=True, exist_ok=True)

    payload = {
        "id": case["id"],
        "day": case["day"],
        "take": case["take"],
        "series": case["series"],
        "seriesFull": case["series_full"],
        "title": case["title"],
        "fixed": FIXED_LABEL,
        "tone": case["tone"],
        "tag": case["tag"],
        "note": case["note"],
        "featured": case["featured"],
        "qc": case["qc"],
        "qcReason": case["qc_reason"],
        "inWorkbook": case["in_workbook"],
        "order": case["order"],
        "summary": f"{case['series']} → {FIXED_LABEL}",
        "method": "Centroid translation → Rigid (bone/landmark)",
        "scores": {},
        "views": {},
        **read_annotations(case["batch"]),
    }

    for organ, organ_name in ORGANS.items():
        y0, y1, x0, x1 = crops[organ]
        z0, z1, cx0, cx1, y_slice = coronal_crops[organ]
        axial = []
        for index, z in enumerate(picks[organ]):
            before_path = case_out / f"{organ_name}_axial_{index}_before.jpg"
            after_path = case_out / f"{organ_name}_axial_{index}_after.jpg"
            ct = fixed_ct_array[z, y0:y1, x0:x1]
            fixed_slice = fixed_array[z, y0:y1, x0:x1] == organ
            base.render(ct, fixed_slice, before_array[z, y0:y1, x0:x1] == organ,
                        before_path, size=640, quality=86)
            base.render(ct, fixed_slice, after_array[z, y0:y1, x0:x1] == organ,
                        after_path, size=640, quality=86)
            axial.append({
                "label": f"z {z}",
                "before": f"assets/{case['id']}/{before_path.name}",
                "after": f"assets/{case['id']}/{after_path.name}",
            })

        coronal_before = case_out / f"{organ_name}_coronal_before.jpg"
        coronal_after = case_out / f"{organ_name}_coronal_after.jpg"
        ct_c = fixed_ct_array[z0:z1, y_slice, cx0:cx1]
        base.render(ct_c, fixed_array[z0:z1, y_slice, cx0:cx1] == organ,
                    before_array[z0:z1, y_slice, cx0:cx1] == organ,
                    coronal_before, size=640, quality=86)
        base.render(ct_c, fixed_array[z0:z1, y_slice, cx0:cx1] == organ,
                    after_array[z0:z1, y_slice, cx0:cx1] == organ,
                    coronal_after, size=640, quality=86)

        organ_views = {
            "axial": axial,
            "coronal": [{
                "label": f"y {y_slice}",
                "before": f"assets/{case['id']}/{coronal_before.name}",
                "after": f"assets/{case['id']}/{coronal_after.name}",
            }],
        }

        if case["featured"]:
            all_slices = []
            for z in z_indices[organ]:
                z = int(z)
                all_before = case_out / f"{organ_name}_all_{z}_before.jpg"
                all_after = case_out / f"{organ_name}_all_{z}_after.jpg"
                ct = fixed_ct_array[z, y0:y1, x0:x1]
                fixed_slice = fixed_array[z, y0:y1, x0:x1] == organ
                base.render(ct, fixed_slice, before_array[z, y0:y1, x0:x1] == organ,
                            all_before, size=560, quality=84)
                base.render(ct, fixed_slice, after_array[z, y0:y1, x0:x1] == organ,
                            all_after, size=560, quality=84)
                all_slices.append({
                    "label": f"z {z}",
                    "before": f"assets/{case['id']}/{all_before.name}",
                    "after": f"assets/{case['id']}/{all_after.name}",
                })
            organ_views["all"] = all_slices

        payload["views"][organ_name] = organ_views
        payload["scores"][organ_name] = {
            "before": round(base.dice(fixed_array, before_array, organ), 5),
            "after": round(base.dice(fixed_array, after_array, organ), 5),
        }

    payload["gallery"] = make_gallery(case_out, payload["views"])

    if not case["featured"]:
        gain = sum(
            s["after"] - s["before"] for s in payload["scores"].values()
        ) / max(1, len(payload["scores"]))
        if case["qc"]:
            payload["tone"] = "amber"
            payload["tag"] = "QC flag"
            payload["note"] = f"QC: {case['qc_reason']}. Kayıt yine de gösteriliyor."
        elif not case["in_workbook"]:
            payload["tone"] = "amber"
            payload["tag"] = "Workbook dışı"
            payload["note"] = "Workbook dışı bırakılan take; kayıt yine de gösteriliyor."
        else:
            payload["tone"] = "mint" if gain > 0.08 else "sky" if gain > 0.03 else "violet"
            payload["tag"] = "Take"
            display = {
                "aorta": "Aorta",
                "spleen": "Dalak",
                "kidney_right": "Böbrek sağ",
                "kidney_left": "Böbrek sol",
            }
            payload["note"] = "; ".join(
                f"{display.get(name, name)} {s['before']:.3f}→{s['after']:.3f}"
                for name, s in payload["scores"].items()
            ) + "."

    print(
        f"{case['series']:16s} " + "  ".join(
            f"{name} {s['before']:.3f}->{s['after']:.3f}"
            for name, s in payload["scores"].items()
        )
    )
    return payload


def main() -> None:
    fixed_ct = sitk.ReadImage(
        str(LAB / f"{FIXED_DAY}{SESSION_SUFFIX}" / "nifti" / FIXED_BASE / "ct.nii.gz")
    )
    fixed_seg = sitk.ReadImage(
        str(LAB / f"{FIXED_DAY}{SESSION_SUFFIX}" / "segmentation" / FIXED_BASE / "seg.nii.gz")
    )
    fixed_ct_array = sitk.GetArrayFromImage(fixed_ct)
    fixed_array = sitk.GetArrayFromImage(fixed_seg)

    crops = {
        organ: fixed_crop(fixed_array, organ, MARGIN_AXIAL[organ])
        for organ in ORGANS
    }
    coronal_crops = {
        organ: fixed_coronal(fixed_array, organ, MARGIN_CORONAL[organ])
        for organ in ORGANS
    }
    z_indices = {
        organ: np.where((fixed_array == organ).any(axis=(1, 2)))[0]
        for organ in ORGANS
    }
    picks = {}
    for organ, indices in z_indices.items():
        values = sorted(set(int(round(v)) for v in np.percentile(indices, [20, 50, 80])))
        while len(values) < 3:
            values.append(int(indices[len(values)]))
        picks[organ] = sorted(set(values))[:3]

    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True, exist_ok=True)

    cases = discover_cases()
    payloads = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}]", end=" ")
        payloads.append(write_case(
            case, fixed_ct_array, fixed_array, fixed_seg,
            crops, coronal_crops, z_indices, picks,
        ))

    data = {
        "fixed": FIXED_LABEL,
        "beforeLabel": "Centroid translation",
        "afterLabel": "Rigid (bone/landmark)",
        "cases": payloads,
    }
    (OUT / "data.js").write_text(
        "window.CASES = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(payloads)} cases and {OUT / 'data.js'}")


def refresh_galleries() -> None:
    raw = (OUT / "data.js").read_text(encoding="utf-8")
    payload = json.loads(raw.split("=", 1)[1].strip().rstrip(";"))
    for case in payload["cases"]:
        make_gallery(ASSETS / case["id"], case["views"])
    print(f"Refreshed {len(payload['cases'])} gallery composites")


def annotate_existing_data() -> None:
    raw = (OUT / "data.js").read_text(encoding="utf-8")
    payload = json.loads(raw.split("=", 1)[1].strip().rstrip(";"))
    for case in payload["cases"]:
        batch = BATCH / f"{case['day']}{SESSION_SUFFIX}__{case['seriesFull']}"
        case.update(read_annotations(batch))
    (OUT / "data.js").write_text(
        "window.CASES = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"Annotated {len(payload['cases'])} cases with residual metrics")


if __name__ == "__main__":
    if "--composites-only" in sys.argv:
        refresh_galleries()
    elif "--annotate-only" in sys.argv:
        annotate_existing_data()
    else:
        main()
