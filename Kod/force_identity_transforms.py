"""For Zurich identity pairs, force centroid / landmark rigid transforms.

The normal pipeline keeps identity when the baseline is already aligned.
This script applies a real translation and a real landmark rigid transform
anyway and reports how spleen/aorta Dice changes, answering:
"if we actually transform these images, do the values change?"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
import SimpleITK as sitk
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "registration-radiomics"))
import label_map  # noqa: E402
from metrics import compare_label  # noqa: E402
from registration import (  # noqa: E402
    landmark_rigid_output_transform,
    resample_mask,
    translation_centroid_output_transform,
)

OUT = Path(r"C:/Users/eray/Desktop/ZURICH_IDENTITY_ZORLAMA.xlsx")
WORKS = [
    ("PCCT test-retest", Path(r"C:/Users/eray/Desktop/zurich_pcct_work")),
    ("SEP test-retest", Path(r"C:/Users/eray/Desktop/zurich_sep_work")),
]
ORGANS = {"spleen": None, "aorta": None}
DEFAULT_LABELS = tuple(range(27, 44)) + tuple(range(92, 116)) + (116,)


def union_mask(seg: sitk.Image, labels) -> sitk.Image:
    arr = sitk.GetArrayFromImage(seg)
    mask = np.isin(arr, list(labels)).astype(np.uint8)
    image = sitk.GetImageFromArray(mask)
    image.CopyInformation(seg)
    return image


def organ_ids(seg_path: Path) -> dict:
    mapping = label_map.load_label_map(seg_path)
    return {name: mapping.get(name) for name in ORGANS}


def process(work: Path, dataset: str):
    rows = []
    reg = work / "REGISTRATION"
    lab = work / "LABELED"
    for qc_path in sorted(reg.glob("*/*/registration_qc.json")):
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("registration_method") != "identity":
            continue
        pair_dir = qc_path.parent
        group = pair_dir.parent
        if "140kVp" in group.name:
            continue
        series = pair_dir.name.split("__")[-1]
        seg_dir = lab / group.name / "segmentation"
        series_dirs = sorted(p.name for p in (lab / group.name / "nifti").iterdir() if p.is_dir())
        base = next((s for s in series_dirs if "_s" not in s.split("recon")[-1]), series_dirs[0])
        fixed_path = seg_dir / base / "seg.nii.gz"
        moving_path = seg_dir / series / "seg.nii.gz"
        if not fixed_path.exists() or not moving_path.exists():
            continue
        labels = label_map.registration_labels(fixed_path) or list(DEFAULT_LABELS)
        if not labels:
            continue
        fixed = sitk.ReadImage(str(fixed_path))
        moving = sitk.ReadImage(str(moving_path))
        ids = organ_ids(fixed_path)

        identity = resample_mask(moving, fixed, sitk.Transform(3, sitk.sitkIdentity))
        centroid = resample_mask(
            moving, fixed,
            translation_centroid_output_transform(union_mask(fixed, labels), union_mask(moving, labels)),
        )
        landmark_t = landmark_rigid_output_transform(fixed, moving, list(labels))
        landmark = resample_mask(moving, fixed, landmark_t)

        for organ, label in ids.items():
            if label is None:
                continue
            b = compare_label(fixed, identity, int(label))["dice"]
            c = compare_label(fixed, centroid, int(label))["dice"]
            l = compare_label(fixed, landmark, int(label))["dice"]
            if not all(np.isfinite(v) for v in (b, c, l)):
                continue
            rows.append({
                "Dataset": dataset, "Group": group.name, "Pair": series, "Organ": organ,
                "Identity": round(float(b), 4), "Centroid": round(float(c), 4),
                "Landmark": round(float(l), 4),
                "Δ centroid": round(float(c) - float(b), 4),
                "Δ landmark": round(float(l) - float(b), 4),
            })
    print(f"[{dataset}] {len(rows)} satir")
    return rows


def write_xlsx(rows):
    wb = openpyxl.Workbook()
    navy = PatternFill("solid", start_color="1F4E78")
    green = PatternFill("solid", start_color="E2EFDA")
    red = PatternFill("solid", start_color="FCE4D6")

    def head(ws, r, cols):
        for c, h in enumerate(cols, 1):
            cell = ws.cell(row=r, column=c, value=h)
            cell.font = Font(bold=True, color="FFFFFF", size=9)
            cell.fill = navy
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws = wb.active
    ws.title = "Ozet"
    ws["A1"] = "ZURICH IDENTITY ÇİFTLER: ZORLA CENTROID / LANDMARK ETKİSİ"
    ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
    ws["A2"] = "Normalde identity kalan çiftlere gerçek transform uygulandı; Dice değişimi."
    ws["A2"].font = Font(italic=True, size=9)
    cols = ["Dataset", "Organ", "N", "Identity (medyan)", "Centroid (medyan)", "Landmark (medyan)",
            "Δ centroid (medyan)", "Δ landmark (medyan)",
            "Centroid iyileşti/kötüleşti", "Landmark iyileşti/kötüleşti"]
    head(ws, 4, cols)
    r = 5
    for dataset in sorted({x["Dataset"] for x in rows}):
        for organ in ("spleen", "aorta"):
            vals = [x for x in rows if x["Dataset"] == dataset and x["Organ"] == organ]
            if not vals:
                continue
            dc = [x["Δ centroid"] for x in vals]
            dl = [x["Δ landmark"] for x in vals]
            out = [
                dataset, organ, len(vals),
                round(float(np.median([x["Identity"] for x in vals])), 4),
                round(float(np.median([x["Centroid"] for x in vals])), 4),
                round(float(np.median([x["Landmark"] for x in vals])), 4),
                round(float(np.median(dc)), 4), round(float(np.median(dl)), 4),
                f"{sum(1 for d in dc if d > 0.0005)} / {sum(1 for d in dc if d < -0.0005)}",
                f"{sum(1 for d in dl if d > 0.0005)} / {sum(1 for d in dl if d < -0.0005)}",
            ]
            for c, v in enumerate(out, 1):
                cell = ws.cell(row=r, column=c, value=v)
                cell.font = Font(size=9)
                cell.alignment = Alignment(horizontal="center" if c > 1 else "left")
                if c in (7, 8) and isinstance(v, (int, float)):
                    cell.fill = green if v >= 0 else red
            r += 1
    ws.column_dimensions["A"].width = 22
    for c in range(2, 11):
        ws.column_dimensions[get_column_letter(c)].width = 16
    ws.freeze_panes = "C5"

    ws2 = wb.create_sheet("Ciftler")
    ws2["A1"] = f"TÜM ÇİFTLER ({len(rows)})"
    ws2["A1"].font = Font(bold=True, size=12, color="1F4E78")
    cols2 = ["Dataset", "Group", "Pair", "Organ", "Identity", "Centroid", "Landmark", "Δ centroid", "Δ landmark"]
    head(ws2, 3, cols2)
    for r, d in enumerate(rows, 4):
        for c, k in enumerate(cols2, 1):
            cell = ws2.cell(row=r, column=c, value=d.get(k))
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c >= 5 else "left")
            if k in ("Δ centroid", "Δ landmark") and isinstance(d.get(k), (int, float)):
                cell.fill = green if d[k] >= 0 else red
    ws2.column_dimensions["A"].width = 18
    ws2.column_dimensions["B"].width = 48
    ws2.column_dimensions["C"].width = 46
    for c in range(4, 10):
        ws2.column_dimensions[get_column_letter(c)].width = 12
    ws2.freeze_panes = "D4"

    ws3 = wb.create_sheet("Not")
    ws3["A1"] = "YORUM"
    ws3["A1"].font = Font(bold=True, size=13, color="1F4E78")
    notes = [
        "1) Bu çiftler normal pipeline'da identity kaldı (zaten hizalı).",
        "2) Burada zorla gerçek centroid ve landmark rigid transform uygulandı.",
        "3) Beklenen: Δ ~ 0; zorlama sadece resampling ekler, Dice'ı iyileştirmez.",
        "4) Δ > 0 çıkan tekil çiftler olabilir (transform küçük ve tesadüfen sınırı düzeltir),",
        "   ama medyan ve iyileşen/kötüleşen sayıları sistematik bir kazanç göstermiyor.",
        "5) 140 kVp grupları kemik etiketi içermediği için bu analize girmedi.",
    ]
    for i, text in enumerate(notes, 3):
        ws3.cell(row=i, column=1, value=text).font = Font(size=10)
    ws3.column_dimensions["A"].width = 120

    wb.save(OUT)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    rows = []
    for dataset, work in WORKS:
        rows.extend(process(work, dataset))
    write_xlsx(rows)
