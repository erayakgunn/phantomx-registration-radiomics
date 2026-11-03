"""Compare HU repeatability inside the VOI vs the full organ (Zurich).

For each protocol and organ, the coefficient of variation across the 10 takes
is computed for mean HU and SD HU, both for the full organ and for the fixed
centered VOI. A lower VOI CV is the legitimate repeatability gain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import SimpleITK as sitk
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "registration-radiomics"))
import label_map  # noqa: E402

OUT = Path(r"C:/Users/eray/Desktop/ZURICH_VOI_CV.xlsx")
WORKS = [
    ("PCCT", Path(r"C:/Users/eray/Desktop/zurich_pcct_work")),
    ("SEP", Path(r"C:/Users/eray/Desktop/zurich_sep_work")),
]


def organ_label(seg_path: Path, organ: str):
    mapping = label_map.load_label_map(seg_path)
    return mapping.get(organ)


def full_hu(work: Path, group: str, series: str, organ: str):
    seg_path = work / "LABELED" / group / "segmentation" / series / "seg.nii.gz"
    ct_path = work / "LABELED" / group / "nifti" / series / "ct.nii.gz"
    label = organ_label(seg_path, organ)
    if label is None or not ct_path.exists():
        return None, None
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path)))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
    values = ct[seg == int(label)]
    if values.size == 0:
        return None, None
    return float(values.mean()), float(values.std())


def cv(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(values) < 2 or abs(values.mean()) < 1e-9:
        return float("nan")
    return float(values.std(ddof=1) / abs(values.mean()) * 100.0)


def main():
    rows = []
    for dataset, work in WORKS:
        voi_path = work / "VOI_SUMMARY.csv"
        if not voi_path.exists():
            print(f"[skip] {voi_path}")
            continue
        voi = pd.read_csv(voi_path)
        for _, r in voi.iterrows():
            group, organ, series = r["group"], r["organ"], r["series"]
            fm, fs = full_hu(work, group, series, organ)
            rows.append({
                "Dataset": dataset, "Group": group, "Organ": organ, "Series": series,
                "Full mean HU": fm, "VOI mean HU": float(r["mean_hu"]),
                "Full SD HU": fs, "VOI SD HU": float(r["sd_hu"]),
            })
        print(f"[{dataset}] {len(voi)} VOI satiri")

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
    ws["A1"] = "ZURICH: VOI vs TAM ORGAN HU TEKRARLANABİLİRLİĞİ (%CV, 10 take)"
    ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
    ws["A2"] = "Düşük %CV daha iyi. VOI, sabit merkezli küp (liver 30 mm, aorta 10 mm)."
    ws["A2"].font = Font(italic=True, size=9)
    cols = ["Dataset", "Group", "Organ", "N", "Full mean CV %", "VOI mean CV %",
            "Mean CV azalma", "Full SD CV %", "VOI SD CV %", "SD CV azalma"]
    head(ws, 4, cols)
    r = 5
    for (dataset, group, organ), g in pd.DataFrame(rows).groupby(["Dataset", "Group", "Organ"]):
        full_mean = cv(g["Full mean HU"].tolist())
        voi_mean = cv(g["VOI mean HU"].tolist())
        full_sd = cv(g["Full SD HU"].tolist())
        voi_sd = cv(g["VOI SD HU"].tolist())
        vals = [dataset, group, organ, len(g), round(full_mean, 3), round(voi_mean, 3),
                round(full_mean - voi_mean, 3), round(full_sd, 3), round(voi_sd, 3),
                round(full_sd - voi_sd, 3)]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c > 1 else "left")
            if c in (7, 10) and isinstance(v, (int, float)):
                cell.fill = green if v > 0 else red
        r += 1
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 52
    ws.column_dimensions["C"].width = 8
    for c in range(4, 11):
        ws.column_dimensions[get_column_letter(c)].width = 14
    ws.freeze_panes = "D5"

    ws2 = wb.create_sheet("Ciftler")
    ws2["A1"] = "HAM VERİ"
    ws2["A1"].font = Font(bold=True, size=12, color="1F4E78")
    cols2 = ["Dataset", "Group", "Organ", "Series", "Full mean HU", "VOI mean HU", "Full SD HU", "VOI SD HU"]
    head(ws2, 3, cols2)
    for r, d in enumerate(rows, 4):
        for c, k in enumerate(cols2, 1):
            cell = ws2.cell(row=r, column=c, value=d.get(k))
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c >= 5 else "left")
    ws2.column_dimensions["A"].width = 10
    ws2.column_dimensions["B"].width = 52
    ws2.column_dimensions["C"].width = 8
    ws2.column_dimensions["D"].width = 60
    ws2.freeze_panes = "E4"

    ws3 = wb.create_sheet("Not")
    ws3["A1"] = "YORUM"
    ws3["A1"].font = Font(bold=True, size=13, color="1F4E78")
    notes = [
        "1) %CV = take'ler arası SD / ortalama × 100. Düşük olması iyi.",
        "2) VOI, referans take1'de organ merkezinden tanımlanan sabit küp.",
        "3) Amaç: registration test-retest'te fayda sağlamıyorsa, VOI ile",
        "   HU/radiomics tekrarlanabilirliğini göstermek.",
        "4) VOI mean HU CV'si tam organa göre genelde daha düşükse, bu",
        "   'veri bozmadan' elde edilen gerçek kazançtır.",
        "5) VOI Dice ayrıca ~0.99 (tam organ ~0.93); ancak VOI iç bölge olduğu",
        "   için Dice doygunluğu göz önünde tutulmalı.",
    ]
    for i, text in enumerate(notes, 3):
        ws3.cell(row=i, column=1, value=text).font = Font(size=10)
    ws3.column_dimensions["A"].width = 120

    wb.save(OUT)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
