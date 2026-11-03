"""Consensus-reference Dice for the longitudinal PCCT and SEP cohorts.

All registered masks live on the 20260805 take1 fixed grid. For every moving
series, the reference becomes the leave-one-out majority consensus of the
cohort instead of the single take1 mask. Per-series masks and images are not
changed.
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

OUT = Path(r"C:/Users/eray/Desktop/PCCT_SEP_LONG_CONSENSUS.xlsx")
PCCT = Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX")
SEP = Path(r"C:/Users/eray/Desktop/SEP - ['Br40f',_'3']_120_100_300_1/PhantomX")
DATASETS = [
    ("PCCT", PCCT,
     PCCT / "REGISTRATION" / "batch_to_20260805_v1.0.0",
     PCCT / "LABELED" / "20260805_PCCT_Aorta_Spleen" / "segmentation" /
     "USZ_Siemens_Healthineers_NAEOTOM_Alpha_20260805_Q3_Qr40f_120_100_300_1_PolyE_take1" / "seg.nii.gz"),
    ("SEP", SEP,
     SEP / "REGISTRATION" / "batch_to_20260805_v6",
     SEP / "LABELED" / "20260805_SEP_Aorta_Spleen" / "segmentation" /
     "USZ_SIEMENS_SOMATOM_Edge_Plus_20260805_S3_Br40f_120_100_300_1_PolyE_take1_recon1mm" / "seg.nii.gz"),
]
ORGANS = {"spleen": 1, "aorta": 52}


def dice(a, b):
    total = int(a.sum()) + int(b.sum())
    return float(2 * np.logical_and(a, b).sum() / total) if total else float("nan")


def bbox(mask, pad=20):
    z, y, x = np.where(mask)
    if not len(z):
        return None
    z0, z1 = max(0, z.min() - pad), min(mask.shape[0], z.max() + pad + 1)
    y0, y1 = max(0, y.min() - pad), min(mask.shape[1], y.max() + pad + 1)
    x0, x1 = max(0, x.min() - pad), min(mask.shape[2], x.max() + pad + 1)
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def process(dataset: str, base: Path, fixed_path: Path):
    fixed = sitk.ReadImage(str(fixed_path))
    fixed_arr = sitk.GetArrayFromImage(fixed)
    pair_dirs = sorted(
        p for p in base.iterdir()
        if p.is_dir() and (p / "seg_registered.nii.gz").exists()
    )
    print(f"[{dataset}] {len(pair_dirs)} pair")

    rows = []
    for organ, label in ORGANS.items():
        crop = bbox(fixed_arr == label, pad=20)
        if crop is None:
            continue
        baseline = (fixed_arr == label)[crop]
        masks = []
        meta = []
        for pair_dir in pair_dirs:
            session, series = pair_dir.name.split("__", 1)
            arr = sitk.GetArrayFromImage(
                sitk.ReadImage(str(pair_dir / "seg_registered.nii.gz"))
            )[crop] == label
            masks.append(arr)
            meta.append((session, series))
        if len(masks) < 5:
            continue
        stack = np.stack([baseline] + masks)  # baseline + moving
        total = stack.sum(axis=0, dtype=np.uint16)
        n = stack.shape[0]
        for i, (session, series) in enumerate(meta, start=1):
            cons = (total - stack[i]) / (n - 1) >= 0.5
            rows.append({
                "Dataset": dataset, "Session": session, "Series": series,
                "Organ": organ,
                "Dice take1": round(dice(stack[i], baseline), 5),
                "Dice consensus": round(dice(stack[i], cons), 5),
            })
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

    import pandas as pd
    frame = pd.DataFrame(rows)
    ws = wb.active
    ws.title = "Tum"
    ws["A1"] = "LONGITUDINAL PCCT/SEP: TAKE1 vs KOHORT KONSENSÜSÜ (leave-one-out)"
    ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
    ws["A2"] = "Per-series maskeler ve görüntüler değişmedi; yalnızca referans daha temsili seçildi."
    ws["A2"].font = Font(italic=True, size=9)
    cols = ["Dataset", "Organ", "N", "Take1 (medyan)", "Konsensüs (medyan)", "Delta", "İyileşen", "Kötüleşen"]
    head(ws, 4, cols)
    r = 5
    for (dataset, organ), g in frame.groupby(["Dataset", "Organ"]):
        t1 = g["Dice take1"].median()
        co = g["Dice consensus"].median()
        vals = [dataset, organ, len(g), round(t1, 4), round(co, 4), round(co - t1, 4),
                int((g["Dice consensus"] > g["Dice take1"]).sum()),
                int((g["Dice consensus"] < g["Dice take1"]).sum())]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c > 1 else "left")
            if c == 6:
                cell.fill = green if v >= 0 else red
        r += 1
    for c in range(1, 9):
        ws.column_dimensions[get_column_letter(c)].width = 16
    ws.column_dimensions["B"].width = 10
    ws.freeze_panes = "C5"

    ws2 = wb.create_sheet("Ozet_Session")
    ws2["A1"] = "OTURUM BAZINDA"
    ws2["A1"].font = Font(bold=True, size=12, color="1F4E78")
    head(ws2, 3, ["Dataset", "Session", "Organ", "N", "Take1", "Konsensüs", "Delta"])
    r = 4
    for (dataset, session, organ), g in frame.groupby(["Dataset", "Session", "Organ"]):
        t1 = g["Dice take1"].median()
        co = g["Dice consensus"].median()
        for c, v in enumerate([dataset, session, organ, len(g), round(t1, 4), round(co, 4), round(co - t1, 4)], 1):
            cell = ws2.cell(row=r, column=c, value=v)
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c > 3 else "left")
            if c == 7:
                cell.fill = green if v >= 0 else red
        r += 1
    ws2.column_dimensions["A"].width = 10
    ws2.column_dimensions["B"].width = 46
    for c in range(3, 8):
        ws2.column_dimensions[get_column_letter(c)].width = 12
    ws2.freeze_panes = "D4"

    ws3 = wb.create_sheet("Ciftler")
    ws3["A1"] = "HAM VERİ"
    ws3["A1"].font = Font(bold=True, size=12, color="1F4E78")
    cols3 = ["Dataset", "Session", "Series", "Organ", "Dice take1", "Dice consensus"]
    head(ws3, 3, cols3)
    for r, d in enumerate(rows, 4):
        for c, k in enumerate(cols3, 1):
            cell = ws3.cell(row=r, column=c, value=d.get(k))
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c >= 5 else "left")
    ws3.column_dimensions["A"].width = 10
    ws3.column_dimensions["B"].width = 46
    ws3.column_dimensions["C"].width = 60
    ws3.column_dimensions["D"].width = 8
    ws3.freeze_panes = "E4"

    ws4 = wb.create_sheet("Not")
    ws4["A1"] = "YORUM"
    ws4["A1"].font = Font(bold=True, size=13, color="1F4E78")
    notes = [
        "1) Tüm maskeler 20260805 take1 fixed grid'inde.",
        "2) Konsensüs, o seri dışlanarak kohortun çoğunluk oyu (leave-one-out).",
        "3) Per-series maskeler ve görüntüler değişmedi; referans daha temsili.",
        "4) Take1 tek bir tarih; konsensüs kohortun medyan şekli.",
        "5) Delta > 0 ise konsensüs referansı Dice'ı iyileştiriyor.",
    ]
    for i, text in enumerate(notes, 3):
        ws4.cell(row=i, column=1, value=text).font = Font(size=10)
    ws4.column_dimensions["A"].width = 120

    wb.save(OUT)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    rows = []
    for dataset, _, base, fixed_path in DATASETS:
        rows.extend(process(dataset, base, fixed_path))
    write_xlsx(rows)
