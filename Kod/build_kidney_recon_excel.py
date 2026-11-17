"""Compare kidney metrics in 1 mm vs 3 mm Zurich PCCT test-retest series."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DESKTOP = Path(r"C:/Users/eray/Desktop")
METRICS = DESKTOP / "KIDNEY_METRICS_PCCT_SEP.csv"
SCAN = DESKTOP / "KIDNEY_CC_SCAN.csv"
OUT = DESKTOP / "BOBREK_1MM_VS_3MM.xlsx"
NAVY = PatternFill("solid", start_color="1F4E78")
GREEN = PatternFill("solid", start_color="E2EFDA")
RED = PatternFill("solid", start_color="FCE4D6")


def recon(group: str) -> str:
    text = str(group)
    if re.search(r"_3_|3mm", text):
        return "3 mm"
    if re.search(r"_1_|1mm", text):
        return "1 mm"
    return "other"


def head(ws, row, columns):
    for index, name in enumerate(columns, 1):
        cell = ws.cell(row=row, column=index, value=name)
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.fill = NAVY
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def main() -> None:
    metrics = pd.read_csv(METRICS)
    metrics = metrics[
        (metrics["Scope"] == "test-retest")
        & (metrics["Dataset"] == "PCCT test-retest")
    ].copy()
    metrics["Recon"] = metrics["Group"].map(recon)

    scan = pd.read_csv(SCAN)
    scan = scan[
        (scan["Scope"] == "test-retest")
        & (scan["Dataset"] == "PCCT test-retest")
    ].copy()
    scan["Recon"] = scan["Group"].map(recon)
    scan = scan[[
        "Group", "Pair", "Organ", "Recon",
        "After_components", "Largest_fraction", "Small_volume_mm3",
        "Largest_ASSD_mm", "Largest_MSSD_mm", "Largest_HD95_mm",
    ]]
    merged = metrics.merge(
        scan, on=["Group", "Pair", "Organ", "Recon"], how="left"
    )

    rows = []
    for (recon_name, organ), group in merged.groupby(["Recon", "Organ"], sort=True):
        rows.append({
            "Rekon": recon_name,
            "Organ": {"kidney_right": "Sağ böbrek", "kidney_left": "Sol böbrek"}[organ],
            "N": len(group),
            "Dice (med)": round(group["Dice_after"].median(), 4),
            "Dice (ort)": round(group["Dice_after"].mean(), 4),
            "ASSD (med)": round(group["ASSD_after"].median(), 3),
            "ASSD (ort)": round(group["ASSD_after"].mean(), 3),
            "MSSD (med)": round(group["MSSD_after"].median(), 3),
            "MSSD (ort)": round(group["MSSD_after"].mean(), 3),
            "HD95 (med)": round(group["HD95_after"].median(), 3),
            "HD95 (ort)": round(group["HD95_after"].mean(), 3),
            "Adacık": int((group["After_components"] > 1).sum()),
            "ASSD largest (med)": round(group["Largest_ASSD_mm"].median(), 3),
            "MSSD largest (med)": round(group["Largest_MSSD_mm"].median(), 3),
        })
    ozet = pd.DataFrame(rows)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Ozet"
    ws["A1"] = "BÖBREK: 1 mm vs 3 mm REKONSTRÜKSİYON (Zurich PCCT test-retest)"
    ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
    ws["A2"] = (
        "1 mm tüm test-retest çiftlerinde identity; 3 mm'de 8 adacık vardır. "
        "3 mm sol böbrek medyan ASSD'si smoothing/partial volume nedeniyle düşük "
        "görünebilir; sağ böbrek ortalama/HD95 adacıklar nedeniyle bozulur."
    )
    ws["A2"].font = Font(italic=True, size=9)
    cols = list(ozet.columns)
    head(ws, 4, cols)
    for r, row in enumerate(ozet.itertuples(index=False), 5):
        for c, value in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c > 2 else "left")
            if cols[c - 1] == "Adacık" and value:
                cell.fill = RED
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 14
    for c in range(3, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 14
    ws.freeze_panes = "C5"

    ws = wb.create_sheet("Ciftler")
    ws["A1"] = f"TEST-RETEST BÖBREK ÇİFTLERİ ({len(merged)} satır)"
    ws["A1"].font = Font(bold=True, size=12, color="1F4E78")
    cols = [
        "Recon", "Group", "Pair", "Organ",
        "Dice_before", "Dice_after", "ASSD_before", "ASSD_after",
        "MSSD_before", "MSSD_after", "HD95_before", "HD95_after",
        "After_components", "Largest_fraction", "Small_volume_mm3",
        "Largest_ASSD_mm", "Largest_MSSD_mm", "Largest_HD95_mm",
    ]
    head(ws, 3, cols)
    for r, row in enumerate(merged[cols].itertuples(index=False), 4):
        for c, value in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.font = Font(size=8.5)
            cell.alignment = Alignment(horizontal="center" if c >= 5 else "left")
            if cols[c - 1] == "After_components" and isinstance(value, (int, float)) and value > 1:
                cell.fill = RED
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 44
    ws.column_dimensions["C"].width = 50
    ws.column_dimensions["D"].width = 14
    for c in range(5, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 12
    ws.freeze_panes = "E4"

    ws = wb.create_sheet("Not")
    notes = [
        "1) Bu analiz Zurich PCCT test-retest verisindedir; cross-date değildir.",
        "2) 1 mm: 72 böbrek satırı, 0 adacık. 3 mm: 36 böbrek satırı, 8 adacık (hepsi sağ).",
        "3) 1 mm medyan ASSD/MSSD sağ/sol: 0.350/0.381 ve 0.328/0.361 mm; HD95 1.0 mm.",
        "4) 3 mm sol medyan ASSD/MSSD 0.226/0.193 mm görünür; bu smoothing/partial volume etkisidir, doğruluk artışı değildir.",
        "5) 3 mm sağ ortalama ASSD/MSSD 7.76/5.11 mm, HD95 108.7 mm; adacıklar ortalamayı bozar.",
        "6) Sonuç: böbrek yüzey metrikleri için 1 mm tercih edilmelidir; 3 mm ayrı raporlanmalı veya dışlanmalıdır.",
        "7) 3 mm görüntüler 1 mm'ye dönüştürülemez; acquisition/recon değişmeden 'tüm görüntüleri 1 mm yapmak' mümkün değildir.",
    ]
    ws["A1"] = "YÖNTEM / YORUM"
    ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
    for i, text in enumerate(notes, 3):
        ws.cell(row=i, column=1, value=text).font = Font(size=10)
    ws.column_dimensions["A"].width = 130

    wb.save(OUT)
    print(f"[OK] {OUT}")
    print(ozet.to_string(index=False))


if __name__ == "__main__":
    main()
