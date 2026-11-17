"""Build the primary 1 mm kidney analysis workbook.

Primary = cross-date (all 1 mm) + test-retest 1 mm.
3 mm test-retest is kept as a separate sensitivity sheet and is never pooled.
"""

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
OUT = DESKTOP / "BOBREK_1MM_PRIMARY.xlsx"
ORGAN_LABEL = {"kidney_right": "Sağ böbrek", "kidney_left": "Sol böbrek"}
NAVY = PatternFill("solid", start_color="1F4E78")
GREEN = PatternFill("solid", start_color="E2EFDA")
RED = PatternFill("solid", start_color="FCE4D6")


def recon_of(group: str) -> str:
    text = str(group)
    if re.search(r"_3_|3mm", text):
        return "3 mm"
    return "1 mm"


def head(ws, row, columns):
    for index, name in enumerate(columns, 1):
        cell = ws.cell(row=row, column=index, value=name)
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.fill = NAVY
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def main() -> None:
    frame = pd.read_csv(METRICS)
    frame["Recon"] = frame["Group"].map(recon_of)
    frame["Primary"] = np.where(
        (frame["Scope"] == "cross-date") | (frame["Recon"] == "1 mm"),
        "yes", "no",
    )

    scan = pd.read_csv(SCAN)[[
        "Scope", "Dataset", "Group", "Pair", "Organ",
        "After_components", "Largest_fraction", "Small_volume_mm3",
        "Largest_ASSD_mm", "Largest_MSSD_mm", "Largest_HD95_mm",
    ]].rename(columns={
        "After_components": "Components_after",
        "Largest_fraction": "Largest_fraction_after",
        "Largest_ASSD_mm": "ASSD_after_largest",
        "Largest_MSSD_mm": "MSSD_after_largest",
        "Largest_HD95_mm": "HD95_after_largest",
    })
    merged = frame.merge(
        scan, on=["Scope", "Dataset", "Group", "Pair", "Organ"], how="left"
    )
    primary = merged[merged["Primary"] == "yes"].copy()
    sensitivity = merged[merged["Recon"] == "3 mm"].copy()

    # Ozet
    rows = []
    for (scope, dataset, organ), group in primary.groupby(
        ["Scope", "Dataset", "Organ"], sort=True
    ):
        n = len(group)
        label = (
            "PCCT 1 mm" if (dataset == "PCCT test-retest") else
            "SEP 1 mm" if (dataset == "SEP test-retest") else
            dataset.replace(" cross-date", "")
        )
        rows.append({
            "Kapsam": "test-retest" if scope == "test-retest" else "cross-date",
            "Veri": label,
            "Organ": ORGAN_LABEL[organ],
            "N": n,
            "Dice önce": round(group["Dice_before"].median(), 4),
            "Dice sonra": round(group["Dice_after"].median(), 4),
            "ΔDice": round((group["Dice_after"] - group["Dice_before"]).median(), 4),
            "ASSD önce": round(group["ASSD_before"].median(), 3),
            "ASSD sonra": round(group["ASSD_after"].median(), 3),
            "MSSD önce": round(group["MSSD_before"].median(), 3),
            "MSSD sonra": round(group["MSSD_after"].median(), 3),
            "HD95 önce": round(group["HD95_before"].median(), 3),
            "HD95 sonra": round(group["HD95_after"].median(), 3),
            "İyileşen": int((group["Dice_after"] - group["Dice_before"] > 0.0005).sum()),
            "Kötüleşen": int((group["Dice_after"] - group["Dice_before"] < -0.0005).sum()),
            "Adacık": int((group["Components_after"] > 1).sum()),
            "ASSD largest": round(group["ASSD_after_largest"].median(), 3),
            "MSSD largest": round(group["MSSD_after_largest"].median(), 3),
        })
    ozet = pd.DataFrame(rows)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Ozet"
    ws["A1"] = "BÖBREK — PRIMARY 1 mm ANALİZİ"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = (
        "Primary = cross-date (tümü 1 mm) + Zurich test-retest 1 mm. "
        "3 mm seriler pooled edilmedi; ayrı duyarlılık sayfasındadır. "
        "Önce = identity; sonra = rigid."
    )
    ws["A2"].font = Font(italic=True, size=9)
    cols = list(ozet.columns)
    head(ws, 4, cols)
    for r, row in enumerate(ozet.itertuples(index=False), 5):
        for c, value in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c > 3 else "left")
            if cols[c - 1] == "ΔDice" and isinstance(value, (int, float)):
                cell.fill = GREEN if value >= 0 else RED
            if cols[c - 1] == "Adacık" and value:
                cell.fill = RED
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14
    for c in range(4, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 14
    ws.freeze_panes = "D5"

    # Primary pairs
    ws = wb.create_sheet("Ciftler_Primary")
    ws["A1"] = f"PRIMARY 1 mm BÖBREK ÇİFTLERİ ({len(primary)} satır)"
    ws["A1"].font = Font(bold=True, size=12, color="1F4E78")
    cols = [
        "Scope", "Dataset", "Group", "Pair", "Organ", "Recon",
        "Dice_before", "Dice_after", "Dice_delta",
        "ASSD_before", "ASSD_after", "MSSD_before", "MSSD_after",
        "HD95_before", "HD95_after", "Durum",
        "Components_after", "Largest_fraction_after", "Small_volume_mm3",
        "ASSD_after_largest", "MSSD_after_largest", "HD95_after_largest",
    ]
    head(ws, 3, cols)
    for r, row in enumerate(primary[cols].itertuples(index=False), 4):
        for c, value in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.font = Font(size=8.5)
            cell.alignment = Alignment(horizontal="center" if c >= 7 else "left")
            if cols[c - 1] == "Dice_delta" and isinstance(value, (int, float)):
                cell.fill = GREEN if value >= 0 else RED
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 28
    ws.column_dimensions["D"].width = 48
    ws.column_dimensions["E"].width = 14
    for c in range(6, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 12
    ws.freeze_panes = "F4"

    # 3 mm sensitivity
    ws = wb.create_sheet("Sensitivity_3mm")
    ws["A1"] = f"3 mm DUYARLILIK (pooled edilmedi; {len(sensitivity)} satır)"
    ws["A1"].font = Font(bold=True, size=12, color="9C0006")
    cols = [
        "Dataset", "Group", "Pair", "Organ",
        "Dice_before", "Dice_after", "ASSD_after", "MSSD_after",
        "HD95_after", "Components_after", "Small_volume_mm3",
        "ASSD_after_largest", "MSSD_after_largest", "HD95_after_largest",
    ]
    head(ws, 3, cols)
    for r, row in enumerate(sensitivity[cols].itertuples(index=False), 4):
        for c, value in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.font = Font(size=8.5)
            cell.alignment = Alignment(horizontal="center" if c >= 5 else "left")
            if cols[c - 1] == "Components_after" and isinstance(value, (int, float)) and value > 1:
                cell.fill = RED
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 42
    ws.column_dimensions["C"].width = 48
    ws.column_dimensions["D"].width = 14
    for c in range(5, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 12
    ws.freeze_panes = "E4"

    # Decision
    ws = wb.create_sheet("Karar")
    lines = [
        "KARAR",
        "1) Böbrek primary analizi yalnızca 1 mm üzerinden yapılır.",
        "2) Cross-date PCCT/SEP zaten 1 mm'dir; değişiklik gerekmez.",
        "3) Zurich test-retest 3 mm serileri primary analize dahil edilmez.",
        "4) 3 mm yalnızca duyarlılık/karşılaştırma olarak raporlanır; pooled edilmez.",
        "5) 3 mm'de 8 adacık (sağ böbrek), ortalama ASSD 7.76 mm, HD95 108.7 mm; 1 mm'de adacık yok.",
        "6) 3 mm sol medyan ASSD 0.226 mm daha düşük görünür; bu smoothing/partial volume etkisidir, doğruluk artışı değildir.",
        "7) Kaynak: KIDNEY_METRICS_PCCT_SEP.csv, KIDNEY_CC_SCAN.csv, BOBREK_1MM_VS_3MM.xlsx.",
    ]
    ws["A1"] = lines[0]
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    for i, text in enumerate(lines[1:], 3):
        ws.cell(row=i, column=1, value=text).font = Font(size=10)
    ws.column_dimensions["A"].width = 140

    wb.save(OUT)
    print(f"[OK] {OUT}")
    print(ozet.to_string(index=False))


if __name__ == "__main__":
    main()
