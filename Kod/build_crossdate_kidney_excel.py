"""Build CROSSDATE_BOBREK.xlsx and annotate related Excel workbooks.

Sources:
  KIDNEY_METRICS_PCCT_SEP.csv  (cross-date and test-retest kidney metrics)
  KIDNEY_CC_SCAN.csv           (connected-component diagnostics)

Outputs:
  CROSSDATE_BOBREK.xlsx
  ASSD_MSSD_OUTLIERS.xlsx  + Bobrek sheet
  PCCT_SEP_LONG_CONSENSUS.xlsx + Bobrek_Not sheet
  KARSILASTIRMA_PCCT_DUZELTILMIS.xlsx + Bobrek_Not sheet
  KARSILASTIRMA_PCCT_SKIP5.xlsx + Bobrek_Not sheet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DESKTOP = Path(r"C:/Users/eray/Desktop")
METRICS = DESKTOP / "KIDNEY_METRICS_PCCT_SEP.csv"
SCAN = DESKTOP / "KIDNEY_CC_SCAN.csv"
OUT = DESKTOP / "CROSSDATE_BOBREK.xlsx"
ORGAN_LABEL = {"kidney_right": "Sağ böbrek", "kidney_left": "Sol böbrek"}
NAVY = PatternFill("solid", start_color="1F4E78")
GREEN = PatternFill("solid", start_color="E2EFDA")
RED = PatternFill("solid", start_color="FCE4D6")
AMBER = PatternFill("solid", start_color="FCE4D6")


def head(ws, row, columns):
    for index, name in enumerate(columns, 1):
        cell = ws.cell(row=row, column=index, value=name)
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.fill = NAVY
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def widths(ws, mapping):
    for column, width in mapping.items():
        ws.column_dimensions[column].width = width


def build_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS)
    metrics = metrics[metrics["Scope"] == "cross-date"].copy()
    scan = pd.read_csv(SCAN)
    scan = scan[scan["Scope"] == "cross-date"].copy()
    scan = scan[[
        "Dataset", "Group", "Pair", "Organ",
        "After_components", "Largest_fraction", "Small_components",
        "Small_volume_mm3", "Largest_ASSD_mm", "Largest_MSSD_mm",
        "Largest_HD95_mm",
    ]].rename(columns={
        "After_components": "Components_after",
        "Largest_fraction": "Largest_fraction_after",
        "Largest_ASSD_mm": "ASSD_after_largest",
        "Largest_MSSD_mm": "MSSD_after_largest",
        "Largest_HD95_mm": "HD95_after_largest",
    })
    merged = metrics.merge(
        scan, on=["Dataset", "Group", "Pair", "Organ"], how="left"
    )
    return merged


def _median_if_valid(series: pd.Series, n: int, digits: int):
    return round(float(series.median()), digits) if series.notna().sum() >= n * 0.5 else "—"


def build_ozet(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, organ), group in frame.groupby(["Dataset", "Organ"], sort=True):
        n = len(group)
        rows.append({
            "Dataset": dataset.replace(" cross-date", ""),
            "Organ": ORGAN_LABEL[organ],
            "N": n,
            "Dice önce (med)": round(group["Dice_before"].median(), 4),
            "Dice sonra (med)": round(group["Dice_after"].median(), 4),
            "ΔDice (med)": round(group["Dice_delta"].median(), 4),
            "ASSD önce (med)": _median_if_valid(group["ASSD_before"], n, 3),
            "ASSD sonra (med)": round(group["ASSD_after"].median(), 3),
            "MSSD önce (med)": _median_if_valid(group["MSSD_before"], n, 3),
            "MSSD sonra (med)": round(group["MSSD_after"].median(), 3),
            "HD95 önce (med)": _median_if_valid(group["HD95_before"], n, 3),
            "HD95 sonra (med)": round(group["HD95_after"].median(), 3),
            "İyileşen": int((group["Dice_delta"] > 0.0005).sum()),
            "Kötüleşen": int((group["Dice_delta"] < -0.0005).sum()),
            "Adacık": int((group["Components_after"] > 1).sum()),
            "ASSD largest (med)": round(group["ASSD_after_largest"].median(), 3),
            "MSSD largest (med)": round(group["MSSD_after_largest"].median(), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    frame = build_metrics()
    ozet = build_ozet(frame)
    islands = frame[frame["Components_after"] > 1].copy()
    failures = frame[
        (frame["Components_after"] == 1) & (frame["ASSD_after"] > 1.0)
    ].copy()

    wb = openpyxl.Workbook()

    # Ozet
    ws = wb.active
    ws.title = "Ozet"
    ws["A1"] = "CROSS-DATE BÖBREK DEĞERLENDİRMESİ"
    ws["A1"].font = Font(bold=True, size=14, color="1F4E78")
    ws["A2"] = (
        "Önce = identity (çakıştırma yok); sonra = rigid registration. "
        "ASSD/MSSD/HD95 mm; largest = yalnız en büyük bağlı bileşen (tanısal). "
        "Kaynak maskeler değiştirilmedi."
    )
    ws["A2"].font = Font(italic=True, size=9)
    cols = list(ozet.columns)
    head(ws, 4, cols)
    for r, row in enumerate(ozet.itertuples(index=False), 5):
        for c, value in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c > 2 else "left")
            if cols[c - 1] == "ΔDice (med)" and isinstance(value, (int, float)):
                cell.fill = GREEN if value >= 0 else RED
    widths(ws, {"A": 14, "B": 14})
    for c in range(3, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 14
    ws.freeze_panes = "C5"

    # Ciftler
    ws = wb.create_sheet("Ciftler")
    ws["A1"] = f"CROSS-DATE BÖBREK ÇİFTLERİ ({len(frame)} satır)"
    ws["A1"].font = Font(bold=True, size=12, color="1F4E78")
    cols = [
        "Dataset", "Group", "Pair", "Organ",
        "Dice_before", "Dice_after", "Dice_delta",
        "ASSD_before", "ASSD_after", "MSSD_before", "MSSD_after",
        "HD95_before", "HD95_after", "Durum",
        "Components_after", "Largest_fraction_after", "Small_volume_mm3",
        "ASSD_after_largest", "MSSD_after_largest", "HD95_after_largest",
        "Method",
    ]
    head(ws, 3, cols)
    for r, row in enumerate(frame[cols].itertuples(index=False), 4):
        for c, value in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.font = Font(size=8.5)
            cell.alignment = Alignment(horizontal="center" if c >= 5 else "left")
            if cols[c - 1] == "Dice_delta" and isinstance(value, (int, float)):
                cell.fill = GREEN if value >= 0 else RED
            if cols[c - 1] == "Durum":
                if value == "İyileşti":
                    cell.font = Font(size=8.5, bold=True, color="006100")
                elif value == "Kötüleşti":
                    cell.font = Font(size=8.5, bold=True, color="9C0006")
    widths(ws, {"A": 14, "B": 30, "C": 48, "D": 14})
    for c in range(5, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 12
    ws.freeze_panes = "E4"

    # Adaciklar
    ws = wb.create_sheet("Adaciklar")
    ws["A1"] = f"ADACIKLI BÖBREK MASKEleri ({len(islands)} satır)"
    ws["A1"].font = Font(bold=True, size=12, color="1F4E78")
    cols = [
        "Dataset", "Pair", "Organ", "Components_after",
        "Small_volume_mm3", "Dice_after", "ASSD_after", "ASSD_after_largest",
        "MSSD_after", "MSSD_after_largest", "HD95_after", "HD95_after_largest",
    ]
    head(ws, 3, cols)
    for r, row in enumerate(islands[cols].itertuples(index=False), 4):
        for c, value in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.font = Font(size=8.5)
            cell.alignment = Alignment(horizontal="center" if c >= 4 else "left")
            if c >= 7 and isinstance(value, (int, float)):
                cell.fill = AMBER
    widths(ws, {"A": 16, "B": 48, "C": 14})
    for c in range(4, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13
    ws.freeze_panes = "D4"

    # SegHatasi
    ws = wb.create_sheet("SegHatasi")
    ws["A1"] = f"TEK BİLEŞENLİ GERÇEK SEGMENTASYON AYKIRILARI ({len(failures)} satır)"
    ws["A1"].font = Font(bold=True, size=12, color="9C0006")
    cols = [
        "Dataset", "Pair", "Organ", "Dice_after", "ASSD_after",
        "MSSD_after", "HD95_after", "Components_after",
    ]
    head(ws, 3, cols)
    for r, row in enumerate(failures[cols].itertuples(index=False), 4):
        for c, value in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=value)
            cell.font = Font(size=8.5)
            cell.alignment = Alignment(horizontal="center" if c >= 4 else "left")
    widths(ws, {"A": 16, "B": 50, "C": 14})
    for c in range(4, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 13

    # Not
    ws = wb.create_sheet("Not")
    notes = [
        "1) Cross-date: her moving seri 20260805 take1 referansına göre değerlendirildi.",
        "2) Önce = identity; sonra = rigid (kemik/landmark). Centroid ara adımdır; burada yok.",
        "3) Böbrekler TotalSegmentator label 2 (sağ) ve 3 (sol) üzerinden ayrı hesaplandı.",
        "4) ASSD/MSSD/HD95 mm; Dice 0–1. Dice yüksek iyi; mesafeler düşük iyi.",
        "5) Largest = en büyük bağlı bileşen; yalnızca tanısal duyarlılık analizidir.",
        "6) Bu cross-date kitabında 10 adacık (7 sağ, 3 sol) ve 2 gerçek segmentasyon aykırısı (SEP 20260731_s80 sağ/sol) vardır. Test-retest'teki 8 adacık ayrıca KIDNEY_CC_SCAN.csv içindedir.",
        "7) Kaynak dosyalar: KIDNEY_METRICS_PCCT_SEP.csv, KIDNEY_CC_SCAN.csv.",
    ]
    ws["A1"] = "YÖNTEM / YORUM"
    ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
    for index, text in enumerate(notes, 3):
        ws.cell(row=index, column=1, value=text).font = Font(size=10)
    ws.column_dimensions["A"].width = 120

    wb.save(OUT)
    print(f"[OK] {OUT}")

    # Add Bobrek sheet to ASSD_MSSD_OUTLIERS.xlsx
    outlier_path = DESKTOP / "ASSD_MSSD_OUTLIERS.xlsx"
    if outlier_path.exists():
        wb2 = openpyxl.load_workbook(outlier_path)
        if "Bobrek" in wb2.sheetnames:
            del wb2["Bobrek"]
        ws = wb2.create_sheet("Bobrek")
        ws["A1"] = "BÖBREK ASSD/MSSD AYKIRILARI (adacık + gerçek segmentasyon)"
        ws["A1"].font = Font(bold=True, size=12, color="1F4E78")
        cols = [
            "Dataset", "Pair", "Organ", "Tip", "Components",
            "Small_volume_mm3", "ASSD_ham", "ASSD_largest",
            "MSSD_ham", "MSSD_largest", "HD95_ham", "HD95_largest",
        ]
        head(ws, 3, cols)
        rows = []
        for _, row in islands.iterrows():
            rows.append([
                row["Dataset"], row["Pair"], ORGAN_LABEL[row["Organ"]], "Adacık",
                int(row["Components_after"]), round(float(row["Small_volume_mm3"]), 1),
                round(float(row["ASSD_after"]), 3), round(float(row["ASSD_after_largest"]), 3),
                round(float(row["MSSD_after"]), 3), round(float(row["MSSD_after_largest"]), 3),
                round(float(row["HD95_after"]), 3), round(float(row["HD95_after_largest"]), 3),
            ])
        for _, row in failures.iterrows():
            rows.append([
                row["Dataset"], row["Pair"], ORGAN_LABEL[row["Organ"]], "Segmentasyon",
                int(row["Components_after"]), 0.0,
                round(float(row["ASSD_after"]), 3), round(float(row["ASSD_after"]), 3),
                round(float(row["MSSD_after"]), 3), round(float(row["MSSD_after"]), 3),
                round(float(row["HD95_after"]), 3), round(float(row["HD95_after"]), 3),
            ])
        for r, values in enumerate(rows, 4):
            for c, value in enumerate(values, 1):
                cell = ws.cell(row=r, column=c, value=value)
                cell.font = Font(size=8.5)
                cell.alignment = Alignment(horizontal="center" if c >= 5 else "left")
                if c == 4:
                    cell.fill = GREEN if value == "Segmentasyon" else AMBER
        widths(ws, {"A": 16, "B": 48, "C": 14, "D": 14})
        for c in range(5, len(cols) + 1):
            ws.column_dimensions[get_column_letter(c)].width = 13
        ws.freeze_panes = "D4"
        wb2.save(outlier_path)
        print(f"[OK] {outlier_path} + Bobrek")

    # Notes in other workbooks
    notes = {
        "PCCT_SEP_LONG_CONSENSUS.xlsx": (
            "Konsensüs referansı bu çalışma kitabında dalak ve aorta için "
            "hesaplanmıştır; böbrek konsensüsü bu sürümde yoktur. Böbrek "
            "cross-date değerlendirmesi CROSSDATE_BOBREK.xlsx ve "
            "KAYIT_IYILESEN_KOTULESEN.xlsx/CrossDate_Bobrek içindedir."
        ),
        "KARSILASTIRMA_PCCT_DUZELTILMIS.xlsx": (
            "Bu çalışma kitabı aynı oturumdaki take-to-take (Zurich PCTT) "
            "karşılaştırmalarını içerir; cross-date değildir ve böbrek "
            "içermez. Cross-date böbrek değerlendirmesi CROSSDATE_BOBREK.xlsx "
            "içindedir."
        ),
        "KARSILASTIRMA_PCCT_SKIP5.xlsx": (
            "Bu çalışma kitabı aynı oturumdaki take-to-take (Zurich PCTT) "
            "karşılaştırmalarını içerir; cross-date değildir ve böbrek "
            "içermez. Cross-date böbrek değerlendirmesi CROSSDATE_BOBREK.xlsx "
            "içindedir."
        ),
    }
    for name, text in notes.items():
        path = DESKTOP / name
        if not path.exists():
            continue
        book = openpyxl.load_workbook(path)
        if "Bobrek_Not" in book.sheetnames:
            del book["Bobrek_Not"]
        sheet = book.create_sheet("Bobrek_Not")
        sheet["A1"] = "BÖBREK KAPSAMI"
        sheet["A1"].font = Font(bold=True, size=12, color="1F4E78")
        sheet["A3"] = text
        sheet["A3"].alignment = Alignment(wrap_text=True, vertical="top")
        sheet.column_dimensions["A"].width = 120
        book.save(path)
        print(f"[OK] {path} + Bobrek_Not")


if __name__ == "__main__":
    main()
