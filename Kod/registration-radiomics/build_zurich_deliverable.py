"""Build the Zurich PCCT deliverable workbook for the collaborator.

Sheets:
  * Ozet_FullOrgan  - full-organ before/after, skip0 (forced) vs skip5 (identity)
  * VOI_Kup_Kure    - cube vs sphere VOI Dice (+ full-organ Dice)
  * CrossDate       - longitudinal before/after (registration benefit)
  * Yorum           - conclusions / caveats
"""

import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PCCT_WORK = Path(r"C:/Users/eray/Desktop/zurich_pcct_work")
SEP_WORK = Path(r"C:/Users/eray/Desktop/zurich_sep_work")
PCCT_LONG = Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX/REGISTRATION/batch_to_20260805_v1.0.0")
SEP_LONG = Path(r"C:/Users/eray/Desktop/SEP - ['Br40f',_'3']_120_100_300_1/PhantomX/REGISTRATION/batch_to_20260805_v6")
VOI_CSV = PCCT_WORK / "VOI_DICE_SUMMARY_shapes.csv"
OUT = Path(r"C:/Users/eray/Desktop/ZURICH_PCCT_DEGERLENDIRME.xlsx")

SPLEEN_LABELS = {1, 3}
AORTA_LABELS = {6, 52}


def group_metrics(base: Path):
    """Return DataFrame with columns group,label,before_dice,after_dice from metrics.csv."""
    rows = []
    for f in base.glob("*/*/metrics.csv"):
        try:
            df = pd.read_csv(f)
            df["group"] = f.parent.parent.name
            rows.append(df[["group", "label", "before_dice", "after_dice"]])
        except Exception:
            continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def organ_of(label):
    if int(label) in SPLEEN_LABELS:
        return "spleen"
    if int(label) in AORTA_LABELS:
        return "aorta"
    return None


def summarize(df: pd.DataFrame, group_prefix=""):
    out = {}
    if df.empty:
        return out
    df = df.copy()
    df["organ"] = df["label"].map(organ_of)
    df = df[df["organ"].notna()]
    for group, g in df.groupby("group"):
        key = group
        key = key.replace("PCTT_20260929_1__", "").replace("PCTT_20260929_2__", "")
        out[key] = {}
        for organ in ("spleen", "aorta"):
            o = g[g["organ"] == organ]
            if o.empty:
                continue
            out[key][organ] = {
                "n": len(o),
                "before": float(o["before_dice"].median()),
                "after": float(o["after_dice"].median()),
                "delta": float((o["after_dice"] - o["before_dice"]).median()),
            }
    return out


def cross_summary(base: Path):
    frames = []
    for f in base.glob("*/metrics.csv"):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df["organ"] = df["label"].map(organ_of)
    out = {}
    for organ in ("spleen", "aorta"):
        o = df[df["organ"] == organ]
        if o.empty:
            continue
        out[organ] = {
            "n": len(o),
            "before": float(o["before_dice"].median()),
            "after": float(o["after_dice"].median()),
            "delta": float((o["after_dice"] - o["before_dice"]).median()),
        }
    return out


def make_workbook(skip0, skip5, cross_pcct, cross_sep, sep_skip0, sep_skip5):
    wb = openpyxl.Workbook()
    navy = PatternFill("solid", start_color="1F4E78")
    green = PatternFill("solid", start_color="E2EFDA")
    red = PatternFill("solid", start_color="FCE4D6")

    def head(ws, row, cols):
        for c, h in enumerate(cols, 1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = Font(bold=True, color="FFFFFF", size=9)
            cell.fill = navy
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def add_overview(ws, title, note, s0_data, s5_data):
        ws["A1"] = title
        ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
        ws["A2"] = note
        ws["A2"].font = Font(italic=True, size=9)
        cols = ["Protokol", "N", "Dalak Once (skip0)", "Dalak Sonra (skip0)", "Dalak Fark (skip0)",
                "Dalak Once (skip5)", "Dalak Sonra (skip5)", "Dalak Fark (skip5)",
                "Aort Once (skip0)", "Aort Sonra (skip0)", "Aort Fark (skip0)",
                "Aort Once (skip5)", "Aort Sonra (skip5)", "Aort Fark (skip5)"]
        head(ws, 4, cols)
        groups = sorted(set(s0_data) | set(s5_data))
        r = 5
        for g in groups:
            s0 = s0_data.get(g, {}).get("spleen", {})
            a0 = s0_data.get(g, {}).get("aorta", {})
            s5 = s5_data.get(g, {}).get("spleen", {})
            a5 = s5_data.get(g, {}).get("aorta", {})
            vals = [g, s0.get("n", s5.get("n", 0)),
                    s0.get("before"), s0.get("after"), s0.get("delta"),
                    s5.get("before"), s5.get("after"), s5.get("delta"),
                    a0.get("before"), a0.get("after"), a0.get("delta"),
                    a5.get("before"), a5.get("after"), a5.get("delta")]
            for c, v in enumerate(vals, 1):
                cell = ws.cell(row=r, column=c, value=v)
                cell.font = Font(size=9)
                cell.alignment = Alignment(horizontal="center" if c > 1 else "left")
                if c in (5, 8, 11, 14) and isinstance(v, (int, float)):
                    cell.fill = green if v >= 0 else red
            r += 1
        ws.column_dimensions["A"].width = 52
        for c in range(2, 15):
            ws.column_dimensions[get_column_letter(c)].width = 13
        ws.freeze_panes = "B5"

    note = ("skip0 = registration zorlandi (eski kosu); skip5 = 5 mm esigi (identity). "
            "Degerler metrics.csv'den (tam fixed grid).")
    ws = wb.active
    ws.title = "Ozet_FullOrgan"
    add_overview(ws, "ZURICH PCCT TEST-RETEST: TUM ORGAN DICE (before/after)", note, skip0, skip5)
    add_overview(wb.create_sheet("Ozet_SEP"), "ZURICH SEP TEST-RETEST: TUM ORGAN DICE (before/after)",
                 note, sep_skip0, sep_skip5)

    # VOI
    ws2 = wb.create_sheet("VOI_Kup_Kure")
    ws2["A1"] = "VOI (1 test - 9 retest / 1 test - 9 cross-date): KUP vs KURE"
    ws2["A1"].font = Font(bold=True, size=13, color="1F4E78")
    ws2["A2"] = ("DIKKAT: VOI organin ICINDE; kucuk kure tamamen organ icinde kalirsa "
                 "(aorta 10 mm) maskenin tamami 1 olur ve Dice trivially 1.0 cikar. "
                 "VOI Dice registration QC degil, feature stabilitesi icindir.")
    ws2["A2"].font = Font(italic=True, size=9)
    head(ws2, 4, ["Dataset", "Organ", "Shape", "N", "VOI Dice (medyan)", "Full-organ Dice (medyan)", "Not"])
    if VOI_CSV.exists():
        v = pd.read_csv(VOI_CSV)
        v["organ"] = v["organ"].astype(str).str.lower()
        v["dataset"] = v["dataset"].astype(str).str.split(":").str[0]
        r = 5
        for (ds, organ, shape), g in v.groupby(["dataset", "organ", "shape"]):
            note = ""
            if shape == "sphere" and float(g["voi_dice"].median()) >= 0.9999:
                note = "Doygun (kure tamamen organ icinde)"
            vals = [ds, organ, shape, len(g), round(float(g["voi_dice"].median()), 5),
                    round(float(g["full_dice"].median()), 5), note]
            for c, val in enumerate(vals, 1):
                cell = ws2.cell(row=r, column=c, value=val)
                cell.font = Font(size=9)
                cell.alignment = Alignment(horizontal="center" if c > 1 else "left")
            r += 1
        ws2.column_dimensions["A"].width = 42
        ws2.column_dimensions["B"].width = 9
        ws2.column_dimensions["C"].width = 9
        ws2.column_dimensions["E"].width = 16
        ws2.column_dimensions["F"].width = 20
        ws2.column_dimensions["G"].width = 34
        ws2.freeze_panes = "A5"

    # Cross-date
    ws3 = wb.create_sheet("CrossDate")
    ws3["A1"] = "CROSS-DATE (farkli tarih): registration'in gercek faydasi"
    ws3["A1"].font = Font(bold=True, size=13, color="1F4E78")
    head(ws3, 3, ["Dataset", "Organ", "N", "Once (identity)", "Sonra (registered)", "Fark"])
    r = 4
    for ds, data in (("PCCT longitudinal", cross_pcct), ("SEP longitudinal", cross_sep)):
        for organ in ("spleen", "aorta"):
            if organ not in data:
                continue
            d = data[organ]
            vals = [ds, organ, d["n"], round(d["before"], 4), round(d["after"], 4), round(d["delta"], 4)]
            for c, val in enumerate(vals, 1):
                cell = ws3.cell(row=r, column=c, value=val)
                cell.font = Font(size=9)
                cell.alignment = Alignment(horizontal="center" if c > 1 else "left")
                if c == 6:
                    cell.fill = green if val >= 0 else red
            r += 1
    ws3.column_dimensions["A"].width = 22
    ws3.column_dimensions["B"].width = 10
    ws3.column_dimensions["C"].width = 8
    ws3.column_dimensions["D"].width = 16
    ws3.column_dimensions["E"].width = 18
    ws3.column_dimensions["F"].width = 10
    ws3.freeze_panes = "A4"

    # Yorum
    ws4 = wb.create_sheet("Yorum")
    ws4["A1"] = "SONUC VE ONERILER"
    ws4["A1"].font = Font(bold=True, size=13, color="1F4E78")
    notes = [
        "1) Test-retest verisi zaten hizali (72/72 pair centroid farki <= 0.94 mm).",
        "2) Full fixed grid + 1 test-9 retest: skip0 (zorla registration) karisik, cogunlukla notr/hafif +;",
        "   skip5 (identity) before == after (tam esit). Yani registration zorlaninca sistematik dusus YOK.",
        "3) Ogrencinin 45-pair karsilastirmasindaki dusus iki artifact'ten: (a) Ref crop'a, Target tam",
        "   fixed grid'e resample ediliyor; (b) 45 all-pairs vs 9 test-retest. Ayni grid + 1-vs-9'da dusus yok.",
        "4) Dogru karsilastirma: ayni grid (tam fixed) ve ayni pair seti. metrics.csv bunu sagliyor.",
        "5) VOI (kup/kure) organin icinde; VOI Dice 0.99-1.00 arasi doyar. Aorta 10 mm kure tamamen",
        "   organ icinde kaldigi icin Dice=1.0 (ayirt edici degil). VOI Dice registration QC icin uygun degil.",
        "6) VOI'nin amaci feature (HU/radiomics) stabilitesi; registration QC icin tam-organ Dice kullanilmali.",
        "7) Cross-date'te registration sart: dalak 0.03->0.96, aort 0.13->0.93.",
        "8) Oneri: test-retest'te registration zorlanmasin (skip 5 mm, identity); cross-date'te registration.",
    ]
    for i, text in enumerate(notes, 3):
        ws4.cell(row=i, column=1, value=text).font = Font(size=10)
    ws4.column_dimensions["A"].width = 120

    wb.save(OUT)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    skip0 = summarize(group_metrics(PCCT_WORK / "REGISTRATION"))
    skip5 = summarize(group_metrics(PCCT_WORK / "REGISTRATION_skip5"))
    sep_skip0 = summarize(group_metrics(SEP_WORK / "REGISTRATION"))
    sep_skip5 = summarize(group_metrics(SEP_WORK / "REGISTRATION_skip5"))
    cross_pcct = cross_summary(PCCT_LONG)
    cross_sep = cross_summary(SEP_LONG)
    make_workbook(skip0, skip5, cross_pcct, cross_sep, sep_skip0, sep_skip5)
    print("skip0 gruplari:", len(skip0), "skip5 gruplari:", len(skip5),
          "| SEP:", len(sep_skip0), len(sep_skip5))
    print("cross PCCT:", cross_pcct)
    print("cross SEP:", cross_sep)
