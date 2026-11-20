"""Cross-check kidney numbers across the v2.1 report, primary Excel, all-pairs
workbooks and the site data.

Writes KONSISTENSLIK_RAPORU.md on the Desktop.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document

DESKTOP = Path(r"C:/Users/eray/Desktop")
ROOT = Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE")
PCCT_BATCH = ROOT / "PhantomX/REGISTRATION/batch_to_20260805_v1.0.0"
SEP_ROOT = next(DESKTOP.glob("SEP -*")) / "PhantomX"
SEP_BATCH = SEP_ROOT / "REGISTRATION/batch_to_20260805_v6_skip2"
ZURICH_PCCT = DESKTOP / "zurich_pcct_work/REGISTRATION_skip5"
ZURICH_SEP = DESKTOP / "zurich_sep_work/REGISTRATION_skip5"
REPORT = ROOT / "RAPOR_AKADEMIK_REGISTRATION_RADIOMICS_v2.1.docx"
KIDNEY = DESKTOP / "KIDNEY_METRICS_PCCT_SEP.csv"
PRIMARY = DESKTOP / "BOBREK_1MM_PRIMARY.xlsx"
RECON = DESKTOP / "BOBREK_1MM_VS_3MM.xlsx"
SITE = DESKTOP / "phantomx-overlay-site/data.js"
ALL_PAIRS = {
    "PCCT": ROOT / "PhantomX/REGISTRATION/all_pairs_report_PCCT_pairwise_registration.xlsx",
    "SEP": SEP_ROOT / "REGISTRATION/all_pairs_report_SEP_pairwise_registration_v1.0.0.xlsx",
}
ORGAN = {"Aorta": "aorta", "Dalak": "spleen", "Sağ böbrek": "kidney_right", "Sol böbrek": "kidney_left"}
TOL = {"Dice sonra": 0.0006, "ASSD (mm)": 0.0051, "MSSD (mm)": 0.0051, "HD95 (mm)": 0.0051}
DIGITS = {"Dice sonra": 3, "ASSD (mm)": 2, "MSSD (mm)": 2, "HD95 (mm)": 2}
checks = []


def add(name, ok, detail=""):
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def read_metrics(root: Path, recursive: bool = False) -> pd.DataFrame:
    rows = []
    paths = root.rglob("metrics.csv") if recursive else root.glob("*/metrics.csv")
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        for _, row in frame.iterrows():
            organ = {1: "spleen", 52: "aorta"}.get(int(row["label"]))
            if not organ:
                continue
            rows.append({
                "organ": organ,
                "dice": row["after_dice"],
                "assd": row["after_assd_mm"],
                "mssd": row["after_mssd_mm"],
                "hd95": row["after_hd95_mm"],
            })
    return pd.DataFrame(rows)


def kidney_expected() -> dict[tuple[str, str], dict]:
    frame = pd.read_csv(KIDNEY)
    out = {}
    cross = frame[frame["Scope"] == "cross-date"]
    for (dataset_name, organ), group in cross.groupby(["Dataset", "Organ"]):
        dataset = "PCCT" if "PCCT" in dataset_name else "SEP"
        out[(dataset, organ)] = {
            "dice": group["Dice_after"].median(),
            "assd": group["ASSD_after"].median(),
            "mssd": group["MSSD_after"].median(),
            "hd95": group["HD95_after"].median(),
        }
    test = frame[
        (frame["Scope"] == "test-retest")
        & (~frame["Group"].str.contains("_3_|3mm", regex=True))
    ]
    for (dataset_name, organ), group in test.groupby(["Dataset", "Organ"]):
        dataset = "PCCT test-retest" if "PCCT" in dataset_name else "SEP test-retest"
        out[(dataset, organ)] = {
            "dice": group["Dice_after"].median(),
            "assd": group["ASSD_after"].median(),
            "mssd": group["MSSD_after"].median(),
            "hd95": group["HD95_after"].median(),
        }
    return out


def expected_means() -> dict[tuple[str, str], dict]:
    out = {}
    for dataset, root in (("PCCT", PCCT_BATCH), ("SEP", SEP_BATCH)):
        frame = read_metrics(root)
        for organ, group in frame.groupby("organ"):
            out[(dataset, organ)] = {
                "dice": group["dice"].median(),
                "assd": group["assd"].median(),
                "mssd": group["mssd"].median(),
                "hd95": group["hd95"].median(),
            }
    for dataset, root in (("PCCT test-retest", ZURICH_PCCT), ("SEP test-retest", ZURICH_SEP)):
        frame = read_metrics(root, recursive=True)
        for organ, group in frame.groupby("organ"):
            out[(dataset, organ)] = {
                "dice": group["dice"].median(),
                "assd": group["assd"].median(),
                "mssd": group["mssd"].median(),
                "hd95": group["hd95"].median(),
            }
    return {**out, **kidney_expected()}


def report_rows(doc: Document, index: int, value_indexes: dict[int, str]) -> list[dict]:
    rows = []
    table = doc.tables[index]
    headers = [c.text for c in table.rows[0].cells]
    for row in table.rows[1:]:
        values = [c.text for c in row.cells]
        item = {"dataset": values[0], "organ": ORGAN.get(values[1], values[1])}
        for col, name in value_indexes.items():
            item[name] = values[col]
        rows.append(item)
    return rows


def compare_rows(rows, expected, label):
    bad = 0
    for row in rows:
        key = (row["dataset"], row["organ"])
        exp = expected.get(key)
        if exp is None:
            add(f"{label} {key}", False, "kaynakta yok")
            bad += 1
            continue
        for column, digit in DIGITS.items():
            report_value = row.get(column)
            if report_value is None:
                continue
            try:
                value = float(report_value)
            except ValueError:
                continue
            exp_value = round(float(exp[column.split()[0].lower() if column != "Dice sonra" else "dice"]), digit)
            if abs(value - exp_value) > TOL[column]:
                add(f"{label} {key[0]} {key[1]} {column}", False, f"rapor={value} kaynak={exp_value}")
                bad += 1
    add(f"{label} tablo", bad == 0, f"{bad} uyumsuz" if bad else "tüm satırlar eşleşti")


def main() -> None:
    expected = expected_means()
    doc = Document(REPORT)
    cross_rows = report_rows(doc, 2, {4: "Dice sonra", 7: "ASSD (mm)", 8: "MSSD (mm)", 9: "HD95 (mm)"})
    test_rows = report_rows(doc, 4, {4: "Dice sonra", 6: "ASSD (mm)", 7: "MSSD (mm)", 8: "HD95 (mm)"})
    compare_rows(cross_rows, expected, "Cross-date")
    compare_rows(test_rows, expected, "Test-retest 1 mm")

    # Primary Excel
    try:
        ozet = pd.read_excel(PRIMARY, sheet_name="Ozet", header=3)
        test_rows_x = ozet[ozet["Kapsam"] == "test-retest"]
        n_ok = (
            len(test_rows_x[test_rows_x["Veri"] == "PCCT 1 mm"]) == 2
            and len(test_rows_x[test_rows_x["Veri"] == "SEP 1 mm"]) == 2
            and set(test_rows_x[test_rows_x["Veri"] == "PCCT 1 mm"]["N"]) == {36}
            and set(test_rows_x[test_rows_x["Veri"] == "SEP 1 mm"]["N"]) == {55}
        )
        add("Primary Excel test-retest n", n_ok, "PCCT 36 / SEP 55")
        primary_pairs = pd.read_excel(PRIMARY, sheet_name="Ciftler_Primary", header=2)
        add("Primary Excel 3 mm yok", not primary_pairs["Group"].astype(str).str.contains("_3_|3mm", regex=True).any(),
            f"{len(primary_pairs)} primary satır")
        sens = pd.read_excel(PRIMARY, sheet_name="Sensitivity_3mm", header=2)
        add("Primary Excel 3 mm ayrı", len(sens) == 36, f"{len(sens)} satır")
    except Exception as exc:  # noqa: BLE001
        add("Primary Excel", False, str(exc))

    # 3 mm Excel
    try:
        recon = pd.read_excel(RECON, sheet_name="Ozet", header=3)
        islands = int(recon[recon["Rekon"] == "3 mm"]["Adacık"].sum())
        add("1mm vs 3mm adacık", islands == 8, f"3mm adacık={islands}")
    except Exception as exc:  # noqa: BLE001
        add("1mm vs 3mm Excel", False, str(exc))

    # All-pairs
    for dataset, path in ALL_PAIRS.items():
        try:
            x = pd.ExcelFile(path)
            sheets = set(x.sheet_names)
            pairs = pd.read_excel(path, sheet_name="Pairs")
            ok = all(f"Kidney_{side}_{metric}_Series" in sheets for side in ("Right", "Left") for metric in ("Dice", "ASSD", "HD95", "clDice"))
            ok = ok and all(f"Kidney {side.lower()} Dice" in pairs.columns for side in ("right", "left"))
            add(f"{dataset} all-pairs böbrek", ok, f"pairs={len(pairs)}")
        except Exception as exc:  # noqa: BLE001
            add(f"{dataset} all-pairs", False, str(exc))

    # Site
    try:
        text = SITE.read_text(encoding="utf-8")
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        for key, expected_n in (("pcct", 88), ("sep", 67)):
            cases = data[key]["cases"]
            missing = [
                case["id"] for case in cases
                if not all(organ in case.get("views", {}) for organ in ("aorta", "spleen", "kidney_right", "kidney_left"))
            ]
            add(f"Site {key} 4 organ", len(cases) == expected_n and not missing,
                f"cases={len(cases)} missing={len(missing)}")
    except Exception as exc:  # noqa: BLE001
        add("Site data.js", False, str(exc))

    lines = [
        "# Böbrek Tutarlılık Raporu",
        "",
        f"Tarih: 17 Kasım 2026",
        "",
        "| Kontrol | Durum | Detay |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item['check']} | {item['status']} | {item['detail']} |")
    lines.append("")
    lines.append(f"Toplam: {sum(c['status']=='PASS' for c in checks)} PASS, "
                 f"{sum(c['status']=='FAIL' for c in checks)} FAIL")
    out = DESKTOP / "KONSISTENSLIK_RAPORU.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {out}")
    for item in checks:
        print(f"{item['status']:4s} {item['check']} | {item['detail']}")


if __name__ == "__main__":
    main()
