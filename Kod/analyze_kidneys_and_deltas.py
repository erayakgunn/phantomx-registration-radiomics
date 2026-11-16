"""Improved/worsened tables for spleen, aorta and kidneys.

Spleen/aorta test-retest and cross-date values come from the existing
``metrics.csv`` files. Kidneys are recomputed from the stored transforms:

    before = identity resample
    after  = moving_to_fixed.tfm resample

Both Dice and surface metrics (ASSD, HD95) are written to
``KIDNEY_METRICS_PCCT_SEP.csv`` and to the workbook.
"""

from __future__ import annotations

import json
import math
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
from metrics import compare_labels  # noqa: E402
from registration import resample_mask  # noqa: E402

OUT = Path(r"C:/Users/eray/Desktop/KAYIT_IYILESEN_KOTULESEN.xlsx")
CSV_OUT = Path(r"C:/Users/eray/Desktop/KIDNEY_METRICS_PCCT_SEP.csv")
WORKS = [
    ("PCCT test-retest", Path(r"C:/Users/eray/Desktop/zurich_pcct_work"), "zpcct"),
    ("SEP test-retest", Path(r"C:/Users/eray/Desktop/zurich_sep_work"), "zsep"),
]
ORGAN_LABELS = {1: "spleen", 3: "spleen", 52: "aorta", 6: "aorta"}
KIDNEYS = [("kidney_right", 2), ("kidney_left", 3)]
TOL = 0.0005
CROSSDATE = [
    (
        "PCCT cross-date",
        Path(
            r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/"
            r"PhantomX/REGISTRATION/batch_to_20260805_v1.0.0"
        ),
    ),
    (
        "SEP cross-date",
        Path(
            r"C:/Users/eray/Desktop/SEP - ['Br40f',_'3']_120_100_300_1/"
            r"PhantomX/REGISTRATION/batch_to_20260805_v6_skip2"
        ),
    ),
]


def organ_of(label):
    try:
        return ORGAN_LABELS.get(int(label))
    except Exception:
        return None


def status(delta):
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return "-"
    if delta > TOL:
        return "İyileşti"
    if delta < -TOL:
        return "Kötüleşti"
    return "Değişmedi"


def _num(value, digits: int) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    if not math.isfinite(value):
        return float("nan")
    return round(value, digits)


def kidney_pair_rows(
    fixed_seg: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Transform,
    dataset: str,
    group: str,
    pair: str,
    method: str | None = None,
) -> list[dict]:
    before = resample_mask(moving, fixed_seg, sitk.Transform(3, sitk.sitkIdentity))
    after = resample_mask(moving, fixed_seg, transform)
    labels = [label for _, label in KIDNEYS]
    bm = {int(m["label"]): m for m in compare_labels(fixed_seg, before, labels)}
    am = {int(m["label"]): m for m in compare_labels(fixed_seg, after, labels)}

    rows = []
    for kidney, label in KIDNEYS:
        b = bm.get(label, {})
        a = am.get(label, {})
        dice_b = _num(b.get("dice"), 4)
        dice_a = _num(a.get("dice"), 4)
        if not (math.isfinite(dice_b) and math.isfinite(dice_a)):
            continue
        delta = dice_a - dice_b
        rows.append({
            "Dataset": dataset,
            "Group": group,
            "Pair": pair,
            "Organ": kidney,
            "Dice_before": dice_b,
            "Dice_after": dice_a,
            "Dice_delta": round(delta, 4),
            "ASSD_before": _num(b.get("assd_mm"), 4),
            "ASSD_after": _num(a.get("assd_mm"), 4),
            "MSSD_before": _num(b.get("mssd_mm"), 4),
            "MSSD_after": _num(a.get("mssd_mm"), 4),
            "HD95_before": _num(b.get("hd95_mm"), 3),
            "HD95_after": _num(a.get("hd95_mm"), 3),
            # aliases used by the summary helper
            "Before": dice_b,
            "After": dice_a,
            "Delta": round(delta, 4),
            "Durum": status(delta),
            "Method": method,
        })
    return rows


def read_spleen_aorta(work: Path, dataset: str):
    rows = []
    for mc in work.glob("REGISTRATION/*/*/metrics.csv"):
        pair_dir = mc.parent
        group = pair_dir.parent.name
        series = pair_dir.name.split("__")[-1]
        qc_path = pair_dir / "registration_qc.json"
        method = None
        if qc_path.exists():
            method = json.loads(qc_path.read_text(encoding="utf-8")).get("registration_method")
        try:
            df = pd.read_csv(mc)
        except Exception:
            continue
        for _, r in df.iterrows():
            organ = organ_of(r["label"])
            if not organ:
                continue
            before, after = float(r["before_dice"]), float(r["after_dice"])
            delta = after - before
            rows.append({
                "Dataset": dataset, "Group": group, "Pair": series, "Organ": organ,
                "Before": round(before, 4), "After": round(after, 4), "Delta": round(delta, 4),
                "Durum": status(delta), "Method": method,
            })
    return rows


def read_kidneys(work: Path, dataset: str, prefix: str):
    rows = []
    reg = work / "REGISTRATION"
    lab = work / "LABELED"
    if not lab.exists():
        return rows
    for group in sorted(p for p in lab.iterdir() if p.is_dir()):
        if "140kVp" in group.name:
            continue
        seg_dir = group / "segmentation"
        nii_dir = group / "nifti"
        series = sorted(p.name for p in nii_dir.iterdir() if p.is_dir())
        if not series:
            continue
        base = next((s for s in series if "_s" not in s.split("recon")[-1]), series[0])
        fixed_seg_path = seg_dir / base / "seg.nii.gz"
        manifest_path = reg / group.name / "batch_manifest.json"
        if not fixed_seg_path.exists() or not manifest_path.exists():
            continue
        fixed_seg = sitk.ReadImage(str(fixed_seg_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("pairs", []):
            moving_name = item["series"]
            pair_dir = reg / group.name / f"{group.name}__{moving_name}"
            tfm_path = pair_dir / "moving_to_fixed.tfm"
            seg_path = seg_dir / moving_name / "seg.nii.gz"
            if not tfm_path.exists() or not seg_path.exists():
                continue
            qc_path = pair_dir / "registration_qc.json"
            method = None
            if qc_path.exists():
                method = json.loads(qc_path.read_text(encoding="utf-8")).get("registration_method")
            rows.extend(kidney_pair_rows(
                fixed_seg,
                sitk.ReadImage(str(seg_path)),
                sitk.ReadTransform(str(tfm_path)),
                dataset,
                group.name,
                moving_name,
                method,
            ))
    print(f"[kidney] {dataset}: {len(rows)} satir")
    return rows


def read_crossdate():
    rows = []
    for dataset, base in CROSSDATE:
        if not base.exists():
            continue
        for mc in base.glob("*/metrics.csv"):
            try:
                df = pd.read_csv(mc)
            except Exception:
                continue
            for _, r in df.iterrows():
                organ = organ_of(r["label"])
                if not organ:
                    continue
                b, a = float(r["before_dice"]), float(r["after_dice"])
                delta = a - b
                rows.append({
                    "Dataset": dataset, "Organ": organ,
                    "Before": round(b, 4), "After": round(a, 4), "Delta": round(delta, 4),
                })
    return rows


def read_crossdate_kidneys():
    rows = []
    for dataset, base in CROSSDATE:
        if not base.exists():
            print(f"[cross-kidney] atlandi (yok): {base}")
            continue
        for pair_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            state_path = pair_dir / "pilot_state.json"
            tfm_path = pair_dir / "moving_to_fixed.tfm"
            if not state_path.exists() or not tfm_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                fixed_seg = sitk.ReadImage(state["fixed_seg"])
                moving = sitk.ReadImage(state["moving_seg"])
            except Exception as exc:  # noqa: BLE001
                print(f"[cross-kidney] okunamadi {pair_dir.name}: {exc}")
                continue
            qc_path = pair_dir / "registration_qc.json"
            method = None
            if qc_path.exists():
                method = json.loads(qc_path.read_text(encoding="utf-8")).get("registration_method")
            rows.extend(kidney_pair_rows(
                fixed_seg,
                moving,
                sitk.ReadTransform(str(tfm_path)),
                dataset,
                pair_dir.parent.name,
                pair_dir.name.split("__")[-1],
                method,
            ))
    print(f"[cross-kidney] toplam: {len(rows)} satir")
    return rows


def summarize(rows, organs):
    out = []
    for dataset in sorted({r["Dataset"] for r in rows}):
        for organ in organs:
            vals = [r for r in rows if r["Dataset"] == dataset and r["Organ"] == organ]
            if not vals:
                continue
            deltas = [r["Delta"] for r in vals]
            out.append({
                "Dataset": dataset, "Organ": organ, "N": len(vals),
                "İyileşti": sum(1 for d in deltas if d > TOL),
                "Kötüleşti": sum(1 for d in deltas if d < -TOL),
                "Değişmedi": sum(1 for d in deltas if abs(d) <= TOL),
                "Before (medyan)": round(float(np.median([r["Before"] for r in vals])), 4),
                "After (medyan)": round(float(np.median([r["After"] for r in vals])), 4),
                "Delta (medyan)": round(float(np.median(deltas)), 4),
            })
    return out


def write_csv(kidney_rows, cross_kidney_rows):
    rows = []
    for scope, data in (("test-retest", kidney_rows), ("cross-date", cross_kidney_rows)):
        for row in data:
            rows.append({"Scope": scope, **row})
    if not rows:
        print("[csv] satir yok")
        return
    df = pd.DataFrame(rows)
    columns = [
        "Scope", "Dataset", "Group", "Pair", "Organ",
        "Dice_before", "Dice_after", "Dice_delta",
        "ASSD_before", "ASSD_after", "MSSD_before", "MSSD_after",
        "HD95_before", "HD95_after",
        "Durum", "Method",
    ]
    df = df[[c for c in columns if c in df.columns]]
    df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"[OK] {CSV_OUT} ({len(df)} satir)")


def write_xlsx(rows, kidney_rows, cross_rows, cross_kidney_rows):
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
    ws["A1"] = "TEST-RETEST: REGISTRATION SONRASI İYİLEŞEN / KÖTÜLEŞEN"
    ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
    ws["A2"] = "Eşik: |Δ| ≤ 0.0005 → Değişmedi. Spleen/aorta metrics.csv; böbrekler kayıtlı transformdan (Dice)."
    ws["A2"].font = Font(italic=True, size=9)
    cols = ["Dataset", "Organ", "N", "İyileşti", "Kötüleşti", "Değişmedi",
            "Before (medyan)", "After (medyan)", "Delta (medyan)"]
    head(ws, 4, cols)
    summary = summarize(
        rows + kidney_rows,
        ["spleen", "aorta", "kidney_right", "kidney_left"],
    )
    r = 5
    for s in summary:
        for c, v in enumerate([s[k] for k in cols], 1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c > 1 else "left")
            if c == 9 and isinstance(v, (int, float)):
                cell.fill = green if v >= 0 else red
        r += 1
    ws.column_dimensions["A"].width = 22
    for c in range(2, 10):
        ws.column_dimensions[get_column_letter(c)].width = 14
    ws.freeze_panes = "B5"

    def add_detail(name, data, columns):
        ws = wb.create_sheet(name)
        ws["A1"] = name
        ws["A1"].font = Font(bold=True, size=12, color="1F4E78")
        head(ws, 3, columns)
        for r, d in enumerate(data, 4):
            for c, k in enumerate(columns, 1):
                cell = ws.cell(row=r, column=c, value=d.get(k))
                cell.font = Font(size=9)
                cell.alignment = Alignment(horizontal="center" if c >= 5 else "left")
                if k in ("Delta", "Dice_delta") and isinstance(d.get(k), (int, float)):
                    cell.fill = green if d[k] >= 0 else red
                if k == "Durum":
                    if d[k] == "İyileşti":
                        cell.font = Font(size=9, bold=True, color="006100")
                    elif d[k] == "Kötüleşti":
                        cell.font = Font(size=9, bold=True, color="9C0006")
        widths = {
            "A": 18, "B": 48, "C": 46, "D": 16, "E": 11, "F": 11, "G": 11,
            "H": 11, "I": 11, "J": 11, "K": 11, "L": 12, "M": 24, "N": 24,
        }
        for col, width in widths.items():
            if col <= get_column_letter(len(columns)):
                ws.column_dimensions[col].width = width
        ws.freeze_panes = "E4"

    add_detail(
        "Spleen_Aorta",
        rows,
        ["Dataset", "Group", "Pair", "Organ", "Before", "After", "Delta", "Durum", "Method"],
    )
    add_detail(
        "Bobrek",
        kidney_rows,
        ["Dataset", "Group", "Pair", "Organ", "Dice_before", "Dice_after", "Dice_delta",
         "ASSD_before", "ASSD_after", "MSSD_before", "MSSD_after",
         "HD95_before", "HD95_after", "Durum", "Method"],
    )
    add_detail(
        "CrossDate_Bobrek",
        cross_kidney_rows,
        ["Dataset", "Group", "Pair", "Organ", "Dice_before", "Dice_after", "Dice_delta",
         "ASSD_before", "ASSD_after", "MSSD_before", "MSSD_after",
         "HD95_before", "HD95_after", "Durum", "Method"],
    )

    ws_c = wb.create_sheet("CrossDate")
    ws_c["A1"] = "CROSS-DATE (farklı tarih): registration'ın gerçek faydası"
    ws_c["A1"].font = Font(bold=True, size=12, color="1F4E78")
    cols_c = ["Dataset", "Organ", "N", "İyileşti", "Kötüleşti", "Değişmedi",
              "Before (medyan)", "After (medyan)", "Delta (medyan)"]
    head(ws_c, 3, cols_c)
    cross_all = cross_rows + cross_kidney_rows
    for r, s in enumerate(
        summarize(cross_all, ["spleen", "aorta", "kidney_right", "kidney_left"]), 4
    ):
        for c, k in enumerate(cols_c, 1):
            cell = ws_c.cell(row=r, column=c, value=s[k])
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c > 1 else "left")
            if k == "Delta (medyan)" and isinstance(s[k], (int, float)):
                cell.fill = green if s[k] >= 0 else red
    ws_c.column_dimensions["A"].width = 22
    for c in range(2, 10):
        ws_c.column_dimensions[get_column_letter(c)].width = 14
    ws_c.freeze_panes = "B4"

    ws_not = wb.create_sheet("Not")
    ws_not["A1"] = "YORUM"
    ws_not["A1"].font = Font(bold=True, size=13, color="1F4E78")
    notes = [
        "1) Test-retest verisi zaten hizalı; registration sonrası çoğu pair'de |Δ| < 0.005.",
        "2) Sistematik büyük düşüş yok; bazı pair'lerde ±0.01–0.02 oynama var (resampling/partial volume).",
        "3) 3 mm ve ince aorta'da düşüş daha sık; 140 kVp'de bazı pair'lerde hafif artış var.",
        "4) Öğrencinin evaluator'ında Ref crop'a, Target tam grid'e resample edildiği için identity'de",
        "   bile ~0.001–0.003 yapay fark oluşabiliyor; bu tablo metrics.csv (tam grid) kullanır.",
        "5) Böbrekler 120 kVp PCCT ve SEP'te var; 140 kVp maskelerinde böbrek etiketi yok.",
        "6) Aorta phantomda normal anatomik aortadan farklı olabilir; segmentation sınırı ve",
        "   partial volume aorta Dice'ını spleen'den daha çok etkiler (aorta daha ince/tübüler).",
        "7) Böbrek ASSD/HD95 değerleri KIDNEY_METRICS_PCCT_SEP.csv içinde tam olarak verilmiştir.",
        "8) Cross-date böbrek satırları batch_to_20260805_v1.0.0 (PCCT) ve v6_skip2 (SEP) transformlarından.",
    ]
    for i, text in enumerate(notes, 3):
        ws_not.cell(row=i, column=1, value=text).font = Font(size=10)
    ws_not.column_dimensions["A"].width = 120

    wb.save(OUT)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    rows, kidney_rows = [], []
    for dataset, work, prefix in WORKS:
        rows.extend(read_spleen_aorta(work, dataset))
        kidney_rows.extend(read_kidneys(work, dataset, prefix))
    cross_rows = read_crossdate()
    cross_kidney_rows = read_crossdate_kidneys()
    write_csv(kidney_rows, cross_kidney_rows)
    write_xlsx(rows, kidney_rows, cross_rows, cross_kidney_rows)
    print(
        "spleen/aorta satir:", len(rows),
        "| bobrek satir:", len(kidney_rows),
        "| cross-date satir:", len(cross_rows),
        "| cross-date bobrek:", len(cross_kidney_rows),
    )
