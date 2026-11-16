"""List ASSD/MSSD outliers across the current registration batches.

Uses the recomputed Metrics Reloaded-compatible metrics.csv files and flags
cases with a robust z-score > 3.5 for ASSD or MSSD (per dataset/organ).
"""

from __future__ import annotations

import json
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

OUT = Path(r"C:/Users/eray/Desktop/ASSD_MSSD_OUTLIERS.xlsx")
PCCT = Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX")
SEP = next(Path(r"C:/Users/eray/Desktop").glob("SEP -*")) / "PhantomX"
ZURICH = [
    ("Zurich PCCT", Path(r"C:/Users/eray/Desktop/zurich_pcct_work")),
    ("Zurich SEP", Path(r"C:/Users/eray/Desktop/zurich_sep_work")),
]
Z = 3.5


def datasets():
    yield "PCCT longitudinal", PCCT / "REGISTRATION" / "batch_to_20260805_v1.0.0", PCCT / "LABELED"
    yield "SEP longitudinal", SEP / "REGISTRATION" / "batch_to_20260805_v6_skip2", SEP / "LABELED"
    for name, work in ZURICH:
        for group in sorted(p for p in (work / "REGISTRATION").iterdir() if p.is_dir()):
            yield name, group, work / "LABELED"


def organ_of(label: int):
    return "spleen" if label in (1, 3) else "aorta" if label in (52, 6) else None


def volume_cm3(fixed_path: Path, organ: str) -> float:
    try:
        seg = sitk.ReadImage(str(fixed_path))
        mapping = label_map.load_label_map(fixed_path)
        label = mapping.get(organ)
        if label is None:
            return float("nan")
        st = sitk.LabelShapeStatisticsImageFilter()
        st.Execute(sitk.Cast(seg == int(label), sitk.sitkUInt8))
        if not st.HasLabel(1):
            return float("nan")
        return float(st.GetPhysicalSize(1) / 1000.0)
    except Exception:
        return float("nan")


def main():
    rows = []
    for dataset, base, lab in datasets():
        manifest_path = base / "batch_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fixed_path = lab / manifest["fixed_session"] / "segmentation" / manifest["fixed_series"] / "seg.nii.gz"
        group = manifest.get("fixed_session", base.name)
        for pair in manifest.get("pairs", []):
            pair_dir = base / f"{pair['session']}__{pair['series']}"
            metrics_path = pair_dir / "metrics.csv"
            if not metrics_path.exists():
                continue
            qc_path = pair_dir / "registration_qc.json"
            qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
            df = pd.read_csv(metrics_path)
            for _, r in df.iterrows():
                organ = organ_of(int(r["label"]))
                if organ is None:
                    continue
                rows.append({
                    "Dataset": dataset, "Group": group, "Series": pair["series"], "Organ": organ,
                    "after_assd": float(r["after_assd_mm"]), "after_mssd": float(r["after_mssd_mm"]),
                    "after_hd95": float(r["after_hd95_mm"]), "after_dice": float(r["after_dice"]),
                    "before_dice": float(r["before_dice"]),
                    "after_nsd": float(r["after_nsd_1mm"]), "after_cldice": float(r["after_cldice"]),
                    "after_centroid": float(r["after_centroid_displacement_mm"]),
                    "touches_border": bool(r.get("after_touches_border", False)),
                    "volume_cm3": volume_cm3(fixed_path, organ),
                    "method": qc.get("registration_method"),
                    "max_disp": qc.get("target_max_centroid_displacement_mm"),
                    "holdout_tre": qc.get("holdout_tre_mm"),
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("no data")
        return

    records = []
    summary = []
    for (dataset, organ), g in frame.groupby(["Dataset", "Organ"]):
        for metric in ("after_assd", "after_mssd"):
            vals = g[metric].to_numpy(dtype=float)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            scale = 1.4826 * mad
            idx = (vals - med) / scale > Z if scale > 0 else np.zeros(len(vals), dtype=bool)
            for pos in np.nonzero(idx)[0]:
                records.append((g.index[pos], metric, float((vals[pos] - med) / scale)))
            summary.append({
                "Dataset": dataset, "Organ": organ, "Metric": metric, "N": len(vals),
                "Median": round(med, 4), "MAD": round(mad, 4),
                "Outlier": int(idx.sum()), "Threshold": round(med + Z * scale, 4) if scale > 0 else None,
            })

    outlier_index = {}
    for index, metric, zval in records:
        outlier_index.setdefault(index, {"z_assd": None, "z_mssd": None})
        outlier_index[index]["z_assd" if metric == "after_assd" else "z_mssd"] = round(zval, 2)

    out = frame.loc[list(outlier_index)].copy()
    out["z_assd"] = [outlier_index[i]["z_assd"] for i in out.index]
    out["z_mssd"] = [outlier_index[i]["z_mssd"] for i in out.index]
    out["abs_z"] = out[["z_assd", "z_mssd"]].abs().max(axis=1)
    out = out.sort_values("abs_z", ascending=False)

    def reason(row) -> str:
        flags = []
        if row["touches_border"]:
            flags.append("FOV/border")
        if row["after_dice"] < 0.85:
            flags.append("Dice<0.85")
        if row["after_nsd"] < 0.80:
            flags.append("NSD@1mm<0.80")
        if row["after_cldice"] < 0.95:
            flags.append("clDice<0.95")
        if row["before_dice"] < 0.5:
            flags.append("before<0.5")
        if row["method"] == "identity" and (row["max_disp"] or 0) > 2:
            flags.append("identity + kayma")
        if "3mm" in str(row["Group"]):
            flags.append("3 mm")
        if "140kVp" in str(row["Group"]):
            flags.append("140 kVp")
        if np.isfinite(row["volume_cm3"]) and row["volume_cm3"] < 10:
            flags.append("küçük organ")
        if row["after_assd"] > 1.0:
            flags.append("ASSD>1mm")
        if row["after_mssd"] > 1.0:
            flags.append("MSSD>1mm")
        return "; ".join(flags) if flags else "belirgin bayrak yok"

    out["reason"] = out.apply(reason, axis=1)

    write_xlsx(out, summary)
    print(f"[OK] {OUT}  ({len(out)} outlier satiri)")
    print("\nEn yüksek 15:")
    for _, r in out.head(15).iterrows():
        print(f"  {r['Dataset']:18s} {r['Organ']:6s} ASSD={r['after_assd']:.3f} MSSD={r['after_mssd']:.3f} "
              f"Dice={r['after_dice']:.3f} HD95={r['after_hd95']:.2f} | {r['Series'][-45:]} | {r['reason']}")


def write_xlsx(frame, summary):
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
    ws["A1"] = "ASSD / MSSD AYKIRI ÖZETİ (robust z > 3.5)"
    ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
    cols = ["Dataset", "Organ", "Metric", "N", "Median", "MAD", "Outlier", "Threshold"]
    head(ws, 3, cols)
    for r, s in enumerate(summary, 4):
        for c, k in enumerate(cols, 1):
            cell = ws.cell(row=r, column=c, value=s[k])
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c > 2 else "left")
    for c in range(1, len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 16
    ws.freeze_panes = "C4"

    ws2 = wb.create_sheet("Aykirilar")
    ws2["A1"] = f"ASSD/MSSD AYKIRILARI ({len(frame)} satır)"
    ws2["A1"].font = Font(bold=True, size=12, color="1F4E78")
    cols2 = ["Dataset", "Group", "Series", "Organ", "after_assd", "after_mssd", "after_hd95",
             "after_dice", "before_dice", "after_nsd", "after_cldice", "after_centroid",
             "volume_cm3", "touches_border", "method", "max_disp", "holdout_tre",
             "z_assd", "z_mssd", "abs_z", "reason"]
    head(ws2, 3, cols2)
    for r, (_, d) in enumerate(frame.iterrows(), 4):
        for c, k in enumerate(cols2, 1):
            cell = ws2.cell(row=r, column=c, value=d[k])
            cell.font = Font(size=9)
            cell.alignment = Alignment(horizontal="center" if c >= 5 else "left")
            if k == "abs_z" and isinstance(d[k], (int, float)):
                cell.fill = red if d[k] >= 5 else green
    widths = {"A": 18, "B": 46, "C": 60, "D": 8}
    for col, w in widths.items():
        ws2.column_dimensions[col].width = w
    for c in range(5, len(cols2) + 1):
        ws2.column_dimensions[get_column_letter(c)].width = 12
    ws2.freeze_panes = "E4"

    ws3 = wb.create_sheet("Not")
    ws3["A1"] = "YÖNTEM / YORUM"
    ws3["A1"].font = Font(bold=True, size=13, color="1F4E78")
    notes = [
        "1) ASSD = yönlü ortalama yüzey mesafelerinin ortalaması (surface-distance).",
        "2) MSSD = pooled ortalama yüzey mesafesi.",
        "3) Robust z = (x - median) / (1.4826 * MAD); |z| > 3.5 aykırı.",
        "4) Yalnızca yüksek taraf (kötü) aykırılar listelenir; düşük ASSD/MSSD",
        "   (aynı oturum/identity) iyi olduğu için listeye alınmaz.",
        "5) Bayraklar: FOV/border, Dice<0.85, NSD<0.80, clDice<0.95, before<0.5,",
        "   identity + kayma, 3 mm, 140 kVp, küçük organ (<10 cm³), ASSD/MSSD>1 mm.",
        "6) Olası nedenler: sınır/FOV, segmentasyon, ince aorta, 3 mm partial volume,",
        "   140 kVp label/seg farkı, kayıt başarısızlığı veya identity kalması.",
    ]
    for i, text in enumerate(notes, 3):
        ws3.cell(row=i, column=1, value=text).font = Font(size=10)
    ws3.column_dimensions["A"].width = 120

    wb.save(OUT)


if __name__ == "__main__":
    main()
