"""Add kidney metrics to the PCCT all-pairs registration workbook.

The workbook's existing Aorta/Spleen values were reproduced before this script
was written:
  - same-session: identity on the session take1 grid
  - different-session: identity (before) and fixed-reference registration (after)

This script precomputes kidney crops on those grids, evaluates 3,916 pairs
with two kidneys, and appends columns/sheets to the workbook.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import SimpleITK as sitk
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX")
WB = ROOT / "REGISTRATION" / "all_pairs_report_PCCT_pairwise_registration.xlsx"
BATCH = ROOT / "REGISTRATION" / "batch_to_20260805_v1.0.0"
FIXED_DAY = "20260805"
PAD = 8
WORKERS = 8
LABELS = {"kidney_right": 2, "kidney_left": 3}
ORGAN_LABEL = {"kidney_right": "Kidney right", "kidney_left": "Kidney left"}

sys.path.insert(0, str(Path(__file__).resolve().parent / "registration-radiomics"))
from metrics import compare_label  # noqa: E402
from registration import resample_mask  # noqa: E402

NAVY = PatternFill("solid", start_color="1F4E78")


@dataclass
class Crop:
    array: np.ndarray
    z0: int
    y0: int
    x0: int


def make_crop(image: sitk.Image, label: int) -> Crop | None:
    array = sitk.GetArrayFromImage(image) == label
    if not array.any():
        return None
    idx = np.where(array)
    z0 = max(0, int(idx[0].min()) - PAD)
    y0 = max(0, int(idx[1].min()) - PAD)
    x0 = max(0, int(idx[2].min()) - PAD)
    z1 = min(array.shape[0], int(idx[0].max()) + PAD + 1)
    y1 = min(array.shape[1], int(idx[1].max()) + PAD + 1)
    x1 = min(array.shape[2], int(idx[2].max()) + PAD + 1)
    return Crop(array[z0:z1, y0:y1, x0:x1].astype(np.uint8), z0, y0, x0)


def compare_crops(a: Crop | None, b: Crop | None, reference: sitk.Image):
    if a is None or b is None:
        return None
    z0 = min(a.z0, b.z0)
    y0 = min(a.y0, b.y0)
    x0 = min(a.x0, b.x0)
    z1 = max(a.z0 + a.array.shape[0], b.z0 + b.array.shape[0])
    y1 = max(a.y0 + a.array.shape[1], b.y0 + b.array.shape[1])
    x1 = max(a.x0 + a.array.shape[2], b.x0 + b.array.shape[2])
    shape = (z1 - z0, y1 - y0, x1 - x0)
    aa = np.zeros(shape, dtype=np.uint8)
    bb = np.zeros(shape, dtype=np.uint8)
    aa[a.z0 - z0:a.z0 - z0 + a.array.shape[0],
       a.y0 - y0:a.y0 - y0 + a.array.shape[1],
       a.x0 - x0:a.x0 - x0 + a.array.shape[2]] = a.array
    bb[b.z0 - z0:b.z0 - z0 + b.array.shape[0],
       b.y0 - y0:b.y0 - y0 + b.array.shape[1],
       b.x0 - x0:b.x0 - x0 + b.array.shape[2]] = b.array
    ia = sitk.GetImageFromArray(aa)
    ia.SetSpacing(reference.GetSpacing())
    ia.SetOrigin(reference.TransformIndexToPhysicalPoint((z0, y0, x0)))
    ib = sitk.GetImageFromArray(bb)
    ib.SetSpacing(reference.GetSpacing())
    ib.SetOrigin(ia.GetOrigin())
    return compare_label(ia, ib, 1)


def series_info(pairs: pd.DataFrame, root: Path, batch: Path, suffix: str) -> dict[str, dict]:
    labels = sorted(
        set(pairs["Series A"].astype(str)) | set(pairs["Series B"].astype(str))
    )
    info = {}
    for label in labels:
        match = re.match(r"(\d{8})_t(\d+)$", label)
        if not match:
            continue
        day, take = match.group(1), int(match.group(2))
        session = root / "LABELED" / f"{day}{suffix}"
        nifti = session / "nifti"
        folders = sorted(p for p in nifti.iterdir() if p.is_dir())
        full = folders[take - 1].name
        seg = session / "segmentation" / full / "seg.nii.gz"
        tfm_matches = list(batch.glob(f"*__{full}/moving_to_fixed.tfm"))
        info[label] = {
            "day": day,
            "take": take,
            "full": full,
            "seg": seg,
            "tfm": tfm_matches[0] if tfm_matches else None,
        }
    return info


def main(args: argparse.Namespace) -> None:
    global ROOT, WB, BATCH
    ROOT = args.root
    WB = args.workbook
    BATCH = args.batch
    suffix = args.session_suffix

    pairs = pd.read_excel(WB, sheet_name="Pairs")
    info = series_info(pairs, ROOT, BATCH, suffix)
    print(f"series: {len(info)} | pairs: {len(pairs)}")

    def first_seg(day: str) -> Path:
        session = ROOT / "LABELED" / f"{day}{suffix}"
        folders = sorted(p for p in (session / "nifti").iterdir() if p.is_dir())
        return session / "segmentation" / folders[0].name / "seg.nii.gz"

    fixed_ref = sitk.ReadImage(str(first_seg(args.fixed_day)))
    identity = sitk.Transform(3, sitk.sitkIdentity)
    session_refs = {
        day: sitk.ReadImage(str(first_seg(day)))
        for day in sorted({item["day"] for item in info.values()})
    }

    t0 = time.time()
    masks = {}
    for index, (label, item) in enumerate(sorted(info.items()), 1):
        native = sitk.ReadImage(str(item["seg"]))
        sref = session_refs[item["day"]]
        session = resample_mask(native, sref, identity)
        fixed_id = resample_mask(native, fixed_ref, identity)
        if item["tfm"] is not None:
            fixed_reg = resample_mask(
                native, fixed_ref, sitk.ReadTransform(str(item["tfm"]))
            )
        else:
            fixed_reg = fixed_id
        masks[label] = {
            "session": {name: make_crop(session, value) for name, value in LABELS.items()},
            "fixed_id": {name: make_crop(fixed_id, value) for name, value in LABELS.items()},
            "fixed_reg": {name: make_crop(fixed_reg, value) for name, value in LABELS.items()},
        }
        if index % 20 == 0:
            print(f"  precompute {index}/{len(info)} ({time.time()-t0:.1f}s)")

    def one_pair(row) -> dict:
        a = str(row["Series A"])
        b = str(row["Series B"])
        same = str(row["Same session"]).strip().lower() == "yes"
        day_a = info[a]["day"]
        day_b = info[b]["day"]
        result = {"right": {}, "left": {}}
        for organ, crop_key, label in (
            ("right", "kidney_right", 2),
            ("left", "kidney_left", 3),
        ):
            if same:
                metric = compare_crops(
                    masks[a]["session"][crop_key],
                    masks[b]["session"][crop_key],
                    session_refs[day_a],
                )
                result[organ] = {"before": metric, "after": metric}
            else:
                result[organ] = {
                    "before": compare_crops(
                        masks[a]["fixed_id"][crop_key],
                        masks[b]["fixed_id"][crop_key],
                        fixed_ref,
                    ),
                    "after": compare_crops(
                        masks[a]["fixed_reg"][crop_key],
                        masks[b]["fixed_reg"][crop_key],
                        fixed_ref,
                    ),
                }
        return result

    rows = [row for _, row in pairs.iterrows()]
    cache = WB.with_name(WB.stem + "_kidney_cache.pkl")
    results = None
    if cache.exists():
        try:
            results = pickle.loads(cache.read_bytes())
        except Exception:  # noqa: BLE001
            results = None
        if results is not None and len(results) != len(rows):
            results = None
        if results is not None:
            print(f"loaded cache: {cache}")

    if results is None:
        print("evaluating pairs ...")
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(one_pair, rows))
        print(f"evaluated {len(results)} pairs in {time.time()-t1:.1f}s")
        cache.write_bytes(pickle.dumps(results))
        print(f"cache written: {cache}")

    # Prepare matrices
    book = openpyxl.load_workbook(WB)
    matrix_ws = book["Spleen_Dice_Series"]
    series_order = [
        matrix_ws.cell(row=3, column=col).value
        for col in range(2, matrix_ws.max_column + 1)
    ]
    series_order = [str(value) for value in series_order if value]
    order_index = {name: idx for idx, name in enumerate(series_order)}
    matrices = {
        (organ, metric): np.full((len(series_order), len(series_order)), np.nan)
        for organ in ("right", "left")
        for metric in ("Dice", "ASSD", "HD95", "clDice")
    }

    def metric_value(metric: dict, key: str):
        if metric is None:
            return np.nan
        mapping = {
            "Dice": "dice",
            "ASSD": "assd_mm",
            "HD95": "hd95_mm",
            "clDice": "cldice",
        }
        value = metric.get(mapping[key], np.nan)
        return float(value) if value is not None else np.nan

    for row, result in zip(rows, results):
        a = str(row["Series A"])
        b = str(row["Series B"])
        i, j = order_index[a], order_index[b]
        for organ in ("right", "left"):
            for metric in ("Dice", "ASSD", "HD95", "clDice"):
                value = metric_value(result[organ]["after"], metric)
                matrices[(organ, metric)][i, j] = value
                matrices[(organ, metric)][j, i] = value

    # Pairs columns
    ws = book["Pairs"]
    header_row = 1
    start_col = ws.max_column + 1
    new_headers = []
    for organ in ("right", "left"):
        for metric in ("Dice", "ASSD (mm)", "HD95 (mm)", "clDice"):
            new_headers.append(f"{ORGAN_LABEL['kidney_'+organ]} {metric}")
    for offset, name in enumerate(new_headers):
        cell = ws.cell(row=header_row, column=start_col + offset, value=name)
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.fill = NAVY
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for row_index, result in enumerate(results, start=2):
        for organ_index, organ in enumerate(("right", "left")):
            metric = result[organ]["after"]
            values = [
                metric_value(metric, "Dice"),
                metric_value(metric, "ASSD"),
                metric_value(metric, "HD95"),
                metric_value(metric, "clDice"),
            ]
            for offset, value in enumerate(values):
                ws.cell(row=row_index, column=start_col + organ_index * 4 + offset, value=value)
    ws.freeze_panes = "A2"

    # Improvements columns
    ws = book["Improvements"]
    start_col = ws.max_column + 1
    headers = []
    for organ in ("right", "left"):
        label = ORGAN_LABEL["kidney_" + organ]
        headers.extend([
            f"{label} before", f"{label} after", f"{label} gain", f"{label} improved",
        ])
    for offset, name in enumerate(headers):
        cell = ws.cell(row=1, column=start_col + offset, value=name)
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.fill = NAVY
    for row_index, result in enumerate(results, start=2):
        for organ_index, organ in enumerate(("right", "left")):
            before = metric_value(result[organ]["before"], "Dice")
            after = metric_value(result[organ]["after"], "Dice")
            gain = after - before if np.isfinite(before) and np.isfinite(after) else np.nan
            improved = "yes" if np.isfinite(gain) and gain > 0.0005 else "no"
            for offset, value in enumerate([before, after, gain, improved]):
                ws.cell(row=row_index, column=start_col + organ_index * 4 + offset, value=value)

    # Date summary columns
    ws = book["Date_Summary"]
    start_col = ws.max_column + 1
    headers = []
    for organ in ("right", "left"):
        label = ORGAN_LABEL["kidney_" + organ]
        headers.extend([f"{label} before", f"{label} after", f"{label} gain"])
    for offset, name in enumerate(headers):
        cell = ws.cell(row=1, column=start_col + offset, value=name)
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.fill = NAVY
    date_groups = {}
    for row, result in zip(rows, results):
        key = (str(row["Session A"]), str(row["Session B"]))
        date_groups.setdefault(key, []).append(result)
    for row_index in range(2, ws.max_row + 1):
        key = (
            str(ws.cell(row=row_index, column=1).value),
            str(ws.cell(row=row_index, column=2).value),
        )
        items = date_groups.get(key, [])
        for organ_index, organ in enumerate(("right", "left")):
            before = [metric_value(item[organ]["before"], "Dice") for item in items]
            after = [metric_value(item[organ]["after"], "Dice") for item in items]
            before = [v for v in before if np.isfinite(v)]
            after = [v for v in after if np.isfinite(v)]
            values = [
                float(np.mean(before)) if before else np.nan,
                float(np.mean(after)) if after else np.nan,
                float(np.mean(after) - np.mean(before)) if before and after else np.nan,
            ]
            for offset, value in enumerate(values):
                ws.cell(row=row_index, column=start_col + organ_index * 3 + offset, value=value)

    # Summary rows
    ws = book["Summary"]
    next_row = ws.max_row + 1
    for organ in ("right", "left"):
        label = ORGAN_LABEL["kidney_" + organ]
        for comparison, selector in (
            ("Same session", lambda r: str(r["Same session"]).lower() == "yes"),
            ("Different sessions", lambda r: str(r["Same session"]).lower() != "yes"),
        ):
            selected = [res for row, res in zip(rows, results) if selector(row)]
            for metric, key in (
                ("Dice", "Dice"), ("ASSD (mm)", "ASSD"),
                ("HD95 (mm)", "HD95"), ("clDice", "clDice"),
            ):
                values = [metric_value(item[organ]["after"], key) for item in selected]
                values = np.array([v for v in values if np.isfinite(v)], dtype=float)
                if not len(values):
                    continue
                row_values = [
                    label, metric, comparison, len(values),
                    float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    float(np.median(values)), float(values.min()), float(values.max()),
                ]
                for offset, value in enumerate(row_values, 1):
                    cell = ws.cell(row=next_row, column=offset, value=value)
                    cell.font = Font(size=9)
                next_row += 1

    # Matrices
    for organ in ("right", "left"):
        for metric in ("Dice", "ASSD", "HD95", "clDice"):
            name = f"Kidney_{organ.capitalize()}_{metric}_Series"
            if name in book.sheetnames:
                del book[name]
            sheet = book.create_sheet(name)
            sheet.cell(row=1, column=1, value=f"Kidney {organ} - {metric} - series x series")
            sheet.cell(row=1, column=1).font = Font(bold=True, size=12, color="1F4E78")
            sheet.cell(row=3, column=1, value="")
            for col, series in enumerate(series_order, 2):
                cell = sheet.cell(row=3, column=col, value=series)
                cell.font = Font(bold=True, size=8)
                cell.alignment = Alignment(horizontal="center")
            for row, series in enumerate(series_order, 4):
                cell = sheet.cell(row=row, column=1, value=series)
                cell.font = Font(bold=True, size=8)
                for col, value in enumerate(matrices[(organ, metric)][row - 4], 2):
                    sheet.cell(row=row, column=col, value=float(value) if np.isfinite(value) else None)
            sheet.freeze_panes = "B4"

    note = book.create_sheet("Kidney_Recon_Not") if "Kidney_Recon_Not" not in book.sheetnames else book["Kidney_Recon_Not"]
    note["A1"] = "BÖBREK / REKONSTRÜKSİYON NOTU"
    note["A1"].font = Font(bold=True, size=12, color="1F4E78")
    note["A3"] = (
        "Bu çalışma kitabındaki 89 serinin tamamı 1 mm'dir; 3 mm seri yoktur. "
        "1 mm vs 3 mm böbrek karşılaştırması Zurich test-retest verisinde ayrı "
        "olarak yapılmıştır (BOBREK_1MM_VS_3MM.xlsx)."
    )
    note["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    note.column_dimensions["A"].width = 120

    out_path = args.output or WB
    if out_path == WB:
        backup = WB.with_name(WB.stem + "_backup.xlsx")
        if not backup.exists():
            shutil.copy2(WB, backup)
        print(f"[backup] {backup}")
    book.save(out_path)
    print(f"[OK] {out_path}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Add kidney metrics to an all-pairs workbook")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/PhantomX"),
    )
    ap.add_argument(
        "--workbook",
        type=Path,
        default=Path(
            r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/"
            r"PhantomX/REGISTRATION/all_pairs_report_PCCT_pairwise_registration.xlsx"
        ),
    )
    ap.add_argument(
        "--batch",
        type=Path,
        default=Path(
            r"C:/Users/eray/Desktop/PCCT - Q3_Qr40f_120_100_300_1_PolyE/"
            r"PhantomX/REGISTRATION/batch_to_20260805_v1.0.0"
        ),
    )
    ap.add_argument("--fixed-day", default="20260805")
    ap.add_argument("--session-suffix", default="_PCCT_Aorta_Spleen")
    ap.add_argument("--output", type=Path, default=None)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
