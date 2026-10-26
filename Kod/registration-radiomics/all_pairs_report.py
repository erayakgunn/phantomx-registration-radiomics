"""Generate an all-pairs pairwise registration report (Excel) from registered segmentations.

Methodology:
- same-session pairs      -> identity comparison on a session-local reference grid
- different-session pairs -> fixed-reference registered masks

Metrics: Dice, ASSD (mm), HD95 (mm), clDice.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import SimpleITK as sitk
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy import ndimage
from skimage.morphology import skeletonize

ORGANS = [("Aorta", 52), ("Spleen", 1)]
MARGIN = 15
STRUCTURE = ndimage.generate_binary_structure(3, 1)


def dice(a, b):
    denom = int(a.sum() + b.sum())
    return 0.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def prepare(mask, spacing):
    surface = mask & ~ndimage.binary_erosion(mask, STRUCTURE)
    if surface.any():
        dt = ndimage.distance_transform_edt(~surface, sampling=spacing).astype(np.float32)
        skel = skeletonize(mask)
    else:
        dt = np.zeros(mask.shape, dtype=np.float32)
        skel = np.zeros_like(mask)
    return {"surface": surface, "dt": dt, "skel": skel, "vol": int(mask.sum())}


def pair_metrics(mask_a, mask_b, prep_a, prep_b):
    if prep_a["surface"].any() and prep_b["surface"].any():
        dist = np.concatenate([prep_b["dt"][prep_a["surface"]], prep_a["dt"][prep_b["surface"]]])
        assd = float(dist.mean())
        hd95 = float(np.percentile(dist, 95))
    else:
        assd = hd95 = float("nan")
    if prep_a["skel"].any() and prep_b["skel"].any():
        tprec = float((prep_a["skel"] & mask_b).sum() / prep_a["skel"].sum())
        trec = float((prep_b["skel"] & mask_a).sum() / prep_b["skel"].sum())
        cld = 0.0 if tprec + trec == 0 else 2 * tprec * trec / (tprec + trec)
    else:
        cld = float("nan")
    return assd, hd95, cld


def crop_bbox(array, label, margin):
    idx = np.where(array == label)
    if len(idx[0]) == 0:
        return None
    z0, z1 = max(0, int(idx[0].min()) - margin), int(idx[0].max()) + margin + 1
    y0, y1 = max(0, int(idx[1].min()) - margin), int(idx[1].max()) + margin + 1
    x0, x1 = max(0, int(idx[2].min()) - margin), int(idx[2].max()) + margin + 1
    return (z0, z1, y0, y1, x0, x1)


def build_series(labeled, batch, session_suffix, fixed_day, fixed_series, exclude):
    by_day = defaultdict(list)
    for path in batch.iterdir():
        match = re.match(r"^(?P<day>\d{8})_(?P<session>.+?)__(?P<series>.+)$", path.name)
        if not path.is_dir() or not match:
            continue
        day = match.group("day")
        series = match.group("series")
        if series in exclude:
            continue
        sort_match = re.search(r"_s(\d+)$", series) or re.search(r"_take(\d+)$", series)
        sort_key = int(sort_match.group(1)) if sort_match else 0
        by_day[day].append((sort_key, path, series))
    result = []
    for day, entries in sorted(by_day.items()):
        offset = 1 if day == fixed_day else 0
        for position, (_, batch_path, series) in enumerate(sorted(entries, key=lambda x: x[0])):
            result.append({"id": f"{day}_t{position + 1 + offset}", "day": day,
                           "take": position + 1 + offset, "series": series,
                           "batch": batch_path, "is_fixed": False})
    result.append({"id": f"{fixed_day}_t1", "day": fixed_day, "take": 1,
                   "series": fixed_series, "batch": None, "is_fixed": True})
    result.sort(key=lambda item: (item["day"], item["take"]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--session-suffix", required=True)
    ap.add_argument("--fixed-day", required=True)
    ap.add_argument("--fixed-series", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--exclude-series", nargs="*", default=[])
    args = ap.parse_args()

    labeled, batch = Path(args.labeled), Path(args.batch)
    fixed_day_dir = labeled / f"{args.fixed_day}_{args.session_suffix}"
    fixed_seg = sitk.ReadImage(str(fixed_day_dir / "segmentation" / args.fixed_series / "seg.nii.gz"))
    fixed_array = sitk.GetArrayFromImage(fixed_seg)
    spacing = np.array(fixed_seg.GetSpacing(), dtype=float)

    series = build_series(labeled, batch, args.session_suffix, args.fixed_day, args.fixed_series,
                          set(args.exclude_series))
    n = len(series)
    print(f"series={n} pairs={n*(n-1)//2}", flush=True)
    by_day = defaultdict(list)
    for i, item in enumerate(series):
        by_day[item["day"]].append(i)

    def native_path(item):
        return labeled / f"{item['day']}_{args.session_suffix}" / "segmentation" / item["series"] / "seg.nii.gz"

    data = {}
    for organ_name, label in ORGANS:
        fixed_bbox = crop_bbox(fixed_array, label, MARGIN)
        print(f"[{organ_name}] fixed crop {fixed_bbox}", flush=True)
        store = {}
        for day, idxs in sorted(by_day.items()):
            ref_idx = next((i for i in idxs if series[i]["is_fixed"]), idxs[0])
            ref_item = series[ref_idx]
            ref_seg = fixed_seg if ref_item["is_fixed"] else sitk.ReadImage(str(native_path(ref_item)))
            ref_array = sitk.GetArrayFromImage(ref_seg)
            local_bbox = crop_bbox(ref_array, label, MARGIN) or fixed_bbox
            lz0, lz1, ly0, ly1, lx0, lx1 = local_bbox
            fz0, fz1, fy0, fy1, fx0, fx1 = fixed_bbox
            for i in idxs:
                item = series[i]
                if item["is_fixed"]:
                    native = fixed_seg
                    after = fixed_array
                else:
                    native = sitk.ReadImage(str(native_path(item)))
                    after = sitk.GetArrayFromImage(sitk.ReadImage(str(item["batch"] / "seg_registered.nii.gz")))
                identity_fixed = sitk.GetArrayFromImage(sitk.Resample(
                    native, fixed_seg, sitk.Transform(3, sitk.sitkIdentity),
                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
                ))
                local = sitk.GetArrayFromImage(sitk.Resample(
                    native, ref_seg, sitk.Transform(3, sitk.sitkIdentity),
                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
                ))
                before_mask = identity_fixed[fz0:fz1, fy0:fy1, fx0:fx1] == label
                local_mask = local[lz0:lz1, ly0:ly1, lx0:lx1] == label
                after_mask = after[fz0:fz1, fy0:fy1, fx0:fx1] == label
                store[i] = {"before": before_mask, "local": local_mask, "after": after_mask,
                            "after_prep": prepare(after_mask, spacing)}
            if (len(store)) % 20 == 0:
                print(f"  [{organ_name}] {len(store)}/{n}", flush=True)
        data[organ_name] = store

    local_cache = {}

    def local_prep(organ_name, i):
        key = (organ_name, i)
        if key not in local_cache:
            local_cache[key] = prepare(data[organ_name][i]["local"], spacing)
        return local_cache[key]

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = series[i], series[j]
            same = si["day"] == sj["day"]
            row = {"Series A": si["id"], "Series B": sj["id"], "Session A": si["day"],
                   "Session B": sj["day"], "Same session": "yes" if same else "no",
                   "Same irradiation": "", "row_i": i, "row_j": j}
            for organ_name, _ in ORGANS:
                a, b = data[organ_name][i], data[organ_name][j]
                if same:
                    ma, mb = a["local"], b["local"]
                    pa, pb = local_prep(organ_name, i), local_prep(organ_name, j)
                    before = after = dice(ma, mb)
                else:
                    ma, mb = a["after"], b["after"]
                    pa, pb = a["after_prep"], b["after_prep"]
                    before = dice(a["before"], b["before"])
                    after = dice(a["after"], b["after"])
                assd, hd95, cld = pair_metrics(ma, mb, pa, pb)
                vol_ratio = float(pa["vol"] / pb["vol"]) if pb["vol"] else float("nan")
                com_a = np.array(ndimage.center_of_mass(ma), dtype=float) if ma.any() else np.full(3, np.nan)
                com_b = np.array(ndimage.center_of_mass(mb), dtype=float) if mb.any() else np.full(3, np.nan)
                cdiff = float(np.linalg.norm((com_a - com_b) * spacing))
                row[f"{organ_name} before"] = before
                row[f"{organ_name} after"] = after
                row[f"{organ_name} gain"] = after - before
                row[f"{organ_name} ASSD"] = assd
                row[f"{organ_name} HD95"] = hd95
                row[f"{organ_name} clDice"] = cld
                row[f"{organ_name} vol_ratio"] = vol_ratio
                row[f"{organ_name} centroid_diff"] = cdiff
            pairs.append(row)
        if (i + 1) % 10 == 0:
            print(f"  pairs {i+1}/{n}", flush=True)
    print(f"computed {len(pairs)} pairs", flush=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append([f"PhantomX - {args.session_suffix} - all series vs all series"])
    ws.append([f"{n} series, {len(by_day)} sessions, {len(pairs)} pairs. "
               "Same-session pairs use identity on a session-local grid; different-session pairs use "
               "fixed-reference registration. ASSD/HD95/clDice are post-alignment values."])
    ws.append([])
    ws.append(["Organ", "Metric", "Comparison", "N pairs", "Mean", "Std", "Median", "Min", "Max"])
    for organ_name, _ in ORGANS:
        for metric, key in [("Dice", "after"), ("ASSD (mm)", "ASSD"), ("HD95 (mm)", "HD95"), ("clDice", "clDice")]:
            for comp, same in [("Different sessions", False), ("Same session", True)]:
                vals = [p[f"{organ_name} {key}"] for p in pairs
                        if (p["Same session"] == "yes") == same and np.isfinite(p[f"{organ_name} {key}"])]
                if vals:
                    ws.append([organ_name, metric, comp, len(vals), round(float(np.mean(vals)), 5),
                               round(float(np.std(vals)), 5), round(float(np.median(vals)), 5),
                               round(float(np.min(vals)), 5), round(float(np.max(vals)), 5)])

    index = {}
    for p in pairs:
        index[(p["row_i"], p["row_j"])] = p
        index[(p["row_j"], p["row_i"])] = p
    for organ_name, _ in ORGANS:
        for metric, key in [("Dice", "after"), ("ASSD", "ASSD"), ("HD95", "HD95"), ("clDice", "clDice")]:
            sheet = wb.create_sheet(f"{organ_name}_{metric}_Series")
            sheet.append([f"{organ_name} - {metric} - series x series"])
            sheet.append([])
            sheet.append([""] + [s["id"] for s in series])
            for i in range(n):
                sheet.append([series[i]["id"]] + [None] * n)
                for j in range(n):
                    if i != j:
                        sheet.cell(row=4 + i, column=2 + j).value = round(float(index[(i, j)][f"{organ_name} {key}"]), 5)

    ws = wb.create_sheet("Pairs")
    ws.append(["Series A", "Series B", "Session A", "Session B", "Same session", "Same irradiation",
               "Aorta Dice", "Aorta ASSD (mm)", "Aorta HD95 (mm)", "Aorta clDice",
               "Spleen Dice", "Spleen ASSD (mm)", "Spleen HD95 (mm)", "Spleen cut by FOV (either)"])
    for p in pairs:
        ws.append([p["Series A"], p["Series B"], p["Session A"], p["Session B"], p["Same session"],
                   p["Same irradiation"], round(p["Aorta after"], 5), round(p["Aorta ASSD"], 4),
                   round(p["Aorta HD95"], 4), round(p["Aorta clDice"], 5), round(p["Spleen after"], 5),
                   round(p["Spleen ASSD"], 4), round(p["Spleen HD95"], 4), "no"])

    ws = wb.create_sheet("Improvements")
    ws.append(["Date A", "Take A", "Date B", "Take B", "Series A", "Series B", "Same session",
               "Aorta before", "Aorta after", "Aorta gain", "Spleen before", "Spleen after", "Spleen gain",
               "Aorta improved", "Spleen improved"])
    by_id = {s["id"]: s for s in series}
    for p in pairs:
        sa, sb = by_id[p["Series A"]], by_id[p["Series B"]]
        ws.append([sa["day"], f"t{sa['take']}", sb["day"], f"t{sb['take']}", p["Series A"], p["Series B"],
                   p["Same session"], round(p["Aorta before"], 5), round(p["Aorta after"], 5), round(p["Aorta gain"], 5),
                   round(p["Spleen before"], 5), round(p["Spleen after"], 5), round(p["Spleen gain"], 5),
                   "yes" if p["Aorta gain"] >= 0 else "no", "yes" if p["Spleen gain"] >= 0 else "no"])

    grouped = defaultdict(list)
    for p in pairs:
        grouped[tuple(sorted([p["Session A"], p["Session B"]]))].append(p)
    ws = wb.create_sheet("Date_Summary")
    ws.append(["Date A", "Date B", "N take pairs", "Aorta before", "Aorta after", "Aorta gain",
               "Spleen before", "Spleen after", "Spleen gain", "Aorta improved", "Spleen improved"])
    for (da, db), items in sorted(grouped.items()):
        ws.append([da, db, len(items),
                   round(float(np.mean([x["Aorta before"] for x in items])), 5),
                   round(float(np.mean([x["Aorta after"] for x in items])), 5),
                   round(float(np.mean([x["Aorta gain"] for x in items])), 5),
                   round(float(np.mean([x["Spleen before"] for x in items])), 5),
                   round(float(np.mean([x["Spleen after"] for x in items])), 5),
                   round(float(np.mean([x["Spleen gain"] for x in items])), 5),
                   sum(1 for x in items if x["Aorta gain"] >= 0),
                   sum(1 for x in items if x["Spleen gain"] >= 0)])

    ws = wb.create_sheet("Dice_Diagnostics")
    ws.append(["Series A", "Series B", "Date A", "Date B", "Same session", "Aorta Dice", "Spleen Dice",
               "Aorta centroid diff (mm)", "Aorta volume ratio", "Spleen centroid diff (mm)", "Spleen volume ratio",
               "Flag A", "Flag B", "Low Aorta (<0.95)", "Low Spleen (<0.95)", "Category", "Severity", "Suggested action"])
    for p in pairs:
        low_a = "yes" if p["Aorta after"] < 0.95 else "no"
        low_s = "yes" if p["Spleen after"] < 0.95 else "no"
        category = "low_dice" if (low_a == "yes" or low_s == "yes") else ""
        severity = "high" if min(p["Aorta after"], p["Spleen after"]) < 0.90 else ("medium" if category else "")
        ws.append([p["Series A"], p["Series B"], p["Session A"], p["Session B"], p["Same session"],
                   round(p["Aorta after"], 5), round(p["Spleen after"], 5),
                   round(p["Aorta centroid_diff"], 3), round(p["Aorta vol_ratio"], 5),
                   round(p["Spleen centroid_diff"], 3), round(p["Spleen vol_ratio"], 5),
                   "", "", low_a, low_s, category, severity, "deformable/segmentation QC" if category else ""])

    # ---- styling (mirrors the SEP all-pairs workbook) ----
    header_fill = PatternFill("solid", fgColor="2F5496")
    diag_fill = PatternFill("solid", fgColor="1F4E78")
    diff_fill = PatternFill("solid", fgColor="FCE4D6")
    same_fill = PatternFill("solid", fgColor="E2EFDA")

    def style_header(sheet, row, ncols, size=10, fill=header_fill):
        for col in range(1, ncols + 1):
            cell = sheet.cell(row=row, column=col)
            cell.fill = fill
            cell.font = Font(bold=True, color="FFFFFF", size=size)
            cell.alignment = Alignment(horizontal="center", vertical="center")

    def scale(sheet, rng, higher_better=True):
        if higher_better:
            sheet.conditional_formatting.add(rng, ColorScaleRule(
                start_type="min", start_color="FFC7CE",
                mid_type="percentile", mid_value=50, mid_color="FFEB9C",
                end_type="max", end_color="C6EFCE"))
        else:
            sheet.conditional_formatting.add(rng, ColorScaleRule(
                start_type="min", start_color="C6EFCE",
                mid_type="percentile", mid_value=50, mid_color="FFEB9C",
                end_type="max", end_color="FFC7CE"))

    ws = wb["Summary"]
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"].font = Font(size=9)
    style_header(ws, 4, 9, size=10)
    for row in range(5, ws.max_row + 1):
        comp = ws.cell(row=row, column=3).value
        fill = diff_fill if comp == "Different sessions" else same_fill
        for col in range(1, 10):
            cell = ws.cell(row=row, column=col)
            cell.fill = fill
            cell.font = Font(size=10)
    ws.freeze_panes = "A5"
    for col, width in {"A": 10, "B": 12, "C": 20, "D": 9, "E": 11, "F": 11, "G": 11, "H": 11, "I": 11}.items():
        ws.column_dimensions[col].width = width

    for organ_name, _ in ORGANS:
        for metric, key in [("Dice", "after"), ("ASSD", "ASSD"), ("HD95", "HD95"), ("clDice", "clDice")]:
            sheet = wb[f"{organ_name}_{metric}_Series"]
            sheet["A1"].font = Font(bold=True, size=12)
            ncols = sheet.max_column
            style_header(sheet, 3, ncols, size=8)
            for row in range(4, sheet.max_row + 1):
                label = sheet.cell(row=row, column=1)
                label.fill = header_fill
                label.font = Font(bold=True, color="FFFFFF", size=8)
            sheet.freeze_panes = "B4"
            sheet.column_dimensions["A"].width = 14
            for col in range(2, ncols + 1):
                sheet.column_dimensions[get_column_letter(col)].width = 6.5
            scale(sheet, f"B4:{get_column_letter(ncols)}{sheet.max_row}", metric in ("Dice", "clDice"))

    ws = wb["Pairs"]
    style_header(ws, 1, ws.max_column, size=10)
    ws.freeze_panes = "C2"
    for col in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col)].width = 16
        for row in range(2, ws.max_row + 1):
            ws.cell(row=row, column=col).font = Font(size=9)
    for col, higher in [("G", True), ("H", False), ("I", False), ("J", True),
                        ("K", True), ("L", False), ("M", False)]:
        scale(ws, f"{col}2:{col}{ws.max_row}", higher)

    ws = wb["Improvements"]
    style_header(ws, 1, ws.max_column, size=10)
    ws.freeze_panes = "C2"
    for col, width in {"A": 11, "B": 7, "C": 11, "D": 7, "E": 18, "F": 18, "G": 11,
                       "H": 11, "I": 11, "J": 10, "K": 11, "L": 11, "M": 11, "N": 10, "O": 10}.items():
        ws.column_dimensions[col].width = width
    for row in range(2, ws.max_row + 1):
        for col in range(1, ws.max_column + 1):
            ws.cell(row=row, column=col).font = Font(size=9)
    for col in "HIJKLM":
        scale(ws, f"{col}2:{col}{ws.max_row}", True)

    ws = wb["Date_Summary"]
    style_header(ws, 1, ws.max_column, size=10)
    ws.freeze_panes = "C2"
    for col, width in {"A": 11, "B": 11, "C": 12, "D": 11, "E": 11, "F": 10,
                       "G": 11, "H": 11, "I": 10, "J": 10, "K": 10}.items():
        ws.column_dimensions[col].width = width
    for row in range(2, ws.max_row + 1):
        for col in range(1, ws.max_column + 1):
            ws.cell(row=row, column=col).font = Font(size=9)
    for col in "DEFGHI":
        scale(ws, f"{col}2:{col}{ws.max_row}", True)

    ws = wb["Dice_Diagnostics"]
    style_header(ws, 1, ws.max_column, size=10, fill=diag_fill)
    ws.freeze_panes = "C2"
    ws.sheet_properties.tabColor = "F4B183"
    for col, width in {"A": 17, "B": 17, "C": 11, "D": 11, "E": 10, "F": 10, "G": 10,
                       "H": 13, "I": 13, "J": 13, "K": 13, "L": 10, "M": 10, "N": 10,
                       "O": 10, "P": 10, "Q": 10, "R": 24}.items():
        ws.column_dimensions[col].width = width
    for row in range(2, ws.max_row + 1):
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = Font(size=11)
            if col in (6, 7):
                cell.number_format = "0.0000"
            elif col in (8, 10):
                cell.number_format = "0.000"
    notes_start = ws.max_row + 2
    title_cell = ws.cell(row=notes_start, column=1, value="Dice diagnostics — notes")
    title_cell.font = Font(bold=True, size=11)
    ws.merge_cells(start_row=notes_start, start_column=1, end_row=notes_start, end_column=18)
    notes = [
        ("Same-session", "Aynı oturum çiftleri kendi oturum referans grid'inde identity ile karşılaştırılır; düşük Dice segmentasyon/şekil kaynaklıdır."),
        ("Cross-session Aorta", "Rigid centroid residual küçük; kalan fark Aorta boundary/segmentasyon farkıdır."),
        ("Cross-session Spleen", "Düşük Spleen vakalarında centroid residual daha büyük; refinement yardımcı olabilir."),
        ("QC", "QC işaretli seriler ayrı değerlendirilmeli; overlay yalnızca görsel kontrol içindir."),
    ]
    for offset, (label_text, note_text) in enumerate(notes, start=1):
        ws.cell(row=notes_start + offset, column=1, value=label_text).font = Font(bold=True, size=11)
        ws.cell(row=notes_start + offset, column=2, value=note_text).font = Font(size=11)

    wb.save(args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
