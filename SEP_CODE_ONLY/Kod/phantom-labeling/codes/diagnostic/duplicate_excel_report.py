"""Generate an Excel (.xlsx) duplicate-audit report with per-section
coronal data extracted from the labeler's run.log.

Each row covers one input series and shows:
  - Output folder name (the labeled name produced by label_series.py)
  - Input file count, unique SliceLocation count, duplicate-pair count
  - Whether the duplicates are bit-identical pixel pairs (= benign re-
    anonymization) or real differing reconstructions
  - Patient ID, scanner model, series description
  - Slice thickness, reconstruction spacing (mm)
  - Spectral type (Virtual Monoenergetic Image, etc.)
  - Status verdict (CLEAN / DUPLICATE-ANONYMIZATION / DUPLICATE-DIFFERING)
  - For each section (STAR16, STAR8, INSERT1, INSERT2) the values
    rendered on the coronal overlay:
      L mm  — coronal-slab length (= span of section_classifier's
              detected extent)
      N slices — kept-window slice count
      N unique z — kept-window unique z positions
      Phys mm  — physical extent of the kept window (unique z × dz)
      midpoint z mm
      kept window z range mm

NOTHING is filtered or deleted — the pipeline still labels and exports
every file as before; this script just AUDITS the input and produces a
reviewer-friendly Excel attachment.

Usage:
    python diagnostic/duplicate_excel_report.py INPUT_ROOT OUTPUT_ROOT [-o REPORT.xlsx]

Example:
    python diagnostic/duplicate_excel_report.py C:/input C:/output \\
        -o C:/output/duplicate_audit.xlsx
"""
import os
import re
import sys
import glob
import hashlib
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pydicom
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import labeler helpers to mirror the actual output naming.
from labeling.header_extractor import (  # noqa: E402
    extract_from_file, estimate_recon_spacing, UNKNOWN,
)
from labeling.name_builder import (  # noqa: E402
    build_name, build_pcct_base, compute_recon_type,
    format_recon_spacing, is_siemens_pcct,
)


SECTION_NAMES = ["STAR16", "STAR8", "INSERT1", "INSERT2"]


# Regex for the section-trim log line. Captures every field shown on
# the coronal overlay text labels.
_SECTION_LOG_RE = re.compile(
    r"\s+(?P<sec>STAR16|STAR8|INSERT1|INSERT2): "
    r"detected\s+(?P<det_lo>-?\d+\.?\d*)\.\.(?P<det_hi>-?\d+\.?\d*)\s+mm\s+"
    r"\(span\s+(?P<span>-?\d+\.?\d*)\s+mm\),\s+"
    r"midpoint\s+z=(?P<mid>-?\d+\.?\d*)\s+mm,\s+"
    r"target\s+(?P<target_unique>\d+)\s+unique-z\s*=\s*(?P<target_total>\d+)\s+slices.*?"
    r"kept\s+(?P<kept_n>\d+)\s+slices\s*\((?P<kept_unique>\d+)\s+unique\s+z\s*=\s*"
    r"(?P<phys_mm>-?\d+\.?\d*)\s+mm\s+physical\)\s+spanning\s+"
    r"(?P<kept_lo>-?\d+\.?\d*)\.\.(?P<kept_hi>-?\d+\.?\d*)\s+mm",
    re.IGNORECASE,
)

# Older log format (before unique-z accounting). Captures whichever
# subset of fields exists so we still get something useful for old logs.
_SECTION_LOG_LEGACY_RE = re.compile(
    r"\s+(?P<sec>STAR16|STAR8|INSERT1|INSERT2): "
    r"detected\s+(?P<det_lo>-?\d+\.?\d*)\.\.(?P<det_hi>-?\d+\.?\d*)\s+mm\s+"
    r"\(span\s+(?P<span>-?\d+\.?\d*)\s+mm\),.*?"
    r"kept\s+(?P<kept_n>\d+)\s+slices",
    re.IGNORECASE,
)

_ANNOTATING_RE = re.compile(r"Annotating:\s+(\S+)")


def parse_run_log(log_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Parse run.log into ``{ output_folder_name: { section_name: {field: value} } }``.

    Only the MOST RECENT block for each output folder is kept (re-runs
    overwrite earlier blocks). Numeric values stay as strings so they
    can be written verbatim to the Excel.
    """
    if not os.path.exists(log_path):
        return {}
    result: Dict[str, Dict[str, Dict[str, str]]] = {}
    current_folder: Optional[str] = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = _ANNOTATING_RE.search(line)
            if m:
                current_folder = m.group(1).strip()
                result[current_folder] = {}
                continue
            if current_folder is None:
                continue
            m = _SECTION_LOG_RE.search(line)
            if not m:
                m = _SECTION_LOG_LEGACY_RE.search(line)
                if not m:
                    continue
            sec = m.group("sec")
            d = m.groupdict()
            result[current_folder][sec] = d
    return result


# ----------------------------------------------------------------------
# 1. SCAN: input audit (counts, hashes, tag diffs)
# ----------------------------------------------------------------------

def scan_series(series_dir: str) -> Dict[str, object]:
    files = sorted(glob.glob(os.path.join(series_dir, "**", "*.dcm"), recursive=True))
    n_files = len(files)
    out: Dict[str, object] = {
        "series_dir": series_dir,
        "n_files": n_files,
        "n_unique_z": 0,
        "duplicate_pairs": 0,
        "identical_pixel_pairs": 0,
        "diff_pairs": 0,
        "sample_diff_tags": [],
        "sample_pair": None,
        # Full per-pair record: list of dicts, one per duplicate z
        "all_pairs": [],
        "manufacturer": "",
        "model": "",
        "series_description": "",
        "image_type": [],
        "patient_id": "",
        "study_date": "",
        "z_range_mm": (None, None),
        "slice_thickness": "",
        "recon_spacing_mm": None,
        "spectral_type": "",
    }
    if not files:
        return out
    ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)
    out["manufacturer"] = str(getattr(ds0, "Manufacturer", "") or "")
    out["model"] = str(getattr(ds0, "ManufacturerModelName", "") or "")
    out["series_description"] = str(getattr(ds0, "SeriesDescription", "") or "")
    out["image_type"] = list(getattr(ds0, "ImageType", []) or [])
    out["patient_id"] = str(getattr(ds0, "PatientID", "") or "")
    out["study_date"] = str(getattr(ds0, "StudyDate", "") or "")
    out["slice_thickness"] = str(getattr(ds0, "SliceThickness", "") or "")

    sbs = getattr(ds0, "SpacingBetweenSlices", None)
    if sbs is not None:
        try:
            out["recon_spacing_mm"] = float(sbs)
        except (TypeError, ValueError):
            pass

    it_upper = [str(t).upper() for t in out["image_type"]]
    if "VMI" in it_upper or any("ME" in t and "KEV" in t for t in it_upper):
        out["spectral_type"] = "VMI (single virtual monoenergetic)"
    elif "THRESHOLD" in it_upper:
        out["spectral_type"] = "PolyE / Threshold"
    else:
        out["spectral_type"] = "Single reconstruction"

    # Bucket files by SliceLocation
    by_z: Dict[float, List[str]] = defaultdict(list)
    locs: List[float] = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            sl = getattr(ds, "SliceLocation", None)
            if sl is not None:
                z = round(float(sl), 4)
                by_z[z].append(f)
                locs.append(z)
        except Exception:
            continue
    out["n_unique_z"] = len(by_z)
    if locs:
        out["z_range_mm"] = (min(locs), max(locs))

    # Recon spacing fallback from unique z diffs
    if out["recon_spacing_mm"] is None and len(by_z) > 1:
        sorted_uz = sorted(by_z.keys())
        diffs = [sorted_uz[i + 1] - sorted_uz[i] for i in range(len(sorted_uz) - 1)]
        pos = [d for d in diffs if d > 1e-6]
        if pos:
            out["recon_spacing_mm"] = min(pos)

    sample_collected = False
    for z, paths in by_z.items():
        if len(paths) < 2:
            continue
        out["duplicate_pairs"] += 1
        try:
            ds_a = pydicom.dcmread(paths[0])
            ds_b = pydicom.dcmread(paths[1])
            h_a = hashlib.md5(ds_a.PixelData).hexdigest()
            h_b = hashlib.md5(ds_b.PixelData).hexdigest()
        except Exception:
            continue
        ident = (h_a == h_b)
        if ident:
            out["identical_pixel_pairs"] += 1
        else:
            out["diff_pairs"] += 1

        # Record every pair so the dedicated sheet can list them all.
        out["all_pairs"].append({
            "z_mm": z,
            "file_a": os.path.basename(paths[0]),
            "file_b": os.path.basename(paths[1]),
            "pixel_hash_a": h_a,
            "pixel_hash_b": h_b,
            "pixel_identical": ident,
        })

        if not sample_collected:
            a_tags = {e.tag: e for e in ds_a.iterall()}
            b_tags = {e.tag: e for e in ds_b.iterall()}
            diff_tags = []
            for tag in set(a_tags) | set(b_tags):
                ea = a_tags.get(tag)
                eb = b_tags.get(tag)
                va = str(ea.value) if ea else None
                vb = str(eb.value) if eb else None
                if va == vb:
                    continue
                name = (ea.keyword if ea else eb.keyword) or str(tag)
                diff_tags.append(name)
            out["sample_diff_tags"] = sorted(set(diff_tags))
            out["sample_pair"] = out["all_pairs"][-1]
            sample_collected = True
    return out


def classify_verdict(audit: Dict[str, object]) -> str:
    """Short verdict string for the Excel STATUS column."""
    if audit["duplicate_pairs"] == 0:
        return "CLEAN"
    meta_only = {"OriginalAttributesSequence", "AttributeModificationDateTime",
                 "ModifyingSystem", "ReasonForTheAttributeModification",
                 "SOPInstanceUID"}
    non_meta = [t for t in audit["sample_diff_tags"] if t not in meta_only]
    if audit["identical_pixel_pairs"] > 0 and not non_meta:
        return "DUPLICATE (re-anonymization, harmless)"
    if audit["diff_pairs"] > 0:
        return "DUPLICATE-DIFFERING (investigate)"
    return f"DUPLICATE (differs in: {', '.join(non_meta[:3])})"


# ----------------------------------------------------------------------
# 2. NAME: predict the labeler's output folder name for this input
# ----------------------------------------------------------------------

def predict_output_folder_name(series_dir: str) -> Optional[str]:
    """Mirror the labeler's naming so we can find the matching output
    folder for an input series."""
    dcms = sorted(glob.glob(os.path.join(series_dir, "**", "*.dcm"),
                            recursive=True))
    if not dcms:
        return None
    meta = extract_from_file(dcms[0])
    if meta.recon_spacing == UNKNOWN and len(dcms) > 1:
        est = estimate_recon_spacing(dcms)
        if est is not None and est > 0:
            meta.recon_spacing = f"{est:.4f}".rstrip("0").rstrip(".")
    recon_token = format_recon_spacing(meta.recon_spacing)
    if is_siemens_pcct(meta):
        recon_type = compute_recon_type(meta.image_type, meta.image_comments)
        base = build_pcct_base(meta)
        # take_idx is assigned across series — for the audit we just look
        # for an output folder STARTING with this prefix and ending with
        # the recon_token. If multiple takes match, we pick the lowest.
        return f"{base}_{recon_type}_TAKEPREFIX_{recon_token}"
    base_name = build_name(meta) + "_" + recon_token
    return base_name


def find_output_folder(predicted: str, output_root: str) -> Optional[str]:
    if predicted is None:
        return None
    if "_TAKEPREFIX_" in predicted:
        prefix, suffix = predicted.split("_TAKEPREFIX_", 1)
        candidates = [
            d for d in os.listdir(output_root)
            if os.path.isdir(os.path.join(output_root, d))
            and d.startswith(prefix + "_take")
            and d.endswith("_" + suffix)
        ]
        if not candidates:
            return None
        # Lowest take number first
        candidates.sort()
        return os.path.join(output_root, candidates[0])
    full = os.path.join(output_root, predicted)
    return full if os.path.isdir(full) else None


def find_coronal_png(output_folder: str) -> Optional[str]:
    if not output_folder:
        return None
    p = os.path.join(output_folder, "coronal", "coronal.png")
    return p if os.path.exists(p) else None


# ----------------------------------------------------------------------
# 3. EXCEL: build the workbook
# ----------------------------------------------------------------------

HEADER_FILL = PatternFill(start_color="FF1F4E78", end_color="FF1F4E78", fill_type="solid")
HEADER_FONT = Font(color="FFFFFFFF", bold=True, size=11)
CLEAN_FILL = PatternFill(start_color="FFD9F2D9", end_color="FFD9F2D9", fill_type="solid")
DUP_FILL = PatternFill(start_color="FFFFF2CC", end_color="FFFFF2CC", fill_type="solid")
WARN_FILL = PatternFill(start_color="FFF8CBAD", end_color="FFF8CBAD", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style="thin", color="FFAAAAAA"),
    right=Side(style="thin", color="FFAAAAAA"),
    top=Side(style="thin", color="FFAAAAAA"),
    bottom=Side(style="thin", color="FFAAAAAA"),
)


def _fmt_num(v: Optional[str], suffix: str = "") -> str:
    if v in (None, "", "None"):
        return ""
    try:
        f = float(v)
        if abs(f - round(f)) < 1e-6:
            s = str(int(round(f)))
        else:
            s = f"{f:g}"
    except (TypeError, ValueError):
        s = str(v)
    return s + suffix


def write_xlsx(audits: List[Dict[str, object]],
               input_to_output: Dict[str, Optional[str]],
               log_data: Dict[str, Dict[str, Dict[str, str]]],
               output_path: str) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Audit"

    # Base columns (series + duplicate audit info, including a
    # REPRESENTATIVE pair's pixel hashes so reviewers can see the
    # evidence at a glance without opening the dedicated pairs sheet).
    base_columns = [
        ("Input series", 18),
        ("Output folder", 56),
        ("Patient ID", 12),
        ("Study Date", 11),
        ("Scanner", 26),
        ("Series Description", 32),
        ("Spectral type", 28),
        ("Slice thickness (mm)", 16),
        ("Recon spacing (mm)", 16),
        ("Input files", 10),
        ("Unique z", 10),
        ("Dup pairs", 10),
        ("Identical-pixel pairs", 16),
        ("Differing-pixel pairs", 16),
        ("Verdict", 36),
        ("Sample pair z (mm)", 14),
        ("Sample file A", 50),
        ("Sample file B", 50),
        ("Sample pixel hash A (MD5)", 36),
        ("Sample pixel hash B (MD5)", 36),
        ("Sample pixels identical", 14),
        ("Sample diff tags", 40),
    ]
    # Per-section columns (6 fields per section)
    section_columns: List[Tuple[str, int]] = []
    for sec in SECTION_NAMES:
        section_columns.extend([
            (f"{sec} L (mm)", 12),
            (f"{sec} kept slices", 14),
            (f"{sec} kept unique z", 16),
            (f"{sec} physical (mm)", 16),
            (f"{sec} midpoint z (mm)", 16),
            (f"{sec} kept z range (mm)", 22),
        ])
    columns = base_columns + section_columns

    for col_idx, (header, width) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=True)
        cell.border = THIN_BORDER
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.freeze_panes = "C2"
    ws.row_dimensions[1].height = 42

    for i, audit in enumerate(audits):
        row = i + 2
        ws.row_dimensions[row].height = 26

        verdict = classify_verdict(audit)
        fill = (
            CLEAN_FILL if verdict == "CLEAN" else
            WARN_FILL if "INVESTIGATE" in verdict.upper() else
            DUP_FILL
        )

        recon_disp = (f"{audit['recon_spacing_mm']:.3g}"
                      if audit.get("recon_spacing_mm") is not None else "")
        output_folder_path = input_to_output.get(audit["series_dir"])
        output_folder_name = (os.path.basename(output_folder_path)
                              if output_folder_path else "")

        sp = audit.get("sample_pair") or {}
        diff_tags_repr = ", ".join(audit.get("sample_diff_tags", [])[:6]) or "(none)"
        values: List[object] = [
            os.path.basename(audit["series_dir"]),
            output_folder_name or "(no output)",
            audit["patient_id"],
            audit["study_date"],
            f"{audit['manufacturer']} / {audit['model']}",
            audit["series_description"],
            audit["spectral_type"],
            audit["slice_thickness"],
            recon_disp,
            audit["n_files"],
            audit["n_unique_z"],
            audit["duplicate_pairs"],
            audit["identical_pixel_pairs"],
            audit["diff_pairs"],
            verdict,
            sp.get("z_mm", ""),
            sp.get("file_a", ""),
            sp.get("file_b", ""),
            sp.get("pixel_hash_a", ""),
            sp.get("pixel_hash_b", ""),
            ("Yes" if sp.get("pixel_identical") else
             ("No" if sp.get("pixel_identical") is False else "")),
            diff_tags_repr if audit["duplicate_pairs"] > 0 else "(no duplicates)",
        ]

        # Per-section values from parsed log
        sec_data = log_data.get(output_folder_name, {}) if output_folder_name else {}
        for sec in SECTION_NAMES:
            d = sec_data.get(sec, {})
            kept_lo = d.get("kept_lo")
            kept_hi = d.get("kept_hi")
            kept_range = ""
            if kept_lo is not None and kept_hi is not None:
                try:
                    a = min(float(kept_lo), float(kept_hi))
                    b = max(float(kept_lo), float(kept_hi))
                    kept_range = f"{a:.1f} .. {b:.1f}"
                except (TypeError, ValueError):
                    kept_range = f"{kept_lo} .. {kept_hi}"
            values.extend([
                _fmt_num(d.get("span")),
                _fmt_num(d.get("kept_n")),
                _fmt_num(d.get("kept_unique")),
                _fmt_num(d.get("phys_mm")),
                _fmt_num(d.get("mid")),
                kept_range,
            ])

        for col_idx, v in enumerate(values, start=1):
            cell = ws.cell(row=row, column=col_idx, value=v)
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            cell.border = THIN_BORDER

    # --- Duplicate Pairs sheet (one row per pair across all series) ---
    ws_pairs = wb.create_sheet("Duplicate Pairs")
    pair_cols = [
        ("Input series", 18),
        ("Output folder", 56),
        ("SliceLocation (mm)", 16),
        ("File A", 56),
        ("File B", 56),
        ("Pixel hash A (MD5)", 36),
        ("Pixel hash B (MD5)", 36),
        ("Pixels identical", 14),
    ]
    for col_idx, (header, width) in enumerate(pair_cols, start=1):
        c = ws_pairs.cell(row=1, column=col_idx, value=header)
        c.fill = HEADER_FILL
        c.font = HEADER_FONT
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
        c.border = THIN_BORDER
        ws_pairs.column_dimensions[get_column_letter(col_idx)].width = width
    ws_pairs.freeze_panes = "C2"
    ws_pairs.row_dimensions[1].height = 32
    pair_row = 2
    for audit in audits:
        out_path = input_to_output.get(audit["series_dir"])
        out_name = os.path.basename(out_path) if out_path else ""
        for p in audit.get("all_pairs", []):
            ident = p.get("pixel_identical", False)
            pf = CLEAN_FILL if ident else WARN_FILL
            cells = [
                os.path.basename(audit["series_dir"]),
                out_name,
                p["z_mm"],
                p["file_a"],
                p["file_b"],
                p["pixel_hash_a"],
                p["pixel_hash_b"],
                "Yes" if ident else "No",
            ]
            for col_idx, v in enumerate(cells, start=1):
                cell = ws_pairs.cell(row=pair_row, column=col_idx, value=v)
                cell.fill = pf
                cell.alignment = Alignment(vertical="center", wrap_text=True)
                cell.border = THIN_BORDER
            pair_row += 1
    if pair_row == 2:
        ws_pairs.cell(row=2, column=1,
                      value="(no duplicate pairs in any series)")

    # --- Summary sheet ---
    ws2 = wb.create_sheet("Summary")
    n_clean = sum(1 for a in audits if a["duplicate_pairs"] == 0)
    n_dup = sum(1 for a in audits if a["duplicate_pairs"] > 0)
    n_anon = sum(
        1 for a in audits
        if a["duplicate_pairs"] > 0
        and "re-anonymization" in classify_verdict(a)
    )

    summary_rows = [
        ("DICOM Duplicate-Slice Audit Summary", ""),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Total series audited", len(audits)),
        ("Clean (no duplicates)", n_clean),
        ("With duplicates", n_dup),
        ("Of which: harmless re-anonymization duplicates", n_anon),
        ("", ""),
        ("METHODOLOGY", ""),
        ("1. Group input files by SliceLocation", ""),
        ("2. For buckets with >1 file, hash PixelData", ""),
        ("3. Diff DICOM tags; if only OriginalAttributesSequence + ", ""),
        ("   AttributeModificationDateTime differ -> harmless duplicate", ""),
        ("", ""),
        ("PIPELINE BEHAVIOUR", ""),
        ("- No files are filtered out; full pipeline runs as before", ""),
        ("- label_series.py counts UNIQUE z positions so the labeled", ""),
        ("  physical volume is correct regardless of duplicate count", ""),
        ("", ""),
        ("PIXEL HASH EVIDENCE", ""),
        ("- Main 'Audit' sheet shows a representative pair (file A,", ""),
        ("  file B, MD5 hash A, MD5 hash B, identical Y/N) per series.", ""),
        ("- 'Duplicate Pairs' sheet lists EVERY duplicate pair across", ""),
        ("  all series with both files and both MD5 hashes — reviewers", ""),
        ("  can verify any single pair (e.g. spot-check 10 random rows", ""),
        ("  in Excel and confirm hash A == hash B for each).", ""),
        ("", ""),
        ("PER-SECTION COLUMN MEANINGS", ""),
        ("L (mm)", "Coronal-slab length (= span of section_classifier's "
                   "detected extent for that section)"),
        ("kept slices", "Number of raw DICOM files kept after the inner-window trim"),
        ("kept unique z", "Number of distinct physical z positions kept (= kept "
                          "slices / slices-per-z multiplicity)"),
        ("physical (mm)", "True physical extent of the kept window "
                           "(= kept unique z * recon spacing). 30 mm for all "
                           "protocols in this dataset."),
        ("midpoint z (mm)", "Signal-weighted z centroid the kept window is centred on"),
        ("kept z range (mm)", "Physical z range of the kept window"),
    ]
    for row_idx, (k, v) in enumerate(summary_rows, start=1):
        ws2.cell(row=row_idx, column=1, value=k).font = Font(bold=(row_idx <= 6))
        ws2.cell(row=row_idx, column=2, value=v)
    ws2.column_dimensions["A"].width = 60
    ws2.column_dimensions["B"].width = 50

    wb.save(output_path)
    print(f"Wrote Excel audit to: {output_path}")


# ----------------------------------------------------------------------
# 4. CLI
# ----------------------------------------------------------------------

def run_audit(input_root: str, output_root: str,
              xlsx_path: Optional[str] = None,
              log_path: Optional[str] = None,
              progress: bool = True) -> str:
    """Programmatic entry point.

    Scans ``input_root`` for series subfolders, matches each to its
    labeled folder under ``output_root``, parses the labeler's
    ``run.log`` for per-section coronal data, and writes the audit
    workbook to ``xlsx_path`` (default: ``<output_root>/duplicate_audit.xlsx``).

    Returns the path of the written xlsx.
    Raises any underlying I/O / pydicom exception — callers should
    wrap the call in try/except if the audit failure must not crash
    the parent pipeline.
    """
    if xlsx_path is None:
        xlsx_path = os.path.join(output_root, "duplicate_audit.xlsx")
    if log_path is None:
        log_path = os.path.join(output_root, "run.log")

    # 1) Enumerate input series
    series_dirs: List[str] = []
    for entry in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, entry)
        if os.path.isdir(path) and glob.glob(
                os.path.join(path, "**", "*.dcm"), recursive=True):
            series_dirs.append(path)
    if not series_dirs:
        series_dirs = [input_root]

    if progress:
        print(f"Auditing {len(series_dirs)} series ...")
    audits = []
    input_to_output: Dict[str, Optional[str]] = {}
    for sd in series_dirs:
        if progress:
            print(f"  -> {os.path.basename(sd)}")
        audits.append(scan_series(sd))
        predicted = predict_output_folder_name(sd)
        match = find_output_folder(predicted, output_root)
        input_to_output[sd] = match

    log_data = parse_run_log(log_path)
    if not log_data and progress:
        print(f"NOTE: no labeler log found at {log_path}. Per-section "
              f"columns will be empty.")

    write_xlsx(audits, input_to_output, log_data, xlsx_path)
    return xlsx_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_root", help="Directory with one or more input series subfolders")
    p.add_argument("output_root", help="Labeled output root (where labeled series folders + coronal/coronal.png live)")
    p.add_argument("-o", "--output", default=None,
                   help="Output xlsx (default: <output_root>/duplicate_audit.xlsx)")
    p.add_argument("--log", default=None,
                   help="Path to the labeler's run.log (default: <output_root>/run.log)")
    args = p.parse_args()
    run_audit(args.input_root, args.output_root,
              xlsx_path=args.output, log_path=args.log, progress=True)


if __name__ == "__main__":
    main()
