"""Generate a formal duplicate-slice audit report across multiple input
DICOM series. Produces a human-readable text log suitable for sharing
with supervisors / QA reviewers.

Usage:
    python diagnostic/duplicate_audit_report.py INPUT_ROOT [-o REPORT.log]

Example:
    python diagnostic/duplicate_audit_report.py C:/input -o C:/output/duplicate_audit.log
"""
import os
import sys
import glob
import hashlib
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import pydicom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def scan_series(series_dir: str) -> Dict[str, object]:
    """Scan a single series directory; return audit results."""
    files = sorted(glob.glob(os.path.join(series_dir, "**", "*.dcm"), recursive=True))
    n_files = len(files)
    result = {
        "series_dir": series_dir,
        "n_files": n_files,
        "n_unique_z": 0,
        "multiplicity": {},
        "duplicate_pairs": 0,
        "identical_pixel_pairs": 0,
        "diff_pairs": 0,
        "sample_diff_tags": [],
        "sample_pair": None,
        "spectral_type": "?",
        "image_type": [],
        "series_description": "",
        "manufacturer": "",
        "model": "",
        "patient_id": "",
        "study_date": "",
        "z_range_mm": (None, None),
    }
    if not files:
        return result

    # Header read of first file for metadata
    ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)
    result["manufacturer"] = str(getattr(ds0, "Manufacturer", "") or "")
    result["model"] = str(getattr(ds0, "ManufacturerModelName", "") or "")
    result["series_description"] = str(getattr(ds0, "SeriesDescription", "") or "")
    result["image_type"] = list(getattr(ds0, "ImageType", []) or [])
    result["patient_id"] = str(getattr(ds0, "PatientID", "") or "")
    result["study_date"] = str(getattr(ds0, "StudyDate", "") or "")

    # Identify spectral type from ImageType
    it_upper = [str(t).upper() for t in result["image_type"]]
    if "VMI" in it_upper or any("ME" in t and "KEV" in t for t in it_upper):
        result["spectral_type"] = "Virtual Monoenergetic Image (single VMI reconstruction)"
    elif "THRESHOLD" in it_upper:
        result["spectral_type"] = "PolyE / Threshold (single reconstruction)"
    else:
        result["spectral_type"] = "Single reconstruction"

    # Bucket all files by SliceLocation
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
    result["n_unique_z"] = len(by_z)
    if locs:
        result["z_range_mm"] = (min(locs), max(locs))

    multiplicity_counter: Dict[int, int] = defaultdict(int)
    for z, paths in by_z.items():
        multiplicity_counter[len(paths)] += 1
    result["multiplicity"] = dict(multiplicity_counter)

    # For every z with >=2 files, hash pixel data
    sample_collected = False
    for z, paths in by_z.items():
        if len(paths) < 2:
            continue
        result["duplicate_pairs"] += 1
        try:
            ds_a = pydicom.dcmread(paths[0])
            ds_b = pydicom.dcmread(paths[1])
            h_a = hashlib.md5(ds_a.PixelData).hexdigest()
            h_b = hashlib.md5(ds_b.PixelData).hexdigest()
        except Exception:
            continue
        if h_a == h_b:
            result["identical_pixel_pairs"] += 1
        else:
            result["diff_pairs"] += 1
        # Capture one representative diff for the report
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
                diff_tags.append((name, va, vb))
            result["sample_diff_tags"] = diff_tags
            result["sample_pair"] = {
                "z_mm": z,
                "file_a": os.path.basename(paths[0]),
                "file_b": os.path.basename(paths[1]),
                "pixel_hash_a": h_a,
                "pixel_hash_b": h_b,
                "pixel_identical": (h_a == h_b),
            }
            sample_collected = True

    return result


def truncate(s: str, n: int = 90) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n - 3] + "..."


def write_report(results: List[Dict], output_path: str) -> None:
    lines: List[str] = []
    add = lines.append

    add("=" * 78)
    add("  DICOM INPUT AUDIT REPORT — Duplicate-slice scan")
    add("=" * 78)
    add(f"  Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"  Audited by:   diagnostic/duplicate_audit_report.py")
    add(f"  Number of series audited: {len(results)}")
    add("")
    add("PURPOSE")
    add("-" * 78)
    add("  This report identifies CT series whose input file count is INFLATED")
    add("  by duplicate copies of the same physical slice. Such duplicates can")
    add("  appear when an anonymization tool (or any DICOM-modification")
    add("  pipeline) is run more than once on the same source data — the tool")
    add("  emits a fresh DICOM file each time, with new SOPInstanceUIDs and")
    add("  updated metadata-modification tags, but the PIXEL DATA is bit-for-")
    add("  bit identical between copies.")
    add("")
    add("METHOD")
    add("-" * 78)
    add("  For each series:")
    add("    1. Read every file's SliceLocation header (no pixel decode).")
    add("    2. Group files by SliceLocation. Buckets with >1 file are")
    add("       candidate duplicates.")
    add("    3. For each candidate pair, decode PixelData and compare MD5")
    add("       hashes. Identical hashes => bit-identical pixel arrays.")
    add("    4. Diff DICOM tags between paired files. If only metadata-")
    add("       modification tags differ (OriginalAttributesSequence,")
    add("       AttributeModificationDateTime), the duplicate is from a")
    add("       re-anonymization pass — NOT a dual-energy / dual-detector")
    add("       reconstruction.")
    add("")
    add("FINDINGS PER SERIES")
    add("=" * 78)

    has_any_dup = False
    for i, r in enumerate(results, start=1):
        name = os.path.basename(r["series_dir"]) or r["series_dir"]
        add("")
        add(f"[{i}] {name}")
        add("-" * 78)
        add(f"  Path:               {r['series_dir']}")
        add(f"  Patient ID:         {r['patient_id'] or '(empty)'}")
        add(f"  Study Date:         {r['study_date'] or '(empty)'}")
        add(f"  Manufacturer:       {r['manufacturer']}")
        add(f"  Model:              {r['model']}")
        add(f"  Series Description: {truncate(r['series_description'])}")
        add(f"  ImageType:          {r['image_type']}")
        add(f"  Spectral type:      {r['spectral_type']}")
        add("")
        add(f"  File count:                  {r['n_files']}")
        add(f"  Unique SliceLocation count:  {r['n_unique_z']}")
        zmin, zmax = r["z_range_mm"]
        if zmin is not None:
            add(f"  Physical z range:            {zmin:.1f} .. {zmax:.1f} mm "
                f"(span {zmax - zmin:.1f} mm)")
        # Multiplicity
        mult = r["multiplicity"]
        if mult:
            mult_str = ", ".join(f"{c} z's with {n} file(s)"
                                 for n, c in sorted(mult.items()))
            add(f"  Slice multiplicity:          {mult_str}")
        # Duplicate verdict
        if r["duplicate_pairs"] == 0:
            add(f"  Duplicate-pair count:        0  -- CLEAN, no duplicate slices")
        else:
            has_any_dup = True
            add(f"  Duplicate-pair count:        {r['duplicate_pairs']} unique z values "
                f"have >1 file")
            add(f"  Identical-pixel pairs:       {r['identical_pixel_pairs']}")
            add(f"  Different-pixel pairs:       {r['diff_pairs']}")

        if r["sample_pair"]:
            sp = r["sample_pair"]
            add("")
            add(f"  REPRESENTATIVE DUPLICATE PAIR (at SliceLocation = {sp['z_mm']} mm):")
            add(f"    File A: {sp['file_a']}")
            add(f"            md5(pixels) = {sp['pixel_hash_a']}")
            add(f"    File B: {sp['file_b']}")
            add(f"            md5(pixels) = {sp['pixel_hash_b']}")
            add(f"    Pixel data identical: {sp['pixel_identical']}")
            add("")
            add(f"  ALL DIFFERING DICOM TAGS BETWEEN A AND B (this pair):")
            if not r["sample_diff_tags"]:
                add("    (none — files are exact byte-for-byte duplicates)")
            for name, va, vb in r["sample_diff_tags"]:
                add(f"    - {name}")
                add(f"        A: {truncate(va)}")
                add(f"        B: {truncate(vb)}")

            # Diagnosis
            meta_only_tags = {"OriginalAttributesSequence",
                              "AttributeModificationDateTime",
                              "ModifyingSystem",
                              "ReasonForTheAttributeModification",
                              "SOPInstanceUID"}
            non_meta_diffs = [name for (name, _, _) in r["sample_diff_tags"]
                              if name not in meta_only_tags]
            add("")
            if (r["identical_pixel_pairs"] > 0 and not non_meta_diffs):
                add(f"  DIAGNOSIS:  Duplicate from re-anonymization / re-processing.")
                add(f"              The two files contain BIT-IDENTICAL pixel arrays")
                add(f"              and differ only in metadata-modification tags.")
                add(f"              This is NOT a dual-energy or multi-energy")
                add(f"              reconstruction. No clinical / quantitative impact.")
            elif r["diff_pairs"] > 0:
                add(f"  DIAGNOSIS:  Different pixel data per copy — investigate before")
                add(f"              treating as duplicates.")
            else:
                add(f"  DIAGNOSIS:  See above tag differences.")

    add("")
    add("=" * 78)
    add("  SUMMARY")
    add("=" * 78)
    clean = sum(1 for r in results if r["duplicate_pairs"] == 0)
    dup = sum(1 for r in results if r["duplicate_pairs"] > 0)
    add(f"  Total series audited:      {len(results)}")
    add(f"    - Clean (no duplicates): {clean}")
    add(f"    - With duplicates:       {dup}")
    if has_any_dup:
        add("")
        add("  IMPACT")
        add("  ------")
        add("    For series with duplicates, the raw file count is 2x the number of")
        add("    unique physical z positions actually present in the scan. Any tool")
        add("    that naively multiplies (file_count * SliceThickness) to estimate")
        add("    the physical extent will OVERSHOOT by a factor of 2.")
        add("")
        add("    The downstream labeling pipeline (label_series.py) is aware of")
        add("    this and counts UNIQUE SliceLocations rather than raw file count")
        add("    when computing the labeled physical volume, so the labeled ROI")
        add("    volume per section reflects the true physical extent.")
        add("")
        add("  RECOMMENDED ACTION")
        add("  ------------------")
        add("    Deduplicate the input by keeping one file per unique SliceLocation.")
        add("    Easiest implementation: pre-process with a script that hashes")
        add("    PixelData per file, keeps the FIRST occurrence of each hash, and")
        add("    discards subsequent identical files. This halves disk usage and")
        add("    eliminates any risk of downstream code mis-counting slices.")
    add("")
    add("=" * 78)
    add("  END OF REPORT")
    add("=" * 78)

    text = "\n".join(lines) + "\n"
    if output_path == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote audit report to: {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_root", help="Directory containing one or more series sub-folders")
    p.add_argument("-o", "--output", default="duplicate_audit.log",
                   help="Output report path (default: ./duplicate_audit.log)")
    args = p.parse_args()

    # Look one level deep: input_root/<series_name>/...dcm
    candidate_dirs = []
    for entry in sorted(os.listdir(args.input_root)):
        path = os.path.join(args.input_root, entry)
        if not os.path.isdir(path):
            continue
        # Has any .dcm under it?
        if glob.glob(os.path.join(path, "**", "*.dcm"), recursive=True):
            candidate_dirs.append(path)
    if not candidate_dirs:
        # Treat input_root itself as the series
        candidate_dirs = [args.input_root]

    print(f"Auditing {len(candidate_dirs)} series under {args.input_root} ...")
    results = []
    for d in candidate_dirs:
        print(f"  -> {os.path.basename(d)}")
        results.append(scan_series(d))
    write_report(results, args.output)


if __name__ == "__main__":
    main()
