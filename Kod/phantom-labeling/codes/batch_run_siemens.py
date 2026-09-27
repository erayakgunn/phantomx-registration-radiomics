"""Run the phantom analysis pipeline on every RECON series in an
organized Siemens PCCT folder and write a summary CSV.

Skips non-recon folders (Topogram / PatientProtocol / Dose / Exam reports).
Each series' full log is written to ``<output>/_logs/<series>.log`` so
any failing series can be inspected afterwards.

Usage
-----
    python batch_run_siemens.py \
        --data-dir  <organized_folder> \
        --output-dir <annotations_folder>
"""

import argparse
import csv
import os
import sys
import time
import traceback
from typing import List


def _is_recon(name: str) -> bool:
    """True only for per-series RECON folders (contain 'VMI' token)."""
    return "_VMI" in name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root (default: same as --data-dir for unified layout)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N series (debug / smoke test)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "_logs")
    os.makedirs(logs_dir, exist_ok=True)

    series: List[str] = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and _is_recon(d)
    )
    if args.limit:
        series = series[: args.limit]

    print(f"Found {len(series)} RECON series to process")
    print(f"Output:  {output_dir}")
    print(f"Logs:    {logs_dir}")
    print()

    # Lazy imports so argparse --help stays fast.
    from utils.logger import log, setup_logger
    import logging

    summary_path = os.path.join(output_dir, "_batch_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "series", "status", "elapsed_s",
            "accepted_rois", "rejected_rois", "errors", "warnings",
            "star8_start", "star8_end", "star16_start", "star16_end",
            "insert1_start", "insert1_end", "insert2_start", "insert2_end",
            "failure_reason",
        ])
        csvf.flush()

        t_all = time.time()
        for idx, name in enumerate(series, 1):
            series_dir = os.path.join(args.data_dir, name)
            log_path = os.path.join(logs_dir, f"{name}.log")
            t0 = time.time()

            print(f"[{idx}/{len(series)}] {name}", flush=True)

            # Re-configure logger per series so each series' log is isolated.
            setup_logger(level=logging.INFO, log_file=log_path)

            try:
                # Fresh import of process_series each iteration keeps module
                # state from leaking between runs (main.py defers its imports
                # inside process_series too).
                from main import process_series
                validation = process_series(series_dir, output_dir)

                # Pull per-section ranges from the last-classified result.
                # process_series does not expose sections directly, so we
                # reclassify cheaply (phantom detect + classify) for summary.
                from core.dicom_loader import load_series
                from core.phantom_detector import detect_phantom
                from core.section_classifier import classify_sections

                dv = load_series(series_dir)
                phantom = detect_phantom(dv)
                sec = classify_sections(dv, phantom)
                ranges = {s.name: (s.start_idx, s.end_idx) for s in sec.sections}

                writer.writerow([
                    name, "OK", f"{time.time() - t0:.1f}",
                    validation.accepted_rois, validation.rejected_rois,
                    len(validation.errors), len(validation.warnings),
                    *ranges.get("STAR8",   ("", "")),
                    *ranges.get("STAR16",  ("", "")),
                    *ranges.get("INSERT1", ("", "")),
                    *ranges.get("INSERT2", ("", "")),
                    "",
                ])
                csvf.flush()
                print(
                    f"   OK {validation.accepted_rois} accepted, "
                    f"{validation.rejected_rois} rejected, "
                    f"{len(validation.errors)} errors, "
                    f"{len(validation.warnings)} warnings "
                    f"({time.time() - t0:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                tb = traceback.format_exc()
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write("\n\n=== UNCAUGHT EXCEPTION ===\n" + tb)
                writer.writerow([
                    name, "FAILED", f"{time.time() - t0:.1f}",
                    "", "", "", "",
                    "", "", "", "", "", "", "", "",
                    f"{type(e).__name__}: {e}",
                ])
                csvf.flush()
                print(f"   FAILED: {type(e).__name__}: {e}", flush=True)

        total = time.time() - t_all
        print()
        print(f"Done. Total: {total/60:.1f} min. Summary: {summary_path}")


if __name__ == "__main__":
    main()
