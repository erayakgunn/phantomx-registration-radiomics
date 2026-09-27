"""CLI entry point for batch DICOM labeling.

Reads DICOM headers from every CT reconstruction series under the
given input directory and copies each series to a new folder under
a standardized name of the form::

    {Merkez}_{Marka}_{Cihaz}_{Tarih}_{ReconKernel}_{ConvKernel}_{KVP}_{mAs}_{FOV}_{SliceThickness}

Usage
-----
    python label_series.py --input "C:/Users/Administrator/Desktop/CALIBRAMICS_TEST 2615"
    python label_series.py --input <dir> --output-dir <dir> [--dry-run] [--verbose]

If ``--output-dir`` is omitted, results are written to
``<Desktop>/<input_basename>_LABELED/``.
"""

import argparse
import logging
import os
import sys

import config
from labeling.labeler import label_folder
from utils.logger import log, setup_logger


def parse_insert_sizes(text: str) -> tuple:
    """Parse ``"5,10"`` -> ``(5, 10)``; raises ``argparse.ArgumentTypeError`` on junk."""
    try:
        sizes = tuple(sorted({int(t) for t in text.split(",") if t.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--insert-sizes expects comma-separated integers in mm, got {text!r}"
        ) from exc
    if not sizes or any(s <= 0 for s in sizes):
        raise argparse.ArgumentTypeError(
            f"--insert-sizes needs at least one positive diameter, got {text!r}"
        )
    return sizes


def _default_output_dir(input_dir: str) -> str:
    """Return ``<Desktop>/<basename>_LABELED`` for the given input."""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    basename = os.path.basename(os.path.abspath(input_dir).rstrip("/\\"))
    return os.path.join(desktop, f"{basename}_LABELED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch DICOM series labeler",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input DICOM root directory (e.g. a CALIBRAMICS_TEST folder)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: Desktop/<input_name>_LABELED)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not copy files; only generate labels.csv / labels.json",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help=(
            "Skip the annotation pipeline and only copy DICOMs into "
            "<labeled>/dicoms/ subfolders. Useful when you want to run "
            "annotation separately, or when debugging the labeler itself."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to a log file (in addition to stdout)",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help=(
            "Process at most N CT reconstruction series (useful for smoke-"
            "testing the annotation pipeline on a new dataset before running "
            "on every series). Skipped series (topograms, dose reports) do "
            "not count towards the limit."
        ),
    )
    parser.add_argument(
        "--institution",
        default=None,
        help=(
            "Institution token to prepend to standardized names, overriding "
            "the DICOM InstitutionName tag. Needed when the tag is "
            "anonymized (e.g. Siemens exports carrying 'Ser.Nr.NNNN'). "
            "Example: --institution AcibademKent"
        ),
    )
    parser.add_argument(
        "--insert-sizes",
        type=parse_insert_sizes,
        default=None,
        metavar="MM[,MM...]",
        help=(
            "Insert sub-ROI diameters in mm, comma-separated (default: "
            + ",".join(str(d) for d in config.INSERT_SUBROI_DIAMETERS_MM)
            + "). Example: --insert-sizes 5,10,15 restores the old three-size layout."
        ),
    )

    args = parser.parse_args()

    if args.insert_sizes:
        config.INSERT_SUBROI_DIAMETERS_MM = args.insert_sizes

    # Match main.py's logger configuration pattern.
    if args.verbose:
        setup_logger(level=logging.DEBUG, log_file=args.log_file)
    elif args.log_file:
        setup_logger(log_file=args.log_file)

    if not os.path.isdir(args.input):
        log.error(f"Input not a directory: {args.input}")
        sys.exit(1)

    output_dir = args.output_dir or _default_output_dir(args.input)

    annotate = not args.no_annotate

    log.info("=" * 60)
    log.info("DICOM Auto-Labeler")
    log.info("=" * 60)
    log.info(f"Input:    {args.input}")
    log.info(f"Output:   {output_dir}")
    log.info(f"Dry run:  {args.dry_run}")
    log.info(f"Annotate: {annotate}")
    log.info(f"Insert sub-ROI sizes: {', '.join(str(d) for d in config.INSERT_SUBROI_DIAMETERS_MM)} mm")

    results = label_folder(
        args.input,
        output_dir,
        dry_run=args.dry_run,
        annotate=annotate,
        max_series=args.max_series,
        institution=args.institution,
    )

    n_ok = sum(1 for r in results if not r.skipped)
    n_skip = sum(1 for r in results if r.skipped)
    log.info(f"Labeled {n_ok} series, skipped {n_skip}.")

    # Only report annotation stats when the pipeline actually ran.
    if annotate and not args.dry_run:
        n_annotated = sum(1 for r in results if r.annotated)
        n_ann_fail = sum(
            1 for r in results
            if (not r.skipped and not r.annotated)
        )
        log.info(
            f"Annotated {n_annotated}/{n_ok} series "
            f"({n_ann_fail} pipeline failures)."
        )

    log.info(f"Report:  {os.path.join(output_dir, 'labels.csv')}")

    # --- Duplicate-slice audit (Excel) ---
    # Runs after every non-dry-run labeling. Generates
    # <output_dir>/duplicate_audit.xlsx with:
    #   - Audit sheet: one row per input series with file count, unique-
    #     z count, duplicate-pair count, verdict, and a representative
    #     pair's pixel hashes
    #   - Duplicate Pairs sheet: every duplicate pair across all series
    #     with both files and their MD5 hashes (reviewers can spot-check)
    #   - Summary sheet: totals + methodology + column meanings
    # The audit reads the same run.log the labeler just wrote, so the
    # "per-section coronal data" columns (L, kept slices, physical mm,
    # midpoint z, kept z range) are populated for every section.
    if not args.dry_run and n_ok > 0:
        try:
            from diagnostic.duplicate_excel_report import run_audit
            xlsx_path = os.path.join(output_dir, "duplicate_audit.xlsx")
            audit_log = args.log_file or os.path.join(output_dir, "run.log")
            log.info("Generating duplicate-slice audit Excel ...")
            run_audit(
                args.input, output_dir,
                xlsx_path=xlsx_path,
                log_path=audit_log,
                progress=False,
            )
            log.info(f"Duplicate audit: {xlsx_path}")
        except Exception as e:
            log.warning(
                f"Duplicate-audit generation failed (labeling itself "
                f"completed successfully): {e}"
            )

    if n_ok == 0:
        log.error("No series were labeled. Check input layout and logs above.")
        sys.exit(2)


if __name__ == "__main__":
    main()
