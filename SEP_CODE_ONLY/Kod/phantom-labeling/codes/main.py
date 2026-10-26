"""CT Phantom Analysis Pipeline - CLI Orchestrator."""

import argparse
import os
import sys
import time

from utils.logger import log, setup_logger
from utils.file_utils import list_series_dirs, ensure_dir
import config


def process_series(series_dir: str, output_dir: str):
    """Run the full pipeline on a single DICOM series."""
    series_name = os.path.basename(series_dir)
    log.info(f"{'='*60}")
    log.info(f"Processing series: {series_name}")
    log.info(f"{'='*60}")

    # Phase 0: Load DICOM
    from core.dicom_loader import load_series
    dv = load_series(series_dir)

    # Phase 1: Phantom detection
    from core.phantom_detector import detect_phantom
    phantom = detect_phantom(dv)

    # Phase 2: Section classification
    from core.section_classifier import classify_sections
    sections = classify_sections(dv, phantom)

    # Phase 2b: Orientation correction — when the phantom was placed upside-
    # down in the scanner, the volume is flipped here so subsequent phases
    # see the canonical orientation (STAR16 first in z, textures unmirrored).
    from core.orientation_corrector import canonicalize_orientation
    canonicalize_orientation(dv, phantom, sections)

    # Phase 2c: Inner-window trim — each section is restricted to the inner
    # fixed-mm window centred on its data-driven content centroid. Makes
    # the kept volume protocol-independent so cross-protocol comparisons
    # aren't dominated by per-protocol TRANSITION sensitivity.
    from core.section_trim import trim_sections_to_inner_window
    trim_sections_to_inner_window(dv, sections, phantom=phantom)

    # Phase 3: Rotation detection
    from core.rotation_detector import detect_rotation
    rotation = detect_rotation(dv, phantom, sections)

    # Phase 4: ROI extraction
    from core.roi_extractor import extract_rois
    extraction = extract_rois(dv, phantom, sections, rotation)

    # Phase 5: Validation
    from core.validator import validate_all
    validation = validate_all(dv, phantom, sections, extraction)

    # Phase 6: Visualization and export
    from output.visualizer import generate_overlays
    from output.insert_painter import paint_insert_rois
    from output.coronal_overlay import generate_coronal_overlay
    from output.exporter import export_results

    try:
        generate_overlays(dv, phantom, sections, extraction, output_dir, series_name)
    except Exception as e:
        log.error(f"Overlay generation failed for {series_name}: {e}", exc_info=True)

    try:
        paint_insert_rois(dv, phantom, sections, extraction, output_dir, series_name)
    except Exception as e:
        log.error(f"Insert paint generation failed for {series_name}: {e}", exc_info=True)

    try:
        generate_coronal_overlay(dv, phantom, sections, output_dir, series_name)
    except Exception as e:
        log.error(f"Coronal overlay generation failed for {series_name}: {e}", exc_info=True)

    export_results(
        extraction, validation, output_dir, series_name,
        source_metadata=dv.metadata,
    )

    log.info(f"Series {series_name} complete.")
    return validation


def main():
    parser = argparse.ArgumentParser(
        description="CT Phantom Analysis Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        default=config.DATA_ROOT,
        help="Root directory containing DICOM series folders",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_ROOT,
        help="Output directory for results",
    )
    parser.add_argument(
        "--series",
        default=None,
        help="Process only this series directory name (default: all)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Process only series whose directory name starts with this date (e.g. 20260114)",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Maximum number of series to process",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (in addition to stdout)",
    )
    parser.add_argument(
        "--insert-sizes",
        default=None,
        metavar="MM[,MM...]",
        help=(
            "Insert sub-ROI diameters in mm, comma-separated (default: "
            + ",".join(str(d) for d in config.INSERT_SUBROI_DIAMETERS_MM)
            + "). Example: --insert-sizes 5,10,15 restores the old three-size layout."
        ),
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        setup_logger(level=logging.DEBUG, log_file=args.log_file)
    elif args.log_file:
        setup_logger(log_file=args.log_file)

    if args.insert_sizes:
        try:
            sizes = tuple(sorted({int(t) for t in args.insert_sizes.split(",") if t.strip()}))
        except ValueError:
            log.error(f"--insert-sizes expects comma-separated integers in mm, got {args.insert_sizes!r}")
            sys.exit(2)
        if not sizes or any(s <= 0 for s in sizes):
            log.error(f"--insert-sizes needs at least one positive diameter, got {args.insert_sizes!r}")
            sys.exit(2)
        config.INSERT_SUBROI_DIAMETERS_MM = sizes
    log.info(f"Insert sub-ROI sizes: {', '.join(str(d) for d in config.INSERT_SUBROI_DIAMETERS_MM)} mm")

    ensure_dir(args.output_dir)

    if args.series:
        series_path = os.path.join(args.data_dir, args.series)
        if not os.path.isdir(series_path):
            log.error(f"Series directory not found: {series_path}")
            sys.exit(1)
        series_dirs = [series_path]
    else:
        series_dirs = list_series_dirs(args.data_dir)

    if args.date:
        series_dirs = [d for d in series_dirs if os.path.basename(d).startswith(args.date)]

    if not series_dirs:
        log.error(f"No series directories found in {args.data_dir}")
        sys.exit(1)

    if args.max_series:
        series_dirs = series_dirs[:args.max_series]

    log.info(f"Found {len(series_dirs)} series to process")

    t0 = time.time()
    results = {}

    failed_count = 0
    for series_dir in series_dirs:
        try:
            validation = process_series(series_dir, args.output_dir)
            results[os.path.basename(series_dir)] = {
                "status": "OK",
                "accepted": validation.accepted_rois,
                "rejected": validation.rejected_rois,
                "errors": len(validation.errors),
                "warnings": len(validation.warnings),
            }
        except Exception as e:
            log.error(f"Failed on {series_dir}: {e}", exc_info=True)
            failed_count += 1
            results[os.path.basename(series_dir)] = {
                "status": "FAILED",
                "error": str(e),
            }

    # Summary
    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"Pipeline complete. Processed {len(results)} series.")
    log.info(f"Elapsed: {elapsed:.1f}s")
    for name, res in results.items():
        if res["status"] == "OK":
            log.info(
                f"  {name}: {res['accepted']} accepted, "
                f"{res['rejected']} rejected, "
                f"{res['errors']} errors, {res['warnings']} warnings"
            )
        else:
            log.info(f"  {name}: FAILED - {res.get('error', 'unknown')}")

    if failed_count > 0:
        log.error(f"{failed_count}/{len(results)} series failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
