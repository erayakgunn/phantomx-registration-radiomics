#!/usr/bin/env python3
"""
Extract Star8 and Star16 DICOM label files from a labeled PCCT dataset.

Input structure expected:
    DATASET_ROOT/
      <protocol_1>/
        Stars/
          Star8/*.dcm
          Star16/*.dcm
      <protocol_2>/
        Stars/
          Star8/*.dcm
          Star16/*.dcm

Output structure created:
    OUTPUT_DIR/
      <protocol_1>/
        Star8/*.dcm
        Star16/*.dcm
      <protocol_2>/
        Star8/*.dcm
        Star16/*.dcm

Usage:
    python extract_star_dicoms.py --dataset-root "C:\\...\\LABELED"
    python extract_star_dicoms.py --dataset-root "C:\\...\\LABELED" --output-dir "C:\\...\\stars_only"
    python extract_star_dicoms.py --dataset-root "C:\\...\\LABELED" --dry-run
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

STAR_LABELS = ("Star8", "Star16")


def is_protocol_dir(path: Path) -> bool:
    """A protocol folder is accepted if it contains Stars/Star8 or Stars/Star16."""
    if not path.is_dir():
        return False
    stars = path / "Stars"
    return any((stars / label).is_dir() for label in STAR_LABELS)


def discover_protocols(dataset_root: Path) -> List[Path]:
    """Find protocol folders directly under dataset_root.

    If dataset_root itself is a protocol folder, return it as a single item.
    """
    if is_protocol_dir(dataset_root):
        return [dataset_root]

    protocols = []
    for child in sorted(dataset_root.iterdir()):
        if is_protocol_dir(child):
            protocols.append(child)
    return protocols


def collect_star_files(protocol_dir: Path) -> Dict[str, List[Path]]:
    """Return Star8/Star16 DICOM files for one protocol."""
    result: Dict[str, List[Path]] = {}
    for label in STAR_LABELS:
        star_dir = protocol_dir / "Stars" / label
        if not star_dir.is_dir():
            result[label] = []
            continue
        result[label] = sorted(star_dir.glob("*.dcm"))
    return result


def copy_star_files(protocol_dir: Path, output_dir: Path, overwrite: bool = False) -> Dict[str, int]:
    """Copy Star8/Star16 files to output_dir/<protocol_name>/<Star8|Star16>."""
    files_by_label = collect_star_files(protocol_dir)
    protocol_out = output_dir / protocol_dir.name
    counts: Dict[str, int] = {}

    for label, files in files_by_label.items():
        label_out = protocol_out / label
        label_out.mkdir(parents=True, exist_ok=True)
        copied = 0

        for src in files:
            dst = label_out / src.name
            if dst.exists() and not overwrite:
                copied += 1
                continue
            shutil.copy2(src, dst)
            copied += 1

        counts[label] = copied

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy only Star8 and Star16 DICOM label files from PCCT labeled protocols."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root labeled dataset folder that contains protocol folders."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Default: <dataset-root>/stars_only_output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be copied; do not copy files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output folder."
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"ERROR: dataset root does not exist: {dataset_root}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "stars_only_output"
    protocols = discover_protocols(dataset_root)

    print("=" * 70)
    print("Extract Star8 / Star16 DICOM files")
    print("=" * 70)
    print(f"Dataset:  {dataset_root}")
    print(f"Output:   {output_dir}")
    print(f"Protocols found: {len(protocols)}")
    print(f"Mode:     {'DRY-RUN' if args.dry_run else 'COPY'}")
    print("=" * 70)

    if not protocols:
        print("No protocol folders found. Expected each protocol to contain Stars/Star8 and/or Stars/Star16.")
        return 1

    total_star8 = 0
    total_star16 = 0

    for protocol in protocols:
        files_by_label = collect_star_files(protocol)
        n8 = len(files_by_label.get("Star8", []))
        n16 = len(files_by_label.get("Star16", []))
        total_star8 += n8
        total_star16 += n16

        print(f"\n{protocol.name}")
        print(f"  Star8 :  {n8} DICOM files")
        print(f"  Star16:  {n16} DICOM files")

        if not args.dry_run:
            counts = copy_star_files(protocol, output_dir, overwrite=args.overwrite)
            print(f"  copied -> Star8={counts.get('Star8', 0)}, Star16={counts.get('Star16', 0)}")

    print("\n" + "=" * 70)
    print(f"Total Star8 : {total_star8}")
    print(f"Total Star16: {total_star16}")
    if not args.dry_run:
        print(f"Done. Output folder: {output_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
