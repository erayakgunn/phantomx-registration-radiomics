"""Organize a Siemens NAEOTOM Alpha (Photon Counting CT) flat DICOM dump
into per-series folders named from parsed DICOM headers.

Siemens PCCT exports typically drop every SOP instance of a study into a
single directory as a flat list of ``.dcm`` files (mixed modalities and
series). This tool groups them by ``SeriesInstanceUID`` and writes each
series into its own folder with a descriptive, parameter-rich name such as::

    S003_VMI120kVp_100mAs_1mm_300FOV_Qr40_Q1
    S002_Topogram
    S501_PatientProtocol
    SR502_DoseReport

NAEOTOM Alpha-specific notes
----------------------------
The physical tube voltage (``KVP``) on a photon counting CT is fixed
(120 or 140 kVp). The *Virtual Monoenergetic* (VMI) reconstruction kVp
and the other recon parameters live in ``SeriesDescription``, e.g.::

    "120kVp_100mAs_1mm_300FOV 1,00 Qr40 Q3"
     ^VMI   ^mAs  ^th ^FOV   ^slice_mm ^kernel ^Q-level

Usage
-----
    python organize_siemens_pcct.py --input "C:/.../27032026-PCCT"
    python organize_siemens_pcct.py --input <dir> --output-dir <dir> [--copy] [--move] [--dry-run]

By default files are hardlinked (falls back to copy if the filesystem
refuses), so the operation is fast and disk-cheap. The original input
folder is never modified.
"""

import argparse
import collections
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import pydicom


# ---------------------------------------------------------------------------
# Parsing & classification
# ---------------------------------------------------------------------------

# Siemens NAEOTOM Alpha recon SeriesDescription pattern. Example match:
#   "120kVp_100mAs_1mm_300FOV 1,00 Qr40 Q3"
#   "80kVp_200mAs_3mm_500FOV 3,00 Qr64 Q1"
_RECON_DESC = re.compile(
    r"(?P<vmi>\d+)\s*kVp_(?P<mas>\d+)\s*mAs_(?P<slice>\d+)\s*mm_(?P<fov>\d+)\s*FOV"
    r"\s+(?P<slice_dec>[\d,\.]+)\s+(?P<kernel>[A-Za-z]+\d*[A-Za-z]*)\s+Q(?P<q>\d+)",
    re.IGNORECASE,
)

_TOPO_KEYS = ("TOPOGRAM", "SURVIEW", "SCOUT", "LOCALIZER")

# DICOM SOP class UIDs
_SOP_CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"

# Invalid chars on Windows + control chars
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WS = re.compile(r"\s+")


def _sanitize(v: str, maxlen: int = 120) -> str:
    """Return a Windows-safe, collapsed-whitespace token derived from ``v``."""
    if not v:
        return "UNKNOWN"
    s = _WS.sub("_", str(v))
    s = _INVALID_CHARS.sub("", s).strip("._ ")
    return (s or "UNKNOWN")[:maxlen]


def _compact(v: str, maxlen: int = 60) -> str:
    """Like ``_sanitize`` but strips internal whitespace entirely.

    Used for prefix fields where we want ``Radiologie USZ`` to become
    ``RadiologieUSZ`` (not ``Radiologie_USZ``) so that a single ``_``
    unambiguously separates the four prefix tokens.
    """
    if not v:
        return "UNKNOWN"
    s = _WS.sub("", str(v))
    s = _INVALID_CHARS.sub("", s).strip("._")
    return (s or "UNKNOWN")[:maxlen]


@dataclass
class SeriesInfo:
    """Everything needed to name and place a series."""

    uid: str
    kind: str                   # RECON / TOPOGRAM / PROTOCOL / SR_DOSE / SR_EXAM / OTHER
    modality: str               # "CT", "SR", ...
    series_number: int
    series_description: str
    kvp_physical: str           # tube kVp from the DICOM tag (120 or 140 on NAEOTOM)
    institution: str = ""       # InstitutionName
    manufacturer: str = ""      # Manufacturer (vendor / marka)
    device_model: str = ""      # ManufacturerModelName (fallback: StationName)
    study_date: str = ""        # StudyDate / AcquisitionDate / ContentDate (YYYYMMDD)
    slice_thickness_mm: str = ""  # DICOM SliceThickness, formatted "1p0" / "0p67"
    recon_params: Dict[str, str] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)


def _format_slice_thickness(value) -> str:
    """Render DICOM SliceThickness as a Windows-safe token.

    Always emits a fractional part so DICOM-sourced values are visibly
    distinct from the description-parsed integer (1 vs 1p0):
        1.0  -> "1p0"
        3.0  -> "3p0"
        0.67 -> "0p67"
    """
    if value in (None, ""):
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if v == int(v):
        return f"{int(v)}p0"
    return f"{v:g}".replace(".", "p")


def _classify(ds: pydicom.Dataset) -> str:
    """Return a coarse class tag for a series based on a single file header."""
    modality = str(getattr(ds, "Modality", "") or "").upper()
    desc = str(getattr(ds, "SeriesDescription", "") or "")
    desc_u = desc.upper()

    if modality == "SR":
        if "DOSE" in desc_u:
            return "SR_DOSE"
        if "EXAM" in desc_u:
            return "SR_EXAM"
        return "SR_OTHER"

    if any(k in desc_u for k in _TOPO_KEYS):
        return "TOPOGRAM"

    if "PROTOCOL" in desc_u:
        return "PROTOCOL"

    sop = str(getattr(ds, "SOPClassUID", "") or "")
    if sop == _SOP_CT_IMAGE:
        return "RECON"

    return "OTHER"


def _parse_recon_desc(desc: str) -> Dict[str, str]:
    """Parse a Siemens PCCT recon SeriesDescription into its parameter tokens."""
    m = _RECON_DESC.search(desc or "")
    if not m:
        return {}
    return {k: v for k, v in m.groupdict().items() if v is not None}


def _study_prefix(info: SeriesInfo) -> str:
    """Return the ``{hospital}_{brand}_{device}_{date}`` prefix for folder names.

    Internal whitespace inside each token is stripped (``Radiologie USZ``
    -> ``RadiologieUSZ``) so that a single ``_`` always separates the
    four prefix tokens.
    """
    parts = [
        _compact(info.institution),
        _compact(info.manufacturer),
        _compact(info.device_model),
        _compact(info.study_date),
    ]
    return "_".join(parts)


def _folder_name(info: SeriesInfo) -> str:
    """Build the target folder name for a series (unique-per-UID by caller).

    Format: ``{hospital}_{brand}_{device}_{date}_{descriptor}``
    e.g. ``RadiologieUSZ_SiemensHealthineers_NAEOTOMAlpha_20260327_VMI120kVp_100mAs_1mm_300FOV_Qr40_Q1``.

    Series numbers are deliberately NOT encoded — repeated-parameter
    acquisitions get a ``_01`` suffix via ``_resolve_unique`` in the
    caller. The SeriesNumber is still recorded in the manifest for
    traceability.
    """
    prefix = _study_prefix(info)

    if info.kind == "TOPOGRAM":
        tail = "Topogram"
    elif info.kind == "PROTOCOL":
        tail = "PatientProtocol"
    elif info.kind == "SR_DOSE":
        tail = "DoseReport"
    elif info.kind == "SR_EXAM":
        tail = "ExaminationReport"
    elif info.kind == "SR_OTHER":
        tail = "StructuredReport"
    elif info.kind == "RECON" and info.recon_params:
        p = info.recon_params
        # Prefer the DICOM SliceThickness tag (precise, e.g. 0p67)
        # over the description-parsed integer (e.g. 1).
        slice_token = info.slice_thickness_mm or p['slice']
        tail = (
            f"VMI{p['vmi']}kVp_{p['mas']}mAs_{slice_token}mm"
            f"_{p['fov']}FOV_{p['kernel']}_Q{p['q']}"
        )
    else:
        # Fallback: unparseable recon / OTHER
        tail = _sanitize(info.series_description) or "NoDesc"

    return f"{prefix}_{tail}"


# ---------------------------------------------------------------------------
# File ops
# ---------------------------------------------------------------------------

def _place_file(src: str, dst: str, mode: str) -> None:
    """Copy/hardlink/move ``src`` to ``dst`` according to ``mode``."""
    if os.path.exists(dst):
        return
    if mode == "move":
        shutil.move(src, dst)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    # hardlink (default), fall back to copy if not supported
    try:
        os.link(src, dst)
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _iter_dicoms(input_dir: str) -> List[str]:
    """Return every ``.dcm`` file directly inside ``input_dir``.

    NAEOTOM exports are flat, so we don't need to recurse. Any DICOM
    nested deeper is assumed to already belong to a different organization.
    """
    out = []
    for entry in os.listdir(input_dir):
        full = os.path.join(input_dir, entry)
        if os.path.isfile(full) and entry.lower().endswith(".dcm"):
            out.append(full)
    return out


def _index_series(files: List[str], log_every: int = 1000) -> Dict[str, SeriesInfo]:
    """Read headers of every file and group by SeriesInstanceUID.

    Returns a dict mapping UID -> SeriesInfo, with per-series file lists
    populated. Unreadable files are skipped with a stderr warning.
    """
    groups: Dict[str, SeriesInfo] = {}
    n = len(files)
    unread = 0

    for i, path in enumerate(files, 1):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception as e:  # unreadable file: skip, don't abort
            print(f"[warn] unreadable DICOM {os.path.basename(path)}: {e}",
                  file=sys.stderr)
            unread += 1
            continue

        uid = str(getattr(ds, "SeriesInstanceUID", "") or f"NOUID_{os.path.basename(path)}")
        info = groups.get(uid)
        if info is None:
            try:
                snum = int(str(getattr(ds, "SeriesNumber", "") or "0"))
            except ValueError:
                snum = 0
            kind = _classify(ds)
            desc = str(getattr(ds, "SeriesDescription", "") or "")
            recon_params = _parse_recon_desc(desc) if kind == "RECON" else {}
            info = SeriesInfo(
                uid=uid,
                kind=kind,
                modality=str(getattr(ds, "Modality", "") or ""),
                series_number=snum,
                series_description=desc,
                kvp_physical=str(getattr(ds, "KVP", "") or ""),
                institution=str(getattr(ds, "InstitutionName", "") or ""),
                manufacturer=str(getattr(ds, "Manufacturer", "") or ""),
                device_model=str(
                    getattr(ds, "ManufacturerModelName", "")
                    or getattr(ds, "StationName", "") or ""
                ),
                study_date=str(
                    getattr(ds, "StudyDate", "")
                    or getattr(ds, "AcquisitionDate", "")
                    or getattr(ds, "ContentDate", "") or ""
                ),
                slice_thickness_mm=_format_slice_thickness(
                    getattr(ds, "SliceThickness", None)
                ),
                recon_params=recon_params,
            )
            groups[uid] = info
        info.files.append(path)

        if i % log_every == 0:
            print(f"  indexed {i}/{n}")

    if unread:
        print(f"[warn] {unread} files could not be read; skipped.", file=sys.stderr)
    return groups


def _resolve_unique(name: str, seen: Dict[str, int]) -> str:
    """Append ``_NN`` suffix if ``name`` is already taken."""
    if name not in seen:
        seen[name] = 0
        return name
    seen[name] += 1
    return f"{name}_{seen[name]:02d}"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _write_manifest(
    rows: List[Dict[str, object]],
    output_dir: str,
) -> Tuple[str, str]:
    """Write a JSON + CSV manifest describing every placed series."""
    json_path = os.path.join(output_dir, "_manifest.json")
    csv_path = os.path.join(output_dir, "_manifest.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "folder", "kind", "modality", "series_number", "n_files",
        "institution", "manufacturer", "device_model", "study_date",
        "vmi_kvp", "mas", "slice_mm", "fov_mm", "kernel", "q_level",
        "kvp_physical", "series_description", "series_instance_uid",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    return json_path, csv_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def organize(
    input_dir: str,
    output_dir: str,
    mode: str = "hardlink",
    dry_run: bool = False,
) -> List[Dict[str, object]]:
    """Organize a flat Siemens PCCT DICOM dump into per-series folders.

    Parameters
    ----------
    input_dir
        Folder containing the loose ``.dcm`` files (e.g. ``27032026-PCCT``).
    output_dir
        Destination root. Created if missing.
    mode
        ``"hardlink"`` (default, falls back to copy on failure),
        ``"copy"`` (always copy), or ``"move"`` (destructive — files
        leave ``input_dir``).
    dry_run
        If True, no files are touched; the manifest is still written so
        you can review the planned layout.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if mode not in ("hardlink", "copy", "move"):
        raise ValueError(f"Invalid mode {mode!r}; expected hardlink/copy/move")

    os.makedirs(output_dir, exist_ok=True)

    files = _iter_dicoms(input_dir)
    if not files:
        raise FileNotFoundError(f"No .dcm files directly inside {input_dir}")

    print(f"Scanning {len(files)} DICOM files for SeriesInstanceUID...")
    groups = _index_series(files)
    print(f"Found {len(groups)} unique series.")

    # Stable ordering: CT first (sorted by series number), then SR, then anything else.
    def _sort_key(info: SeriesInfo) -> Tuple[int, int]:
        order = {"CT": 0, "SR": 1}.get(info.modality, 2)
        return (order, info.series_number)

    seen_names: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []

    for info in sorted(groups.values(), key=_sort_key):
        base_name = _folder_name(info)
        folder = _resolve_unique(base_name, seen_names)
        target_dir = os.path.join(output_dir, folder, "dicoms")

        if not dry_run:
            os.makedirs(target_dir, exist_ok=True)
            for src in info.files:
                dst = os.path.join(target_dir, os.path.basename(src))
                _place_file(src, dst, mode)

        p = info.recon_params
        rows.append({
            "folder": folder,
            "kind": info.kind,
            "modality": info.modality,
            "series_number": info.series_number,
            "n_files": len(info.files),
            "institution": info.institution,
            "manufacturer": info.manufacturer,
            "device_model": info.device_model,
            "study_date": info.study_date,
            "vmi_kvp": p.get("vmi", ""),
            "mas": p.get("mas", ""),
            "slice_mm": p.get("slice", ""),
            "fov_mm": p.get("fov", ""),
            "kernel": p.get("kernel", ""),
            "q_level": p.get("q", ""),
            "kvp_physical": info.kvp_physical,
            "series_description": info.series_description,
            "series_instance_uid": info.uid,
        })
        tag = "[dry-run]" if dry_run else f"[{mode}]"
        print(f"  {tag} {info.kind:<9} S{info.series_number:03d} "
              f"({len(info.files):>4} files) -> {folder}")

    jpath, cpath = _write_manifest(rows, output_dir)
    print(f"Manifest: {jpath}")
    print(f"Manifest: {cpath}")

    # Summary by kind
    by_kind = collections.Counter(r["kind"] for r in rows)
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))

    return rows


def _default_output(input_dir: str) -> str:
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    base = os.path.basename(os.path.abspath(input_dir).rstrip("/\\"))
    return os.path.join(desktop, f"{base}_ORGANIZED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize a Siemens NAEOTOM Alpha (PCCT) flat DICOM dump"
                    " into per-series folders named from DICOM headers.",
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Flat DICOM directory (e.g. 27032026-PCCT)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Destination (default: Desktop/<input>_ORGANIZED)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true",
                      help="Copy files (default: hardlink w/ copy fallback)")
    mode.add_argument("--move", action="store_true",
                      help="Move files (destructive — empties input_dir)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only; write the manifest but touch no files")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"error: input not a directory: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.copy:
        mode = "copy"
    elif args.move:
        mode = "move"
    else:
        mode = "hardlink"

    out = args.output_dir or _default_output(args.input)

    print("=" * 60)
    print("Siemens PCCT Organizer")
    print("=" * 60)
    print(f"Input:   {args.input}")
    print(f"Output:  {out}")
    print(f"Mode:    {mode}{' (dry-run)' if args.dry_run else ''}")
    print()

    try:
        organize(args.input, out, mode=mode, dry_run=args.dry_run)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
