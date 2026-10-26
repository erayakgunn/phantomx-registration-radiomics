"""Batch DICOM series labeler.

Walks an input directory (e.g. a Philips CT export), filters to CT
reconstruction series, extracts label metadata and copies each series
to ``output_dir`` under its standardized name. Writes a CSV and JSON
report next to the renamed copies.
"""

import collections
import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pydicom

from labeling.header_extractor import (
    LabelMetadata, UNKNOWN, extract_from_file, estimate_recon_spacing,
)
from labeling.name_builder import (
    build_force_base,
    build_name,
    build_pcct_base,
    compute_recon_type,
    format_recon_spacing,
    is_siemens_force,
    is_siemens_pcct,
)
from utils.file_utils import ensure_dir, find_dicom_files, is_dicom_file
from utils.logger import log


# DICOM SOP class for axial CT images. Used to reject non-image data
# (dose reports, structured reports, key object selections, etc.).
_CT_IMAGE_SOP = "1.2.840.10008.5.1.4.1.1.2"


@dataclass
class LabelResult:
    """Result of labeling (and optionally annotating) a single series.

    ``new_path`` points at the labeled folder itself; the actual DICOM
    files live under ``<new_path>/dicoms/``. When the annotation pipeline
    runs successfully, sibling folders (``overlays/``, ``Textures/``,
    ``Stars/``, ``Inserts/``) and ``metadata.json`` appear next to
    ``dicoms/`` inside ``new_path``.
    """

    original_path: str
    new_name: str
    new_path: str
    meta: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""
    # Annotation pipeline results (populated when ``annotate=True``)
    annotated: bool = False
    annotation_error: str = ""
    accepted_rois: int = 0
    rejected_rois: int = 0


def _is_ct_recon_series(series_dir_name: str, first_ds: pydicom.Dataset) -> bool:
    """Return True only for CT reconstruction series.

    Filters out:
      * ``SR_*`` Philips structured reports (dose info, etc.)
      * ``Surview`` / ``Scout`` / ``Localizer`` scans
      * Anything whose SOP class is not CT Image Storage
    """
    name = (series_dir_name or "").upper()
    if name.startswith("SR_"):
        return False
    # Scout / planning / topogram scans: Philips uses "Surview", Siemens uses
    # "Topogram", GE uses "Scout", and DICOM-standard uses "Localizer". We
    # treat them all as the same class (2D planning image, not a recon volume).
    if any(k in name for k in ("SURVIEW", "SCOUT", "LOCALIZER", "TOPOGRAM")):
        return False

    sop = str(getattr(first_ds, "SOPClassUID", "") or "")
    if sop and sop != _CT_IMAGE_SOP:
        return False

    # Also filter by series description when available — some vendors ship
    # topograms under generic series-folder names but set SeriesDescription
    # accordingly. Keeping this check after the SOP-class gate avoids false
    # rejections on real recon series that happen to mention those terms.
    desc = str(getattr(first_ds, "SeriesDescription", "") or "").upper()
    if any(k in desc for k in ("TOPOGRAM", "SURVIEW", "SCOUT", "LOCALIZER")):
        return False

    return True


_SAFE_FILENAME_TRANS = str.maketrans({c: "_" for c in '<>:"/\\|?*'})


def _sanitize_for_tempname(s: str, maxlen: int = 80) -> str:
    """Make a string safe for use as a temp folder name."""
    s = (s or "").strip().translate(_SAFE_FILENAME_TRANS)
    s = "_".join(s.split())  # collapse whitespace
    return s[:maxlen] or "UNKNOWN"


def _is_flat_dicom_dump(input_dir: str) -> bool:
    """True if input_dir needs SeriesInstanceUID-based staging before labeling.

    Covers three export styles where DICOMs are not already organized into
    per-series subfolders:

    * Flat dump (Siemens PCCT NAEOTOM): all ``.dcm`` files sit directly inside
      the study folder.
    * DICOMDIR export: a ``DICOMDIR`` index file at the root plus a ``DICOM/``
      subtree with extension-less files nested 2+ levels deep
      (``DICOM/<study>/<series>/<instance>``).
    * Any layout where DICOM files exist only below depth 1 (so the direct
      children of ``input_dir`` are not themselves series directories).
    """
    # .dcm files directly at root -> classic flat dump.
    for entry in os.listdir(input_dir):
        full = os.path.join(input_dir, entry)
        if os.path.isfile(full) and entry.lower().endswith(".dcm"):
            return True

    # DICOMDIR index at root -> nested export.
    if os.path.isfile(os.path.join(input_dir, "DICOMDIR")):
        return True

    # Fallback: walk one level down — if none of the direct subdirs contain
    # DICOM files themselves but DICOMs exist deeper, treat as flat/nested.
    direct_subdirs = [
        os.path.join(input_dir, e)
        for e in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, e))
    ]
    if not direct_subdirs:
        return False

    def _has_dicom_here(d: str) -> bool:
        for f in os.listdir(d):
            if f.startswith(".") or f.upper() == "DICOMDIR":
                continue
            fp = os.path.join(d, f)
            if not os.path.isfile(fp):
                continue
            if f.lower().endswith(".dcm"):
                return True
            if is_dicom_file(fp):
                return True
        return False

    any_direct_has_dicom = any(_has_dicom_here(d) for d in direct_subdirs)
    if any_direct_has_dicom:
        # Regular per-series or two-level layout — let _iter_series_dirs handle it.
        return False

    # Deep nesting — any DICOM files below?
    for d in direct_subdirs:
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.startswith(".") or f.upper() == "DICOMDIR":
                    continue
                fp = os.path.join(root, f)
                if f.lower().endswith(".dcm") or is_dicom_file(fp):
                    return True
    return False


def _stage_flat_dicom_dump(input_dir: str, staging_root: str) -> str:
    """Group every DICOM under ``input_dir`` by SeriesInstanceUID.

    Recurses through the whole input tree, accepting both ``.dcm`` and
    extension-less DICOM files (DICOMDIR exports), and writes them into
    ``staging_root/<series_name>/<file>.dcm`` via hardlinks (falls back to
    copy on volumes that don't support them). Staged filenames always carry
    a ``.dcm`` suffix so downstream tooling that relies on the extension
    keeps working.
    """
    if os.path.isdir(staging_root):
        shutil.rmtree(staging_root)
    os.makedirs(staging_root, exist_ok=True)

    all_files = find_dicom_files(input_dir)
    log.info(
        f"Flat/DICOMDIR dump detected: indexing {len(all_files)} files by "
        f"SeriesInstanceUID..."
    )

    groups: Dict[str, List[str]] = collections.defaultdict(list)
    descs: Dict[str, str] = {}
    for i, path in enumerate(all_files):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            uid = str(getattr(ds, "SeriesInstanceUID", "") or "UNKNOWN")
        except Exception as e:
            log.warning(f"  Skipping unreadable DICOM {os.path.basename(path)}: {e}")
            continue
        groups[uid].append(path)
        if uid not in descs:
            try:
                snum = int(str(getattr(ds, "SeriesNumber", "") or "0"))
            except ValueError:
                snum = 0
            raw_desc = str(getattr(ds, "SeriesDescription", "") or "NoDesc")
            # Append a per-UID discriminator so two DISTINCT acquisitions that
            # share SeriesNumber + SeriesDescription (e.g. a test-retest pair:
            # same protocol scanned twice minutes apart, identical S-number and
            # description but different SeriesInstanceUID) get SEPARATE staging
            # folders instead of being merged into one. Without this, both
            # acquisitions' slices land in the same folder and are annotated as
            # a single interleaved volume — invalid for repeatability studies.
            # The staging name is internal only; the final labeled name is
            # re-derived from DICOM headers downstream, and the PCCT take-index
            # logic then assigns take1/take2 by acquisition time.
            uid_tag = uid[-8:] if uid and uid != "UNKNOWN" else f"{len(descs):04d}"
            descs[uid] = f"S{snum:04d}_{_sanitize_for_tempname(raw_desc)}_{uid_tag}"
        if (i + 1) % 1000 == 0:
            log.info(f"  indexed {i + 1}/{len(all_files)}")

    log.info(f"  Found {len(groups)} unique series, linking into staging dir...")
    for uid, paths in groups.items():
        series_name = descs.get(uid, f"series_{uid[:8]}")
        series_dir = os.path.join(staging_root, series_name)
        os.makedirs(series_dir, exist_ok=True)
        used: set = set()
        for src_path in paths:
            base = os.path.basename(src_path)
            # Ensure .dcm extension so downstream tooling recognizes the file.
            if not base.lower().endswith(".dcm"):
                base = base + ".dcm"
            dst = os.path.join(series_dir, base)
            # Handle rare collisions when different source subdirs hold files
            # with identical basenames (DICOMDIR exports sometimes do this).
            if dst in used or os.path.exists(dst):
                root, ext = os.path.splitext(base)
                n = 1
                while True:
                    candidate = os.path.join(series_dir, f"{root}_{n:04d}{ext}")
                    if candidate not in used and not os.path.exists(candidate):
                        dst = candidate
                        break
                    n += 1
            used.add(dst)
            try:
                os.link(src_path, dst)
            except OSError:
                shutil.copy2(src_path, dst)
    return staging_root


def _iter_series_dirs(input_dir: str):
    """Yield (test_case_name, series_name, series_path) tuples.

    Handles three layouts gracefully:

    * Two-level (Philips CT export with test cases)::

          input_dir/
              CaseA/
                  CT_Seq._201_IMR_BR_1/
              CaseB/
                  CT_Seq._201_IMR_BR_1/

    * Flat series layout::

          input_dir/
              CT_Seq._201_IMR_BR_1/

    * Flat DICOM dump (Siemens PCCT-style: all ``.dcm`` files sit directly
      inside the study folder). Detected and auto-staged by ``label_folder``
      before this function is called.
    """
    for entry in sorted(os.listdir(input_dir)):
        entry_path = os.path.join(input_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        children = [
            c for c in sorted(os.listdir(entry_path))
            if os.path.isdir(os.path.join(entry_path, c))
        ]
        has_dcm_here = any(
            f.lower().endswith(".dcm")
            for f in os.listdir(entry_path)
            if os.path.isfile(os.path.join(entry_path, f))
        )

        if has_dcm_here or not children:
            # Flat layout: entry itself is a series directory.
            yield "", entry, entry_path
        else:
            # Two-level layout: entry is a test case, iterate its children.
            for child in children:
                child_path = os.path.join(entry_path, child)
                yield entry, child, child_path


def _read_first_dicom(series_path: str) -> Optional[pydicom.Dataset]:
    """Return the first header-only dataset in a series dir, or None."""
    dcms = find_dicom_files(series_path)
    if not dcms:
        return None
    try:
        return pydicom.dcmread(dcms[0], stop_before_pixels=True)
    except Exception as e:
        log.warning(f"Failed to read {dcms[0]}: {e}")
        return None


def _resolve_unique_name(base_name: str, seen: Dict[str, int]) -> str:
    """Append an incrementing suffix if ``base_name`` was already used."""
    if base_name not in seen:
        seen[base_name] = 0
        return base_name
    seen[base_name] += 1
    return f"{base_name}_{seen[base_name]:02d}"


def _materialize_one(
    files_or_dir,
    display_name: str,
    meta: LabelMetadata,
    final_name: str,
    output_dir: str,
    dry_run: bool,
    annotate: bool,
    original_path: str,
) -> LabelResult:
    """Copy one (sub-)series into ``<output_dir>/<final_name>/dicoms/`` and
    optionally annotate it. ``files_or_dir`` is either a directory path
    (copy the whole tree — regular flow) or an explicit list of DICOM file
    paths (multi-phantom instance subset)."""
    target_path = os.path.join(output_dir, final_name)
    dicoms_path = os.path.join(target_path, "dicoms")

    if not dry_run:
        if os.path.exists(target_path):
            log.warning(f"Target exists, overwriting: {target_path}")
            shutil.rmtree(target_path)
        if isinstance(files_or_dir, str):
            # Copy directly into the dicoms/ subfolder so annotation outputs
            # can later sit alongside it (overlays/, Textures/, Stars/,
            # Inserts/, metadata.json) inside target_path.
            shutil.copytree(files_or_dir, dicoms_path)
        else:
            os.makedirs(dicoms_path, exist_ok=True)
            for src in files_or_dir:
                shutil.copy2(src, os.path.join(dicoms_path, os.path.basename(src)))
        log.info(f"Labeled: {display_name} -> {final_name}")
    else:
        log.info(f"[dry-run] Would label: {display_name} -> {final_name}")

    result = LabelResult(
        original_path=original_path,
        new_name=final_name,
        new_path=target_path,
        meta=asdict(meta),
        skipped=False,
    )

    if annotate and not dry_run:
        from labeling.annotator import annotate_labeled_series

        validation = annotate_labeled_series(output_dir, final_name)
        if validation is not None:
            result.annotated = True
            result.accepted_rois = getattr(validation, "accepted_rois", 0)
            result.rejected_rois = getattr(validation, "rejected_rois", 0)
        else:
            result.annotation_error = "Pipeline failed (see logs)"

    return result


def _materialize_series(
    series_path: str,
    display_name: str,
    meta: LabelMetadata,
    final_name: str,
    output_dir: str,
    dry_run: bool,
    annotate: bool,
    check_multi_phantom: bool = False,
) -> List[LabelResult]:
    """Copy a series to ``<output_dir>/<final_name>/dicoms/`` and (optionally)
    run the annotation pipeline. Returns the populated LabelResult(s).

    When ``check_multi_phantom`` is set (Siemens SOMATOM Force flow), the
    series is first scanned for MULTIPLE phantom assemblies stacked in one
    long acquisition (see ``labeling.phantom_instances``). If N > 1
    instances are found, the series materializes as N separate labeled
    folders ``<final_name>_P1..PN`` — each holding only its own z-range's
    DICOMs — and each is annotated independently, so the single-phantom
    section classifier sees exactly one STAR16/STAR8/INSERT1/INSERT2 set
    per volume. Air-gap and bare-couch slices belong to no instance and
    are not copied.
    """
    if check_multi_phantom and not dry_run:
        from labeling.phantom_instances import split_series_into_instances
        dcms = find_dicom_files(series_path)
        parts = split_series_into_instances(series_path, dcms)
        if parts:
            results = []
            for k, files in enumerate(parts, start=1):
                results.append(_materialize_one(
                    files, f"{display_name} [phantom {k}/{len(parts)}]",
                    meta, f"{final_name}_P{k}", output_dir,
                    dry_run, annotate, original_path=series_path,
                ))
            return results

    return [_materialize_one(
        series_path, display_name, meta, final_name, output_dir,
        dry_run, annotate, original_path=series_path,
    )]


def label_folder(
    input_dir: str,
    output_dir: str,
    dry_run: bool = False,
    annotate: bool = True,
    max_series: Optional[int] = None,
    institution: Optional[str] = None,
) -> List[LabelResult]:
    """Label every CT reconstruction series under ``input_dir``.

    For each accepted series the original folder is copied to
    ``<output_dir>/<standardized_name>/dicoms/`` and (optionally) the
    annotation pipeline is run in-place, producing sibling
    ``overlays/``, ``Textures/``, ``Stars/``, ``Inserts/`` folders and
    ``metadata.json``. A CSV and JSON report are written to
    ``output_dir`` describing every series found (including the
    skipped ones).

    Parameters
    ----------
    input_dir
        Root folder handed in by the user (e.g. a CALIBRAMICS_TEST case).
    output_dir
        Destination folder. Created if it does not exist.
    dry_run
        If True, no files are copied and no annotation is run. Reports
        are still written so the planned renames can be reviewed safely.
    annotate
        If True (default), run the annotation pipeline on each labeled
        series immediately after copying its DICOMs. Set to False to
        keep this as a pure rename/copy step (useful for debugging or
        when the annotation pipeline needs to be re-run separately).
        Ignored when ``dry_run=True``.
    max_series
        If set, cap the number of accepted CT-recon series actually
        processed. Skipped series (topograms, SR reports) are still
        reported but don't count towards the cap. Useful for smoke
        testing on large datasets.
    institution
        If set, overrides the DICOM ``InstitutionName`` on every extracted
        series before name building. Use when the tag is anonymized (e.g.
        Siemens service exports carry ``Ser.Nr.NNNN``) so standardized
        names still start with a meaningful center token.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ensure_dir(output_dir)

    # Flat DICOM dump (Siemens PCCT-style) -> stage into per-series subfolders
    # via SeriesInstanceUID first, so _iter_series_dirs can walk them normally.
    # Staging is done via hardlinks when possible, so disk cost is near-zero
    # and the original input_dir is never modified.
    walk_dir = input_dir
    staging_root: Optional[str] = None
    if _is_flat_dicom_dump(input_dir):
        staging_root = os.path.join(output_dir, "_staging")
        walk_dir = _stage_flat_dicom_dump(input_dir, staging_root)
        log.info(f"Staged flat DICOM dump into: {walk_dir}")

    results: List[LabelResult] = []
    seen_names: Dict[str, int] = {}
    processed_recon = 0

    # Siemens PCCT serileri: take_idx atamak için cross-series sıralama gerekiyor;
    # bu yüzden ilk geçişte metadata'yı toplayıp kopyalama/annotation'ı erteleriz.
    # Her entry: (series_path, display_name, dcms, meta, base, recon_type)
    pending_pcct: List[tuple] = []

    for test_case, series_name, series_path in _iter_series_dirs(walk_dir):
        display_name = f"{test_case}/{series_name}" if test_case else series_name

        dcms = find_dicom_files(series_path)
        if not dcms:
            log.info(f"Skip (no DICOM files): {display_name}")
            results.append(LabelResult(
                original_path=series_path,
                new_name="",
                new_path="",
                skipped=True,
                skip_reason="No DICOM files",
            ))
            continue

        first_ds = _read_first_dicom(series_path)
        if first_ds is None:
            results.append(LabelResult(
                original_path=series_path,
                new_name="",
                new_path="",
                skipped=True,
                skip_reason="DICOM read error",
            ))
            continue

        if not _is_ct_recon_series(series_name, first_ds):
            log.info(f"Skip (not CT recon): {display_name}")
            results.append(LabelResult(
                original_path=series_path,
                new_name="",
                new_path="",
                skipped=True,
                skip_reason="Not a CT reconstruction series",
            ))
            continue

        if max_series is not None and processed_recon >= max_series:
            log.info(f"Skip (max-series={max_series} reached): {display_name}")
            results.append(LabelResult(
                original_path=series_path,
                new_name="",
                new_path="",
                skipped=True,
                skip_reason=f"max-series limit ({max_series}) reached",
            ))
            continue
        processed_recon += 1

        meta = extract_from_file(dcms[0])

        # Institution override (CLI --institution): replaces anonymized
        # InstitutionName tags (e.g. "Ser.Nr.246663") for name building.
        if institution:
            meta.institution = institution

        # Fill in recon_spacing from a sample of SliceLocations when the
        # DICOM header doesn't have SpacingBetweenSlices (Siemens NAEOTOM
        # PCCT scans don't populate it; the estimator gets the same
        # number from a quick scan of ~20 file headers).
        if meta.recon_spacing == UNKNOWN and len(dcms) > 1:
            est = estimate_recon_spacing(dcms)
            if est is not None and est > 0:
                meta.recon_spacing = f"{est:.4f}".rstrip("0").rstrip(".")

        if is_siemens_pcct(meta):
            recon_type = compute_recon_type(meta.image_type, meta.image_comments)
            base = build_pcct_base(meta)
            pending_pcct.append(
                (series_path, display_name, dcms, meta, base, recon_type)
            )
            continue

        # Siemens SOMATOM Force (Dual-Source): same deferred take-index
        # grouping as PCCT — repeated same-protocol acquisitions (identical
        # SeriesDescription/kernel/kV but different IrradiationEventUID /
        # AcquisitionDateTime) become take1..takeN by acquisition order.
        # All Force recons are conventional polychromatic, so the recon
        # type slot is a constant "PolyE" (no VMI/MonoE on this scanner).
        if is_siemens_force(meta):
            pending_pcct.append(
                (series_path, display_name, dcms, meta,
                 build_force_base(meta), "PolyE")
            )
            continue

        # --- Non-PCCT akışı (mevcut davranış) ---
        base_name = build_name(meta) + "_" + format_recon_spacing(meta.recon_spacing)
        final_name = _resolve_unique_name(base_name, seen_names)
        results.extend(_materialize_series(
            series_path=series_path,
            display_name=display_name,
            meta=meta,
            final_name=final_name,
            output_dir=output_dir,
            dry_run=dry_run,
            annotate=annotate,
        ))

    # --- PCCT take_idx atama: (base, recon_type) gruplarını
    # (acquisition_datetime, series_number) ile sırala, take1..takeN ata. ---
    pcct_groups: Dict[tuple, List[tuple]] = collections.defaultdict(list)
    for entry in pending_pcct:
        _, _, _, meta, base, recon_type = entry
        pcct_groups[(base, recon_type)].append(entry)

    pcct_with_takes: List[tuple] = []  # (entry, take_idx)
    for key, grp in pcct_groups.items():
        grp.sort(key=lambda e: (e[3].acquisition_datetime or "", e[3].series_number))
        for i, entry in enumerate(grp, start=1):
            pcct_with_takes.append((entry, i))

    # PCCT serilerini işle. Beklenmedik final_name çakışmalarında
    # _resolve_unique_name fallback'i kullan (ör. aynı (base,recon,acq_dt,series_no)
    # birden fazla seri — pratikte olmamalı ama güvenlik kemeri).
    for entry, take_idx in pcct_with_takes:
        series_path, display_name, dcms, meta, base, recon_type = entry
        recon_token = format_recon_spacing(meta.recon_spacing)
        candidate = f"{base}_{recon_type}_take{take_idx}_{recon_token}"
        if candidate in seen_names:
            log.warning(
                f"PCCT name collision on {candidate!r}; falling back to "
                f"_NN suffix to keep folders distinct"
            )
            final_name = _resolve_unique_name(candidate, seen_names)
        else:
            seen_names[candidate] = 0
            final_name = candidate

        results.extend(_materialize_series(
            series_path=series_path,
            display_name=display_name,
            meta=meta,
            final_name=final_name,
            output_dir=output_dir,
            dry_run=dry_run,
            annotate=annotate,
            # SOMATOM Force acquisitions can carry several phantom
            # assemblies end-to-end in one scan — split before annotating.
            check_multi_phantom=is_siemens_force(meta),
        ))

    _write_csv_report(results, os.path.join(output_dir, "labels.csv"))
    _write_json_report(results, os.path.join(output_dir, "labels.json"))

    n_ok = sum(1 for r in results if not r.skipped)
    n_skip = sum(1 for r in results if r.skipped)
    log.info(f"Done. {n_ok} labeled, {n_skip} skipped.")
    return results


def _write_csv_report(results: List[LabelResult], path: str) -> None:
    """Write a spreadsheet-friendly CSV mapping file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "original_path", "new_name", "new_path",
            "institution", "manufacturer", "device_model", "study_date",
            "recon_kernel", "conv_kernel", "kvp", "mas", "fov", "slice_thickness",
            "series_description", "skipped", "skip_reason",
            "annotated", "accepted_rois", "rejected_rois", "annotation_error",
        ])
        for r in results:
            m = r.meta or {}
            w.writerow([
                r.original_path, r.new_name, r.new_path,
                m.get("institution", ""), m.get("manufacturer", ""),
                m.get("device_model", ""), m.get("study_date", ""),
                m.get("recon_kernel", ""), m.get("conv_kernel", ""),
                m.get("kvp", ""), m.get("mas", ""),
                m.get("fov", ""), m.get("slice_thickness", ""),
                m.get("series_description", ""),
                r.skipped, r.skip_reason,
                r.annotated, r.accepted_rois, r.rejected_rois, r.annotation_error,
            ])


def _write_json_report(results: List[LabelResult], path: str) -> None:
    """Write the full structured report (including per-field warnings)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
