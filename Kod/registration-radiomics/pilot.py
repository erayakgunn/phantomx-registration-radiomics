"""Run one resumable fixed/moving PhantomX registration pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import SimpleITK as sitk

from metrics import centroid_displacement, compare_label
from registration import (
    read_image,
    centroid_initial_transform,
    translation_centroid_initial_transform,
    resample_for_registration,
    resample_image,
    resample_mask,
    rigid_register,
    translation_centroid_output_transform,
)


DEFAULT_REGISTRATION_LABELS = (
    *range(27, 44),       # vertebrae T/L/S
    *range(92, 116),      # ribs
    116,                  # sternum
)
REGISTRATION_METRIC = "correlation"
WORKFLOW_VERSION = 5


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def label_union_mask(segmentation: sitk.Image, labels: list[int]) -> sitk.Image:
    array = sitk.GetArrayFromImage(segmentation)
    mask = sum((array == label for label in labels), start=(array == -1))
    result = sitk.GetImageFromArray(mask.astype("uint8"))
    result.CopyInformation(segmentation)
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_image(image: sitk.Image, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp.nii.gz")
    sitk.WriteImage(image, str(temporary))
    os.replace(temporary, path)


def _metric_row(fixed_seg: sitk.Image, moving_seg: sitk.Image,
                registered_seg: sitk.Image, label: int,
                fixed_array, moving_array, registered_array) -> dict:
    before = compare_label(
        fixed_seg, moving_seg, label, fixed_array, moving_array
    )
    after = compare_label(
        fixed_seg, registered_seg, label, fixed_array, registered_array
    )
    return {
        "label": label,
        "before_dice": before["dice"],
        "after_dice": after["dice"],
        "before_centroid_displacement_mm": before["centroid_displacement_mm"],
        "after_centroid_displacement_mm": after["centroid_displacement_mm"],
        "before_hd95_mm": before["hd95_mm"],
        "after_hd95_mm": after["hd95_mm"],
        "before_assd_mm": before["assd_mm"],
        "after_assd_mm": after["assd_mm"],
    }


def _stable_score(fixed_mask: sitk.Image, moving_mask: sitk.Image) -> dict:
    fixed_array = sitk.GetArrayFromImage(fixed_mask)
    moving_array = sitk.GetArrayFromImage(moving_mask)
    return compare_label(
        fixed_mask,
        moving_mask,
        1,
        reference_array=fixed_array,
        candidate_array=moving_array,
    )


def _run(args: argparse.Namespace, output: Path) -> None:
    print("[1/5] CT ve segmentation dosyaları okunuyor", flush=True)
    fixed_ct = read_image(Path(args.fixed_ct), sitk.sitkFloat32)
    moving_ct = read_image(Path(args.moving_ct), sitk.sitkFloat32)
    fixed_seg = read_image(Path(args.fixed_seg), sitk.sitkUInt16)
    moving_seg = read_image(Path(args.moving_seg), sitk.sitkUInt16)

    if fixed_ct.GetDimension() != 3 or moving_ct.GetDimension() != 3:
        raise ValueError("Only 3D CT volumes are supported")
    if fixed_ct.GetSize() != fixed_seg.GetSize():
        raise ValueError("Fixed CT and fixed segmentation are not on the same grid")
    if moving_ct.GetSize() != moving_seg.GetSize():
        raise ValueError("Moving CT and moving segmentation are not on the same grid")
    moving_seg_on_fixed = resample_mask(
        moving_seg,
        fixed_seg,
        sitk.Transform(3, sitk.sitkIdentity),
    )

    registration_labels = args.registration_labels or list(DEFAULT_REGISTRATION_LABELS)
    if not registration_labels:
        raise ValueError("Provide --registration-labels or --labels")
    labels = args.labels or sorted(
        set(sitk.GetArrayViewFromImage(fixed_seg).ravel())
        & set(sitk.GetArrayViewFromImage(moving_seg).ravel())
        - {0}
    )
    metadata = {
        "workflow_version": WORKFLOW_VERSION,
        "fixed_ct": str(Path(args.fixed_ct).resolve()),
        "fixed_seg": str(Path(args.fixed_seg).resolve()),
        "moving_ct": str(Path(args.moving_ct).resolve()),
        "moving_seg": str(Path(args.moving_seg).resolve()),
        "labels": sorted(labels),
        "registration_labels": sorted(registration_labels),
        "registration_spacing": args.registration_spacing,
        "registration_metric": REGISTRATION_METRIC,
        "skip_threshold_mm": args.skip_threshold_mm,
        "transform_kind": args.transform_kind,
    }
    state_path = output / "pilot_state.json"
    can_resume_registration = False
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if {key: state.get(key) for key in metadata} != metadata:
            raise RuntimeError("Output klasörü farklı bir pilotla daha önce kullanılmış")
        can_resume_registration = bool(state.get("registered"))
    else:
        state = {**metadata, "registered": False, "metrics": []}
        _atomic_json(state_path, state)

    transform_path = output / "moving_to_fixed.tfm"
    registered_ct_path = output / "ct_registered.nii.gz"
    registered_seg_path = output / "seg_registered.nii.gz"
    if can_resume_registration and transform_path.exists() and registered_ct_path.exists() and registered_seg_path.exists():
        print("[2/5] Mevcut registration çıktısı kullanılıyor", flush=True)
        transform = sitk.ReadTransform(str(transform_path))
        registered_ct = read_image(registered_ct_path, sitk.sitkFloat32)
        registered_seg = read_image(registered_seg_path, sitk.sitkUInt16)
    else:
        registration_error = None
        fixed_stable_mask = label_union_mask(fixed_seg, registration_labels)
        moving_stable_mask = label_union_mask(moving_seg, registration_labels)
        moving_stable_on_fixed = resample_mask(
            moving_stable_mask,
            fixed_stable_mask,
            sitk.Transform(3, sitk.sitkIdentity),
        )
        stable_before = _stable_score(fixed_stable_mask, moving_stable_on_fixed)
        registration_method = "identity"
        baseline_displacements = [
            centroid_displacement(fixed_seg, moving_seg_on_fixed, label)
            for label in labels
        ]
        valid_displacements = [
            value for value in baseline_displacements if math.isfinite(value)
        ]
        max_displacement = max(valid_displacements, default=0.0)
        if not valid_displacements:
            registration_method = "identity_no_valid_target_centroid"
        if max_displacement <= args.skip_threshold_mm:
            print(
                f"[2/5] Baseline yeterli ({max_displacement:.2f} mm); "
                "rigid registration atlanıyor",
                flush=True,
            )
            transform = sitk.Transform(3, sitk.sitkIdentity)
            registered_ct = resample_image(
                moving_ct,
                fixed_ct,
                transform,
                sitk.sitkLinear,
                default_value=-1024.0,
            )
            registered_seg = moving_seg_on_fixed
            stable_after = stable_before
            accepted = False
        elif not valid_displacements:
            transform = sitk.Transform(3, sitk.sitkIdentity)
            registered_ct = resample_image(
                moving_ct, fixed_ct, transform,
                sitk.sitkLinear, default_value=-1024.0
            )
            registered_seg = moving_seg_on_fixed
            stable_after = stable_before
            accepted = False
            print("[qc] Geçerli target centroid yok; identity kullanılıyor", flush=True)
        else:
            print("[2/5] Rigid registration başlatılıyor", flush=True)
            registration_fixed_ct = resample_for_registration(
                fixed_ct, args.registration_spacing, sitk.sitkLinear
            )
            registration_moving_ct = resample_for_registration(
                moving_ct, args.registration_spacing, sitk.sitkLinear
            )
            fixed_registration_mask = resample_for_registration(
                label_union_mask(fixed_seg, registration_labels),
                args.registration_spacing,
                sitk.sitkNearestNeighbor,
            )
            moving_registration_mask = resample_for_registration(
                label_union_mask(moving_seg, registration_labels),
                args.registration_spacing,
                sitk.sitkNearestNeighbor,
            )
            if args.transform_kind == "translation":
                initial_transform = translation_centroid_initial_transform(
                    fixed_registration_mask, moving_registration_mask
                )
            else:
                initial_transform = centroid_initial_transform(
                    registration_fixed_ct,
                    registration_moving_ct,
                    fixed_registration_mask,
                    moving_registration_mask,
                )
            accepted = False
            if float(stable_before["dice"]) <= 0.0:
                candidate_transform = None
                registration_error = "stable mask has no baseline overlap"
                print("[qc] Stabil maskede overlap yok; rigid atlanıyor", flush=True)
            else:
                try:
                    candidate_transform = rigid_register(
                        registration_fixed_ct,
                        registration_moving_ct,
                        fixed_mask=fixed_registration_mask,
                        initial_transform=initial_transform,
                        metric=REGISTRATION_METRIC,
                        transform_kind=args.transform_kind,
                    )
                except Exception as exc:  # noqa: BLE001
                    registration_error = repr(exc)
                    candidate_transform = None
                    print(f"[qc] Registration geçersiz: {exc}; identity kullanılıyor", flush=True)
            if candidate_transform is not None:
                print("[3/5] Transform CT ve maskeye uygulanıyor", flush=True)
                output_transform = candidate_transform.GetInverse()
                candidate_ct = resample_image(
                    moving_ct, fixed_ct, output_transform,
                    sitk.sitkLinear, default_value=-1024.0
                )
                candidate_seg = resample_mask(moving_seg, fixed_seg, output_transform)
                candidate_stable = resample_mask(
                    moving_stable_mask, fixed_stable_mask, output_transform
                )
                stable_after = _stable_score(fixed_stable_mask, candidate_stable)
                accepted = (
                    math.isfinite(float(stable_after["dice"]))
                    and float(stable_after["dice"]) > float(stable_before["dice"]) + 1e-6
                )
                candidate_transform_path = output / "candidate_moving_to_fixed.tfm"
                sitk.WriteTransform(output_transform, str(candidate_transform_path))
                if accepted:
                    transform = output_transform
                    registered_ct = candidate_ct
                    registered_seg = candidate_seg
                    registration_method = args.transform_kind
                    print("[qc] Stabil kemik maskesi iyileşti; transform kabul edildi", flush=True)
                else:
                    print("[qc] Stabil kemik maskesi iyileşmedi; identity kullanılıyor", flush=True)
            if candidate_transform is None or not accepted:
                fallback_transform = translation_centroid_output_transform(
                    fixed_stable_mask, moving_stable_mask
                )
                fallback_stable = resample_mask(
                    moving_stable_mask, fixed_stable_mask, fallback_transform
                )
                fallback_score = _stable_score(fixed_stable_mask, fallback_stable)
                fallback_ok = (
                    math.isfinite(float(fallback_score["dice"]))
                    and float(fallback_score["dice"]) >= float(stable_before["dice"])
                )
                if fallback_ok:
                    transform = fallback_transform
                    registered_ct = resample_image(
                        moving_ct, fixed_ct, transform,
                        sitk.sitkLinear, default_value=-1024.0
                    )
                    registered_seg = resample_mask(moving_seg, fixed_seg, transform)
                    stable_after = fallback_score
                    accepted = True
                    registration_method = "centroid_translation_fallback"
                    print("[qc] Centroid translation fallback kabul edildi", flush=True)
                else:
                    transform = sitk.Transform(3, sitk.sitkIdentity)
                    registered_ct = resample_image(
                        moving_ct, fixed_ct, transform,
                        sitk.sitkLinear, default_value=-1024.0
                    )
                    registered_seg = moving_seg_on_fixed
                    stable_after = stable_before
                    registration_method = "identity"
        registration_qc = {
            "stable_before": stable_before,
            "stable_after": stable_after,
            "registration_attempted": max_displacement > args.skip_threshold_mm,
            "registration_accepted": accepted,
            "registration_method": registration_method,
            "registration_error": registration_error,
            "target_max_centroid_displacement_mm": max_displacement,
        }
        _atomic_json(output / "registration_qc.json", registration_qc)
        sitk.WriteTransform(transform, str(transform_path))
        _atomic_write_image(registered_ct, registered_ct_path)
        _atomic_write_image(registered_seg, registered_seg_path)
        state["registered"] = True
        _atomic_json(state_path, state)

    print("[4/5] Geometrik metrikler hesaplanıyor", flush=True)
    fixed_array = sitk.GetArrayFromImage(fixed_seg)
    moving_array = sitk.GetArrayFromImage(moving_seg_on_fixed)
    registered_array = sitk.GetArrayFromImage(registered_seg)
    partial_path = output / "metrics.partial.csv"
    rows_by_label: dict[int, dict] = {}
    if partial_path.exists():
        with partial_path.open(newline="", encoding="utf-8") as handle:
            rows_by_label = {int(row["label"]): row for row in csv.DictReader(handle)}
        print(f"[metrics] checkpoint bulundu: {len(rows_by_label)} label", flush=True)

    missing = [label for label in labels if label not in rows_by_label]
    workers = min(max(1, os.cpu_count() or 1), len(missing)) if missing else 0
    if workers:
        print(f"[metrics] {len(missing)} label, {workers} worker", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _metric_row,
                    fixed_seg,
                    moving_seg_on_fixed,
                    registered_seg,
                    label,
                    fixed_array,
                    moving_array,
                    registered_array,
                ): label
                for label in missing
            }
            for future in as_completed(futures):
                label = futures[future]
                rows_by_label[label] = future.result()
                print(f"[metrics] label {label} tamamlandı", flush=True)
                with partial_path.open("w", newline="", encoding="utf-8") as handle:
                    ordered = [rows_by_label[key] for key in sorted(rows_by_label)]
                    writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
                    writer.writeheader()
                    writer.writerows(ordered)
                state["metrics"] = sorted(rows_by_label)
                _atomic_json(state_path, state)

    rows = [rows_by_label[label] for label in labels]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    state["metrics_complete"] = True
    _atomic_json(state_path, state)
    print(f"[5/5] Tamamlandı: {output}", flush=True)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "pilot.lock"
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            previous_pid = int(lock_path.read_text(encoding="ascii").strip())
            os.kill(previous_pid, 0)
        except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError, OSError):
            lock_path.unlink(missing_ok=True)
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f"Bu output klasörü zaten çalışıyor: {lock_path}")
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    try:
        _run(args, output)
    finally:
        lock_path.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run one resumable rigid PhantomX registration pair")
    ap.add_argument("--fixed-ct", required=True)
    ap.add_argument("--fixed-seg", required=True)
    ap.add_argument("--moving-ct", required=True)
    ap.add_argument("--moving-seg", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--labels", type=int, nargs="*")
    ap.add_argument("--registration-labels", type=int, nargs="*")
    ap.add_argument("--registration-spacing", type=float, default=2.0)
    ap.add_argument("--skip-threshold-mm", type=float, default=5.0)
    ap.add_argument("--transform-kind", choices=("rigid", "translation"), default="rigid")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
