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

import numpy as np
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
    landmark_rigid_output_transform,
    translation_centroid_output_transform,
)


DEFAULT_REGISTRATION_LABELS = (
    *range(27, 44),       # vertebrae T/L/S
    *range(92, 116),      # ribs
    116,                  # sternum
)
REGISTRATION_METRIC = "correlation"
WORKFLOW_VERSION = "1.0.0"
VERTEBRA_LABELS = tuple(range(27, 44))


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(
        int(os.environ.get("PHANTOMX_SITK_THREADS", "1"))
    )
except Exception:
    pass


def label_union_mask(segmentation: sitk.Image, labels: list[int]) -> sitk.Image:
    array = sitk.GetArrayViewFromImage(segmentation)
    mask = np.isin(array, labels).astype("uint8")
    result = sitk.GetImageFromArray(mask)
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
        "before_touches_border": before["candidate_touches_border"],
        "after_touches_border": after["candidate_touches_border"],
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


def _within_tolerance(after_value, before_value, tolerance: float) -> bool:
    """Accept a usable after value; skip the regression check if the baseline is undefined."""
    if not math.isfinite(float(after_value)):
        return False
    if not math.isfinite(float(before_value)):
        return True
    return float(after_value) <= float(before_value) + tolerance


def _passes_stable_qc(before: dict, after: dict) -> bool:
    """Require overlap improvement without material surface/center regression."""
    return (
        math.isfinite(float(after["dice"]))
        and float(after["dice"]) > float(before["dice"]) + 1e-6
        and _within_tolerance(after["centroid_displacement_mm"],
                              before["centroid_displacement_mm"], 0.5)
        and _within_tolerance(after["hd95_mm"], before["hd95_mm"], 1.0)
        and _within_tolerance(after["assd_mm"], before["assd_mm"], 0.25)
    )


HOLDOUT_LABELS = (*range(92, 116), 116)  # ribs + sternum, not used by the vertebrae fit


def _holdout_tre(fixed_seg: sitk.Image, moving_seg: sitk.Image,
                 transform: sitk.Transform, labels) -> float:
    """Mean target registration error on labels that were not used for the fit."""
    fixed_stats = sitk.LabelShapeStatisticsImageFilter()
    moving_stats = sitk.LabelShapeStatisticsImageFilter()
    fixed_stats.Execute(fixed_seg)
    moving_stats.Execute(moving_seg)
    inverse = transform.GetInverse()
    distances = []
    for label in labels:
        if fixed_stats.HasLabel(label) and moving_stats.HasLabel(label):
            fixed_centroid = np.array(fixed_stats.GetCentroid(label), dtype=float)
            moving_centroid = np.array(moving_stats.GetCentroid(label), dtype=float)
            moved = np.array(
                inverse.TransformPoint(tuple(float(v) for v in moving_centroid)),
                dtype=float,
            )
            distances.append(float(np.linalg.norm(moved - fixed_centroid)))
    return float(np.mean(distances)) if distances else float("inf")


def _pid_alive(pid: int) -> bool:
    """Check process liveness without killing it (os.kill(pid, 0) kills on Windows)."""
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(
                handle, ctypes.byref(exit_code)
            )
            return bool(ok) and exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _landmark_label_sets(registration_labels: list[int]):
    """Return robust landmark label subsets, vertebrae first.

    Ribs are frequently truncated asymmetrically between sessions, which biases
    the Kabsch centroid fit; vertebrae are more consistently visible and give a
    stable rigid estimate. The full set is kept as a fallback.
    """
    sets: list[tuple[list[int], str]] = []
    vertebrae = [label for label in registration_labels if label in VERTEBRA_LABELS]
    if len(vertebrae) >= 3:
        sets.append((vertebrae, "landmark_rigid_vertebrae"))
    sets.append((list(registration_labels), "landmark_rigid_fallback"))
    return sets


def _run(args: argparse.Namespace, output: Path) -> None:
    print("[1/5] CT ve segmentation dosyaları okunuyor", flush=True)
    ct_interpolator = {
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
        "windowed_sinc": sitk.sitkLanczosWindowedSinc,
    }[args.ct_interpolator]
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
    fixed_labels = np.unique(sitk.GetArrayViewFromImage(fixed_seg))
    moving_labels = np.unique(sitk.GetArrayViewFromImage(moving_seg))
    labels = args.labels or sorted(
        int(value) for value in np.intersect1d(fixed_labels, moving_labels)
        if int(value) != 0
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
    state = None
    if args.force:
        state = None
    if state_path.exists():
        if not args.force:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if {key: state.get(key) for key in metadata} != metadata:
                raise RuntimeError("Output klasörü farklı bir pilotla daha önce kullanılmış")
            can_resume_registration = bool(state.get("registered"))
    if state is None:
        state = {**metadata, "registered": False, "metrics": []}
        _atomic_json(state_path, state)
        (output / "metrics.partial.csv").unlink(missing_ok=True)

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
        selected_tre = float("inf")
        baseline_displacements = [
            centroid_displacement(fixed_seg, moving_seg, label)
            for label in labels
        ]
        valid_displacements = [
            value for value in baseline_displacements if math.isfinite(value)
        ]
        max_displacement = max(valid_displacements, default=0.0)
        if not valid_displacements:
            registration_method = "identity_no_valid_target_centroid"
            # The target organ is outside the current fixed grid, but stable
            # structures may still provide a valid physical translation.
            max_displacement = args.skip_threshold_mm + 1.0
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
                ct_interpolator,
                default_value=-1024.0,
            )
            registered_seg = moving_seg_on_fixed
            stable_after = stable_before
            accepted = False
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
            # Evaluate every candidate transform and keep the one that maximises
            # stable bone-mask Dice, so a weakly-improving rigid estimate can no
            # longer mask a much better landmark solution.
            accepted_candidates = []
            if candidate_transform is not None:
                print("[3/5] Transform CT ve maskeye uygulanıyor", flush=True)
                output_transform = candidate_transform
                candidate_stable = resample_mask(
                    moving_stable_mask, fixed_stable_mask, output_transform
                )
                candidate_score = _stable_score(fixed_stable_mask, candidate_stable)
                candidate_transform_path = output / "candidate_moving_to_fixed.tfm"
                sitk.WriteTransform(output_transform, str(candidate_transform_path))
                if _passes_stable_qc(stable_before, candidate_score):
                    accepted_candidates.append(
                        (float(candidate_score["dice"]), output_transform,
                         args.transform_kind, candidate_score)
                    )
                    print(
                        f"[qc] Correlation rigid iyileştirdi "
                        f"(dice={float(candidate_score['dice']):.3f})",
                        flush=True,
                    )
                else:
                    print("[qc] Correlation rigid iyileştirmedi", flush=True)

            for label_set, method_name in _landmark_label_sets(registration_labels):
                landmark_transform = landmark_rigid_output_transform(
                    fixed_seg, moving_seg, label_set
                )
                landmark_stable = resample_mask(
                    moving_stable_mask, fixed_stable_mask, landmark_transform
                )
                landmark_score = _stable_score(fixed_stable_mask, landmark_stable)
                if _passes_stable_qc(stable_before, landmark_score):
                    accepted_candidates.append(
                        (float(landmark_score["dice"]), landmark_transform,
                         method_name, landmark_score)
                    )
                    print(
                        f"[qc] {method_name} uygun "
                        f"(dice={float(landmark_score['dice']):.3f})",
                        flush=True,
                    )

            if accepted_candidates:
                scored = []
                for dice_value, candidate, method_name, score in accepted_candidates:
                    if method_name == "landmark_rigid_fallback":
                        tre = float("inf")  # fits ribs as well; no clean hold-out
                    else:
                        tre = _holdout_tre(fixed_seg, moving_seg, candidate, HOLDOUT_LABELS)
                    scored.append((tre, dice_value, candidate, method_name, score))
                selected_tre, best_dice, transform, registration_method, stable_after = min(
                    scored, key=lambda item: (item[0], -item[1])
                )
                registered_ct = resample_image(
                    moving_ct, fixed_ct, transform,
                    ct_interpolator, default_value=-1024.0
                )
                registered_seg = resample_mask(moving_seg, fixed_seg, transform)
                accepted = True
                print(
                    f"[qc] Seçilen transform: {registration_method} "
                    f"(stable dice={best_dice:.3f})",
                    flush=True,
                )
            else:
                fallback_transform = translation_centroid_output_transform(
                    fixed_stable_mask, moving_stable_mask
                )
                fallback_stable = resample_mask(
                    moving_stable_mask, fixed_stable_mask, fallback_transform
                )
                fallback_score = _stable_score(fixed_stable_mask, fallback_stable)
                if _passes_stable_qc(stable_before, fallback_score):
                    transform = fallback_transform
                    registered_ct = resample_image(
                        moving_ct, fixed_ct, transform,
                        ct_interpolator, default_value=-1024.0
                    )
                    registered_seg = resample_mask(moving_seg, fixed_seg, transform)
                    stable_after = fallback_score
                    accepted = True
                    registration_method = "centroid_translation_fallback"
                    selected_tre = _holdout_tre(
                        fixed_seg, moving_seg, fallback_transform, HOLDOUT_LABELS
                    )
                    print("[qc] Centroid translation fallback kabul edildi", flush=True)
                else:
                    transform = sitk.Transform(3, sitk.sitkIdentity)
                    registered_ct = resample_image(
                        moving_ct, fixed_ct, transform,
                        ct_interpolator, default_value=-1024.0
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
            "holdout_tre_mm": None if not math.isfinite(selected_tre) else round(selected_tre, 3),
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
        except (FileNotFoundError, ValueError):
            previous_pid = None
        if previous_pid is not None and _pid_alive(previous_pid):
            raise RuntimeError(f"Bu output klasörü zaten çalışıyor: {lock_path}")
        lock_path.unlink(missing_ok=True)
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
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
    ap.add_argument("--ct-interpolator", choices=("linear", "bspline", "windowed_sinc"), default="linear")
    ap.add_argument("--force", action="store_true")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
