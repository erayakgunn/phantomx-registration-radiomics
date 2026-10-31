"""Extract MIRP features from registered PhantomX CT/mask pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

ORGAN_RAD = Path(__file__).parents[1] / "organ-radiomics"
sys.path.insert(0, str(ORGAN_RAD))
import config  # noqa: E402
import feature_extraction_mirp as fx  # noqa: E402

import mirp  # noqa: E402


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


ROI_MAP = {"1": "spleen", "52": "aorta"}
MIRP_VERSION = package_version("mirp")


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_metadata(pair_dir: Path) -> dict:
    qc_path = pair_dir / "registration_qc.json"
    if not qc_path.exists():
        return {}
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    return {
        "registration_method": qc.get("registration_method"),
        "target_max_centroid_displacement_mm": qc.get("target_max_centroid_displacement_mm"),
        "stable_before_dice": qc.get("stable_before", {}).get("dice"),
        "stable_after_dice": qc.get("stable_after", {}).get("dice"),
        "holdout_tre_mm": qc.get("holdout_tre_mm"),
    }


def _extract_mode(ct: Path, seg: Path, pair: str, mode: str, method: str,
                  metadata: dict) -> pd.DataFrame:
    result = mirp.extract_features(
        image=str(ct),
        mask=str(seg),
        sample_name=pair,
        roi_name=ROI_MAP,
        export_features=True,
        write_features=False,
        num_cpus=1,
        parallel_backend="none",
        **config.MIRP_SETTINGS,
        **config.MIRP_MODES[mode],
    )
    if not result:
        raise RuntimeError(f"MIRP feature üretmedi: {pair} [{mode}]")
    frame = fx.drop_unreportable(pd.concat(result, ignore_index=True)).copy()
    frame.insert(0, "registration_pair", pair)
    frame.insert(1, "analysis_method", method)
    frame.insert(2, "radiomics_mode", mode)
    for key, value in metadata.items():
        frame[key] = value
    return frame


def _prepare_original(pair_dir: Path, dataset_root: Path) -> tuple[Path, Path] | None:
    if "__" not in pair_dir.name:
        return None
    session, series = pair_dir.name.split("__", 1)
    moving_ct = dataset_root / session / "nifti" / series / "ct.nii.gz"
    registered_seg = pair_dir / "seg_registered.nii.gz"
    transform_path = pair_dir / "moving_to_fixed.tfm"
    if not (moving_ct.exists() and registered_seg.exists() and transform_path.exists()):
        return None
    seg_original = pair_dir / "seg_original.nii.gz"
    if not seg_original.exists():
        ct_image = sitk.ReadImage(str(moving_ct))
        seg_image = sitk.ReadImage(str(registered_seg))
        transform = sitk.ReadTransform(str(transform_path))
        moved = sitk.Resample(
            seg_image, ct_image, transform.GetInverse(),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16,
        )
        sitk.WriteImage(moved, str(seg_original))
    return moving_ct, seg_original


def _process_pair(pair_dir: Path, image_name: str, mask_name: str,
                  state_name: str, output_prefix: str, method: str,
                  force: bool = False, input_kind: str = "registered",
                  dataset_root: Path | None = None) -> dict:
    if input_kind == "original":
        prepared = _prepare_original(pair_dir, dataset_root) if dataset_root else None
        if prepared is None:
            return {"pair": pair_dir.name, "status": "skipped_missing_input"}
        ct, seg = prepared
    else:
        ct = pair_dir / image_name
        seg = pair_dir / mask_name
        if not ct.exists() or not seg.exists():
            return {"pair": pair_dir.name, "status": "skipped_missing_input"}
    state_path = pair_dir / state_name
    if state_path.exists() and not force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete"):
            return {"pair": pair_dir.name, "status": "resumed"}
    metadata = _load_metadata(pair_dir)
    frames = {}
    try:
        for mode in config.MIRP_MODES:
            frames[mode] = _extract_mode(ct, seg, pair_dir.name, mode, method, metadata)
            frames[mode].to_csv(pair_dir / f"{output_prefix}_{mode.lower()}.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        return {"pair": pair_dir.name, "status": "failed", "error": repr(exc)}
    _atomic_json(
        state_path,
        {
            "pair": pair_dir.name,
            "complete": True,
            "modes": sorted(frames),
            "mirp_version": MIRP_VERSION,
            "mirp_settings": config.MIRP_SETTINGS,
            "mode_settings": config.MIRP_MODES,
            "metadata": metadata,
        },
    )
    return {"pair": pair_dir.name, "status": "completed"}


def main(args: argparse.Namespace) -> int:
    if args.input_kind == "registered":
        image_name, mask_name = "ct_registered.nii.gz", "seg_registered.nii.gz"
        state_name, output_prefix, method = "features_registered_state.json", "features_registered", "registration_based"
    elif args.input_kind == "worldspace":
        image_name, mask_name = "ct_worldspace.nii.gz", "seg_worldspace.nii.gz"
        state_name, output_prefix, method = "features_worldspace_state.json", "features_worldspace", "worldspace_identity"
    else:
        image_name, mask_name = "", ""
        state_name, output_prefix, method = "features_original_state.json", "features_original", "registration_original_mask"
        if args.dataset_root is None:
            raise SystemExit("--dataset-root, --input-kind original için zorunlu")
    pairs = sorted(
        path for path in args.registration_root.iterdir()
        if path.is_dir()
    )
    if args.input_kind == "original":
        pairs = [p for p in pairs if (p / "seg_registered.nii.gz").exists() and (p / "moving_to_fixed.tfm").exists()]
    else:
        pairs = [p for p in pairs if (p / image_name).exists() and (p / mask_name).exists()]
    if args.pair_filter:
        pairs = [path for path in pairs if args.pair_filter in path.name]
    if args.limit is not None:
        pairs = pairs[:args.limit]
    args.registration_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.registration_root / "registered_features_manifest.json"
    suffix = "" if args.input_kind != "original" else "_original"
    manifest = {"n_candidates": len(pairs), "input_kind": args.input_kind, "pairs": []}
    workers = max(1, args.workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_pair, pair, image_name, mask_name, state_name,
                        output_prefix, method, args.force, args.input_kind,
                        args.dataset_root): pair
            for pair in pairs
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                pair = futures[future].name
                result = {"pair": pair, "status": "failed", "error": repr(exc)}
            manifest["pairs"].append(result)
            print(f"[features] {result['status']}: {result['pair']}", flush=True)
            _atomic_json(manifest_path, manifest)
    for mode in config.MIRP_MODES:
        files = sorted(args.registration_root.glob(f"*/{output_prefix}_{mode.lower()}.csv"))
        if files:
            combined = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
            combined.to_csv(
                args.registration_root / f"all_features_{output_prefix.split('_', 1)[1]}{suffix}_{mode.lower()}.csv",
                index=False,
            )
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract MIRP features from registered pairs")
    ap.add_argument("registration_root", type=Path)
    ap.add_argument("--dataset-root", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--input-kind", choices=("registered", "worldspace", "original"), default="registered")
    ap.add_argument("--pair-filter")
    ap.add_argument("--force", action="store_true")
    return ap


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
