"""Extract MIRP features from registered PhantomX CT/mask pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mirp
import pandas as pd

ORGAN_RAD = Path(__file__).parents[1] / "organ-radiomics"
sys.path.insert(0, str(ORGAN_RAD))
import config  # noqa: E402
import feature_extraction_mirp as fx  # noqa: E402


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


ROI_MAP = {"1": "spleen", "52": "aorta"}


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _extract_mode(ct: Path, seg: Path, pair: str, mode: str) -> pd.DataFrame:
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
    frame.insert(1, "analysis_method", "registration_based")
    frame.insert(2, "radiomics_mode", mode)
    return frame


def _process_pair(pair_dir: Path) -> dict:
    ct = pair_dir / "ct_registered.nii.gz"
    seg = pair_dir / "seg_registered.nii.gz"
    state_path = pair_dir / "features_registered_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete"):
            return {"pair": pair_dir.name, "status": "resumed"}
    frames = {}
    for mode in config.MIRP_MODES:
        frames[mode] = _extract_mode(ct, seg, pair_dir.name, mode)
        frames[mode].to_csv(pair_dir / f"features_registered_{mode.lower()}.csv", index=False)
    _atomic_json(
        state_path,
        {
            "pair": pair_dir.name,
            "complete": True,
            "modes": sorted(frames),
            "mirp_version": "2.5.0",
            "mirp_settings": config.MIRP_SETTINGS,
            "mode_settings": config.MIRP_MODES,
        },
    )
    return {"pair": pair_dir.name, "status": "completed"}


def main(args: argparse.Namespace) -> int:
    pairs = sorted(
        path for path in args.registration_root.iterdir()
        if path.is_dir()
        and (path / "ct_registered.nii.gz").exists()
        and (path / "seg_registered.nii.gz").exists()
    )
    if args.limit is not None:
        pairs = pairs[:args.limit]
    args.registration_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.registration_root / "registered_features_manifest.json"
    manifest = {"n_candidates": len(pairs), "pairs": []}
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_pair, pair): pair for pair in pairs}
        for future in as_completed(futures):
            result = future.result()
            manifest["pairs"].append(result)
            print(f"[features] {result['status']}: {result['pair']}", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for mode in config.MIRP_MODES:
        files = sorted(args.registration_root.glob(f"*/features_registered_{mode.lower()}.csv"))
        if files:
            combined = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
            combined.to_csv(
                args.registration_root / f"all_features_registered_{mode.lower()}.csv",
                index=False,
            )
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract MIRP features from registered pairs")
    ap.add_argument("registration_root", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    return ap


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
