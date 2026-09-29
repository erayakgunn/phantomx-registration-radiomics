"""Create identity/world-space resampled CT and masks for the control analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import SimpleITK as sitk

from registration import read_image, resample_image, resample_mask


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def atomic_image(image: sitk.Image, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp.nii.gz")
    sitk.WriteImage(image, str(temporary))
    os.replace(temporary, path)


def main(args: argparse.Namespace) -> None:
    fixed_series = next(
        args.dataset_root.glob(
            f"{args.fixed_session}/nifti/*_take1/ct.nii.gz"
        )
    )
    fixed_ct = read_image(fixed_series, sitk.sitkFloat32)
    pairs = sorted(
        path for path in args.registration_root.iterdir()
        if path.is_dir() and "__" in path.name
    )
    if args.limit is not None:
        pairs = pairs[:args.limit]
    for pair in pairs:
        session, series = pair.name.split("__", 1)
        moving_ct_path = args.dataset_root / session / "nifti" / series / "ct.nii.gz"
        moving_seg_path = args.dataset_root / session / "segmentation" / series / "seg.nii.gz"
        if not moving_ct_path.exists() or not moving_seg_path.exists():
            print(f"[worldspace] eksik girdi: {pair.name}", flush=True)
            continue
        state_path = pair / "worldspace_state.json"
        if state_path.exists() and json.loads(state_path.read_text(encoding="utf-8")).get("complete"):
            print(f"[worldspace] atlandı: {pair.name}", flush=True)
            continue
        moving_ct = read_image(moving_ct_path, sitk.sitkFloat32)
        moving_seg = read_image(moving_seg_path, sitk.sitkUInt16)
        identity = sitk.Transform(3, sitk.sitkIdentity)
        world_ct = resample_image(moving_ct, fixed_ct, identity, sitk.sitkLinear, -1024.0)
        world_seg = resample_mask(moving_seg, fixed_ct, identity)
        atomic_image(world_ct, pair / "ct_worldspace.nii.gz")
        atomic_image(world_seg, pair / "seg_worldspace.nii.gz")
        state_path.write_text(json.dumps({"complete": True}, indent=2), encoding="utf-8")
        print(f"[worldspace] tamamlandı: {pair.name}", flush=True)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Create identity world-space control volumes")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("registration_root", type=Path)
    ap.add_argument("--fixed-session", required=True)
    ap.add_argument("--limit", type=int)
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
