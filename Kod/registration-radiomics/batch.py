"""Sequential, resumable batch runner for PhantomX registration pairs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


WORKFLOW_VERSION = 3


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def series_files(session: Path) -> list[tuple[str, Path, Path]]:
    result = []
    for ct in sorted(session.glob("nifti/*/ct.nii.gz")):
        name = ct.parent.name
        seg = session / "segmentation" / name / "seg.nii.gz"
        if seg.exists():
            result.append((name, ct, seg))
    return result


def _main(args: argparse.Namespace) -> int:
    dataset = args.dataset_root
    sessions = sorted(
        path for path in dataset.iterdir()
        if path.is_dir() and (path / "nifti").is_dir() and (path / "segmentation").is_dir()
    )
    all_series = [(session.name, *item) for session in sessions for item in series_files(session)]
    fixed_candidates = [
        item for item in all_series
        if item[0] == args.fixed_session and item[1].endswith("_take1")
    ]
    if len(fixed_candidates) != 1:
        raise RuntimeError("Fixed session/take1 bulunamadı veya benzersiz değil")
    fixed_session, fixed_name, fixed_ct, fixed_seg = fixed_candidates[0]
    moving = [item for item in all_series if item[1] != fixed_name or item[0] != fixed_session]
    moving.sort(key=lambda item: (item[0], item[1]))
    if args.limit is not None:
        moving = moving[:args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fixed_session": fixed_session,
        "fixed_series": fixed_name,
        "n_candidates": len(moving),
        "pairs": [],
    }
    manifest_path = args.output / "batch_manifest.json"
    for session, name, ct, seg in moving:
        pair_output = args.output / f"{session}__{name}"
        state_path = pair_output / "pilot_state.json"
        print(f"[batch] {session}/{name}", flush=True)
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
            if state.get("metrics_complete") and state.get("workflow_version") == WORKFLOW_VERSION:
                print("[batch]   tamamlanmış, atlanıyor", flush=True)
                manifest["pairs"].append({"session": session, "series": name, "status": "resumed"})
                continue

        command = [
            sys.executable,
            str(Path(__file__).with_name("pilot.py")),
            "--fixed-ct", str(fixed_ct),
            "--fixed-seg", str(fixed_seg),
            "--moving-ct", str(ct),
            "--moving-seg", str(seg),
            "--output", str(pair_output),
            "--labels", *[str(label) for label in args.labels],
        ]
        result = subprocess.run(command, cwd=Path(__file__).parents[2])
        status = "completed" if result.returncode == 0 else f"failed:{result.returncode}"
        manifest["pairs"].append({"session": session, "series": name, "status": status})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if result.returncode != 0 and args.stop_on_error:
            return result.returncode
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[batch] tamamlandı: {manifest_path}", flush=True)
    return 0


def main(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / "batch.lock"
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
            print(f"[batch] zaten çalışıyor: {lock_path}", flush=True)
            return 2
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    try:
        return _main(args)
    finally:
        lock_path.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run resumable PhantomX registration pairs")
    ap.add_argument("dataset_root", type=Path, help="PhantomX/LABELED")
    ap.add_argument("--fixed-session", required=True, help="e.g. 20260805_PCCT_Aorta_Spleen")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--labels", type=int, nargs="+", default=[1, 52])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stop-on-error", action="store_true")
    return ap


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
