"""Sequential, resumable batch runner for PhantomX registration pairs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


WORKFLOW_VERSION = 5


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


def _process_pair(index: int, total: int, item, fixed_ct: Path,
                  fixed_seg: Path, output: Path, labels: list[int],
                  threads_per_process: int, force: bool) -> dict:
    session, name, ct, seg = item
    pair_output = output / f"{session}__{name}"
    state_path = pair_output / "pilot_state.json"
    print(f"[batch {index}/{total}] {session}/{name}", flush=True)
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
        if not force and state.get("metrics_complete") and state.get("workflow_version") == WORKFLOW_VERSION:
            print(f"[batch {index}/{total}] tamamlanmış, atlanıyor", flush=True)
            return {"session": session, "series": name, "status": "resumed"}

    command = [
        sys.executable,
        str(Path(__file__).with_name("pilot.py")),
        "--fixed-ct", str(fixed_ct),
        "--fixed-seg", str(fixed_seg),
        "--moving-ct", str(ct),
        "--moving-seg", str(seg),
        "--output", str(pair_output),
        "--labels", *[str(label) for label in labels],
    ]
    if force:
        command.append("--force")
    environment = os.environ.copy()
    environment["PHANTOMX_SITK_THREADS"] = str(threads_per_process)
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[2],
        env=environment,
    )
    status = "completed" if result.returncode == 0 else f"failed:{result.returncode}"
    print(f"[batch {index}/{total}] sonuç: {status}", flush=True)
    return {"session": session, "series": name, "status": status}


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
    if args.session_filter:
        moving = [item for item in moving if item[0] == args.session_filter]
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
    total = len(moving)
    threads_per_process = max(1, (os.cpu_count() or 4) // max(1, args.workers))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_pair, index, total, item,
                fixed_ct, fixed_seg, args.output, args.labels,
                threads_per_process, args.force,
            ): index
            for index, item in enumerate(moving, start=1)
        }
        for future in as_completed(futures):
            result = future.result()
            manifest["pairs"].append(result)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if result["status"].startswith("failed:") and args.stop_on_error:
                return int(result["status"].split(":", 1)[1])
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
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--session-filter")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    return ap


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
