"""Dispatch the matched correct-mask IEMOCAP Nested completion on GPU0/1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


GPU_UUIDS = {
    0: "GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45",
    1: "GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def jobs(root: Path) -> list[dict]:
    rows = []
    for classes, gpu in ((4, 0), (6, 1)):
        for seed in (66, 67, 68):
            for fold in range(1, 6):
                name = f"iemocap{classes}_nested_s{seed}_f{fold}"
                rows.append({
                    "name": name,
                    "classes": classes,
                    "seed": seed,
                    "fold": fold,
                    "gpu": gpu,
                    "command": [
                        sys.executable,
                        "-u",
                        "-m",
                        "experiments.osram_gcnet_random_mask_20261013.run",
                        "--classes",
                        str(classes),
                        "--seed",
                        str(seed),
                        "--fold",
                        str(fold),
                        "--model-variant",
                        "nested",
                    ],
                    "output": str(root / f"iemocap{classes}" / f"seed_{seed}" / f"fold_{fold}"),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--per-gpu", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    source = Path(__file__).resolve().parents[2]
    rows = jobs(root)
    state = {
        "status": "running",
        "started_utc": _now(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "per_gpu": args.per_gpu,
        "jobs": rows,
    }
    for row in rows:
        row["status"] = "pending"
    _write(root / "DISPATCH.json", state)
    active = []
    while any(row["status"] == "pending" for row in rows) or active:
        for process, stream, row in list(active):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            row.update(
                status="complete" if code == 0 else "failed",
                exit_code=code,
                finished_utc=_now(),
            )
            active.remove((process, stream, row))
        for gpu in GPU_UUIDS:
            running = sum(1 for _, _, row in active if row["gpu"] == gpu)
            while running < args.per_gpu:
                row = next(
                    (item for item in rows if item["status"] == "pending" and item["gpu"] == gpu),
                    None,
                )
                if row is None:
                    break
                log = root / "logs" / f"{row['name']}.log"
                stream = log.open("x")
                environment = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=GPU_UUIDS[gpu],
                    OSRAM_GCNET_MASK_OUTPUT=str(root),
                    GCNET_CPU_THREADS="1",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                )
                process = subprocess.Popen(
                    row["command"], cwd=source, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT,
                )
                row.update(
                    status="running",
                    pid=process.pid,
                    gpu_uuid=GPU_UUIDS[gpu],
                    log=str(log),
                    started_utc=_now(),
                )
                active.append((process, stream, row))
                running += 1
                time.sleep(1)
        _write(root / "DISPATCH.json", state)
        if active:
            time.sleep(15)
    state.update(
        status="complete" if all(row["status"] == "complete" for row in rows) else "failed",
        finished_utc=_now(),
    )
    _write(root / "DISPATCH.json", state)
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
