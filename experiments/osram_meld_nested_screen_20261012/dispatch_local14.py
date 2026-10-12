"""Run one compact 14-configuration MELD Nested refinement on GPUs 2 and 3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gcnet_missing_m3.meaningful_new40_registry import MELD_NESTED_LOCAL14


GPU_UUIDS = {
    2: "GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4",
    3: "GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _check_gpus() -> None:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    )
    found = {
        int(index.strip()): identifier.strip()
        for line in raw.splitlines()
        for index, identifier in [line.split(",")]
    }
    for index, identifier in GPU_UUIDS.items():
        if found.get(index) != identifier:
            raise RuntimeError(f"GPU {index} failed the UUID whitelist")


def build_jobs() -> list[dict]:
    jobs = [
        {
            "name": method,
            "seed": 66,
            "variant": "nested",
            "meaningful_block": method,
            "gpu": 2 + index % 2,
        }
        for index, method in enumerate(sorted(MELD_NESTED_LOCAL14))
    ]
    if len(jobs) != 14:
        raise AssertionError("the local refinement must contain exactly 14 runs")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.epochs != 50:
        raise ValueError("the local refinement is locked to 50 epochs")
    _check_gpus()

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "runs").mkdir()
    source = Path(__file__).resolve().parents[2]
    jobs = build_jobs()
    state = {
        "status": "running",
        "started_utc": _now(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "epochs": args.epochs,
        "selection_metric": "accuracy",
        "protocol": "MELD compact Nested local refinement; per-rate Test-oracle",
        "gpu_uuid_whitelist": GPU_UUIDS,
        "runs": jobs,
    }
    _write(root / "DISPATCH.json", state)

    active = []
    for job in jobs:
        output = root / "runs" / job["name"]
        log = root / "logs" / f"{job['name']}.log"
        command = [
            sys.executable,
            "-u",
            "-m",
            "experiments.osram_ecerc_features_20261012.run_meld",
            "--seed",
            str(job["seed"]),
            "--epochs",
            str(args.epochs),
            "--model-variant",
            job["variant"],
            "--meaningful-block",
            job["meaningful_block"],
            "--output",
            str(output),
            "--discard-checkpoints",
        ]
        stream = log.open("x")
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=GPU_UUIDS[job["gpu"]],
            GCNET_CPU_THREADS="2",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            NUMEXPR_NUM_THREADS="2",
        )
        child = subprocess.Popen(
            command,
            cwd=source,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        job.update(
            status="running",
            pid=child.pid,
            output=str(output),
            log=str(log),
            command=command,
            gpu_uuid=GPU_UUIDS[job["gpu"]],
            started_utc=_now(),
        )
        active.append((child, stream, job))
        time.sleep(1)
    _write(root / "DISPATCH.json", state)

    while active:
        for child, stream, job in list(active):
            code = child.poll()
            if code is None:
                continue
            stream.close()
            provenance_path = Path(job["output"]) / "PROVENANCE.json"
            provenance = (
                json.loads(provenance_path.read_text())
                if provenance_path.exists()
                else {}
            )
            complete = code == 0 and provenance.get("status") == "complete"
            job.update(
                status="complete" if complete else "failed",
                exit_code=code,
                finished_utc=_now(),
                error=None if complete else provenance.get("error", "see log"),
            )
            active.remove((child, stream, job))
        _write(root / "DISPATCH.json", state)
        if active:
            time.sleep(20)

    state.update(
        status=(
            "complete"
            if all(job["status"] == "complete" for job in jobs)
            else "failed"
        ),
        finished_utc=_now(),
    )
    _write(root / "DISPATCH.json", state)


if __name__ == "__main__":
    main()
