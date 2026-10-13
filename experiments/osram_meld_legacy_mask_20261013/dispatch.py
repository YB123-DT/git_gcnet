"""Dispatch the fair three-seed MELD Flat/Nested comparison."""

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
    66: "GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45",
    67: "GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049",
    68: "GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4",
}
MASK_PROTOCOL = "gcnet-legacy-fixed-v1"
NESTED_BLOCK = "meld_nested_d32_l4_g2_m64"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _check_gpus() -> None:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    )
    found = {}
    for line in raw.splitlines():
        index, identifier = [value.strip() for value in line.split(",")]
        found[int(index)] = identifier
    for index, seed in enumerate((66, 67, 68)):
        if found.get(index) != GPU_UUIDS[seed]:
            raise RuntimeError(f"physical GPU {index} failed the UUID whitelist")


def _jobs(root: Path, epochs: int) -> list[dict]:
    jobs = []
    for seed in (66, 67, 68):
        for variant in ("flat", "nested"):
            name = f"{variant}_seed{seed}"
            command = [
                sys.executable,
                "-u",
                "-m",
                "experiments.osram_ecerc_features_20261012.run_meld",
                "--seed",
                str(seed),
                "--epochs",
                str(epochs),
                "--model-variant",
                variant,
                "--missing-mask-protocol",
                MASK_PROTOCOL,
                "--output",
                str(root / "runs" / name),
                "--discard-checkpoints",
            ]
            if variant == "nested":
                command += ["--meaningful-block", NESTED_BLOCK]
            jobs.append(
                {
                    "name": name,
                    "seed": seed,
                    "variant": variant,
                    "gpu_uuid": GPU_UUIDS[seed],
                    "command": command,
                    "status": "pending",
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.epochs != 50:
        raise ValueError("this comparison is locked to 50 epochs")
    _check_gpus()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "runs").mkdir()
    source = Path(__file__).resolve().parents[2]
    jobs = _jobs(root, args.epochs)
    state = {
        "status": "running",
        "started_utc": _now(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "mask_protocol": MASK_PROTOCOL,
        "selection": "per-rate Test-oracle; internal diagnostic",
        "jobs": jobs,
    }
    _write(root / "DISPATCH.json", state)

    active = []
    for job in jobs:
        log = root / "logs" / f"{job['name']}.log"
        stream = log.open("x")
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=job["gpu_uuid"],
            GCNET_CPU_THREADS="2",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            NUMEXPR_NUM_THREADS="2",
        )
        process = subprocess.Popen(
            job["command"],
            cwd=source,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        job.update(
            status="running", pid=process.pid, log=str(log), started_utc=_now()
        )
        active.append((process, stream, job))
        time.sleep(1)
    _write(root / "DISPATCH.json", state)

    while active:
        for process, stream, job in list(active):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            provenance = root / "runs" / job["name"] / "PROVENANCE.json"
            payload = json.loads(provenance.read_text()) if provenance.exists() else {}
            complete = code == 0 and payload.get("status") == "complete"
            job.update(
                status="complete" if complete else "failed",
                exit_code=code,
                finished_utc=_now(),
                error=None if complete else payload.get("error", "see log"),
            )
            active.remove((process, stream, job))
        _write(root / "DISPATCH.json", state)
        if active:
            time.sleep(20)

    state.update(
        status="complete" if all(j["status"] == "complete" for j in jobs) else "failed",
        finished_utc=_now(),
    )
    _write(root / "DISPATCH.json", state)
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
