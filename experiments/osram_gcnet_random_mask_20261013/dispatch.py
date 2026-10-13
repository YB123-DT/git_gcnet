"""Dispatch IEMOCAP on GPU0/1 and MELD on GPU3."""

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
    3: "GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: object):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _jobs(root: Path) -> list[dict]:
    jobs = []
    for classes, gpu in ((4, 0), (6, 1)):
        for seed in (66, 67, 68):
            for fold in range(1, 6):
                name = f"iemocap{classes}_flat_s{seed}_f{fold}"
                jobs.append({
                    "name": name, "gpu": gpu, "kind": "iemocap",
                    "command": [sys.executable, "-u", "-m",
                        "experiments.osram_gcnet_random_mask_20261013.run",
                        "--classes", str(classes), "--seed", str(seed), "--fold", str(fold)],
                })
    for seed in (66, 67, 68):
        for variant in ("flat", "nested"):
            name = f"meld_{variant}_s{seed}"
            command = [sys.executable, "-u", "-m",
                "experiments.osram_ecerc_features_20261012.run_meld",
                "--seed", str(seed), "--epochs", "50", "--model-variant", variant,
                "--missing-mask-protocol", "gcnet-random-mask-v1",
                "--output", str(root / "meld" / name), "--discard-checkpoints"]
            if variant == "nested":
                command += ["--meaningful-block", "meld_nested_d32_l4_g2_m64"]
            jobs.append({"name": name, "gpu": 3, "kind": "meld", "command": command})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    source = Path(__file__).resolve().parents[2]
    jobs = _jobs(root)
    state = {"status": "running", "started_utc": _now(), "jobs": jobs,
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()}
    for job in jobs:
        job["status"] = "pending"
    _write(root / "DISPATCH.json", state)
    active = []
    limits = {0: 3, 1: 3, 3: 6}
    while any(job["status"] == "pending" for job in jobs) or active:
        for process, stream, job in list(active):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            job.update(status="complete" if code == 0 else "failed", exit_code=code,
                       finished_utc=_now())
            active.remove((process, stream, job))
        for gpu, limit in limits.items():
            running = sum(1 for _, _, job in active if job["gpu"] == gpu)
            while running < limit:
                job = next((row for row in jobs if row["status"] == "pending" and row["gpu"] == gpu), None)
                if job is None:
                    break
                log = root / "logs" / f"{job['name']}.log"
                stream = log.open("x")
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_UUIDS[gpu],
                    OSRAM_GCNET_MASK_OUTPUT=str(root), GCNET_CPU_THREADS="1",
                    ECERC_MELD_OSRAM_OUTPUT=str(root / "meld_runtime"),
                    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                process = subprocess.Popen(job["command"], cwd=source, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT)
                job.update(status="running", pid=process.pid, gpu_uuid=GPU_UUIDS[gpu],
                           log=str(log), started_utc=_now())
                active.append((process, stream, job)); running += 1; time.sleep(1)
        _write(root / "DISPATCH.json", state)
        if active:
            time.sleep(15)
    state.update(status="complete" if all(j["status"] == "complete" for j in jobs) else "failed",
                 finished_utc=_now())
    _write(root / "DISPATCH.json", state)
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
