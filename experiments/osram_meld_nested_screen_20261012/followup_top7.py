"""Complete the top seven seed-66 MELD grid candidates with seeds 67 and 68."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.osram_meld_nested_screen_20261012.dispatch import (
    GPU_UUIDS,
    _check_gpus,
    _command,
    _write,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean_selected_accuracy(output: Path) -> float:
    metrics = json.loads((output / "metrics.json").read_text())
    per_rate = metrics["selected_accuracy_by_rate"]
    values = [float(per_rate[f"{index / 10:.1f}"]) for index in range(8)]
    return sum(values) / len(values)


def select_top_grid(dispatch: dict, count: int = 7) -> list[dict]:
    candidates = []
    for row in dispatch["runs"]:
        if row.get("kind") != "grid" or row.get("status") != "complete":
            continue
        candidate = dict(row)
        candidate["seed66_mean_accuracy"] = _mean_selected_accuracy(
            Path(row["output"])
        )
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (-row["seed66_mean_accuracy"], row["name"])
    )
    if len(candidates) < count:
        raise RuntimeError(
            f"expected at least {count} completed grid candidates, got {len(candidates)}"
        )
    return candidates[:count]


def _wait_for_screen(path: Path, poll_seconds: int) -> dict:
    while True:
        dispatch = json.loads(path.read_text())
        if dispatch["status"] == "complete":
            return dispatch
        if dispatch["status"] == "failed":
            raise RuntimeError("the seed-66 screen failed; refusing follow-up selection")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.epochs != 50:
        raise ValueError("this follow-up is locked to the 50-epoch screening budget")

    _check_gpus()
    screen_dispatch = args.screen_root.resolve() / "DISPATCH.json"
    dispatch = _wait_for_screen(screen_dispatch, args.poll_seconds)
    selected = select_top_grid(dispatch)

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "runs").mkdir()
    source = Path(__file__).resolve().parents[2]
    jobs = []
    for rank, candidate in enumerate(selected, start=1):
        for seed in (67, 68):
            jobs.append(
                {
                    "name": f"{candidate['name']}_seed{seed}",
                    "rank": rank,
                    "seed": seed,
                    "variant": "nested",
                    "meaningful_block": candidate["meaningful_block"],
                    "seed66_mean_accuracy": candidate["seed66_mean_accuracy"],
                }
            )
    for index, job in enumerate(jobs):
        job["gpu"] = index % 2

    state = {
        "status": "running",
        "started_utc": _now(),
        "source_screen": str(args.screen_root.resolve()),
        "selection_rule": "top-7 grid configurations by seed66 8-rate mean ACC",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "epochs": args.epochs,
        "runs": jobs,
    }
    _write(root / "DISPATCH.json", state)

    active = []
    for job in jobs:
        output = root / "runs" / job["name"]
        log = root / "logs" / f"{job['name']}.log"
        command = _command(job, output, args.epochs)
        stream = log.open("x")
        environment = {
            **dict(os.environ),
            "CUDA_VISIBLE_DEVICES": GPU_UUIDS[job["gpu"]],
            "GCNET_CPU_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
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
