#!/usr/bin/env python3
"""Run the validation-locked CMU-MOSI hidden/window screen."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


HIDDEN_SIZES = (50, 100, 200)
WINDOWS = (1, 2, 3, 4)
FEATURE_ROOT = Path(
    "/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/"
    "dataset/CMUMOSI/features"
)


@dataclass(frozen=True)
class SweepJob:
    seed: int
    gpu: int
    hidden: int
    window: int
    output_dir: Path


def build_jobs(
    *,
    seeds: Sequence[int],
    gpus: Sequence[int],
    output_root: Path,
) -> list[SweepJob]:
    if len(seeds) != len(gpus):
        raise ValueError("seeds and gpus must have the same length")
    jobs = []
    for seed, gpu in zip(seeds, gpus):
        for hidden in HIDDEN_SIZES:
            for window in WINDOWS:
                jobs.append(
                    SweepJob(
                        seed=int(seed),
                        gpu=int(gpu),
                        hidden=hidden,
                        window=window,
                        output_dir=(
                            Path(output_root)
                            / f"seed_{seed}"
                            / f"hidden_{hidden}_window_{window}"
                        ),
                    )
                )
    return jobs


def build_command(
    job: SweepJob,
    *,
    python_executable: Path,
) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "gcnet_missing_m3.train_gcnet",
        "--dataset",
        "CMUMOSI",
        "--feature-root",
        str(FEATURE_ROOT),
        "--audio-feature",
        "wav2vec-large-c-UTT",
        "--text-feature",
        "deberta-large-4-UTT",
        "--video-feature",
        "manet_UTT",
        "--output-dir",
        str(job.output_dir),
        "--seed",
        str(job.seed),
        "--fold",
        "1",
        "--epochs",
        "100",
        "--batch-size",
        "32",
        "--train-rate-mode",
        "all",
        "--hidden",
        str(job.hidden),
        "--fusion-type",
        "slot",
        "--representation-type",
        "slot",
        "--mosi-task-mode",
        "regression",
        "--graph-branch-mode",
        "both",
        "--lr",
        "0.0005",
        "--l2",
        "1e-05",
        "--jepa-weight",
        "0.1",
        "--windowp",
        str(job.window),
        "--windowf",
        str(job.window),
        "--num-threads",
        "2",
    ]


def pending_jobs(jobs: Sequence[SweepJob]) -> list[SweepJob]:
    return [job for job in jobs if not (job.output_dir / "metrics.json").is_file()]


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_manifest(path: Path, jobs: Sequence[SweepJob]) -> None:
    _atomic_json(
        path,
        {
            "condition": {
                "dataset": "CMUMOSI",
                "time_attention": False,
                "hidden_sizes": list(HIDDEN_SIZES),
                "windows": list(WINDOWS),
                "train_rate_mode": "all",
                "num_threads": 2,
                "selection_metric": "validation_8rate_mean_weighted_f1",
            },
            "jobs": [
                {
                    "seed": job.seed,
                    "gpu": job.gpu,
                    "hidden": job.hidden,
                    "window": job.window,
                    "output_dir": str(job.output_dir),
                }
                for job in jobs
            ],
        },
    )


def run_jobs(
    jobs: Sequence[SweepJob],
    *,
    python_executable: Path,
    repo_root: Path,
) -> int:
    processes = []
    for job in pending_jobs(jobs):
        job.output_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(job, python_executable=python_executable)
        log_handle = (job.output_dir / "train.log").open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(job.gpu),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            }
        )
        started_at = time.time()
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        _atomic_json(
            job.output_dir / "status.json",
            {
                "state": "running",
                "pid": process.pid,
                "started_at_unix": started_at,
                "gpu": job.gpu,
                "command": command,
            },
        )
        processes.append((job, process, log_handle, started_at))

    failures = 0
    for job, process, log_handle, started_at in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode != 0:
            failures += 1
        _atomic_json(
            job.output_dir / "status.json",
            {
                "state": "complete" if returncode == 0 else "failed",
                "pid": process.pid,
                "returncode": returncode,
                "started_at_unix": started_at,
                "finished_at_unix": time.time(),
                "gpu": job.gpu,
            },
        )
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(66, 67, 68))
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/data2/yb/remote_experiments/"
            "missing_m3_mosi_hidden_window_sweep_20260829"
        ),
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path("/data2/yb/reproduction_envs/gcnet-official/bin/python"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.seeds) != len(args.gpus):
        raise SystemExit("--seeds and --gpus must contain the same number of values")
    if 4 in args.gpus:
        raise SystemExit("GPU 4 is excluded by the workspace protocol")
    jobs = build_jobs(
        seeds=args.seeds,
        gpus=args.gpus,
        output_root=args.output_root,
    )
    write_manifest(args.output_root / "manifest.json", jobs)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "jobs": len(jobs),
                    "pending": len(pending_jobs(jobs)),
                    "per_gpu": {
                        str(gpu): sum(job.gpu == gpu for job in jobs)
                        for gpu in args.gpus
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    failures = run_jobs(
        jobs,
        python_executable=args.python_executable,
        repo_root=args.repo_root,
    )
    _atomic_json(
        args.output_root / "runner_status.json",
        {
            "state": "complete" if failures == 0 else "failed",
            "failures": failures,
            "jobs": len(jobs),
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
