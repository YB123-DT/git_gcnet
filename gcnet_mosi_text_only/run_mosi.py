"""Run five strict CMU-MOSI Text-only seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


SEEDS = (66, 67, 68, 69, 70)


@dataclass(frozen=True)
class Job:
    seed: int
    gpu: int
    output_dir: Path
    command: tuple[str, ...]


def build_jobs(output_root, python, feature_root, gpus):
    if not gpus or 4 in gpus:
        raise ValueError("use healthy GPUs and exclude GPU 4")
    jobs = []
    for index, seed in enumerate(SEEDS):
        output = Path(output_root) / f"seed_{seed}"
        jobs.append(Job(seed, gpus[index % len(gpus)], output, (
            str(python), "-u", "-m", "gcnet_mosi_text_only.train_mosi",
            "--feature-root", str(feature_root), "--output-dir", str(output),
            "--seed", str(seed), "--epochs", "100", "--device", "cuda",
        )))
    return jobs


def _complete(job):
    try:
        metrics = json.loads((job.output_dir / "metrics.json").read_text())
        history = json.loads((job.output_dir / "history.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return metrics.get("variant") == "strict-text-only-bigru" and metrics.get("seed") == job.seed and len(history) == 100 and not metrics.get("collapsed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1, 2, 3))
    args = parser.parse_args()
    jobs = build_jobs(args.output_root.resolve(), args.python, args.feature_root, tuple(args.gpus))
    processes = []
    for job in jobs:
        if _complete(job):
            continue
        job.output_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        log = (job.output_dir / "train.log").open("w", encoding="utf-8")
        process = subprocess.Popen(job.command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        processes.append((job, process, log))
    failures = []
    for job, process, log in processes:
        code = process.wait()
        log.close()
        if code or not _complete(job):
            failures.append({"seed": job.seed, "returncode": code})
    completed = [job for job in jobs if _complete(job)]
    rows = []
    for job in completed:
        metrics = json.loads((job.output_dir / "metrics.json").read_text())
        rows.append({"seed": job.seed, "best_epoch": metrics["best_epoch"], "validation_wf1": metrics["validation"]["weighted_f1"], "test_wf1": metrics["test"]["weighted_f1"], "runtime_seconds": metrics["runtime_seconds"]})
    summary = {"variant": "strict-text-only-bigru", "rows": rows, "mean_test_wf1": sum(row["test_wf1"] for row in rows) / len(rows) if rows else None, "failures": failures}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures or len(rows) != len(SEEDS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
