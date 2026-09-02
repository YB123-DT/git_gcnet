"""Run the current Missing-M3 model on IEMOCAP-4/6 with bounded GPU queues."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DATASETS = ("IEMOCAPSix", "IEMOCAPFour")
SEEDS = (66, 67, 68, 69, 70)


@dataclass(frozen=True)
class Job:
    dataset: str
    seed: int
    gpu: int
    output_dir: Path
    command: tuple[str, ...]


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _gpu_memory_mb(gpu: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip())


def _build_jobs(
    repo_root: Path,
    output_root: Path,
    python: Path,
    gpus: tuple[int, ...],
    datasets: tuple[str, ...] = DATASETS,
    jepa_weight: float = 0.1,
) -> list[Job]:
    jobs: list[Job] = []
    index = 0
    for dataset in datasets:
        for seed in SEEDS:
            gpu = gpus[index % len(gpus)]
            output_dir = output_root / dataset / f"seed_{seed}"
            command = (
                str(python),
                "-u",
                "-m",
                "gcnet_missing_m3.train_gcnet",
                "--dataset",
                dataset,
                "--fold",
                "5",
                "--audio-feature",
                "wav2vec-large-c-UTT",
                "--text-feature",
                "deberta-large-4-UTT",
                "--video-feature",
                "manet_UTT",
                "--output-dir",
                str(output_dir),
                "--seed",
                str(seed),
                "--epochs",
                "100",
                "--batch-size",
                "32",
                "--train-rate-mode",
                "all",
                "--fusion-type",
                "slot",
                "--representation-type",
                "slot",
                "--graph-branch-mode",
                "both",
                "--mmoe-variant",
                "dual-gate",
                "--hidden",
                "200",
                "--latent-dim",
                "256",
                "--num-experts",
                "4",
                "--top-k",
                "2",
                "--windowp",
                "2",
                "--windowf",
                "2",
                "--lr",
                "0.001",
                "--l2",
                "0.00001",
                "--dropout",
                "0.5",
                "--jepa-weight",
                str(jepa_weight),
                "--temperature",
                "0.03",
                "--ema-tau",
                "0.996",
                "--gradient-clip-norm",
                "1.0",
                "--evaluation-protocol",
                "official",
                "--num-threads",
                "2",
            )
            jobs.append(Job(dataset, seed, gpu, output_dir, command))
            index += 1
    return jobs


def _start_job(job: Job, repo_root: Path) -> tuple[subprocess.Popen[str], object]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    environment["PYTHONPATH"] = f"{repo_root}:{repo_root / 'gcnet'}"
    started_at = time.time()
    _atomic_json(
        job.output_dir / "status.json",
        {
            "state": "running",
            "dataset": job.dataset,
            "seed": job.seed,
            "gpu": job.gpu,
            "started_at_unix": started_at,
            "command": list(job.command),
        },
    )
    log_file = (job.output_dir / "train.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        job.command,
        cwd=repo_root,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, log_file


def _is_complete(job: Job) -> bool:
    metrics = job.output_dir / "metrics.json"
    history = job.output_dir / "history.json"
    if not metrics.exists() or not history.exists():
        return False
    try:
        history_payload = json.loads(history.read_text(encoding="utf-8"))
        metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        len(history_payload) == 100
        and int(metrics_payload.get("best_epoch", 0)) in range(1, 101)
        and len(metrics_payload.get("test", {})) == 8
    )


def _runner_payload(
    jobs: list[Job], running: dict[int, tuple[Job, subprocess.Popen[str], object]]
) -> dict[str, object]:
    completed = [job for job in jobs if _is_complete(job)]
    return {
        "total_jobs": len(jobs),
        "completed_jobs": len(completed),
        "running": [
            {"dataset": job.dataset, "seed": job.seed, "gpu": job.gpu, "pid": pid}
            for pid, (job, _, _) in sorted(running.items())
        ],
        "updated_at_unix": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=(2, 3, 7))
    parser.add_argument("--max-concurrent-per-gpu", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    gpus = tuple(args.gpus)
    if not gpus or any(gpu == 4 for gpu in gpus):
        raise ValueError("at least one healthy non-GPU4 device is required")
    if args.max_concurrent_per_gpu <= 0:
        raise ValueError("max-concurrent-per-gpu must be positive")

    jobs = _build_jobs(
        repo_root,
        output_root,
        args.python,
        gpus,
        datasets=tuple(args.datasets),
        jepa_weight=args.jepa_weight,
    )
    if args.dry_run:
        for job in jobs:
            print(job.gpu, job.dataset, job.seed, job.output_dir)
        return 0

    occupied = {gpu: _gpu_memory_mb(gpu) for gpu in gpus}
    occupied = {gpu: value for gpu, value in occupied.items() if value > 512}
    if occupied:
        raise RuntimeError(f"refusing occupied GPUs: {occupied}")

    pending = [job for job in jobs if not _is_complete(job)]
    running: dict[int, tuple[Job, subprocess.Popen[str], object]] = {}
    failures: list[dict[str, object]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_root / "manifest.json",
        {
            "datasets": list(args.datasets),
            "jepa_weight": args.jepa_weight,
            "seeds": list(SEEDS),
            "gpus": list(gpus),
            "max_concurrent_per_gpu": args.max_concurrent_per_gpu,
            "jobs": [
                {
                    **asdict(job),
                    "output_dir": str(job.output_dir),
                    "command": list(job.command),
                }
                for job in jobs
            ],
        },
    )

    while pending or running:
        counts = {gpu: 0 for gpu in gpus}
        for job, _, _ in running.values():
            counts[job.gpu] += 1
        for job in list(pending):
            if counts[job.gpu] >= args.max_concurrent_per_gpu:
                continue
            process, log_file = _start_job(job, repo_root)
            running[process.pid] = (job, process, log_file)
            counts[job.gpu] += 1
            pending.remove(job)

        time.sleep(args.poll_seconds)
        for pid, (job, process, log_file) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log_file.close()
            complete = returncode == 0 and _is_complete(job)
            status = {
                "state": "complete" if complete else "failed",
                "dataset": job.dataset,
                "seed": job.seed,
                "gpu": job.gpu,
                "pid": pid,
                "returncode": returncode,
                "finished_at_unix": time.time(),
                "artifacts_complete": complete,
            }
            _atomic_json(job.output_dir / "status.json", status)
            if not complete:
                failures.append(status)
            del running[pid]
        _atomic_json(output_root / "runner_status.json", _runner_payload(jobs, running))

    _atomic_json(
        output_root / "runner_status.json",
        {**_runner_payload(jobs, running), "failures": failures, "state": "complete"},
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
