"""Run the locked Original/MPFiLM IEMOCAP-6 fold-5 comparison."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, Iterable, List, Sequence, Tuple


ARMS = ("original", "full")
GATE_RATES = (0.0, 0.7)
FORMAL_RATES = tuple(index / 10 for index in range(8))
GATE_SEEDS = (66, 67)
FORMAL_SEEDS = (66, 67, 68, 69, 70)


@dataclass(frozen=True)
class Job:
    stage: str
    arm: str
    missing_rate: float
    seed: int
    output_directory: Path


def _rate_tag(rate: float) -> str:
    return f"{rate:.1f}".replace(".", "p")


def build_jobs(stage: str, output_root: Path) -> List[Job]:
    if stage == "gate":
        rates, seeds = GATE_RATES, GATE_SEEDS
    elif stage == "formal":
        rates, seeds = FORMAL_RATES, FORMAL_SEEDS
    else:
        raise ValueError(f"unknown stage: {stage!r}")
    jobs = []
    for arm in ARMS:
        for rate in rates:
            for seed in seeds:
                output = (
                    Path(output_root)
                    / stage
                    / arm
                    / f"miss_{_rate_tag(rate)}"
                    / f"seed_{seed}"
                    / "fold_5"
                )
                jobs.append(Job(stage, arm, rate, seed, output))
    return jobs


def build_command(
    job: Job,
    python: Path,
    repository: Path,
    data_root: Path,
    mask_bank_root: Path,
) -> List[str]:
    return [
        str(python),
        "-u",
        str(Path(repository) / "gcnet" / "train_gcnet.py"),
        "--audio-feature",
        "wav2vec-large-c-UTT",
        "--text-feature",
        "deberta-large-4-UTT",
        "--video-feature",
        "manet_UTT",
        "--dataset",
        "IEMOCAPSix",
        "--data-root",
        str(data_root),
        "--base-model",
        "LSTM",
        "--windowp",
        "2",
        "--windowf",
        "2",
        "--hidden",
        "200",
        "--lr",
        "0.001",
        "--dropout",
        "0.5",
        "--batch-size",
        "32",
        "--num-threads",
        "6",
        "--epochs",
        "100",
        "--seed",
        str(job.seed),
        "--mask-seed",
        str(job.seed),
        "--mask-type",
        f"constant-{job.missing_rate:.1f}",
        "--fold-index",
        "5",
        "--graph-conv-variant",
        job.arm,
        "--mask-bank-root",
        str(mask_bank_root),
        "--output-dir",
        str(job.output_directory / "saved"),
        "--loss-recon",
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _completed(job: Job) -> bool:
    status_path = job.output_directory / "status.json"
    if not status_path.exists():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status.get("status") == "success" and bool(
        list((job.output_directory / "saved").glob("*.npz"))
    )


def _launch(
    job: Job,
    gpu: str,
    python: Path,
    repository: Path,
    data_root: Path,
    mask_bank_root: Path,
) -> Tuple[subprocess.Popen, object]:
    job.output_directory.mkdir(parents=True, exist_ok=True)
    command = build_command(job, python, repository, data_root, mask_bank_root)
    _write_json(
        job.output_directory / "command.json",
        {
            "stage": job.stage,
            "arm": job.arm,
            "missing_rate": job.missing_rate,
            "seed": job.seed,
            "fold": 5,
            "gpu": gpu,
            "command": command,
        },
    )
    log_handle = (job.output_directory / "train.log").open("w", encoding="utf-8")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository), str(Path(repository) / "gcnet"))
    )
    environment["GCNET_CACHE_ROOT"] = str(mask_bank_root.parent / "dataset_cache")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONHASHSEED"] = "0"
    process = subprocess.Popen(
        command,
        cwd=repository,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_handle


def run_jobs(
    jobs: Iterable[Job],
    gpus: Sequence[str],
    workers_per_gpu: int,
    python: Path,
    repository: Path,
    data_root: Path,
    mask_bank_root: Path,
) -> None:
    pending = [job for job in jobs if not _completed(job)]
    running: List[Tuple[Job, str, subprocess.Popen, object, float]] = []
    failures = []
    while pending or running:
        counts: Dict[str, int] = {gpu: 0 for gpu in gpus}
        for _, gpu, _, _, _ in running:
            counts[gpu] += 1
        while pending:
            available = [gpu for gpu in gpus if counts[gpu] < workers_per_gpu]
            if not available:
                break
            gpu = min(available, key=lambda item: (counts[item], gpus.index(item)))
            job = pending.pop(0)
            process, handle = _launch(
                job, gpu, python, repository, data_root, mask_bank_root
            )
            running.append((job, gpu, process, handle, time.time()))
            counts[gpu] += 1
        time.sleep(1.0)
        survivors = []
        for job, gpu, process, handle, started_at in running:
            return_code = process.poll()
            if return_code is None:
                survivors.append((job, gpu, process, handle, started_at))
                continue
            handle.close()
            status = {
                "status": "success" if return_code == 0 else "failed",
                "return_code": return_code,
                "elapsed_seconds": time.time() - started_at,
                "gpu": gpu,
            }
            _write_json(job.output_directory / "status.json", status)
            if return_code != 0:
                failures.append(str(job.output_directory))
        running = survivors
    if failures:
        raise RuntimeError(f"failed jobs: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-bank-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/yangbin/miniconda3/envs/multimodalerc310/bin/python"),
    )
    parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3"))
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    repository = Path(__file__).resolve().parents[2]
    jobs = build_jobs(args.stage, args.output_root)
    # Create every bank with Original before Full reads it; this also prevents
    # concurrent first-writer races without changing the model comparison.
    for arm in ARMS:
        run_jobs(
            (job for job in jobs if job.arm == arm),
            tuple(args.gpus),
            args.workers_per_gpu,
            args.python,
            repository,
            args.data_root,
            args.mask_bank_root,
        )


if __name__ == "__main__":
    main()
