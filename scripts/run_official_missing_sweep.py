#!/usr/bin/env python3
"""Run the auditable four-dataset GCNet official-protocol sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DATASETS = ("IEMOCAPFour", "IEMOCAPSix", "CMUMOSI", "CMUMOSEI")
METHODS = ("baseline", "jepa")
RATES = tuple(round(index / 10.0, 1) for index in range(8))
SEEDS = tuple(range(66, 76))


@dataclass(frozen=True)
class OfficialJob:
    dataset: str
    method: str
    missing_rate: float
    seed: int
    gpu: int
    slot: int
    output_dir: Path
    command: Tuple[str, ...]

    @property
    def identity(self) -> str:
        return "{}:{}:{:.1f}:{}".format(
            self.dataset, self.method, self.missing_rate, self.seed
        )


def _rate_directory(rate: float) -> str:
    return "miss_{:.1f}".format(rate).replace(".", "p")


def _common_command(
    python: str,
    dataset: str,
    rate: float,
    seed: int,
    epochs: int,
    output_dir: Path,
) -> List[str]:
    command = [
        python,
        "-u",
        "-m",
        "gcnet_modality_jepa.train_gcnet",
        "--audio-feature",
        "wav2vec-large-c-UTT",
        "--text-feature",
        "deberta-large-4-UTT",
        "--video-feature",
        "manet_UTT",
        "--dataset",
        dataset,
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
        "4",
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--mask-type",
        "constant-{:.1f}".format(rate),
        "--evaluation-protocol",
        "official",
        "--stability-aux-mask-rate",
        "0.1",
        "--stability-recon-weight",
        "0.01",
        "--output-dir",
        str(output_dir),
    ]
    if dataset.startswith("IEMOCAP"):
        command.extend(("--fold", "5"))
    if epochs < 60:
        command.append("--allow-short-run")
    return command


def build_jobs(
    output_root: Path,
    python: str,
    gpus: Sequence[int] = (0, 1, 2, 3, 5),
    jobs_per_gpu: int = 3,
    epochs: int = 100,
) -> List[OfficialJob]:
    normalized_gpus = tuple(int(gpu) for gpu in gpus)
    if not normalized_gpus:
        raise ValueError("at least one GPU is required")
    if 4 in normalized_gpus:
        raise ValueError("broken GPU 4 must be excluded")
    if len(set(normalized_gpus)) != len(normalized_gpus):
        raise ValueError("GPU list contains duplicates")
    if not 1 <= jobs_per_gpu <= 3:
        raise ValueError("jobs_per_gpu must be between 1 and 3")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    lanes = [
        (gpu, slot)
        for gpu in normalized_gpus
        for slot in range(jobs_per_gpu)
    ]
    jobs: List[OfficialJob] = []
    job_index = 0
    for dataset in DATASETS:
        for rate in RATES:
            for seed in SEEDS:
                for method in METHODS:
                    gpu, slot = lanes[job_index % len(lanes)]
                    output_dir = (
                        output_root
                        / dataset
                        / _rate_directory(rate)
                        / "seed_{}".format(seed)
                        / method
                    )
                    command = _common_command(
                        python, dataset, rate, seed, epochs, output_dir
                    )
                    if method == "baseline":
                        command.extend(
                            ("--loss-recon", "--jepa-weight", "0", "--model-variant", "addon")
                        )
                    else:
                        command.extend(
                            ("--jepa-weight", "0.1", "--model-variant", "replacement")
                        )
                    jobs.append(
                        OfficialJob(
                            dataset=dataset,
                            method=method,
                            missing_rate=rate,
                            seed=seed,
                            gpu=gpu,
                            slot=slot,
                            output_dir=output_dir,
                            command=tuple(command),
                        )
                    )
                    job_index += 1
    return jobs


def _latest_manifest(output_dir: Path) -> Path | None:
    manifests = sorted(
        output_dir.glob("run_records/*/run_manifest_fold_*.json")
    )
    return manifests[-1] if manifests else None


def is_complete(output_dir: Path) -> bool:
    status_path = output_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("returncode") == 0 and _latest_manifest(output_dir) is not None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _gpu_memory_mb(gpu: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id={}".format(gpu),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip())


def assert_gpus_available(
    gpus: Iterable[int], maximum_idle_memory_mb: int = 768
) -> None:
    occupied = {}
    for gpu in sorted(set(gpus)):
        memory = _gpu_memory_mb(gpu)
        if memory > maximum_idle_memory_mb:
            occupied[gpu] = memory
    if occupied:
        raise RuntimeError("refusing to use occupied GPUs: {}".format(occupied))


def _environment(root: Path, gpu: int) -> Dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "GCNET_DATASET_ROOT": str(root / "dataset"),
            "GCNET_CACHE_ROOT": "/data2/yb/gcnet_unified_cache",
            "PYTHONPATH": str(root),
        }
    )
    return environment


def run_job(job: OfficialJob, root: Path, stop_event: threading.Event) -> bool:
    if is_complete(job.output_dir):
        return True
    if stop_event.is_set():
        return False
    job.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        job.output_dir / "command.json",
        {
            **asdict(job),
            "output_dir": str(job.output_dir),
            "command": list(job.command),
        },
    )
    started_at = time.time()
    with (job.output_dir / "train.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            job.command,
            cwd=str(root),
            env=_environment(root, job.gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    _write_json(
        job.output_dir / "status.json",
        {
            "identity": job.identity,
            "gpu": job.gpu,
            "slot": job.slot,
            "returncode": result.returncode,
            "started_at_unix": started_at,
            "finished_at_unix": time.time(),
        },
    )
    if result.returncode != 0:
        stop_event.set()
        return False
    return True


def _run_lane(
    jobs: Sequence[OfficialJob], root: Path, stop_event: threading.Event
) -> bool:
    for job in jobs:
        if not run_job(job, root, stop_event):
            return False
    return True


def audit_completed_pairs(
    jobs: Sequence[OfficialJob], root: Path, python: str
) -> int:
    pairs: Dict[Tuple[str, float, int], Dict[str, OfficialJob]] = {}
    for job in jobs:
        pairs.setdefault(
            (job.dataset, job.missing_rate, job.seed), {}
        )[job.method] = job
    failures = 0
    audit_script = root / "scripts" / "audit_paired_runs.py"
    for pair_jobs in pairs.values():
        if set(pair_jobs) != set(METHODS):
            continue
        baseline = pair_jobs["baseline"]
        jepa = pair_jobs["jepa"]
        if not is_complete(baseline.output_dir) or not is_complete(jepa.output_dir):
            continue
        baseline_manifest = _latest_manifest(baseline.output_dir)
        jepa_manifest = _latest_manifest(jepa.output_dir)
        audit_path = baseline.output_dir.parent / "paired_audit.log"
        result = subprocess.run(
            [
                python,
                str(audit_script),
                str(baseline_manifest),
                str(jepa_manifest),
            ],
            cwd=str(root),
            env=_environment(root, baseline.gpu),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        audit_path.write_text(result.stdout, encoding="utf-8")
        failures += int(result.returncode != 0)
    return failures


def _parse_gpus(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        default="/data2/yb/reproduction_envs/gcnet-official/bin/python",
    )
    parser.add_argument("--gpus", type=_parse_gpus, default=(0, 1, 2, 3, 5))
    parser.add_argument("--jobs-per-gpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output_root = args.output_root.resolve()
    jobs = build_jobs(
        output_root,
        args.python,
        args.gpus,
        args.jobs_per_gpu,
        args.epochs,
    )
    _write_json(
        output_root / "task_manifest.json",
        [
            {
                **asdict(job),
                "output_dir": str(job.output_dir),
                "command": list(job.command),
            }
            for job in jobs
        ],
    )
    if args.dry_run:
        print("generated {} jobs".format(len(jobs)))
        return 0

    assert_gpus_available(args.gpus)
    lanes: Dict[Tuple[int, int], List[OfficialJob]] = {
        (gpu, slot): []
        for gpu in args.gpus
        for slot in range(args.jobs_per_gpu)
    }
    for job in jobs:
        lanes[(job.gpu, job.slot)].append(job)
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = [
            executor.submit(_run_lane, lane_jobs, root, stop_event)
            for lane_jobs in lanes.values()
        ]
        success = all(future.result() for future in futures)
    audit_failures = audit_completed_pairs(jobs, root, args.python)
    _write_json(
        output_root / "scheduler_status.json",
        {
            "complete_jobs": sum(is_complete(job.output_dir) for job in jobs),
            "total_jobs": len(jobs),
            "worker_success": success,
            "paired_audit_failures": audit_failures,
        },
    )
    return 0 if success and audit_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
