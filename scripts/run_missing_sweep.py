from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


RATES = tuple(round(index / 10, 1) for index in range(8))


@dataclass(frozen=True)
class ExperimentJob:
    method: str
    missing_rate: float
    seed: int
    gpu: int
    output_dir: Path
    command: tuple[str, ...]


def _common_arguments(missing_rate: float, seed: int) -> list[str]:
    return [
        "--audio-feature", "wav2vec-large-c-UTT",
        "--text-feature", "deberta-large-4-UTT",
        "--video-feature", "manet_UTT",
        "--dataset", "IEMOCAPSix",
        "--base-model", "LSTM",
        "--windowp", "2",
        "--windowf", "2",
        "--hidden", "200",
        "--lr", "0.001",
        "--dropout", "0.5",
        "--batch-size", "32",
        "--epochs", "100",
        "--seed", str(seed),
        "--mask-type", f"constant-{missing_rate:.1f}",
        "--loss-recon",
    ]


def _rate_directory(rate: float) -> str:
    return f"miss_{rate:.1f}".replace(".", "p")


def build_jobs(
    root: Path,
    python: Path,
    original_gpus: tuple[int, ...],
    jepa_gpus: tuple[int, ...],
    seed: int,
) -> list[ExperimentJob]:
    if not original_gpus or not jepa_gpus:
        raise ValueError("both methods require at least one GPU")
    jobs: list[ExperimentJob] = []
    method_specs = (
        ("original", original_gpus, "original_missing_sweep_seed66_20260818"),
        ("jepa", jepa_gpus, "modality_jepa_seed66_20260818"),
    )
    for method, gpus, experiment_name in method_specs:
        for index, rate in enumerate(RATES):
            output_dir = root / "experiments" / experiment_name / _rate_directory(rate)
            common = _common_arguments(rate, seed)
            if method == "original":
                command = (str(python), "-u", "gcnet/train_gcnet.py", *common)
            else:
                command = (
                    str(python), "-u", "-m", "gcnet_modality_jepa.train_gcnet",
                    *common,
                    "--jepa-weight", "0.1",
                    "--output-dir", str(output_dir / "saved"),
                )
            jobs.append(
                ExperimentJob(
                    method=method,
                    missing_rate=rate,
                    seed=seed,
                    gpu=gpus[index % len(gpus)],
                    output_dir=output_dir,
                    command=tuple(command),
                )
            )
    return jobs


def _gpu_memory_mb(gpu: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.used", "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip())


def assert_gpus_available(gpus: Iterable[int], maximum_idle_memory_mb: int = 512) -> None:
    occupied = {
        gpu: _gpu_memory_mb(gpu)
        for gpu in sorted(set(gpus))
        if _gpu_memory_mb(gpu) > maximum_idle_memory_mb
    }
    if occupied:
        raise RuntimeError(f"refusing to use occupied GPUs: {occupied}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_job(job: ExperimentJob, root: Path) -> bool:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = job.output_dir / "status.json"
    if completion_path.exists():
        previous = json.loads(completion_path.read_text(encoding="utf-8"))
        if previous.get("returncode") == 0:
            return True
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(job.gpu),
            "GCNET_DATASET_ROOT": str(root / "dataset"),
            "PYTHONPATH": f"{root}:{root / 'gcnet'}",
        }
    )
    if job.method == "original":
        environment["GCNET_SAVED_ROOT"] = str(job.output_dir / "saved")
    _write_json(
        job.output_dir / "COMMAND.json",
        {**asdict(job), "output_dir": str(job.output_dir), "command": list(job.command)},
    )
    started_at = time.time()
    peak_memory_mb = _gpu_memory_mb(job.gpu)
    with (job.output_dir / "train.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            job.command,
            cwd=root,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            peak_memory_mb = max(peak_memory_mb, _gpu_memory_mb(job.gpu))
            time.sleep(2.0)
        returncode = process.returncode
    _write_json(
        completion_path,
        {
            "method": job.method,
            "missing_rate": job.missing_rate,
            "seed": job.seed,
            "gpu": job.gpu,
            "returncode": returncode,
            "started_at_unix": started_at,
            "finished_at_unix": time.time(),
            "peak_nvidia_smi_memory_mb": peak_memory_mb,
        },
    )
    return returncode == 0


def run_gpu_queue(gpu: int, jobs: list[ExperimentJob], root: Path) -> bool:
    for job in jobs:
        if not run_job(job, root):
            return False
    return True


def _parse_gpu_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--original-gpus", type=_parse_gpu_list, default=(1, 2))
    parser.add_argument("--jepa-gpus", type=_parse_gpu_list, default=(3, 4))
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    jobs = build_jobs(root, args.python, args.original_gpus, args.jepa_gpus, args.seed)
    if args.dry_run:
        for job in jobs:
            print(job.gpu, job.method, job.missing_rate, job.output_dir)
        return 0
    assert_gpus_available((*args.original_gpus, *args.jepa_gpus))
    queues = {
        gpu: [job for job in jobs if job.gpu == gpu]
        for gpu in (*args.original_gpus, *args.jepa_gpus)
    }
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        results = list(
            executor.map(lambda item: run_gpu_queue(item[0], item[1], root), queues.items())
        )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
