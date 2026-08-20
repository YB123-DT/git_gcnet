#!/usr/bin/env python3
"""Run the eight-job unified-protocol smoke matrix with per-GPU queues."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DATASETS = ("IEMOCAPFour", "IEMOCAPSix", "CMUMOSI", "CMUMOSEI")


@dataclass(frozen=True)
class SmokeJob:
    dataset: str
    method: str
    gpu: int
    output_dir: Path
    command: Tuple[str, ...]


def build_smoke_jobs(
    output_root: Path,
    python: str,
    gpus: Sequence[int] = (0, 1, 2),
    epochs: int = 2,
    seed: int = 66,
) -> List[SmokeJob]:
    if not gpus or 4 in gpus:
        raise ValueError("provide nonempty GPUs and exclude broken GPU 4")
    if len(DATASETS) * 2 > len(gpus) * 3:
        raise ValueError("at most three smoke jobs may share one GPU queue")
    jobs = []
    job_index = 0
    for dataset in DATASETS:
        for method in ("baseline", "jepa"):
            gpu = int(gpus[job_index % len(gpus)])
            output_dir = output_root / dataset / method
            command = [
                python, "-u", "-m", "gcnet_modality_jepa.train_gcnet",
                "--audio-feature", "wav2vec-large-c-UTT",
                "--text-feature", "deberta-large-4-UTT",
                "--video-feature", "manet_UTT",
                "--dataset", dataset,
                "--base-model", "LSTM",
                "--windowp", "2", "--windowf", "2",
                "--hidden", "200", "--lr", "0.001",
                "--dropout", "0.5", "--batch-size", "32",
                "--epochs", str(epochs), "--seed", str(seed),
                "--mask-type", "constant-0.3",
                "--stability-aux-mask-rate", "0.1",
                "--stability-recon-weight", "0.01",
                "--allow-short-run", "--output-dir", str(output_dir),
            ]
            if dataset.startswith("IEMOCAP"):
                command.extend(("--fold", "5"))
            if method == "baseline":
                command.extend(("--loss-recon", "--jepa-weight", "0", "--model-variant", "addon"))
            else:
                command.extend(("--jepa-weight", "0.1", "--model-variant", "replacement"))
            jobs.append(SmokeJob(dataset, method, gpu, output_dir, tuple(command)))
            job_index += 1
    return jobs


def _run_queue(jobs: Sequence[SmokeJob], cwd: Path, environment: Dict[str, str]) -> None:
    for job in jobs:
        if list(job.output_dir.glob("run_records/*/run_manifest_fold_*.json")):
            continue
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job_environment = dict(environment)
        job_environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        log_path = job.output_dir / "smoke.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                job.command,
                cwd=str(cwd),
                env=job_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        status = {"returncode": result.returncode, "command": list(job.command)}
        (job.output_dir / "status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError("smoke job failed: {} {}".format(job.dataset, job.method))


def _latest_manifest(output_dir: Path) -> Path:
    manifests = sorted(output_dir.glob("run_records/*/run_manifest_fold_*.json"))
    if not manifests:
        raise RuntimeError("missing smoke manifest in {}".format(output_dir))
    return manifests[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default="/data2/yb/reproduction_envs/gcnet-official/bin/python")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=66)
    args = parser.parse_args()
    cwd = Path.cwd().resolve()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    jobs = build_smoke_jobs(args.output_root.resolve(), args.python, gpus, args.epochs, args.seed)
    queues = {gpu: [] for gpu in gpus}
    for job in jobs:
        queues[job.gpu].append(job)
    environment = dict(os.environ)
    environment.setdefault("GCNET_DATASET_ROOT", str(cwd / "dataset"))
    environment.setdefault("GCNET_CACHE_ROOT", "/data2/yb/gcnet_unified_cache")
    environment["PYTHONPATH"] = str(cwd)
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [executor.submit(_run_queue, queue, cwd, environment) for queue in queues.values()]
        for future in futures:
            future.result()
    audit_script = cwd / "scripts" / "audit_paired_runs.py"
    for dataset in DATASETS:
        baseline = _latest_manifest(args.output_root.resolve() / dataset / "baseline")
        jepa = _latest_manifest(args.output_root.resolve() / dataset / "jepa")
        subprocess.run(
            [args.python, str(audit_script), str(baseline), str(jepa)],
            cwd=str(cwd),
            env=environment,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
