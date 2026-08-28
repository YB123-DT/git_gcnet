#!/usr/bin/env python3
"""Run the paired IEMOCAP-6 Single-View PLCI Stage-1 experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METHOD = "plci-single"
DATASET = "IEMOCAPSix"
FOLD = 5
STAGE1_RATES = (0.0, 0.5, 0.7)
STAGE1_SEEDS = (66, 67, 68, 69, 70)
ALLOWED_RATES = tuple(index / 10.0 for index in range(8))
ALLOWED_SEEDS = STAGE1_SEEDS
DEFAULT_GPUS = (0, 1, 2)
DEFAULT_DUAL_VIEW_ROOT = Path(
    "/data2/yb/paper/experiments/plci_jepa_iemocap6_20260826/formal"
)


@dataclass(frozen=True)
class SingleViewJob:
    method: str
    rate: float
    seed: int
    gpu: int
    slot: int
    epochs: int
    output_dir: Path
    dual_view_dir: Path
    command: Tuple[str, ...]

    @property
    def identity(self) -> str:
        return "{}:{}:fold{}:rate{:.1f}:seed{}".format(
            DATASET,
            self.method,
            FOLD,
            self.rate,
            self.seed,
        )


def _rate_directory(rate: float) -> str:
    return "miss_{:.1f}".format(rate)


def _normalize_selection(values, allowed, label):
    normalized = tuple(values)
    if not normalized:
        raise ValueError("at least one {} is required".format(label))
    if len(normalized) != len(set(normalized)):
        raise ValueError("{} contains duplicates".format(label))
    unsupported = tuple(value for value in normalized if value not in allowed)
    if unsupported:
        raise ValueError("unsupported {}: {}".format(label, unsupported))
    return normalized


def _training_command(
    python: str,
    rate: float,
    seed: int,
    epochs: int,
    output_dir: Path,
) -> Tuple[str, ...]:
    return (
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
        DATASET,
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
        "0",
        "--loss-recon",
        "--jepa-weight",
        "0.1",
        "--jepa-architecture",
        METHOD,
        "--model-variant",
        "addon",
        "--fold",
        str(FOLD),
        "--output-dir",
        str(output_dir),
    )


def build_jobs(
    *,
    output_root: Path,
    python: str,
    gpus: Sequence[int] = DEFAULT_GPUS,
    jobs_per_gpu: int = 3,
    rates: Sequence[float] = STAGE1_RATES,
    seeds: Sequence[int] = STAGE1_SEEDS,
    epochs: int = 100,
    dual_view_root: Path = DEFAULT_DUAL_VIEW_ROOT,
) -> List[SingleViewJob]:
    normalized_gpus = tuple(int(gpu) for gpu in gpus)
    if not normalized_gpus:
        raise ValueError("at least one GPU is required")
    if 4 in normalized_gpus:
        raise ValueError("GPU 4 is excluded")
    if len(normalized_gpus) != len(set(normalized_gpus)):
        raise ValueError("GPU list contains duplicates")
    if not 1 <= jobs_per_gpu <= 3:
        raise ValueError("jobs_per_gpu must be between 1 and 3")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    normalized_rates = tuple(
        float(value)
        for value in _normalize_selection(
            tuple(round(float(rate), 1) for rate in rates),
            ALLOWED_RATES,
            "rate",
        )
    )
    normalized_seeds = tuple(
        int(value)
        for value in _normalize_selection(
            tuple(int(seed) for seed in seeds),
            ALLOWED_SEEDS,
            "seed",
        )
    )
    lanes = tuple(
        (gpu, slot)
        for gpu in normalized_gpus
        for slot in range(jobs_per_gpu)
    )
    jobs = []
    for index, (rate, seed) in enumerate(
        (rate, seed)
        for rate in normalized_rates
        for seed in normalized_seeds
    ):
        gpu, slot = lanes[index % len(lanes)]
        relative = Path(_rate_directory(rate)) / "seed_{}".format(seed)
        output_dir = Path(output_root) / relative
        jobs.append(
            SingleViewJob(
                method=METHOD,
                rate=rate,
                seed=seed,
                gpu=gpu,
                slot=slot,
                epochs=epochs,
                output_dir=output_dir,
                dual_view_dir=Path(dual_view_root) / relative,
                command=_training_command(
                    python,
                    rate,
                    seed,
                    epochs,
                    output_dir,
                ),
            )
        )
    return jobs


def _latest_manifest(directory: Path) -> Path | None:
    manifests = sorted(directory.glob("run_records/*/run_manifest_fold_5.json"))
    return manifests[-1] if manifests else None


def audit_dual_view_controls(jobs: Sequence[SingleViewJob]) -> None:
    for job in jobs:
        manifest_path = _latest_manifest(job.dual_view_dir)
        if manifest_path is None:
            raise FileNotFoundError(
                "missing inherited Dual-View manifest: {}".format(
                    job.dual_view_dir
                )
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run"] != {
            "dataset": DATASET,
            "fold": FOLD,
            "master_seed": job.seed,
        }:
            raise ValueError("Dual-View run identity mismatch: {}".format(manifest_path))
        if manifest["masks"]["requested_missing_rate"] != job.rate:
            raise ValueError("Dual-View missing rate mismatch: {}".format(manifest_path))
        if manifest["lifecycle"]["evaluation_protocol"] != "official":
            raise ValueError("Dual-View lifecycle mismatch: {}".format(manifest_path))
        if manifest["method"]["jepa_weight"] != 0.1:
            raise ValueError("Dual-View JEPA weight mismatch: {}".format(manifest_path))
        hashes = manifest["masks"].get("config_hashes", {})
        if set(hashes) != {"train", "validation", "test"}:
            raise ValueError("Dual-View mask hashes are incomplete: {}".format(manifest_path))


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".{}-".format(path.name),
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def is_complete(job: SingleViewJob) -> bool:
    status_path = job.output_dir / "status.json"
    metrics_path = job.output_dir / "fold_metrics.json"
    if not status_path.is_file() or not metrics_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    archives = tuple(job.output_dir.glob("*.npz"))
    return (
        status.get("identity") == job.identity
        and status.get("return_code") == 0
        and len(archives) == 1
    )


def _assert_fresh_or_complete(job: SingleViewJob) -> None:
    if is_complete(job):
        return
    if job.output_dir.exists() and any(job.output_dir.iterdir()):
        raise RuntimeError(
            "refusing to overwrite partial output: {}".format(job.output_dir)
        )


def _run_job(job: SingleViewJob) -> Dict[str, object]:
    if is_complete(job):
        return {"identity": job.identity, "status": "inherited"}
    _assert_fresh_or_complete(job)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        job.output_dir / "command.json",
        {**asdict(job), "command": list(job.command)},
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    started = time.time()
    log_path = job.output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            job.command,
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = {
        "identity": job.identity,
        "return_code": int(process.returncode),
        "elapsed_seconds": time.time() - started,
        "gpu": job.gpu,
        "slot": job.slot,
        "status": "success" if process.returncode == 0 else "failed",
    }
    _write_json_atomic(job.output_dir / "status.json", status)
    if process.returncode != 0:
        raise RuntimeError("training failed: {}".format(job.identity))
    if not is_complete(job):
        raise RuntimeError("training returned success without complete artifacts: {}".format(job.identity))
    return status


def run_jobs(jobs: Sequence[SingleViewJob]) -> List[Dict[str, object]]:
    lanes: Dict[Tuple[int, int], List[SingleViewJob]] = {}
    for job in jobs:
        lanes.setdefault((job.gpu, job.slot), []).append(job)

    def run_lane(lane_jobs):
        return [_run_job(job) for job in lane_jobs]

    results = []
    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = [executor.submit(run_lane, lane_jobs) for lane_jobs in lanes.values()]
        for future in futures:
            results.extend(future.result())
    return results


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS)
    parser.add_argument("--jobs-per-gpu", type=int, default=3)
    parser.add_argument(
        "--missing-rates",
        type=float,
        nargs="+",
        default=list(STAGE1_RATES),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(STAGE1_SEEDS),
    )
    parser.add_argument("--audit-dual-view-controls", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    jobs = build_jobs(
        output_root=args.output_root,
        python=args.python,
        gpus=args.gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        rates=args.missing_rates,
        seeds=args.seeds,
        epochs=args.epochs,
    )
    if args.audit_dual_view_controls:
        audit_dual_view_controls(jobs)
    if args.dry_run:
        print(json.dumps([asdict(job) for job in jobs], indent=2, default=str))
        return 0
    run_jobs(jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
