#!/usr/bin/env python3
"""Run the locked MOSI one-rate-train/one-rate-test experiment matrix.

This runner only launches the registered Slot Missing-M3 treatment. Existing
Original and mixed-rate results are inherited and are never retrained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SEEDS = (66, 67, 68, 69, 70)
MISSING_RATES = tuple(index / 10 for index in range(8))
EXPECTED_GPUS = (0, 1, 2)
FEATURE_ROOT = Path(
    "/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/"
    "dataset/CMUMOSI/features"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data2/yb/remote_experiments/"
    "missing_m3_mosi_fixed_rate_20260830/formal"
)
DEFAULT_PYTHON = Path("/data2/yb/reproduction_envs/gcnet-official/bin/python")
TRAINING_MODULE = "gcnet_missing_m3.train_gcnet"
TREATMENT = "slot-missing-m3-fixed-rate"
SOURCE_FILES = (
    "gcnet_missing_m3/model.py",
    "gcnet_missing_m3/loss.py",
    "gcnet_missing_m3/mixed_rate.py",
    "gcnet_missing_m3/train_gcnet.py",
    "scripts/run_mosi_fixed_rate.py",
)


@dataclass(frozen=True)
class FixedRateJob:
    rate: float
    seed: int
    gpu: int
    output_dir: Path


@dataclass(frozen=True)
class ResultInspection:
    complete: bool
    reason: str


def _rate_key(rate: float) -> str:
    return format(float(rate), ".1f")


def _rate_token(rate: float) -> str:
    return _rate_key(rate).replace(".", "p")


def build_jobs(
    *,
    output_root: Path,
    rates: Sequence[float] = MISSING_RATES,
    seeds: Sequence[int] = SEEDS,
    gpus: Sequence[int] = EXPECTED_GPUS,
) -> list[FixedRateJob]:
    normalized_rates = tuple(float(rate) for rate in rates)
    normalized_seeds = tuple(int(seed) for seed in seeds)
    normalized_gpus = tuple(int(gpu) for gpu in gpus)
    if not normalized_rates or not normalized_seeds or not normalized_gpus:
        raise ValueError("rates, seeds, and gpus must be non-empty")
    if len(set(normalized_rates)) != len(normalized_rates):
        raise ValueError("rates must be unique")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique")
    if any(rate not in MISSING_RATES for rate in normalized_rates):
        raise ValueError("rates must be selected from the official 0.0-0.7 grid")
    if any(gpu == 4 for gpu in normalized_gpus):
        raise ValueError("GPU 4 is forbidden by the remote execution contract")
    if any(gpu not in EXPECTED_GPUS for gpu in normalized_gpus):
        raise ValueError("gpus must be selected from the locked set (0, 1, 2)")

    root = Path(output_root)
    jobs = []
    job_index = 0
    for rate in normalized_rates:
        for seed in normalized_seeds:
            jobs.append(
                FixedRateJob(
                    rate=rate,
                    seed=seed,
                    gpu=normalized_gpus[job_index % len(normalized_gpus)],
                    output_dir=(
                        root
                        / "rate_{}".format(_rate_token(rate))
                        / "seed_{}".format(seed)
                    ),
                )
            )
            job_index += 1
    return jobs


def build_command(
    job: FixedRateJob,
    *,
    python_executable: Path = DEFAULT_PYTHON,
    feature_root: Path = FEATURE_ROOT,
) -> list[str]:
    return [
        str(python_executable),
        "-m",
        TRAINING_MODULE,
        "--dataset",
        "CMUMOSI",
        "--feature-root",
        str(feature_root),
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
        "fixed",
        "--train-missing-rate",
        _rate_key(job.rate),
        "--hidden",
        "200",
        "--latent-dim",
        "256",
        "--num-experts",
        "4",
        "--top-k",
        "2",
        "--mmoe-variant",
        "dual-gate",
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
        "0.00001",
        "--dropout",
        "0.5",
        "--jepa-weight",
        "0.1",
        "--jepa-regression-aggregation",
        "target",
        "--recurrent-padding-mode",
        "legacy",
        "--task-regression-loss",
        "mse",
        "--postgraph-sequence-mode",
        "independent",
        "--jepa-rate-weighting",
        "uniform",
        "--graph-message-calibration",
        "none",
        "--windowp",
        "2",
        "--windowf",
        "2",
        "--evaluation-protocol",
        "official",
        "--device",
        "cuda",
        "--num-threads",
        "2",
    ]


def _read_json(path: Path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, ValueError, TypeError) as error:
        return None, "{} is missing or invalid: {}".format(path.name, error)


def _same_float(left: object, right: float) -> bool:
    return (
        not isinstance(left, bool)
        and isinstance(left, (int, float))
        and math.isfinite(float(left))
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    )


def _config_matches_job(config: object, job: FixedRateJob) -> bool:
    if not isinstance(config, Mapping):
        return False
    exact = {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": job.seed,
        "epochs": 100,
        "batch_size": 32,
        "train_rate_mode": "fixed",
        "hidden": 200,
        "latent_dim": 256,
        "window_past": 2,
        "window_future": 2,
        "time_attention": False,
        "fusion_type": "slot",
        "representation_type": "slot",
        "mmoe_variant": "dual-gate",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "jepa_regression_aggregation": "target",
        "recurrent_padding_mode": "legacy",
        "task_regression_loss": "mse",
        "postgraph_sequence_mode": "independent",
        "jepa_rate_weighting": "uniform",
        "graph_message_calibration": "none",
        "evaluation_protocol": "official",
        "evaluate_test": True,
    }
    if any(config.get(key) != value for key, value in exact.items()):
        return False
    numeric = {
        "fixed_missing_rate": job.rate,
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "jepa_weight": 0.1,
    }
    return all(_same_float(config.get(key), value) for key, value in numeric.items())


def _history_is_complete(history: object) -> bool:
    if not isinstance(history, list) or len(history) != 100:
        return False
    epochs = [
        record.get("epoch") if isinstance(record, Mapping) else None
        for record in history
    ]
    return epochs == list(range(1, 101))


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _prediction_artifact_is_valid(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"predictions", "labels", "availability"}
            if not required.issubset(payload.files):
                return False
            predictions = np.asarray(payload["predictions"])
            labels = np.asarray(payload["labels"])
            availability = np.asarray(payload["availability"])
    except (OSError, ValueError, TypeError):
        return False
    if predictions.ndim != 1 or labels.ndim != 1:
        return False
    if predictions.shape != labels.shape or predictions.size < 1:
        return False
    if availability.ndim != 2 or availability.shape != (predictions.size, 3):
        return False
    return bool(
        np.isfinite(predictions).all()
        and np.isfinite(labels).all()
        and np.isfinite(availability).all()
    )


def inspect_result(job: FixedRateJob) -> ResultInspection:
    """Re-read durable outputs so a stale manifest can never mark a run complete."""
    for name in ("config.json", "history.json", "metrics.json", "train.log"):
        if not (job.output_dir / name).is_file():
            return ResultInspection(False, "missing {}".format(name))

    config, error = _read_json(job.output_dir / "config.json")
    if error is not None:
        return ResultInspection(False, error)
    if not _config_matches_job(config, job):
        return ResultInspection(False, "config.json does not match locked job")

    history, error = _read_json(job.output_dir / "history.json")
    if error is not None:
        return ResultInspection(False, error)
    if not _history_is_complete(history):
        return ResultInspection(False, "history.json must contain exactly 100 epochs")

    metrics, error = _read_json(job.output_dir / "metrics.json")
    if error is not None:
        return ResultInspection(False, error)
    if not isinstance(metrics, Mapping):
        return ResultInspection(False, "metrics.json must contain an object")
    if metrics.get("evaluation_stage") != "train-validation-test":
        return ResultInspection(False, "formal result must include test evaluation")
    if not _same_float(metrics.get("train_missing_rate"), job.rate):
        return ResultInspection(False, "metrics train rate does not match registered rate")
    selection_rates = metrics.get("selection_missing_rates")
    if (
        not isinstance(selection_rates, list)
        or len(selection_rates) != 1
        or not _same_float(selection_rates[0], job.rate)
    ):
        return ResultInspection(False, "selection must use only the registered rate")
    best_epoch = metrics.get("best_epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= 100
    ):
        return ResultInspection(False, "best_epoch must be in the 100-epoch run")

    rate_key = _rate_key(job.rate)
    test_metrics = metrics.get("test")
    mask_hashes = metrics.get("mask_sha256")
    if not isinstance(test_metrics, Mapping) or set(test_metrics) != {rate_key}:
        return ResultInspection(False, "test must contain only the registered rate")
    if not isinstance(mask_hashes, Mapping) or set(mask_hashes) != {rate_key}:
        return ResultInspection(False, "mask hash must contain only the registered rate")
    rate_metrics = test_metrics[rate_key]
    if not isinstance(rate_metrics, Mapping):
        return ResultInspection(False, "registered-rate metrics must be an object")
    weighted_f1 = rate_metrics.get("weighted_f1")
    if (
        isinstance(weighted_f1, bool)
        or not isinstance(weighted_f1, (int, float))
        or not math.isfinite(float(weighted_f1))
        or not 0.0 <= float(weighted_f1) <= 1.0
    ):
        return ResultInspection(False, "weighted_f1 is invalid")
    digest = rate_metrics.get("mask_sha256")
    if not _is_sha256(digest) or mask_hashes[rate_key] != digest:
        return ResultInspection(False, "mask SHA256 provenance is invalid")

    prediction_files = sorted(job.output_dir.glob("predictions_miss_*.npz"))
    expected_prediction = job.output_dir / "predictions_miss_{}.npz".format(
        _rate_token(job.rate)
    )
    if prediction_files != [expected_prediction]:
        return ResultInspection(
            False, "prediction artifact set must contain only the registered rate"
        )
    if not _prediction_artifact_is_valid(expected_prediction):
        return ResultInspection(False, "registered-rate prediction artifact is invalid")
    return ResultInspection(True, "complete-fixed-rate")


def pending_jobs(jobs: Sequence[FixedRateJob]) -> list[FixedRateJob]:
    return [job for job in jobs if not inspect_result(job).complete]


def build_waves(
    jobs: Sequence[FixedRateJob], *, jobs_per_gpu: int
) -> list[list[FixedRateJob]]:
    if isinstance(jobs_per_gpu, bool) or int(jobs_per_gpu) < 1:
        raise ValueError("jobs_per_gpu must be positive")
    capacity = int(jobs_per_gpu)
    grouped = {gpu: [] for gpu in EXPECTED_GPUS}
    for job in jobs:
        if job.gpu not in grouped:
            raise ValueError("job uses a GPU outside the locked mapping")
        grouped[job.gpu].append(job)
    wave_count = max(
        (
            (len(gpu_jobs) + capacity - 1) // capacity
            for gpu_jobs in grouped.values()
        ),
        default=0,
    )
    waves = []
    for wave_index in range(wave_count):
        start = wave_index * capacity
        stop = start + capacity
        wave = []
        for gpu in EXPECTED_GPUS:
            wave.extend(grouped[gpu][start:stop])
        waves.append(wave)
    return waves


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_commit(source_commit: str) -> str:
    normalized = str(source_commit).lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("source_commit must be a full 40-character Git SHA")
    return normalized


def _runtime_provenance(python_executable: Path) -> dict[str, object]:
    command = [
        str(python_executable),
        "-c",
        (
            "import json,platform,torch; "
            "print(json.dumps({'python':platform.python_version(),"
            "'torch':torch.__version__,'cuda':torch.version.cuda}))"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        runtime = json.loads(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        runtime = {"python": sys.version.splitlines()[0], "unavailable": True}
    runtime.update(
        {
            "python_executable": str(python_executable),
            "child_environment": {
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "PYTHONHASHSEED": "0",
            },
        }
    )
    return runtime


def _result_manifest_fields(job: FixedRateJob) -> dict[str, object]:
    config_path = job.output_dir / "config.json"
    metrics_path = job.output_dir / "metrics.json"
    metrics, error = _read_json(metrics_path)
    if error is not None or not isinstance(metrics, Mapping):
        metrics = {}
    return {
        "config_sha256": _sha256_file(config_path) if config_path.is_file() else None,
        "metrics_sha256": (
            _sha256_file(metrics_path) if metrics_path.is_file() else None
        ),
        "mask_sha256": metrics.get("mask_sha256"),
        "best_epoch": metrics.get("best_epoch"),
        "test": metrics.get("test"),
    }


def write_manifest(
    path: Path,
    jobs: Sequence[FixedRateJob],
    *,
    source_commit: str,
    feature_root: Path,
    python_executable: Path,
    repo_root: Path,
) -> None:
    source_commit = _validate_source_commit(source_commit)
    repo_root = Path(repo_root)
    source_hashes = {
        relative_path: _sha256_file(repo_root / relative_path)
        for relative_path in SOURCE_FILES
    }
    entries = []
    for job in jobs:
        inspection = inspect_result(job)
        entries.append(
            {
                "rate": job.rate,
                "seed": job.seed,
                "gpu": job.gpu,
                "output_dir": str(job.output_dir),
                "complete": inspection.complete,
                "completion_reason": inspection.reason,
                "log_path": str(job.output_dir / "train.log"),
                **_result_manifest_fields(job),
            }
        )
    feature_root = Path(feature_root)
    _atomic_json(
        Path(path),
        {
            "schema_version": 1,
            "treatment": TREATMENT,
            "training_module": TRAINING_MODULE,
            "source_commit": source_commit,
            "source_files_sha256": source_hashes,
            "runtime": _runtime_provenance(python_executable),
            "features": {
                "root": str(feature_root),
                "audio": str(feature_root / "wav2vec-large-c-UTT"),
                "text": str(feature_root / "deberta-large-4-UTT"),
                "video": str(feature_root / "manet_UTT"),
            },
            "rates": list(MISSING_RATES),
            "seeds": list(SEEDS),
            "gpus": list(EXPECTED_GPUS),
            "required_epochs": 100,
            "selection_rule": "same-rate validation weighted_f1",
            "test_rule": "same-rate test only",
            "jobs": entries,
        },
    )


def reset_incomplete_output(output_dir: Path) -> None:
    """Clear only this runner's generated files before retrying a partial job."""
    root = Path(output_dir)
    for pattern in ("best.pt", "predictions_miss_*.npz"):
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()
    for name in ("config.json", "history.json", "metrics.json", "status.json", "train.log"):
        path = root / name
        if path.is_file():
            path.unlink()


def _terminate_process_tree(process) -> None:
    try:
        process_group = os.getpgid(process.pid)
    except (OSError, ProcessLookupError):
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process_group, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        return
    try:
        process.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_jobs(
    jobs: Sequence[FixedRateJob],
    *,
    python_executable: Path,
    feature_root: Path,
    repo_root: Path,
    jobs_per_gpu: int,
    timeout_seconds: int,
    manifest_path: Path | None = None,
    source_commit: str | None = None,
) -> int:
    if isinstance(timeout_seconds, bool) or int(timeout_seconds) < 1:
        raise ValueError("timeout_seconds must be positive")
    all_jobs = list(jobs)
    failures = 0
    waves = build_waves(pending_jobs(all_jobs), jobs_per_gpu=jobs_per_gpu)
    for wave_index, wave in enumerate(waves):
        running = []
        handled_processes = set()
        try:
            for job in wave:
                job.output_dir.mkdir(parents=True, exist_ok=True)
                reset_incomplete_output(job.output_dir)
                command = build_command(
                    job,
                    python_executable=python_executable,
                    feature_root=feature_root,
                )
                log_path = job.output_dir / "train.log"
                log_handle = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(job.gpu),
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                        "PYTHONHASHSEED": "0",
                    }
                )
                started_at = time.time()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=Path(repo_root),
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log_handle.close()
                    raise
                _atomic_json(
                    job.output_dir / "status.json",
                    {
                        "state": "running",
                        "pid": process.pid,
                        "rate": job.rate,
                        "seed": job.seed,
                        "gpu": job.gpu,
                        "wave": wave_index,
                        "command": command,
                        "log_path": str(log_path),
                        "started_at_unix": started_at,
                    },
                )
                running.append((job, process, log_handle, command, started_at))

            for job, process, log_handle, command, started_at in running:
                timed_out = False
                returncode = None
                try:
                    remaining = max(0.0, started_at + int(timeout_seconds) - time.time())
                    returncode = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_tree(process)
                finally:
                    log_handle.close()
                handled_processes.add(id(process))

                inspection = inspect_result(job)
                if inspection.complete:
                    state = "complete"
                elif timed_out:
                    state = "timeout"
                    failures += 1
                else:
                    state = "failed"
                    failures += 1
                _atomic_json(
                    job.output_dir / "status.json",
                    {
                        "state": state,
                        "pid": process.pid,
                        "rate": job.rate,
                        "seed": job.seed,
                        "gpu": job.gpu,
                        "wave": wave_index,
                        "command": command,
                        "returncode": returncode,
                        "timeout_seconds": int(timeout_seconds),
                        "completion_reason": inspection.reason,
                        "log_path": str(job.output_dir / "train.log"),
                        "started_at_unix": started_at,
                        "finished_at_unix": time.time(),
                    },
                )
                if manifest_path is not None:
                    if source_commit is None:
                        raise ValueError("source_commit is required for manifest updates")
                    write_manifest(
                        Path(manifest_path),
                        all_jobs,
                        source_commit=source_commit,
                        feature_root=feature_root,
                        python_executable=python_executable,
                        repo_root=repo_root,
                    )
        finally:
            for _, process, log_handle, _, _ in running:
                if id(process) not in handled_processes:
                    _terminate_process_tree(process)
                if not log_handle.closed:
                    log_handle.close()
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpus", type=int, nargs="+", default=EXPECTED_GPUS)
    parser.add_argument("--jobs-per-gpu", type=int, default=5)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        jobs = build_jobs(output_root=args.output_root, gpus=args.gpus)
        source_commit = _validate_source_commit(args.source_commit)
    except ValueError as error:
        raise SystemExit(str(error))
    if args.jobs_per_gpu < 1:
        raise SystemExit("--jobs-per-gpu must be positive")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")

    manifest_path = args.output_root / "manifest.json"
    write_manifest(
        manifest_path,
        jobs,
        source_commit=source_commit,
        feature_root=args.feature_root,
        python_executable=args.python_executable,
        repo_root=args.repo_root,
    )
    if args.dry_run:
        for job in jobs:
            command = build_command(
                job,
                python_executable=args.python_executable,
                feature_root=args.feature_root,
            )
            print("COMMAND {}".format(shlex.join(command)))
        print(
            json.dumps(
                {
                    "jobs": len(jobs),
                    "pending": len(pending_jobs(jobs)),
                    "waves": len(
                        build_waves(pending_jobs(jobs), jobs_per_gpu=args.jobs_per_gpu)
                    ),
                    "jobs_per_gpu": args.jobs_per_gpu,
                    "manifest": str(manifest_path),
                },
                sort_keys=True,
            )
        )
        return 0

    failures = run_jobs(
        jobs,
        python_executable=args.python_executable,
        feature_root=args.feature_root,
        repo_root=args.repo_root,
        jobs_per_gpu=args.jobs_per_gpu,
        timeout_seconds=args.timeout_seconds,
        manifest_path=manifest_path,
        source_commit=source_commit,
    )
    write_manifest(
        manifest_path,
        jobs,
        source_commit=source_commit,
        feature_root=args.feature_root,
        python_executable=args.python_executable,
        repo_root=args.repo_root,
    )
    incomplete = len(pending_jobs(jobs))
    _atomic_json(
        args.output_root / "runner_status.json",
        {
            "state": "complete" if incomplete == 0 else "failed",
            "failures": failures,
            "incomplete": incomplete,
            "jobs": len(jobs),
            "manifest": str(manifest_path),
        },
    )
    return 0 if incomplete == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
