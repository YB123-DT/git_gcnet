#!/usr/bin/env python3
"""Run the locked five-seed MOSI SDT-backbone diagnostic.

This runner intentionally has no Original/control job path.  Existing control
results are inherited during analysis; every subprocess created here trains the
single SDT-style candidate defined in this package.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from gcnet_missing_m3_sdt_backbone.train_gcnet import SDTTrainConfig


SEEDS = (66, 67, 68, 69, 70)
GPU_MAPPING = {66: 0, 67: 1, 68: 2, 69: 0, 70: 1}
EXPECTED_GPUS = (0, 1, 2)
MISSING_RATE_KEYS = tuple(format(index / 10, ".1f") for index in range(8))
FEATURE_ROOT = Path(
    "/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/"
    "dataset/CMUMOSI/features"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "formal"
DEFAULT_PYTHON = Path("/data2/yb/reproduction_envs/gcnet-official/bin/python")
TRAINING_MODULE = "gcnet_missing_m3_sdt_backbone.train_gcnet"
TREATMENT = "sdt-style-full-context"
SOURCE_FILES = (
    "gcnet_missing_m3_sdt_backbone/model.py",
    "gcnet_missing_m3_sdt_backbone/train_gcnet.py",
    "gcnet_missing_m3_sdt_backbone/run_mosi.py",
)


@dataclass(frozen=True)
class SDTJob:
    seed: int
    gpu: int
    output_dir: Path


@dataclass(frozen=True)
class ResultInspection:
    complete: bool
    reason: str
    has_test_metrics: bool = False


def build_jobs(
    *, output_root: Path, gpus: Sequence[int] = EXPECTED_GPUS
) -> list[SDTJob]:
    normalized_gpus = tuple(int(gpu) for gpu in gpus)
    if normalized_gpus != EXPECTED_GPUS:
        raise ValueError("gpus must be exactly (0, 1, 2) for the locked run")
    allowed = set(normalized_gpus)
    if set(GPU_MAPPING.values()) != allowed:
        raise RuntimeError("the locked seed-to-GPU mapping is inconsistent")
    root = Path(output_root)
    return [
        SDTJob(
            seed=seed,
            gpu=GPU_MAPPING[seed],
            output_dir=root / "seed_{}".format(seed),
        )
        for seed in SEEDS
    ]


def build_command(
    job: SDTJob,
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
        "--epochs",
        "100",
        "--train-rate-mode",
        "all",
        "--fusion-type",
        "slot",
        "--lr",
        "0.0005",
        "--device",
        "cuda",
    ]


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, ValueError, TypeError) as error:
        return None, "{} is missing or invalid: {}".format(path.name, error)


def _config_matches_job(config: object, job: SDTJob) -> bool:
    if not isinstance(config, Mapping):
        return False
    expected = asdict(SDTTrainConfig(seed=job.seed))
    return dict(config) == expected


def _history_is_complete(history: object) -> bool:
    if not isinstance(history, list) or len(history) != 100:
        return False
    return [record.get("epoch") if isinstance(record, Mapping) else None
            for record in history] == list(range(1, 101))


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _test_metrics_are_semantically_complete(metrics: Mapping) -> bool:
    test_metrics = metrics.get("test")
    mask_hashes = metrics.get("mask_sha256")
    if not isinstance(test_metrics, Mapping) or not isinstance(mask_hashes, Mapping):
        return False
    if set(test_metrics) != set(MISSING_RATE_KEYS):
        return False
    if set(mask_hashes) != set(MISSING_RATE_KEYS):
        return False
    for rate in MISSING_RATE_KEYS:
        rate_metrics = test_metrics[rate]
        if not isinstance(rate_metrics, Mapping):
            return False
        weighted_f1 = rate_metrics.get("weighted_f1")
        prediction_std = rate_metrics.get("prediction_std")
        sign_count = rate_metrics.get("predicted_sign_count")
        mask_sha256 = rate_metrics.get("mask_sha256")
        if (
            isinstance(weighted_f1, bool)
            or not isinstance(weighted_f1, (int, float))
            or not math.isfinite(float(weighted_f1))
            or not 0.0 <= float(weighted_f1) <= 1.0
        ):
            return False
        if (
            isinstance(prediction_std, bool)
            or not isinstance(prediction_std, (int, float))
            or not math.isfinite(float(prediction_std))
            or float(prediction_std) < 0.0
        ):
            return False
        if (
            isinstance(sign_count, bool)
            or not isinstance(sign_count, int)
            or sign_count < 1
        ):
            return False
        if not _is_sha256(mask_sha256) or mask_hashes[rate] != mask_sha256:
            return False
    expected_parameters = {
        "registered_backbone_parameters": 5_869_754,
        "active_backbone_parameters": 5_869_370,
        "control_active_backbone_parameters": 5_864_700,
    }
    if any(metrics.get(name) != value for name, value in expected_parameters.items()):
        return False
    for name in ("registered_parameters", "trainable_parameters"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return False
    return True


def inspect_result(job: SDTJob) -> ResultInspection:
    """Re-read all durable outputs; never trust a previous manifest state."""
    required = ("config.json", "history.json", "metrics.json", "train.log")
    for name in required:
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
    if metrics.get("backbone") != TREATMENT:
        return ResultInspection(False, "metrics.json has the wrong backbone")

    stage = metrics.get("evaluation_stage")
    test_metrics = metrics.get("test")
    if stage == "train-validation-only":
        return ResultInspection(
            False,
            "formal runner requires test metrics for all 8 rates",
            has_test_metrics=False,
        )

    if stage != "train-validation-test":
        return ResultInspection(False, "metrics.json has an unknown evaluation stage")
    if not isinstance(test_metrics, Mapping) or set(test_metrics) != set(MISSING_RATE_KEYS):
        return ResultInspection(False, "metrics.json must contain exactly 8 test rates")
    if config.get("evaluate_test") is not True:
        return ResultInspection(False, "test metrics require evaluate_test=true")
    if not _test_metrics_are_semantically_complete(metrics):
        return ResultInspection(False, "test metrics failed semantic completion checks")
    return ResultInspection(
        True,
        "complete-test-8-rates",
        has_test_metrics=True,
    )


def pending_jobs(jobs: Sequence[SDTJob]) -> list[SDTJob]:
    return [job for job in jobs if not inspect_result(job).complete]


def build_waves(
    jobs: Sequence[SDTJob], *, jobs_per_gpu: int
) -> list[list[SDTJob]]:
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
    temporary = path.with_name(
        ".{}.{}.tmp".format(path.name, os.getpid())
    )
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
    gpu_names = {}
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            gpu_names = {
                str(index): torch.cuda.get_device_name(index)
                for index in EXPECTED_GPUS
                if index < device_count
            }
        except RuntimeError:
            gpu_names = {}
    return {
        "python_executable": str(python_executable),
        "python_version": sys.version.splitlines()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "child_environment": {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONHASHSEED": "0",
        },
    }


def _result_manifest_fields(job: SDTJob) -> dict[str, object]:
    config_path = job.output_dir / "config.json"
    metrics_path = job.output_dir / "metrics.json"
    metrics, metrics_error = _read_json(metrics_path)
    if metrics_error is not None or not isinstance(metrics, Mapping):
        metrics = {}
    return {
        "config_sha256": (
            _sha256_file(config_path) if config_path.is_file() else None
        ),
        "metrics_sha256": (
            _sha256_file(metrics_path) if metrics_path.is_file() else None
        ),
        "mask_sha256": metrics.get("mask_sha256"),
        "parameter_counts": (
            {
                "registered": metrics.get("registered_parameters"),
                "trainable": metrics.get("trainable_parameters"),
                "backbone_registered": metrics.get(
                    "registered_backbone_parameters"
                ),
                "backbone_active": metrics.get("active_backbone_parameters"),
                "control_backbone_active": metrics.get(
                    "control_active_backbone_parameters"
                ),
            }
            if metrics
            else None
        ),
    }


def write_manifest(
    path: Path,
    jobs: Sequence[SDTJob],
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
                "seed": job.seed,
                "gpu": job.gpu,
                "output_dir": str(job.output_dir),
                "complete": inspection.complete,
                "completion_reason": inspection.reason,
                "has_test_metrics": inspection.has_test_metrics,
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
            "seeds": list(SEEDS),
            "gpu_mapping": {str(seed): GPU_MAPPING[seed] for seed in SEEDS},
            "required_epochs": 100,
            "required_test_rates": list(MISSING_RATE_KEYS),
            "jobs": entries,
        },
    )


def prune_large_artifacts(output_dir: Path) -> list[str]:
    """Remove generated artifacts that must not persist in the GitHub tree."""
    removed = []
    candidates = [Path(output_dir) / "best.pt"]
    candidates.extend(sorted(Path(output_dir).glob("predictions_miss_*.npz")))
    for path in candidates:
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def reset_incomplete_output(output_dir: Path) -> None:
    """Prevent a retry from combining artifacts from different attempts."""
    root = Path(output_dir)
    prune_large_artifacts(root)
    for name in (
        "config.json",
        "history.json",
        "metrics.json",
        "status.json",
        "train.log",
    ):
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
    jobs: Sequence[SDTJob],
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
    for job in all_jobs:
        if inspect_result(job).complete:
            prune_large_artifacts(job.output_dir)

    failures = 0
    waves = build_waves(
        pending_jobs(all_jobs), jobs_per_gpu=jobs_per_gpu
    )
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
                    deadline = started_at + int(timeout_seconds)
                    remaining = max(0.0, deadline - time.time())
                    returncode = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_tree(process)
                finally:
                    log_handle.close()
                handled_processes.add(id(process))

                inspection = inspect_result(job)
                removed = prune_large_artifacts(job.output_dir)
                if inspection.complete:
                    state = "complete"
                elif timed_out:
                    state = "timeout"
                    failures += 1
                else:
                    state = "failed"
                    failures += 1
                status = {
                    "state": state,
                    "pid": process.pid,
                    "seed": job.seed,
                    "gpu": job.gpu,
                    "wave": wave_index,
                    "command": command,
                    "returncode": returncode,
                    "timeout_seconds": int(timeout_seconds),
                    "completion_reason": inspection.reason,
                    "has_test_metrics": inspection.has_test_metrics,
                    "removed_large_artifacts": removed,
                    "log_path": str(job.output_dir / "train.log"),
                    "started_at_unix": started_at,
                    "finished_at_unix": time.time(),
                }
                _atomic_json(job.output_dir / "status.json", status)
                if manifest_path is not None:
                    if source_commit is None:
                        raise ValueError(
                            "source_commit is required when writing a manifest"
                        )
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
    parser.add_argument("--gpus", type=int, nargs=3, default=EXPECTED_GPUS)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.jobs_per_gpu < 1:
        raise SystemExit("--jobs-per-gpu must be positive")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")
    try:
        jobs = build_jobs(output_root=args.output_root, gpus=args.gpus)
    except ValueError as error:
        raise SystemExit(str(error))

    manifest_path = args.output_root / "manifest.json"
    write_manifest(
        manifest_path,
        jobs,
        source_commit=args.source_commit,
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
        source_commit=args.source_commit,
    )
    write_manifest(
        manifest_path,
        jobs,
        source_commit=args.source_commit,
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
