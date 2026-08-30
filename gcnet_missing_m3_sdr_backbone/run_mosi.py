#!/usr/bin/env python3
"""Run the locked two-variant, five-seed MOSI SDR diagnostic."""

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
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from gcnet_missing_m3 import train_gcnet as base_train
from gcnet_missing_m3_sdr_backbone.train_gcnet import SDRTrainConfig


VARIANTS = ("sdr-public", "sdr-paper")
SEEDS = (66, 67, 68, 69, 70)
HEALTHY_GPUS = (2, 3, 5, 6, 7)
MISSING_RATE_KEYS = tuple(format(index / 10.0, ".1f") for index in range(8))
MISSING_RATES = tuple(index / 10.0 for index in range(8))
HIGH_MISSING_RATE_KEYS = ("0.4", "0.5", "0.6", "0.7")
FEATURE_ROOT = Path(
    "/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/"
    "dataset/CMUMOSI/features"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "formal"
DEFAULT_PYTHON = Path("/data2/yb/reproduction_envs/gcnet-official/bin/python")
TRAINING_MODULE = "gcnet_missing_m3_sdr_backbone.train_gcnet"
TREATMENT = "missing-m3-sdr-backbone"
SOURCE_FILES = (
    "gcnet_missing_m3_sdr_backbone/__init__.py",
    "gcnet_missing_m3_sdr_backbone/layers.py",
    "gcnet_missing_m3_sdr_backbone/model.py",
    "gcnet_missing_m3_sdr_backbone/train_gcnet.py",
    "gcnet_missing_m3_sdr_backbone/run_mosi.py",
)
EXPECTED_PARAMETER_COUNTS = {
    "sdr-public": {
        "registered_parameters": 12_486_434,
        "trainable_parameters": 11_626_274,
        "registered_backbone_parameters": 9_444_901,
        "trainable_backbone_parameters": 9_444_901,
    },
    "sdr-paper": {
        "registered_parameters": 21_079_835,
        "trainable_parameters": 20_219_675,
        "registered_backbone_parameters": 18_038_302,
        "trainable_backbone_parameters": 18_038_302,
    },
}


@dataclass(frozen=True)
class SDRJob:
    variant: str
    seed: int
    gpu: int
    output_dir: Path


@dataclass(frozen=True)
class ResultInspection:
    complete: bool
    reason: str
    has_test_metrics: bool = False
    collapsed: bool = False
    collapse_rates: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionInspection:
    weighted_f1: float
    prediction_std: float
    predicted_sign_count: int
    mask_sha256: str
    collapsed: bool


def _validated_gpus(gpus: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(gpus, (str, bytes)):
        raise ValueError("gpus must be a sequence of healthy GPU indices")
    normalized: List[int] = []
    for gpu in gpus:
        if isinstance(gpu, bool) or not isinstance(gpu, int):
            raise ValueError("GPU indices must be integers")
        normalized.append(gpu)
    if not normalized:
        raise ValueError("at least one healthy GPU is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("GPU indices must be unique")
    invalid = sorted(set(normalized) - set(HEALTHY_GPUS))
    if invalid:
        if 4 in invalid:
            raise ValueError("GPU 4 is unhealthy and must never be scheduled")
        raise ValueError(
            "gpus must be selected from {}".format(HEALTHY_GPUS)
        )
    return tuple(normalized)


def build_jobs(
    *,
    output_root: Path,
    gpus: Sequence[int] = HEALTHY_GPUS,
) -> List[SDRJob]:
    """Build exactly 2 variants x 5 seeds with one GPU identity per seed."""

    normalized_gpus = _validated_gpus(gpus)
    seed_gpu = {
        seed: normalized_gpus[index % len(normalized_gpus)]
        for index, seed in enumerate(SEEDS)
    }
    root = Path(output_root)
    return [
        SDRJob(
            variant=variant,
            seed=seed,
            gpu=seed_gpu[seed],
            output_dir=root / variant / "seed_{}".format(seed),
        )
        for variant in VARIANTS
        for seed in SEEDS
    ]


def build_command(
    job: SDRJob,
    *,
    feature_root: Path = FEATURE_ROOT,
) -> List[str]:
    """Return the immutable formal training command for one treatment."""

    if job.variant not in VARIANTS or job.seed not in SEEDS:
        raise ValueError("job is outside the registered treatment matrix")
    return [
        str(DEFAULT_PYTHON),
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
        "--sdr-variant",
        job.variant,
        "--device",
        "cuda",
    ]


def _read_json(path: Path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, ValueError, TypeError) as error:
        return None, "{} is missing or invalid: {}".format(path.name, error)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _same_float(left: object, right: object, tolerance: float = 1e-10) -> bool:
    return (
        _finite_number(left)
        and _finite_number(right)
        and abs(float(left) - float(right)) <= tolerance
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _config_matches_job(config: object, job: SDRJob) -> bool:
    if not isinstance(config, Mapping):
        return False
    expected = asdict(
        SDRTrainConfig(seed=job.seed, sdr_variant=job.variant)
    )
    return dict(config) == expected


def _validated_history(history: object):
    if not isinstance(history, list) or len(history) != 100:
        return None, "history.json must contain exactly 100 epochs"
    expected_epochs = list(range(1, 101))
    actual_epochs = [
        record.get("epoch") if isinstance(record, Mapping) else None
        for record in history
    ]
    if actual_epochs != expected_epochs:
        return None, "history.json must contain ordered epochs 1 through 100"

    best = None
    for record in history:
        validation = record.get("validation")
        if not isinstance(validation, Mapping) or set(validation) != set(
            MISSING_RATE_KEYS
        ):
            return None, "history validation must contain exactly 8 rates"
        scores = []
        for rate in MISSING_RATE_KEYS:
            rate_metrics = validation[rate]
            if not isinstance(rate_metrics, Mapping):
                return None, "history validation rate must contain metrics"
            score = rate_metrics.get("weighted_f1")
            if not _finite_number(score) or not 0.0 <= float(score) <= 1.0:
                return None, "history validation weighted_f1 must be finite"
            scores.append(float(score))
        validation_mean = sum(scores) / len(scores)
        if not _same_float(
            record.get("validation_mean_weighted_f1"), validation_mean
        ):
            return None, "history validation mean is inconsistent"
        if best is None or validation_mean > best[2]:
            best = (record["epoch"], dict(validation), validation_mean)
    return best, None


def _test_metrics_are_semantically_complete(
    metrics: Mapping,
    job: SDRJob,
) -> bool:
    test_metrics = metrics.get("test")
    mask_hashes = metrics.get("mask_sha256")
    if not isinstance(test_metrics, Mapping) or not isinstance(
        mask_hashes, Mapping
    ):
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
            not _finite_number(weighted_f1)
            or not 0.0 <= float(weighted_f1) <= 1.0
        ):
            return False
        if not _finite_number(prediction_std) or float(prediction_std) < 0.0:
            return False
        if (
            isinstance(sign_count, bool)
            or not isinstance(sign_count, int)
            or sign_count < 1
        ):
            return False
        if not _is_sha256(mask_sha256) or mask_hashes[rate] != mask_sha256:
            return False
        for name, value in rate_metrics.items():
            if name == "mask_sha256":
                continue
            if isinstance(value, (int, float)) and not _finite_number(value):
                return False

    expected_counts = EXPECTED_PARAMETER_COUNTS[job.variant]
    if any(metrics.get(name) != value for name, value in expected_counts.items()):
        return False
    ema_steps = metrics.get("ema_steps")
    if isinstance(ema_steps, bool) or not isinstance(ema_steps, int) or ema_steps < 1:
        return False
    if not _finite_number(metrics.get("wall_time_seconds")) or float(
        metrics["wall_time_seconds"]
    ) <= 0.0:
        return False
    peak_memory = metrics.get("peak_memory_bytes")
    if (
        isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory < 0
    ):
        return False
    return True


def _inspect_prediction_archive(path: Path):
    if not Path(path).is_file():
        return None, "archive is missing"
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            if not {"predictions", "labels", "availability"}.issubset(
                archive.files
            ):
                return None, "required arrays are missing"
            predictions = np.asarray(archive["predictions"])
            labels = np.asarray(archive["labels"])
            availability = np.asarray(archive["availability"])
    except (OSError, ValueError, TypeError, EOFError):
        return None, "archive cannot be loaded"
    if predictions.size == 0 or labels.size == 0 or availability.size == 0:
        return None, "archive arrays must be non-empty"
    if predictions.reshape(-1).shape != labels.reshape(-1).shape:
        return None, "predictions and labels have different shapes"
    if availability.ndim != 2 or availability.shape[0] != predictions.size:
        return None, "availability has the wrong row count"
    if availability.shape[1] != 3:
        return None, "availability must have three modality columns"
    if not (
        np.isfinite(predictions).all()
        and np.isfinite(labels).all()
        and np.isfinite(availability).all()
    ):
        return None, "archive arrays must be finite"
    flattened_predictions = predictions.reshape(-1)
    flattened_labels = labels.reshape(-1)
    if not np.any(flattened_labels != 0):
        return None, "MOSI W-F1 requires at least one nonzero label"
    try:
        recomputed = base_train._metrics(
            "CMUMOSI",
            flattened_labels,
            flattened_predictions,
            "regression",
        )
    except (TypeError, ValueError) as error:
        return None, "MOSI metrics cannot be recomputed: {}".format(error)
    availability_tensor = torch.from_numpy(
        np.ascontiguousarray(availability)
    )
    mask_sha256 = base_train._sha256_tensor(availability_tensor)
    prediction_std = float(recomputed["prediction_std"])
    sign_count = int(recomputed["predicted_sign_count"])
    return PredictionInspection(
        weighted_f1=float(recomputed["weighted_f1"]),
        prediction_std=prediction_std,
        predicted_sign_count=sign_count,
        mask_sha256=mask_sha256,
        collapsed=prediction_std <= 0.0 or sign_count < 2,
    ), None


def inspect_result(job: SDRJob) -> ResultInspection:
    """Recompute completion from durable files; never trust prior status."""

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
    selected, error = _validated_history(history)
    if error is not None:
        return ResultInspection(False, error)
    assert selected is not None
    selected_epoch, selected_validation, selected_mean = selected

    metrics, error = _read_json(job.output_dir / "metrics.json")
    if error is not None:
        return ResultInspection(False, error)
    if not isinstance(metrics, Mapping):
        return ResultInspection(False, "metrics.json must contain an object")
    if metrics.get("backbone") != "sdr-gnn-whole-backbone":
        return ResultInspection(False, "metrics.json has the wrong backbone")
    if (
        metrics.get("variant") != job.variant
        or metrics.get("sdr_variant") != job.variant
    ):
        return ResultInspection(False, "metrics.json has the wrong variant")
    if metrics.get("best_epoch") != selected_epoch:
        return ResultInspection(False, "metrics.json is not validation-selected")
    if metrics.get("best_validation") != selected_validation:
        return ResultInspection(False, "best validation metrics are inconsistent")
    if not _same_float(
        metrics.get("best_validation_mean_weighted_f1"), selected_mean
    ):
        return ResultInspection(False, "best validation mean is inconsistent")
    selection_rates = metrics.get("selection_missing_rates")
    if not isinstance(selection_rates, list) or len(selection_rates) != 8:
        return ResultInspection(False, "selection rates must contain all 8 rates")
    if any(
        not _same_float(actual, expected)
        for actual, expected in zip(selection_rates, MISSING_RATES)
    ):
        return ResultInspection(False, "selection rates must contain all 8 rates")

    if metrics.get("evaluation_stage") == "train-validation-only":
        return ResultInspection(
            False,
            "formal runner requires test metrics for all 8 rates",
            has_test_metrics=False,
        )
    if metrics.get("evaluation_stage") != "train-validation-test":
        return ResultInspection(False, "metrics.json has unknown evaluation stage")
    test_metrics = metrics.get("test")
    if not isinstance(test_metrics, Mapping) or set(test_metrics) != set(
        MISSING_RATE_KEYS
    ):
        return ResultInspection(False, "metrics.json must contain exactly 8 test rates")
    if config.get("evaluate_test") is not True:
        return ResultInspection(False, "test metrics require evaluate_test=true")
    if not _test_metrics_are_semantically_complete(metrics, job):
        return ResultInspection(False, "test metrics failed semantic completion checks")

    collapse_rates = []
    for rate in MISSING_RATE_KEYS:
        archive = job.output_dir / "predictions_miss_{}.npz".format(
            rate.replace(".", "p")
        )
        prediction, prediction_error = _inspect_prediction_archive(archive)
        if prediction_error is not None:
            return ResultInspection(
                False,
                "prediction archive is invalid for rate {}: {}".format(
                    rate, prediction_error
                ),
            )
        assert prediction is not None
        rate_metrics = test_metrics[rate]
        comparisons = {
            "weighted_f1": prediction.weighted_f1,
            "prediction_std": prediction.prediction_std,
        }
        for name, recomputed_value in comparisons.items():
            if not _same_float(rate_metrics.get(name), recomputed_value, 1e-8):
                return ResultInspection(
                    False,
                    (
                        "test metrics semantic mismatch: prediction archive "
                        "recomputed {} mismatch "
                        "for rate {}"
                    ).format(name, rate),
                )
        if (
            rate_metrics.get("predicted_sign_count")
            != prediction.predicted_sign_count
        ):
            return ResultInspection(
                False,
                (
                    "test metrics semantic mismatch: prediction archive "
                    "recomputed sign count mismatch for rate {}"
                ).format(rate),
            )
        if (
            rate_metrics.get("mask_sha256") != prediction.mask_sha256
            or metrics["mask_sha256"].get(rate) != prediction.mask_sha256
        ):
            return ResultInspection(
                False,
                "prediction archive availability SHA mismatch for rate {}".format(
                    rate
                ),
            )
        if prediction.collapsed:
            collapse_rates.append(rate)
    return ResultInspection(
        True,
        "complete-test-8-rates",
        has_test_metrics=True,
        collapsed=bool(collapse_rates),
        collapse_rates=tuple(collapse_rates),
    )


def pending_jobs(jobs: Sequence[SDRJob]) -> List[SDRJob]:
    return [job for job in jobs if not inspect_result(job).complete]


def build_waves(
    jobs: Sequence[SDRJob],
    *,
    jobs_per_gpu: int,
) -> List[List[SDRJob]]:
    """Schedule at most two registered variants per GPU."""

    if (
        isinstance(jobs_per_gpu, bool)
        or not isinstance(jobs_per_gpu, int)
        or not 1 <= jobs_per_gpu <= len(VARIANTS)
    ):
        raise ValueError("jobs_per_gpu must be 1 or 2")
    grouped: Dict[int, List[SDRJob]] = {}
    for job in jobs:
        if job.gpu not in HEALTHY_GPUS:
            raise ValueError("job uses a GPU outside the healthy set")
        grouped.setdefault(job.gpu, []).append(job)
    wave_count = max(
        (
            (len(gpu_jobs) + jobs_per_gpu - 1) // jobs_per_gpu
            for gpu_jobs in grouped.values()
        ),
        default=0,
    )
    waves: List[List[SDRJob]] = []
    for wave_index in range(wave_count):
        start = wave_index * jobs_per_gpu
        stop = start + jobs_per_gpu
        wave = []
        for gpu in sorted(grouped):
            wave.extend(grouped[gpu][start:stop])
        waves.append(wave)
    return waves


def _atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
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


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_source_commit(source_commit: str) -> str:
    normalized = str(source_commit).lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("source_commit must be a full 40-character Git SHA")
    return normalized


@lru_cache(maxsize=1)
def _query_training_runtime() -> Dict[str, object]:
    program = """
import json
import sys
import torch

healthy = (2, 3, 5, 6, 7)
gpu_names = {}
if torch.cuda.is_available():
    try:
        count = torch.cuda.device_count()
        gpu_names = {
            str(index): torch.cuda.get_device_name(index)
            for index in healthy
            if index < count
        }
    except RuntimeError:
        gpu_names = {}
print(json.dumps({
    "python_version": sys.version.splitlines()[0],
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "gpu_names": gpu_names,
}, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [str(DEFAULT_PYTHON), "-c", program],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
            timeout=60,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1])
    except (
        IndexError,
        json.JSONDecodeError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        raise RuntimeError(
            "failed to query fixed training Python provenance: {}".format(error)
        )
    if not isinstance(payload, Mapping):
        raise RuntimeError("fixed training Python returned invalid provenance")
    return dict(payload)


def _runtime_provenance() -> Dict[str, object]:
    training = _query_training_runtime()
    runner = {
        "python_executable": sys.executable,
        "python_version": sys.version.splitlines()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    return {
        "training_python_executable": str(DEFAULT_PYTHON),
        "runner_python_executable": sys.executable,
        "python_version": training["python_version"],
        "torch_version": training["torch_version"],
        "cuda_version": training["cuda_version"],
        "cudnn_version": training["cudnn_version"],
        "gpu_names": training["gpu_names"],
        "training": {
            "python_executable": str(DEFAULT_PYTHON),
            **training,
        },
        "runner": runner,
        "child_environment": {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONHASHSEED": "0",
        },
    }


def _result_manifest_fields(job: SDRJob) -> Dict[str, object]:
    config_path = job.output_dir / "config.json"
    metrics_path = job.output_dir / "metrics.json"
    metrics, metrics_error = _read_json(metrics_path)
    if metrics_error is not None or not isinstance(metrics, Mapping):
        metrics = {}
    status, status_error = _read_json(job.output_dir / "status.json")
    if status_error is not None or not isinstance(status, Mapping):
        status = None
    expected_config = asdict(
        SDRTrainConfig(seed=job.seed, sdr_variant=job.variant)
    )
    return {
        "config_sha256": _canonical_json_sha256(expected_config),
        "config_file_sha256": (
            _sha256_file(config_path) if config_path.is_file() else None
        ),
        "metrics_sha256": (
            _sha256_file(metrics_path) if metrics_path.is_file() else None
        ),
        "mask_sha256": metrics.get("mask_sha256"),
        "parameter_counts": (
            {
                name: metrics.get(name)
                for name in EXPECTED_PARAMETER_COUNTS[job.variant]
            }
            if metrics
            else None
        ),
        "status": dict(status) if status is not None else None,
    }


def write_manifest(
    path: Path,
    jobs: Sequence[SDRJob],
    *,
    source_commit: str,
    feature_root: Path,
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
                "variant": job.variant,
                "seed": job.seed,
                "gpu": job.gpu,
                "output_dir": str(job.output_dir),
                "complete": inspection.complete,
                "completion_reason": inspection.reason,
                "has_test_metrics": inspection.has_test_metrics,
                "collapsed": inspection.collapsed,
                "collapse_rates": list(inspection.collapse_rates),
                "log_path": str(job.output_dir / "train.log"),
                **_result_manifest_fields(job),
            }
        )
    feature_root = Path(feature_root)
    seed_gpu = {}
    for job in jobs:
        seed_gpu.setdefault(str(job.seed), job.gpu)
        if seed_gpu[str(job.seed)] != job.gpu:
            raise ValueError("variants for one seed must share a GPU")
    _atomic_json(
        Path(path),
        {
            "schema_version": 1,
            "treatment": TREATMENT,
            "training_module": TRAINING_MODULE,
            "source_commit": source_commit,
            "source_files_sha256": source_hashes,
            "runtime": _runtime_provenance(),
            "features": {
                "root": str(feature_root),
                "audio": str(feature_root / "wav2vec-large-c-UTT"),
                "text": str(feature_root / "deberta-large-4-UTT"),
                "video": str(feature_root / "manet_UTT"),
            },
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "seed_gpu_mapping": seed_gpu,
            "required_epochs": 100,
            "required_test_rates": list(MISSING_RATE_KEYS),
            "jobs": entries,
        },
    )


def reset_incomplete_output(output_dir: Path) -> None:
    """Prevent a retry from combining durable files from different attempts."""

    root = Path(output_dir)
    candidates = [
        root / name
        for name in (
            "best.pt",
            "config.json",
            "history.json",
            "metrics.json",
            "status.json",
            "train.log",
        )
    ]
    candidates.extend(sorted(root.glob("predictions_miss_*.npz")))
    for path in candidates:
        if path.is_file():
            path.unlink()


def _prune_checkpoint(output_dir: Path) -> List[str]:
    path = Path(output_dir) / "best.pt"
    if not path.is_file():
        return []
    path.unlink()
    return [path.name]


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
    jobs: Sequence[SDRJob],
    *,
    feature_root: Path,
    repo_root: Path,
    jobs_per_gpu: int,
    timeout_seconds: int,
    manifest_path: Optional[Path] = None,
    source_commit: Optional[str] = None,
) -> int:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
    ):
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
                command = build_command(job, feature_root=feature_root)
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
                except BaseException as error:
                    log_handle.close()
                    _atomic_json(
                        job.output_dir / "status.json",
                        {
                            "state": "launch-error",
                            "variant": job.variant,
                            "seed": job.seed,
                            "gpu": job.gpu,
                            "command": command,
                            "error": "{}: {}".format(
                                type(error).__name__, error
                            ),
                            "finished_at_unix": time.time(),
                        },
                    )
                    raise
                _atomic_json(
                    job.output_dir / "status.json",
                    {
                        "state": "running",
                        "pid": process.pid,
                        "variant": job.variant,
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
                    deadline = started_at + timeout_seconds
                    remaining = max(0.0, deadline - time.time())
                    returncode = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_tree(process)
                finally:
                    log_handle.close()
                handled_processes.add(id(process))

                inspection = inspect_result(job)
                removed = _prune_checkpoint(job.output_dir)
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
                        "variant": job.variant,
                        "seed": job.seed,
                        "gpu": job.gpu,
                        "wave": wave_index,
                        "command": command,
                        "returncode": returncode,
                        "timeout_seconds": timeout_seconds,
                        "completion_reason": inspection.reason,
                        "has_test_metrics": inspection.has_test_metrics,
                        "removed_large_artifacts": removed,
                        "log_path": str(job.output_dir / "train.log"),
                        "started_at_unix": started_at,
                        "finished_at_unix": time.time(),
                    },
                )
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
                        repo_root=repo_root,
                    )
        except BaseException as error:
            for job, process, _, command, started_at in running:
                if id(process) in handled_processes:
                    continue
                _atomic_json(
                    job.output_dir / "status.json",
                    {
                        "state": "aborted",
                        "pid": process.pid,
                        "variant": job.variant,
                        "seed": job.seed,
                        "gpu": job.gpu,
                        "wave": wave_index,
                        "command": command,
                        "error": "{}: {}".format(type(error).__name__, error),
                        "started_at_unix": started_at,
                        "finished_at_unix": time.time(),
                    },
                )
            raise
        finally:
            for _, process, log_handle, _, _ in running:
                if id(process) not in handled_processes:
                    _terminate_process_tree(process)
                if not log_handle.closed:
                    log_handle.close()
    return failures


def _weighted_f1_map(value: Mapping) -> Dict[str, float]:
    return {
        rate: float(value[rate]["weighted_f1"])
        if isinstance(value[rate], Mapping)
        else float(value[rate])
        for rate in MISSING_RATE_KEYS
    }


def _control_rates(
    control_validation: Optional[Mapping],
    seed: int,
) -> Optional[Dict[str, float]]:
    if control_validation is None:
        return None
    value = control_validation.get(seed)
    if value is None:
        value = control_validation.get(str(seed))
    if not isinstance(value, Mapping):
        return None
    if "best_validation" in value:
        value = value["best_validation"]
    if not isinstance(value, Mapping) or set(value) != set(MISSING_RATE_KEYS):
        return None
    return _weighted_f1_map(value)


def aggregate(
    jobs: Sequence[SDRJob],
    *,
    control_validation: Optional[Mapping] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Aggregate validation-selected runs without using test to rank variants."""

    by_variant: Dict[str, Dict[str, object]] = {}
    for variant in VARIANTS:
        variant_jobs = sorted(
            (job for job in jobs if job.variant == variant),
            key=lambda job: job.seed,
        )
        seed_rows: Dict[str, Dict[str, object]] = {}
        validation_by_rate: Dict[str, List[float]] = {
            rate: [] for rate in MISSING_RATE_KEYS
        }
        test_by_rate: Dict[str, List[float]] = {
            rate: [] for rate in MISSING_RATE_KEYS
        }
        positive_seeds = []
        collapsed_seeds = []
        for job in variant_jobs:
            inspection = inspect_result(job)
            if not inspection.complete:
                raise ValueError(
                    "cannot aggregate incomplete {} seed {}: {}".format(
                        variant, job.seed, inspection.reason
                    )
                )
            metrics, error = _read_json(job.output_dir / "metrics.json")
            if error is not None or not isinstance(metrics, Mapping):
                raise ValueError(error or "metrics.json must be an object")
            validation = _weighted_f1_map(metrics["best_validation"])
            test = _weighted_f1_map(metrics["test"])
            validation_mean = mean(validation.values())
            test_mean = mean(test.values())
            validation_high = mean(
                validation[rate] for rate in HIGH_MISSING_RATE_KEYS
            )
            test_high = mean(test[rate] for rate in HIGH_MISSING_RATE_KEYS)
            reference = _control_rates(control_validation, job.seed)
            paired_delta = None
            paired_high_delta = None
            if reference is not None:
                paired_delta = validation_mean - mean(reference.values())
                paired_high_delta = validation_high - mean(
                    reference[rate] for rate in HIGH_MISSING_RATE_KEYS
                )
                if paired_delta > 0.0:
                    positive_seeds.append(job.seed)
            collapse_rates = [
                rate
                for rate in MISSING_RATE_KEYS
                if float(metrics["test"][rate]["prediction_std"]) <= 0.0
                or int(metrics["test"][rate]["predicted_sign_count"]) < 2
            ]
            if collapse_rates:
                collapsed_seeds.append(job.seed)
            seed_rows[str(job.seed)] = {
                "best_epoch": metrics["best_epoch"],
                "validation": validation,
                "validation_mean": validation_mean,
                "validation_high_missing_mean": validation_high,
                "test": test,
                "test_mean": test_mean,
                "test_high_missing_mean": test_high,
                "paired_validation_delta": paired_delta,
                "paired_validation_high_missing_delta": paired_high_delta,
                "collapse_rates": collapse_rates,
            }
            for rate in MISSING_RATE_KEYS:
                validation_by_rate[rate].append(validation[rate])
                test_by_rate[rate].append(test[rate])

        rates = {
            rate: {
                "validation_mean": mean(validation_by_rate[rate]),
                "test_mean": mean(test_by_rate[rate]),
            }
            for rate in MISSING_RATE_KEYS
        }
        by_variant[variant] = {
            "seeds": seed_rows,
            "rates": rates,
            "validation_mean": mean(
                row["validation_mean"] for row in seed_rows.values()
            ),
            "test_mean": mean(row["test_mean"] for row in seed_rows.values()),
            "validation_high_missing_mean": mean(
                row["validation_high_missing_mean"]
                for row in seed_rows.values()
            ),
            "test_high_missing_mean": mean(
                row["test_high_missing_mean"] for row in seed_rows.values()
            ),
            "positive_seeds": positive_seeds,
            "positive_seed_count": len(positive_seeds),
            "collapse": {
                "any": bool(collapsed_seeds),
                "seeds": collapsed_seeds,
            },
        }

    validation_order = sorted(
        VARIANTS,
        key=lambda variant: (
            -float(by_variant[variant]["validation_mean"]),
            VARIANTS.index(variant),
        ),
    )
    summary: Dict[str, object] = {
        "selection_basis": "validation-eight-rate-mean-weighted-f1",
        "test_used_for_selection": False,
        "validation_order": validation_order,
        "variants": by_variant,
    }
    if output_path is not None:
        _atomic_json(Path(output_path), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpus", type=int, nargs="+", default=HEALTHY_GPUS)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        jobs = build_jobs(output_root=args.output_root, gpus=args.gpus)
        build_waves(jobs, jobs_per_gpu=args.jobs_per_gpu)
    except ValueError as error:
        raise SystemExit(str(error))
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")

    manifest_path = args.output_root / "manifest.json"
    write_manifest(
        manifest_path,
        jobs,
        source_commit=args.source_commit,
        feature_root=args.feature_root,
        repo_root=args.repo_root,
    )
    if args.dry_run:
        for job in jobs:
            print(
                "COMMAND {}".format(
                    shlex.join(build_command(job, feature_root=args.feature_root))
                )
            )
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
