#!/usr/bin/env python3
"""Run the five locked Raw-Residual + SDR-public MOSI treatments."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shlex
import socket
import subprocess
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from gcnet_missing_m3 import train_gcnet as base_train
from gcnet_missing_m3_raw_sdr.train_gcnet import RawSDRTrainConfig
from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule
from gcnet_modality_jepa.protocol import SeedBundle
from gcnet_missing_m3_sdr_backbone.run_mosi import (
    EXPECTED_PARAMETER_COUNTS as LEGACY_SDR_PARAMETER_COUNTS,
    PredictionInspection,
    _atomic_json,
    _canonical_json_sha256,
    _finite_number,
    _inspect_prediction_archive,
    _is_sha256,
    _is_sha256_mapping,
    _query_training_runtime,
    _read_json,
    _resolved_feature_paths,
    _same_float,
    _sha256_file,
    _terminate_process_tree,
    _validate_source_commit,
    _validation_collapse_rates,
    _validation_rates_and_masks,
    _validated_history,
    reset_incomplete_output,
)
from gcnet_missing_m3_sdr_backbone.train_gcnet import SDRTrainConfig


VARIANT = "raw-residual-sdr-public"
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
TRAINING_MODULE = "gcnet_missing_m3_raw_sdr.train_gcnet"
TREATMENT = "missing-m3-raw-residual-sdr-public"
PRODUCER_PROVENANCE_NAME = "producer_provenance.json"
MASK_AUDIT_NAME = "validation_mask_audit.json"
LOCK_NAME = ".formal_run.lock"
SOURCE_FILES = (
    "gcnet_missing_m3/model.py",
    "gcnet_missing_m3/loss.py",
    "gcnet_missing_m3/mixed_rate.py",
    "gcnet_missing_m3/train_gcnet.py",
    "gcnet_modality_jepa/mask_schedule.py",
    "gcnet_modality_jepa/protocol.py",
    "gcnet_missing_m3_sdr_backbone/layers.py",
    "gcnet_missing_m3_sdr_backbone/model.py",
    "gcnet_missing_m3_sdr_backbone/train_gcnet.py",
    "gcnet_missing_m3_raw_sdr/__init__.py",
    "gcnet_missing_m3_raw_sdr/model.py",
    "gcnet_missing_m3_raw_sdr/train_gcnet.py",
    "gcnet_missing_m3_raw_sdr/run_mosi.py",
)

# Task 4 replaces this with the exact counts read from the one formal smoke.
# Formal completion deliberately fails closed while this lock is unset.
EXPECTED_PARAMETER_COUNTS: Optional[Mapping[str, int]] = None
PARAMETER_COUNT_FIELDS = (
    "registered_parameters",
    "trainable_parameters",
    "registered_backbone_parameters",
    "trainable_backbone_parameters",
)

SLOT_REFERENCE_DIRS = {
    seed: Path(
        "/data2/yb/remote_experiments/"
        "missing_m3_sdr_backbone_20260830/formal/sdr-public/"
        "seed_{}".format(seed)
    )
    for seed in SEEDS
}
CONTROL_REFERENCE_DIRS = {
    66: Path(
        "/data2/yb/remote_experiments/missing_m3_mosi_lr_screen_20260829/"
        "screen/lr_5e-4/seed_66"
    ),
    **{
        seed: Path(
            "/data2/yb/remote_experiments/"
            "missing_m3_mosi_lr_screen_20260829/formal/lr_5e-4/"
            "seed_{}".format(seed)
        )
        for seed in (67, 68, 69, 70)
    },
}


@dataclass(frozen=True)
class RawSDRJob:
    seed: int
    gpu: int
    output_dir: Path
    variant: str = VARIANT


@dataclass(frozen=True)
class ResultInspection:
    complete: bool
    reason: str
    collapsed: bool = False
    collapse_rates: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpectedContext:
    """Immutable-by-contract evidence computed before a formal run."""

    producer_provenance: Mapping[Tuple[str, int], Mapping]
    mask_audits: Mapping[int, Mapping]


def _exact_tuple(value: Sequence, expected: Tuple, name: str) -> Tuple:
    if isinstance(value, (str, bytes)):
        raise ValueError("{} must be the exact registered tuple".format(name))
    normalized = tuple(value)
    if normalized != expected or len(set(normalized)) != len(normalized):
        raise ValueError("{} must equal {!r}".format(name, expected))
    return normalized


def build_jobs(
    *,
    output_root: Path,
    gpus: Sequence[int] = HEALTHY_GPUS,
    seeds: Sequence[int] = SEEDS,
    variants: Sequence[str] = (VARIANT,),
) -> List[RawSDRJob]:
    """Build only the registered 5-seed treatment matrix."""

    normalized_gpus = _exact_tuple(gpus, HEALTHY_GPUS, "gpus")
    normalized_seeds = _exact_tuple(seeds, SEEDS, "seeds")
    _exact_tuple(variants, (VARIANT,), "variants")
    root = Path(output_root)
    return [
        RawSDRJob(
            seed=seed,
            gpu=normalized_gpus[index],
            output_dir=root / "seed_{}".format(seed),
        )
        for index, seed in enumerate(normalized_seeds)
    ]


def _validate_job(job: RawSDRJob) -> None:
    if not isinstance(job, RawSDRJob):
        raise TypeError("job must be a RawSDRJob")
    if job.variant != VARIANT or job.seed not in SEEDS:
        raise ValueError("job is outside the registered treatment matrix")
    expected_gpu = HEALTHY_GPUS[SEEDS.index(job.seed)]
    if job.gpu != expected_gpu:
        raise ValueError(
            "seed {} must run on GPU {}".format(job.seed, expected_gpu)
        )


def resolve_feature_root(value) -> Path:
    """Resolve and validate the one immutable formal feature tree."""

    try:
        root = Path(value).expanduser().resolve(strict=True)
    except (TypeError, OSError, RuntimeError, ValueError) as error:
        raise ValueError("feature_root is invalid: {}".format(error))
    if not root.is_dir():
        raise ValueError("feature_root must be a directory")
    for name in (
        "wav2vec-large-c-UTT",
        "deberta-large-4-UTT",
        "manet_UTT",
    ):
        if not (root / name).is_dir():
            raise ValueError("feature_root is missing {}".format(name))
    return root


def protocol_fields(config: Mapping) -> Dict[str, object]:
    """Return only fields that determine the validation protocol identity."""

    if not isinstance(config, Mapping):
        raise ValueError("protocol config must contain an object")
    required = (
        "dataset",
        "seed",
        "fold",
        "evaluation_protocol",
        "validation_fraction",
    )
    if any(name not in config for name in required):
        raise ValueError("protocol config is missing immutable fields")
    fields = {name: config[name] for name in required}
    if (
        fields["dataset"] != "CMUMOSI"
        or fields["evaluation_protocol"] != "official"
        or fields["fold"] != 1
        or fields["seed"] not in SEEDS
        or not _same_float(fields["validation_fraction"], 0.1)
    ):
        raise ValueError("protocol config does not match the locked MOSI protocol")
    return fields


def validation_schedule_identity(config: Mapping) -> Dict[str, object]:
    """Describe validation schedule configuration, not realized masks."""

    fields = protocol_fields(config)
    mask_seed = SeedBundle(int(fields["seed"])).derive("missing_mask")
    rates = {}
    for rate in MISSING_RATES:
        schedule = ConversationMaskSchedule(
            dataset=str(fields["dataset"]),
            split="validation",
            fold=int(fields["fold"]),
            requested_missing_rate=rate,
            mask_seed=mask_seed,
            freeze_evaluation=True,
        )
        rates[format(rate, ".1f")] = schedule.config_hash
    return {
        "kind": "schedule-config-hash-not-realized-mask",
        "split": "validation",
        "mask_seed": mask_seed,
        "rates": rates,
    }


def build_command(
    job: RawSDRJob,
    *,
    feature_root: Path = FEATURE_ROOT,
) -> List[str]:
    _validate_job(job)
    feature_root = resolve_feature_root(feature_root)
    return [
        str(DEFAULT_PYTHON),
        "-m",
        TRAINING_MODULE,
        "--audio-feature",
        "wav2vec-large-c-UTT",
        "--text-feature",
        "deberta-large-4-UTT",
        "--video-feature",
        "manet_UTT",
        "--feature-root",
        str(feature_root),
        "--output-dir",
        str(job.output_dir),
        "--seed",
        str(job.seed),
        "--epochs",
        "100",
        "--train-rate-mode",
        "all",
        "--lr",
        "0.0005",
        "--device",
        "cuda",
    ]


def _locked_counts(
    expected_parameter_counts: Optional[Mapping[str, int]],
):
    counts = (
        EXPECTED_PARAMETER_COUNTS
        if expected_parameter_counts is None
        else expected_parameter_counts
    )
    if not isinstance(counts, Mapping) or set(counts) != set(
        PARAMETER_COUNT_FIELDS
    ):
        return None
    if any(
        isinstance(counts[name], bool)
        or not isinstance(counts[name], int)
        or counts[name] < 1
        for name in PARAMETER_COUNT_FIELDS
    ):
        return None
    return dict(counts)


def _config_matches(config: object, job: RawSDRJob) -> bool:
    return isinstance(config, Mapping) and dict(config) == asdict(
        RawSDRTrainConfig(seed=job.seed)
    )


def _read_producer_provenance(job: RawSDRJob):
    payload, error = _read_json(job.output_dir / PRODUCER_PROVENANCE_NAME)
    if error is not None:
        return None, error
    if not isinstance(payload, Mapping):
        return None, "producer provenance must contain an object"
    features = payload.get("features")
    if not isinstance(features, Mapping):
        return None, "producer provenance features must contain an object"
    try:
        feature_root = resolve_feature_root(features.get("root"))
    except ValueError as error_value:
        return None, str(error_value)
    if dict(features) != _resolved_feature_paths(feature_root):
        return None, "producer provenance feature paths are inconsistent"
    if (
        payload.get("schema_version") != 1
        or payload.get("treatment") != TREATMENT
        or payload.get("training_module") != TRAINING_MODULE
        or payload.get("variant") != VARIANT
        or payload.get("seed") != job.seed
        or not _is_sha256(payload.get("canonical_config_sha256"))
        or not _is_sha256_mapping(payload.get("source_files_sha256"), SOURCE_FILES)
        or not isinstance(payload.get("training_runtime"), Mapping)
        or payload.get("validation_schedule_identity")
        != validation_schedule_identity(asdict(RawSDRTrainConfig(seed=job.seed)))
        or payload.get("command") != build_command(
            job,
            feature_root=feature_root,
        )
    ):
        return None, "producer provenance has invalid or mismatched fields"
    try:
        _validate_source_commit(str(payload.get("source_commit")))
    except ValueError as error_value:
        return None, str(error_value)
    expected_config_hash = _canonical_json_sha256(
        asdict(RawSDRTrainConfig(seed=job.seed))
    )
    if payload["canonical_config_sha256"] != expected_config_hash:
        return None, "producer provenance config hash mismatch"
    return dict(payload), None


def _write_producer_provenance_once(
    job: RawSDRJob,
    expected: Mapping,
) -> None:
    path = job.output_dir / PRODUCER_PROVENANCE_NAME
    existing, error = _read_producer_provenance(job)
    if error is None:
        if existing != dict(expected):
            raise ValueError("producer provenance mismatch; refusing relabel")
        return
    if path.exists():
        raise ValueError("producer provenance is unreadable: {}".format(error))
    job.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, dict(expected))


def _producer_provenance(
    job: RawSDRJob,
    *,
    source_commit: str,
    feature_root: Path,
    repo_root: Path,
) -> Dict[str, object]:
    _validate_job(job)
    source_commit = _validate_source_commit(source_commit)
    feature_root = resolve_feature_root(feature_root)
    repo_root = Path(repo_root).expanduser().resolve()
    return {
        "schema_version": 1,
        "treatment": TREATMENT,
        "training_module": TRAINING_MODULE,
        "variant": VARIANT,
        "seed": job.seed,
        "source_commit": source_commit,
        "source_files_sha256": {
            name: _sha256_file(repo_root / name) for name in SOURCE_FILES
        },
        "features": _resolved_feature_paths(feature_root),
        "training_runtime": dict(_query_training_runtime()),
        "training_python_executable": str(DEFAULT_PYTHON),
        "canonical_config_sha256": _canonical_json_sha256(
            asdict(RawSDRTrainConfig(seed=job.seed))
        ),
        "validation_schedule_identity": validation_schedule_identity(
            asdict(RawSDRTrainConfig(seed=job.seed))
        ),
        "command": build_command(job, feature_root=feature_root),
    }


def _validate_mask_audit(payload: object, config: Mapping):
    if not isinstance(payload, Mapping):
        return None, "mask audit must contain an object"
    expected_protocol = protocol_fields(config)
    if (
        payload.get("schema_version") != 1
        or payload.get("domain")
        != "deterministic-realized-availability-replay"
        or payload.get("protocol") != expected_protocol
        or payload.get("validation_schedule_identity")
        != validation_schedule_identity(config)
        or not _is_sha256_mapping(
            payload.get("validation_mask_sha256"), MISSING_RATE_KEYS
        )
        or not _is_sha256_mapping(
            payload.get("test_mask_sha256"), MISSING_RATE_KEYS
        )
    ):
        return None, "mask audit has invalid or mismatched fields"
    return dict(payload), None


def _read_mask_audit(result_dir: Path, config: Mapping):
    payload, error = _read_json(Path(result_dir) / MASK_AUDIT_NAME)
    if error is not None:
        return None, error
    return _validate_mask_audit(payload, config)


def _write_mask_audit_once(
    result_dir: Path,
    config: Mapping,
    expected: Mapping,
) -> None:
    validated, error = _validate_mask_audit(expected, config)
    if error is not None:
        raise ValueError(error)
    path = Path(result_dir) / MASK_AUDIT_NAME
    existing, existing_error = _read_mask_audit(result_dir, config)
    if existing_error is None:
        if existing != validated:
            raise ValueError("mask audit mismatch; refusing relabel")
        return
    if path.exists():
        raise ValueError("mask audit is unreadable: {}".format(existing_error))
    _atomic_json(path, validated)


def _replay_loader_hashes(loader, schedules, dimensions):
    hashes = {}
    for rate in MISSING_RATES:
        availability = []
        for raw in loader:
            view = base_train._prepare_view(
                raw,
                schedules[rate],
                epoch=0,
                dimensions=dimensions,
            )
            valid = view["umask"].transpose(0, 1).bool()
            availability.append(view["availability"][valid].cpu())
        if not availability:
            raise ValueError("mask replay loader is empty")
        hashes[format(rate, ".1f")] = base_train._sha256_tensor(
            torch.cat(availability, dim=0)
        )
    return hashes


@lru_cache(maxsize=32)
def _replay_mask_audit_cached(
    feature_root: str,
    dataset: str,
    seed: int,
    fold: int,
    evaluation_protocol: str,
    validation_fraction: float,
):
    roots = _resolved_feature_paths(Path(feature_root))
    loaders = base_train.get_loaders(
        audio_root=roots["audio"],
        text_root=roots["text"],
        video_root=roots["video"],
        num_folder=1,
        dataset=dataset,
        batch_size=32,
        num_workers=0,
        seed=seed,
        validation_fraction=validation_fraction,
        evaluation_protocol=evaluation_protocol,
    )
    _, validation_loaders, test_loaders, adim, tdim, vdim = loaders
    config = {
        "dataset": dataset,
        "seed": seed,
        "fold": fold,
        "evaluation_protocol": evaluation_protocol,
        "validation_fraction": validation_fraction,
    }
    validation_schedules = {
        rate: ConversationMaskSchedule(
            dataset=dataset,
            split="validation",
            fold=fold,
            requested_missing_rate=rate,
            mask_seed=SeedBundle(seed).derive("missing_mask"),
            freeze_evaluation=True,
        )
        for rate in MISSING_RATES
    }
    test_schedules = {
        rate: ConversationMaskSchedule(
            dataset=dataset,
            split="test",
            fold=fold,
            requested_missing_rate=rate,
            mask_seed=SeedBundle(seed).derive("missing_mask"),
            freeze_evaluation=True,
        )
        for rate in MISSING_RATES
    }
    dimensions = (adim, tdim, vdim)
    return {
        "schema_version": 1,
        "domain": "deterministic-realized-availability-replay",
        "protocol": protocol_fields(config),
        "validation_schedule_identity": validation_schedule_identity(config),
        "validation_mask_sha256": _replay_loader_hashes(
            validation_loaders[fold - 1], validation_schedules, dimensions
        ),
        "test_mask_sha256": _replay_loader_hashes(
            test_loaders[fold - 1], test_schedules, dimensions
        ),
    }


def replay_mask_audit(config: Mapping, feature_root: Path):
    fields = protocol_fields(config)
    root = resolve_feature_root(feature_root)
    return copy.deepcopy(
        _replay_mask_audit_cached(
            str(root),
            str(fields["dataset"]),
            int(fields["seed"]),
            int(fields["fold"]),
            str(fields["evaluation_protocol"]),
            float(fields["validation_fraction"]),
        )
    )


def build_expected_context(
    jobs: Sequence[RawSDRJob],
    *,
    source_commit: str,
    feature_root: Path,
    repo_root: Path,
    mask_audit_builder=replay_mask_audit,
) -> ExpectedContext:
    """Compute the sole provenance and realized-mask authority for a run."""

    normalized = _validate_job_subset(jobs)
    source_commit = _validate_source_commit(source_commit)
    feature_root = resolve_feature_root(feature_root)
    try:
        repo_root = Path(repo_root).expanduser().resolve(strict=True)
    except (TypeError, OSError, RuntimeError, ValueError) as error:
        raise ValueError("repo_root is invalid: {}".format(error))
    if not repo_root.is_dir():
        raise ValueError("repo_root must be a directory")
    context = ExpectedContext(
        producer_provenance={
            (job.variant, job.seed): _producer_provenance(
                job,
                source_commit=source_commit,
                feature_root=feature_root,
                repo_root=repo_root,
            )
            for job in normalized
        },
        mask_audits={
            job.seed: mask_audit_builder(
                asdict(RawSDRTrainConfig(seed=job.seed)), feature_root
            )
            for job in normalized
        },
    )
    return _validate_expected_context(normalized, context)


def _validate_expected_context(
    jobs: Sequence[RawSDRJob], expected_context: ExpectedContext
) -> ExpectedContext:
    normalized = _validate_job_subset(jobs)
    if not isinstance(expected_context, ExpectedContext):
        raise ValueError("expected_context must be an ExpectedContext")
    expected_keys = {(job.variant, job.seed) for job in normalized}
    expected_seeds = {job.seed for job in normalized}
    if (
        set(expected_context.producer_provenance) != expected_keys
        or set(expected_context.mask_audits) != expected_seeds
    ):
        raise ValueError("expected_context does not match the job matrix")
    for job in normalized:
        producer = expected_context.producer_provenance[
            (job.variant, job.seed)
        ]
        if (
            not isinstance(producer, Mapping)
            or producer.get("variant") != job.variant
            or producer.get("seed") != job.seed
            or producer.get("canonical_config_sha256")
            != _canonical_json_sha256(
                asdict(RawSDRTrainConfig(seed=job.seed))
            )
        ):
            raise ValueError("expected_context provenance is invalid")
        _, audit_error = _validate_mask_audit(
            expected_context.mask_audits[job.seed],
            asdict(RawSDRTrainConfig(seed=job.seed)),
        )
        if audit_error is not None:
            raise ValueError(
                "expected_context mask audit: {}".format(audit_error)
            )
    return expected_context


def inspect_result(
    job: RawSDRJob,
    *,
    expected_parameter_counts: Optional[Mapping[str, int]] = None,
    expected_provenance: Optional[Mapping] = None,
    expected_mask_audit: Optional[Mapping] = None,
) -> ResultInspection:
    """Recompute formal completion from durable artifacts."""

    _validate_job(job)
    counts = _locked_counts(expected_parameter_counts)
    if counts is None:
        return ResultInspection(False, "formal parameter counts are not locked")
    producer, producer_error = _read_producer_provenance(job)
    if producer_error is not None:
        return ResultInspection(False, "producer provenance: {}".format(producer_error))
    if expected_provenance is not None and producer != dict(expected_provenance):
        return ResultInspection(False, "producer provenance mismatch")

    for name in ("config.json", "history.json", "metrics.json", "train.log"):
        if not (job.output_dir / name).is_file():
            return ResultInspection(False, "missing {}".format(name))
    config, error = _read_json(job.output_dir / "config.json")
    if error is not None or not _config_matches(config, job):
        return ResultInspection(False, error or "config.json does not match locked job")
    mask_audit, error = _read_mask_audit(job.output_dir, config)
    if error is not None:
        return ResultInspection(False, "validation mask audit: {}".format(error))
    if expected_mask_audit is not None and mask_audit != dict(expected_mask_audit):
        return ResultInspection(False, "validation mask audit mismatch")
    history, error = _read_json(job.output_dir / "history.json")
    if error is not None:
        return ResultInspection(False, error)
    selected, error = _validated_history(history)
    if error is not None:
        return ResultInspection(False, error)
    selected_epoch, selected_validation, selected_mean = selected
    for record in history:
        for rate in MISSING_RATE_KEYS:
            if any(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not _finite_number(value)
                for value in record["validation"][rate].values()
            ):
                return ResultInspection(False, "history contains a non-finite metric")

    metrics, error = _read_json(job.output_dir / "metrics.json")
    if error is not None or not isinstance(metrics, Mapping):
        return ResultInspection(False, error or "metrics.json must contain an object")
    identities = {
        "variant": VARIANT,
        "sdr_variant": "sdr-public",
        "sdr_input_type": "raw-residual",
        "backbone": VARIANT,
    }
    if any(metrics.get(name) != value for name, value in identities.items()):
        return ResultInspection(False, "metrics identity mismatch")
    if metrics.get("best_epoch") != selected_epoch:
        return ResultInspection(False, "metrics are not validation-selected")
    if metrics.get("best_validation") != selected_validation or not _same_float(
        metrics.get("best_validation_mean_weighted_f1"), selected_mean
    ):
        return ResultInspection(False, "best validation metrics are inconsistent")
    if metrics.get("selection_missing_rates") != list(MISSING_RATES):
        return ResultInspection(False, "selection rates must contain all 8 rates")
    if metrics.get("evaluation_stage") != "train-validation-test":
        return ResultInspection(False, "formal result requires validation and test")
    if config.get("evaluate_test") is not True:
        return ResultInspection(False, "formal config must evaluate test")
    if any(metrics.get(name) != value for name, value in counts.items()):
        return ResultInspection(False, "parameter count mismatch")

    test = metrics.get("test")
    schedule_hashes = metrics.get("mask_sha256")
    archive_hashes = metrics.get("prediction_availability_sha256")
    if (
        not isinstance(test, Mapping)
        or set(test) != set(MISSING_RATE_KEYS)
        or not _is_sha256_mapping(schedule_hashes, MISSING_RATE_KEYS)
        or not _is_sha256_mapping(archive_hashes, MISSING_RATE_KEYS)
    ):
        return ResultInspection(False, "metrics must contain 8 test rates and both hash domains")
    if not _finite_number(metrics.get("wall_time_seconds")) or float(
        metrics["wall_time_seconds"]
    ) <= 0:
        return ResultInspection(False, "wall time is invalid")
    ema_steps = metrics.get("ema_steps")
    peak_memory = metrics.get("peak_memory_bytes")
    if (
        isinstance(ema_steps, bool)
        or not isinstance(ema_steps, int)
        or ema_steps < 1
        or isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory < 0
    ):
        return ResultInspection(False, "runtime metrics are invalid")

    collapsed = []
    for rate in MISSING_RATE_KEYS:
        rate_metrics = test[rate]
        if not isinstance(rate_metrics, Mapping):
            return ResultInspection(False, "test rate metrics are invalid")
        if rate_metrics.get("mask_sha256") != schedule_hashes[rate]:
            return ResultInspection(False, "schedule mask SHA mismatch at {}".format(rate))
        if any(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not _finite_number(value)
            for value in rate_metrics.values()
        ):
            return ResultInspection(False, "test contains a non-finite metric")
        archive = job.output_dir / "predictions_miss_{}.npz".format(
            rate.replace(".", "p")
        )
        prediction, archive_error = _inspect_prediction_archive(archive)
        if archive_error is not None:
            return ResultInspection(
                False,
                "prediction archive {}: {}".format(rate, archive_error),
            )
        assert isinstance(prediction, PredictionInspection)
        if (
            rate_metrics.get("prediction_availability_sha256")
            != prediction.mask_sha256
            or archive_hashes[rate] != prediction.mask_sha256
        ):
            return ResultInspection(
                False, "prediction availability SHA mismatch at {}".format(rate)
            )
        for name in ("weighted_f1", "prediction_std"):
            if not _same_float(rate_metrics.get(name), getattr(prediction, name), 1e-8):
                return ResultInspection(
                    False, "prediction archive recomputed {} mismatch".format(name)
                )
        if rate_metrics.get("predicted_sign_count") != prediction.predicted_sign_count:
            return ResultInspection(False, "prediction archive sign count mismatch")
        if prediction.collapsed:
            collapsed.append(rate)
    if mask_audit["test_mask_sha256"] != dict(schedule_hashes):
        return ResultInspection(False, "test mask replay sentinel mismatch")
    return ResultInspection(
        True,
        "complete-test-8-rates",
        collapsed=bool(collapsed),
        collapse_rates=tuple(collapsed),
    )


def pending_jobs(
    jobs: Sequence[RawSDRJob],
    *,
    expected_parameter_counts: Optional[Mapping[str, int]],
    expected_context: ExpectedContext,
) -> List[RawSDRJob]:
    jobs = _validate_job_subset(jobs)
    context = _validate_expected_context(jobs, expected_context)
    return [
        job
        for job in jobs
        if not inspect_result(
            job,
            expected_parameter_counts=expected_parameter_counts,
            expected_provenance=context.producer_provenance[
                (job.variant, job.seed)
            ],
            expected_mask_audit=context.mask_audits[job.seed],
        ).complete
    ]


def _validate_job_subset(jobs: Sequence[RawSDRJob]) -> List[RawSDRJob]:
    normalized = list(jobs)
    identities = set()
    gpus = set()
    for job in normalized:
        _validate_job(job)
        if job.seed in identities or job.gpu in gpus:
            raise ValueError("jobs must use unique registered seeds and GPUs")
        identities.add(job.seed)
        gpus.add(job.gpu)
    return normalized


def _prune_checkpoint(output_dir: Path):
    path = Path(output_dir) / "best.pt"
    if not path.is_file():
        return None
    evidence = {"name": path.name, "sha256": _sha256_file(path)}
    path.unlink()
    return evidence


@contextmanager
def job_lock(job: RawSDRJob, producer_provenance: Mapping):
    """Hold a kernel-released nonblocking lock for one formal job lifecycle."""

    _validate_job(job)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    path = job.output_dir / LOCK_NAME
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "formal job seed {} is locked by another runner".format(job.seed)
            )
        metadata = {
            "schema_version": 1,
            "variant": VARIANT,
            "seed": job.seed,
            "gpu": job.gpu,
            "owner_pid": os.getpid(),
            "owner_host": socket.gethostname(),
            "started_at_unix": time.time(),
            "producer_provenance_sha256": _canonical_json_sha256(
                dict(producer_provenance)
            ),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield metadata
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def run_jobs(
    jobs: Sequence[RawSDRJob],
    *,
    feature_root: Path,
    repo_root: Path,
    timeout_seconds: int,
    source_commit: str,
    expected_context: ExpectedContext,
    expected_parameter_counts: Optional[Mapping[str, int]] = None,
    manifest_path: Optional[Path] = None,
) -> int:
    """Launch at most one registered treatment on each healthy GPU."""

    all_jobs = _validate_job_subset(jobs)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
    ):
        raise ValueError("timeout_seconds must be positive")
    counts = _locked_counts(expected_parameter_counts)
    if counts is None:
        raise ValueError("formal parameter counts are not locked")
    source_commit = _validate_source_commit(source_commit)
    feature_root = resolve_feature_root(feature_root)
    try:
        repo_root = Path(repo_root).expanduser().resolve(strict=True)
    except (TypeError, OSError, RuntimeError, ValueError) as error:
        raise ValueError("repo_root is invalid: {}".format(error))
    if not repo_root.is_dir():
        raise ValueError("repo_root must be a directory")
    context = _validate_expected_context(all_jobs, expected_context)
    if any(
        context.producer_provenance[(job.variant, job.seed)].get(
            "source_commit"
        )
        != source_commit
        for job in all_jobs
    ):
        raise ValueError("expected_context source commit mismatch")
    running = []
    handled = set()
    failures = 0
    with ExitStack() as lock_stack:
        for job in all_jobs:
            lock_stack.enter_context(
                job_lock(
                    job,
                    context.producer_provenance[(job.variant, job.seed)],
                )
            )
        for job in all_jobs:
            if (job.output_dir / PRODUCER_PROVENANCE_NAME).exists():
                _write_producer_provenance_once(
                    job,
                    context.producer_provenance[(job.variant, job.seed)],
                )
            _write_mask_audit_once(
                job.output_dir,
                asdict(RawSDRTrainConfig(seed=job.seed)),
                context.mask_audits[job.seed],
            )
        pending = pending_jobs(
            all_jobs,
            expected_parameter_counts=counts,
            expected_context=context,
        )
        try:
            for job in pending:
                reset_incomplete_output(job.output_dir)
                _write_producer_provenance_once(
                    job,
                    context.producer_provenance[(job.variant, job.seed)],
                )
                _write_mask_audit_once(
                    job.output_dir,
                    asdict(RawSDRTrainConfig(seed=job.seed)),
                    context.mask_audits[job.seed],
                )
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
                        cwd=repo_root,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log_handle.close()
                    raise
                running.append(
                    (job, process, log_handle, command, started_at)
                )
                _atomic_json(
                    job.output_dir / "status.json",
                    {
                        "state": "running",
                        "pid": process.pid,
                        "variant": VARIANT,
                        "seed": job.seed,
                        "gpu": job.gpu,
                        "command": command,
                        "started_at_unix": started_at,
                    },
                )

            for job, process, log_handle, command, started_at in running:
                timed_out = False
                returncode = None
                try:
                    returncode = process.wait(
                        timeout=max(
                            0.0,
                            started_at + timeout_seconds - time.time(),
                        )
                    )
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_tree(process)
                finally:
                    log_handle.close()
                handled.add(id(process))
                inspection = inspect_result(
                    job,
                    expected_parameter_counts=counts,
                    expected_provenance=context.producer_provenance[
                        (job.variant, job.seed)
                    ],
                    expected_mask_audit=context.mask_audits[job.seed],
                )
                checkpoint = _prune_checkpoint(job.output_dir)
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
                        "variant": VARIANT,
                        "seed": job.seed,
                        "gpu": job.gpu,
                        "command": command,
                        "returncode": returncode,
                        "completion_reason": inspection.reason,
                        "checkpoint_evidence": checkpoint,
                        "started_at_unix": started_at,
                        "finished_at_unix": time.time(),
                    },
                )
            if manifest_path is not None:
                write_manifest(
                    manifest_path,
                    all_jobs,
                    source_commit=source_commit,
                    feature_root=feature_root,
                    repo_root=repo_root,
                    expected_parameter_counts=counts,
                    expected_context=context,
                )
        finally:
            for _, process, log_handle, _, _ in running:
                if id(process) not in handled:
                    _terminate_process_tree(process)
                if not log_handle.closed:
                    log_handle.close()
    return failures


def write_manifest(
    path: Path,
    jobs: Sequence[RawSDRJob],
    *,
    source_commit: str,
    feature_root: Path,
    repo_root: Path,
    expected_parameter_counts: Optional[Mapping[str, int]] = None,
    expected_context: ExpectedContext,
) -> None:
    jobs = _validate_job_subset(jobs)
    context = _validate_expected_context(jobs, expected_context)
    counts = _locked_counts(expected_parameter_counts)
    source_commit = _validate_source_commit(source_commit)
    entries = []
    for job in jobs:
        producer, producer_error = _read_producer_provenance(job)
        inspection = inspect_result(
            job,
            expected_parameter_counts=counts,
            expected_provenance=context.producer_provenance[
                (job.variant, job.seed)
            ],
            expected_mask_audit=context.mask_audits[job.seed],
        )
        entries.append(
            {
                "seed": job.seed,
                "gpu": job.gpu,
                "variant": VARIANT,
                "output_dir": str(job.output_dir),
                "complete": inspection.complete,
                "completion_reason": inspection.reason,
                "producer_provenance": producer,
                "producer_provenance_error": producer_error,
                "expected_producer_provenance_sha256": (
                    _canonical_json_sha256(
                        context.producer_provenance[
                            (job.variant, job.seed)
                        ]
                    )
                ),
                "expected_mask_audit_sha256": _canonical_json_sha256(
                    context.mask_audits[job.seed]
                ),
                "parameter_count_lock": counts,
            }
        )
    _atomic_json(
        Path(path),
        {
            "schema_version": 1,
            "treatment": TREATMENT,
            "training_module": TRAINING_MODULE,
            "source_commit": source_commit,
            "source_files_sha256": {
                name: _sha256_file(Path(repo_root) / name) for name in SOURCE_FILES
            },
            "features": _resolved_feature_paths(feature_root),
            "seeds": list(SEEDS),
            "gpu_mapping": dict(zip(map(str, SEEDS), HEALTHY_GPUS)),
            "required_epochs": 100,
            "required_rates": list(MISSING_RATE_KEYS),
            "jobs": entries,
        },
    )


def expected_control_config(seed: int) -> Dict[str, object]:
    if seed not in SEEDS:
        raise ValueError("control seed is outside the registered matrix")
    return {
        "base_model": "LSTM",
        "batch_size": 32,
        "classification_completion": False,
        "dataset": "CMUMOSI",
        "device": "cuda",
        "dropout": 0.5,
        "ema_tau": 0.996,
        "epochs": 100,
        "evaluation_protocol": "official",
        "fold": 1,
        "fusion_type": "slot",
        "gradient_clip_norm": 1.0,
        "graph_branch_mode": "both",
        "hidden": 200,
        "jepa_weight": 0.1,
        "latent_dim": 256,
        "learning_rate": 5e-4,
        "local_context_residual": False,
        "local_fusion_dropout": 0.2,
        "local_fusion_hidden_dim": 256,
        "mmoe_variant": "dual-gate",
        "mosi_task_mode": "regression",
        "num_experts": 4,
        "predictor_dropout": 0.1,
        "projector_dropout": 0.1,
        "representation_type": "slot",
        "seed": seed,
        "temperature": 0.03,
        "time_attention": False,
        "top_k": 2,
        "train_rate_mode": "all",
        "validation_fraction": 0.1,
        "weight_decay": 1e-5,
        "window_future": 2,
        "window_past": 2,
    }


def load_inherited_reference(
    result_dirs: Mapping[int, Path],
    *,
    reference_kind: str,
    expected_dirs: Optional[Mapping[int, Path]] = None,
    feature_root: Path = FEATURE_ROOT,
    mask_audit_builder=replay_mask_audit,
) -> Dict[int, Mapping]:
    if reference_kind not in {"slot", "control"}:
        raise ValueError("reference_kind must be slot or control")
    normalized = {
        int(seed): Path(path).expanduser().resolve()
        for seed, path in result_dirs.items()
    }
    if set(normalized) != set(SEEDS):
        raise ValueError("inherited reference must contain exactly five seeds")
    if expected_dirs is None:
        expected_dirs = (
            SLOT_REFERENCE_DIRS
            if reference_kind == "slot"
            else CONTROL_REFERENCE_DIRS
        )
    expected_paths = {
        int(seed): Path(path).expanduser().resolve()
        for seed, path in expected_dirs.items()
    }
    if normalized != expected_paths:
        raise ValueError("inherited reference path identity mismatch")
    result = {}
    for seed in SEEDS:
        root = normalized[seed]
        config, config_error = _read_json(root / "config.json")
        history, history_error = _read_json(root / "history.json")
        metrics, metrics_error = _read_json(root / "metrics.json")
        if config_error or history_error or metrics_error:
            raise ValueError(config_error or history_error or metrics_error)
        if not isinstance(config, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError("reference config and metrics must contain objects")
        selected, selection_error = _validated_history(history)
        if selection_error is not None:
            raise ValueError(selection_error)
        best_epoch, best_validation, best_mean = selected
        if (
            metrics.get("best_epoch") != best_epoch
            or not _same_float(
                metrics.get("best_validation_mean_weighted_f1"), best_mean
            )
        ):
            raise ValueError("reference metrics are not validation-selected")

        if reference_kind == "slot":
            expected_config = asdict(
                SDRTrainConfig(seed=seed, sdr_variant="sdr-public")
            )
            if dict(config) != expected_config:
                raise ValueError("Slot reference config identity mismatch")
            expected_counts = LEGACY_SDR_PARAMETER_COUNTS["sdr-public"]
            if (
                metrics.get("variant") != "sdr-public"
                or metrics.get("sdr_variant") != "sdr-public"
                or metrics.get("backbone") != "sdr-gnn-whole-backbone"
                or any(
                    metrics.get(name) != value
                    for name, value in expected_counts.items()
                )
                or metrics.get("best_validation") != best_validation
            ):
                raise ValueError("Slot reference metrics identity mismatch")
            metrics_identity = "slot-sdr-public"
            source_lane = "formal-sdr-public"
        else:
            if dict(config) != expected_control_config(seed):
                raise ValueError("Control reference config identity mismatch")
            old_schema = {
                "best_epoch",
                "best_validation_mean_weighted_f1",
                "ema_steps",
                "mask_sha256",
                "parameter_count",
                "test",
                "trainable_parameter_count",
            }
            if (
                set(metrics) != old_schema
                or metrics.get("parameter_count") != 32_089_733
                or metrics.get("trainable_parameter_count") != 31_229_573
            ):
                raise ValueError("Control reference old metrics schema mismatch")
            metrics_identity = "gcnet-control-old-schema"
            source_lane = "screen-seed66" if seed == 66 else "formal"

        test_hashes = metrics.get("mask_sha256")
        test = metrics.get("test")
        if (
            not _is_sha256_mapping(test_hashes, MISSING_RATE_KEYS)
            or not isinstance(test, Mapping)
            or set(test) != set(MISSING_RATE_KEYS)
            or any(
                not isinstance(test[rate], Mapping)
                or test[rate].get("mask_sha256") != test_hashes[rate]
                for rate in MISSING_RATE_KEYS
            )
        ):
            raise ValueError("reference test schedule hashes are invalid")
        expected_audit = mask_audit_builder(config, feature_root)
        expected_audit, replay_error = _validate_mask_audit(
            expected_audit, config
        )
        if replay_error is not None:
            raise ValueError(
                "reference mask replay is invalid: {}".format(replay_error)
            )
        audit_path = root / MASK_AUDIT_NAME
        if audit_path.exists():
            recorded_audit, audit_error = _read_mask_audit(root, config)
            if audit_error is not None or recorded_audit != expected_audit:
                raise ValueError(
                    "reference mask replay mismatch: {}".format(
                        audit_error or "sidecar differs from fresh replay"
                    )
                )
            audit_source = "sidecar-verified-against-fresh-replay"
        else:
            audit_source = "fresh-replay-no-source-sidecar"
        audit = expected_audit
        if audit["test_mask_sha256"] != dict(test_hashes):
            raise ValueError("reference test replay sentinel mismatch")
        source_files_sha256 = {
            name: _sha256_file(root / name)
            for name in ("config.json", "history.json", "metrics.json")
        }
        result[seed] = {
            "source_path": str(root),
            "source_lane": source_lane,
            "metrics_identity": metrics_identity,
            "protocol": protocol_fields(config),
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "best_validation_mean_weighted_f1": best_mean,
            "test": dict(test),
            "test_schedule_sha256": dict(test_hashes),
            "mask_audit": audit,
            "mask_audit_source": audit_source,
            "source_files_sha256": source_files_sha256,
        }
    return result


def write_reference_audit(
    path: Path,
    *,
    slot_reference: Mapping[int, Mapping],
    control_reference: Mapping[int, Mapping],
) -> None:
    """Record preflight evidence in the new run root, never old references."""

    entries = []
    for kind, reference in (
        ("slot", _reference_validation(slot_reference, "Slot-SDR-public")),
        ("control", _reference_validation(control_reference, "GCNet control")),
    ):
        for seed in SEEDS:
            record = reference[seed]
            source_hashes = record.get("source_files_sha256")
            audit = record.get("mask_audit")
            if (
                not _is_sha256_mapping(
                    source_hashes,
                    ("config.json", "history.json", "metrics.json"),
                )
                or not isinstance(audit, Mapping)
            ):
                raise ValueError("reference audit record is incomplete")
            entries.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "source_path": record["source_path"],
                    "source_files_sha256": dict(source_hashes),
                    "mask_audit_source": record["mask_audit_source"],
                    "validation_mask_sha256": dict(
                        audit["validation_mask_sha256"]
                    ),
                    "test_mask_sha256": dict(audit["test_mask_sha256"]),
                }
            )
    _atomic_json(
        Path(path),
        {
            "schema_version": 1,
            "domain": "inherited-reference-preflight",
            "references": entries,
        },
    )


def _reference_validation(reference: Mapping, name: str):
    normalized = {}
    for raw_seed, metrics in reference.items():
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            raise ValueError("{} reference has an invalid seed".format(name))
        if seed in normalized:
            raise ValueError("{} reference has duplicate seeds".format(name))
        normalized[seed] = metrics
    if set(normalized) != set(SEEDS):
        raise ValueError("{} reference must contain exactly five seeds".format(name))
    return normalized


def _validation_map(metrics: Mapping):
    validation = metrics.get("best_validation", metrics)
    if not isinstance(validation, Mapping) or set(validation) != set(
        MISSING_RATE_KEYS
    ):
        raise ValueError("reference must contain eight validation scores")
    scores = {}
    for rate in MISSING_RATE_KEYS:
        value = validation[rate]
        if not isinstance(value, Mapping) or not _finite_number(
            value.get("weighted_f1")
        ):
            raise ValueError("reference validation score is invalid")
        scores[rate] = float(value["weighted_f1"])
    return scores


def aggregate(
    jobs: Sequence[RawSDRJob],
    *,
    slot_reference: Mapping,
    control_reference: Mapping,
    expected_context: ExpectedContext,
    expected_parameter_counts: Optional[Mapping[str, int]] = None,
    require_complete_results: bool = True,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    jobs = list(jobs)
    if len(jobs) != 5 or {job.seed for job in jobs} != set(SEEDS):
        raise ValueError("aggregate requires exactly five treatment seeds")
    _validate_job_subset(jobs)
    context = _validate_expected_context(jobs, expected_context)
    slot = _reference_validation(slot_reference, "Slot-SDR-public")
    control = _reference_validation(control_reference, "GCNet control")

    seed_rows = {}
    slot_deltas = []
    slot_high_deltas = []
    control_deltas = []
    positive_seeds = []
    all_paired = True
    collapse_free = True
    test_means = []
    for job in sorted(jobs, key=lambda value: value.seed):
        if require_complete_results:
            inspection = inspect_result(
                job,
                expected_parameter_counts=expected_parameter_counts,
                expected_provenance=context.producer_provenance[
                    (job.variant, job.seed)
                ],
                expected_mask_audit=context.mask_audits[job.seed],
            )
            if not inspection.complete:
                raise ValueError(
                    "seed {} is incomplete: {}".format(job.seed, inspection.reason)
                )
        metrics, error = _read_json(job.output_dir / "metrics.json")
        if error is not None or not isinstance(metrics, Mapping):
            raise ValueError(error or "metrics.json must contain an object")
        config, config_error = _read_json(job.output_dir / "config.json")
        if config_error is not None or not isinstance(config, Mapping):
            raise ValueError(config_error or "config.json must contain an object")
        candidate_audit, candidate_audit_error = _read_mask_audit(
            job.output_dir, config
        )
        candidate = _validation_map(metrics)
        slot_record = slot[job.seed]
        control_record = control[job.seed]
        slot_scores = _validation_map(slot_record)
        control_scores = _validation_map(control_record)

        try:
            candidate_protocol = protocol_fields(config)
        except ValueError:
            candidate_protocol = None
        protocols_match = (
            candidate_protocol is not None
            and candidate_protocol == slot_record.get("protocol")
            and candidate_protocol == control_record.get("protocol")
        )
        validation_masks_match = (
            candidate_audit_error is None
            and candidate_audit.get("validation_mask_sha256")
            == slot_record.get("mask_audit", {}).get(
                "validation_mask_sha256"
            )
            == control_record.get("mask_audit", {}).get(
                "validation_mask_sha256"
            )
        )
        candidate_test_hashes = metrics.get("mask_sha256")
        test_sentinels_match = (
            candidate_audit_error is None
            and candidate_audit.get("test_mask_sha256")
            == candidate_test_hashes
            == slot_record.get("mask_audit", {}).get("test_mask_sha256")
            == slot_record.get("test_schedule_sha256")
            == control_record.get("mask_audit", {}).get("test_mask_sha256")
            == control_record.get("test_schedule_sha256")
        )
        paired = (
            protocols_match
            and validation_masks_match
            and test_sentinels_match
        )
        all_paired = all_paired and paired
        validation_mean = mean(candidate.values())
        high_mean = mean(candidate[rate] for rate in HIGH_MISSING_RATE_KEYS)
        slot_mean = mean(slot_scores.values())
        slot_high = mean(slot_scores[rate] for rate in HIGH_MISSING_RATE_KEYS)
        control_mean = mean(control_scores.values())
        slot_delta = validation_mean - slot_mean if paired else None
        slot_high_delta = high_mean - slot_high if paired else None
        control_delta = validation_mean - control_mean if paired else None
        if paired:
            slot_deltas.append(slot_delta)
            slot_high_deltas.append(slot_high_delta)
            control_deltas.append(control_delta)
            if slot_delta > 0:
                positive_seeds.append(job.seed)
        collapsed = _validation_collapse_rates(metrics["best_validation"])
        if collapsed is None or collapsed:
            collapse_free = False
        test = metrics.get("test", {})
        test_mean = (
            mean(float(test[rate]["weighted_f1"]) for rate in MISSING_RATE_KEYS)
            if isinstance(test, Mapping) and set(test) == set(MISSING_RATE_KEYS)
            else None
        )
        if test_mean is not None:
            test_means.append(test_mean)
        seed_rows[str(job.seed)] = {
            "validation_mean": validation_mean,
            "validation_high_missing_mean": high_mean,
            "slot_validation_delta": slot_delta,
            "slot_high_missing_delta": slot_high_delta,
            "control_validation_delta": control_delta,
            "paired": paired,
            "pairing_checks": {
                "protocols_match": protocols_match,
                "validation_masks_match": validation_masks_match,
                "test_replay_sentinels_match": test_sentinels_match,
            },
            "validation_collapse_rates": collapsed,
            "test_mean_descriptive": test_mean,
        }

    paired_complete = all_paired and len(slot_deltas) == 5
    mean_slot_delta = mean(slot_deltas) if paired_complete else None
    mean_high_delta = mean(slot_high_deltas) if paired_complete else None
    mean_control_delta = mean(control_deltas) if paired_complete else None
    primary_criteria = {
        "all_five_seeds_paired": paired_complete,
        "mean_validation_delta_positive": (
            mean_slot_delta is not None and mean_slot_delta > 0
        ),
        "positive_seed_count_at_least_3": len(positive_seeds) >= 3,
        "high_missing_delta_nonnegative": (
            mean_high_delta is not None and mean_high_delta >= 0
        ),
        "validation_collapse_free": collapse_free,
    }
    if not paired_complete:
        primary_status = "not-assessable"
    elif all(primary_criteria.values()):
        primary_status = "pass"
    else:
        primary_status = "fail"
    beats_control = mean_control_delta is not None and mean_control_delta > 0
    formal_status = (
        "not-assessable"
        if primary_status == "not-assessable"
        else "pass"
        if primary_status == "pass" and beats_control
        else "fail"
    )
    summary = {
        "selection_basis": "validation-only",
        "test_used_for_selection": False,
        "pairing_evidence": {
            "kind": "realized-validation-mask-replay",
            "source": MASK_AUDIT_NAME,
            "test_scores_used": False,
            "test_hashes_used_as_replay_sentinel_only": True,
            "all_five_seeds_paired": paired_complete,
        },
        "seeds": seed_rows,
        "primary_gate": {
            "status": primary_status,
            "reference": "slot-sdr-public",
            "criteria": primary_criteria,
            "positive_seeds": positive_seeds,
            "positive_seed_count": len(positive_seeds),
            "mean_validation_delta": mean_slot_delta,
            "mean_high_missing_delta": mean_high_delta,
        },
        "control_comparison": {
            "reference": "gcnet-control",
            "mean_validation_delta": mean_control_delta,
            "beats_control": beats_control,
        },
        "formal_gate": {
            "status": formal_status,
            "uses_test": False,
            "requires_primary_pass_and_control_improvement": True,
        },
        "test_descriptive": {
            "available_seed_count": len(test_means),
            "mean": mean(test_means) if test_means else None,
        },
    }
    if output_path is not None:
        _atomic_json(output_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    jobs = build_jobs(output_root=args.output_root)
    if args.dry_run:
        for job in jobs:
            print(
                "COMMAND {}".format(
                    shlex.join(
                        build_command(job, feature_root=args.feature_root)
                    )
                )
            )
        print(json.dumps({"dry_run": True, "jobs": 5}, sort_keys=True))
        return 0
    counts = _locked_counts(None)
    if counts is None:
        raise SystemExit(
            "formal parameter counts are not locked; run the single smoke first"
        )
    slot = load_inherited_reference(
        SLOT_REFERENCE_DIRS,
        reference_kind="slot",
        expected_dirs=SLOT_REFERENCE_DIRS,
        feature_root=args.feature_root,
    )
    control = load_inherited_reference(
        CONTROL_REFERENCE_DIRS,
        reference_kind="control",
        expected_dirs=CONTROL_REFERENCE_DIRS,
        feature_root=args.feature_root,
    )
    write_reference_audit(
        args.output_root / "reference_audit.json",
        slot_reference=slot,
        control_reference=control,
    )
    context = build_expected_context(
        jobs,
        source_commit=args.source_commit,
        feature_root=args.feature_root,
        repo_root=args.repo_root,
    )
    manifest_path = args.output_root / "manifest.json"
    failures = run_jobs(
        jobs,
        feature_root=args.feature_root,
        repo_root=args.repo_root,
        timeout_seconds=args.timeout_seconds,
        source_commit=args.source_commit,
        expected_context=context,
        expected_parameter_counts=counts,
        manifest_path=manifest_path,
    )
    summary_path = args.output_root / "summary.json"
    runner_status_path = args.output_root / "runner_status.json"
    with ExitStack() as lock_stack:
        for job in jobs:
            lock_stack.enter_context(
                job_lock(
                    job,
                    context.producer_provenance[(job.variant, job.seed)],
                )
            )
        for stale_path in (summary_path, runner_status_path):
            if stale_path.is_file():
                stale_path.unlink()
        incomplete = pending_jobs(
            jobs,
            expected_parameter_counts=counts,
            expected_context=context,
        )
        write_manifest(
            manifest_path,
            jobs,
            source_commit=args.source_commit,
            feature_root=args.feature_root,
            repo_root=args.repo_root,
            expected_parameter_counts=counts,
            expected_context=context,
        )
        if not incomplete:
            aggregate(
                jobs,
                slot_reference=slot,
                control_reference=control,
                expected_context=context,
                expected_parameter_counts=counts,
                output_path=summary_path,
            )
        _atomic_json(
            runner_status_path,
            {
                "state": "complete" if not incomplete else "failed",
                "failures": failures,
                "incomplete": len(incomplete),
                "jobs": 5,
            },
        )
    return 0 if not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
