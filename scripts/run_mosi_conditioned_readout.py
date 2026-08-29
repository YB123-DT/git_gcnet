#!/usr/bin/env python3
"""Run and validation-audit the locked MOSI conditioned-readout experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


FEATURE_ROOT = Path(
    "/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/"
    "dataset/CMUMOSI/features"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data2/yb/remote_experiments/"
    "missing_m3_mosi_conditioned_readout_20260829/results"
)
DEFAULT_CONTROL_ROOT = Path(
    "/data2/yb/remote_experiments/"
    "missing_m3_mosi_conditioned_readout_20260829/results/"
    "deterministic-legacy"
)
VALID_RATES = tuple(index / 10 for index in range(8))
HIGH_MISSING_RATES = (0.4, 0.5, 0.6, 0.7)


@dataclass(frozen=True)
class ReadoutJob:
    seed: int
    gpu: int
    readout_type: str
    rank: int
    output_dir: Path
    jepa_regression_aggregation: str = "target"
    recurrent_padding_mode: str = "legacy"
    task_regression_loss: str = "mse"
    task_smooth_l1_beta: float = 1.0
    postgraph_sequence_mode: str = "independent"
    jepa_rate_weighting: str = "uniform"
    graph_message_calibration: str = "none"


def _variant_directory(
    *,
    readout_type: str,
    rank: int,
    jepa_regression_aggregation: str,
    recurrent_padding_mode: str,
    task_regression_loss: str,
    task_smooth_l1_beta: float,
    postgraph_sequence_mode: str,
    jepa_rate_weighting: str,
    graph_message_calibration: str,
) -> str:
    if graph_message_calibration == "branch-layernorm-residual":
        return "branch-graph-message-calibration"
    if jepa_rate_weighting == "sparsity-budget":
        return "sparsity-weighted-jepa"
    if postgraph_sequence_mode == "shared-bilstm":
        return "shared-postgraph-bilstm"
    if task_regression_loss == "smooth-l1":
        beta = format(task_smooth_l1_beta, "g").replace(".", "p")
        return "smooth-l1-task_beta{}".format(beta)
    if recurrent_padding_mode == "packed":
        return "packed-recurrent"
    if jepa_regression_aggregation == "utterance":
        return "utterance-balanced-jepa"
    if readout_type == "availability-affine":
        return readout_type
    return "{}_rank{}".format(readout_type, rank)


def build_jobs(
    *,
    seeds: Sequence[int],
    gpus: Sequence[int],
    output_root: Path,
    readout_type: str,
    rank: int,
    jepa_regression_aggregation: str = "target",
    recurrent_padding_mode: str = "legacy",
    task_regression_loss: str = "mse",
    task_smooth_l1_beta: float = 1.0,
    postgraph_sequence_mode: str = "independent",
    jepa_rate_weighting: str = "uniform",
    graph_message_calibration: str = "none",
) -> list[ReadoutJob]:
    if len(seeds) != len(gpus):
        raise ValueError("seeds and gpus must have the same length")
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if any(int(gpu) == 4 for gpu in gpus):
        raise ValueError("GPU 4 is forbidden by the remote execution contract")
    if int(rank) <= 0:
        raise ValueError("rank must be positive")
    if jepa_regression_aggregation not in {"target", "utterance"}:
        raise ValueError("unsupported JEPA regression aggregation")
    if recurrent_padding_mode not in {"legacy", "packed"}:
        raise ValueError("unsupported recurrent padding mode")
    if task_regression_loss not in {"mse", "smooth-l1"}:
        raise ValueError("unsupported task regression loss")
    if not math.isfinite(task_smooth_l1_beta) or task_smooth_l1_beta <= 0:
        raise ValueError("task SmoothL1 beta must be finite and positive")
    if postgraph_sequence_mode not in {"independent", "shared-bilstm"}:
        raise ValueError("unsupported postgraph sequence mode")
    if jepa_rate_weighting not in {"uniform", "sparsity-budget"}:
        raise ValueError("unsupported JEPA rate weighting")
    if graph_message_calibration not in {
        "none",
        "branch-layernorm-residual",
    }:
        raise ValueError("unsupported graph message calibration")
    readout_treatments = {
        "availability-low-rank",
        "shared-low-rank-parammatch",
        "availability-affine",
    }
    if readout_type != "shared" and readout_type not in readout_treatments:
        raise ValueError("unsupported candidate readout_type")
    treatment_count = sum(
        (
            readout_type != "shared",
            jepa_regression_aggregation == "utterance",
            recurrent_padding_mode == "packed",
            task_regression_loss == "smooth-l1",
            postgraph_sequence_mode == "shared-bilstm",
            jepa_rate_weighting == "sparsity-budget",
            graph_message_calibration == "branch-layernorm-residual",
        )
    )
    if treatment_count == 0:
        raise ValueError("candidate must change one registered treatment")
    if treatment_count > 1:
        raise ValueError("registered treatments cannot be combined")
    return [
        ReadoutJob(
            seed=int(seed),
            gpu=int(gpu),
            readout_type=readout_type,
            rank=int(rank),
            output_dir=(
                Path(output_root)
                / _variant_directory(
                    readout_type=readout_type,
                    rank=rank,
                    jepa_regression_aggregation=jepa_regression_aggregation,
                    recurrent_padding_mode=recurrent_padding_mode,
                    task_regression_loss=task_regression_loss,
                    task_smooth_l1_beta=task_smooth_l1_beta,
                    postgraph_sequence_mode=postgraph_sequence_mode,
                    jepa_rate_weighting=jepa_rate_weighting,
                    graph_message_calibration=graph_message_calibration,
                )
                / "seed_{}".format(seed)
            ),
            jepa_regression_aggregation=jepa_regression_aggregation,
            recurrent_padding_mode=recurrent_padding_mode,
            task_regression_loss=task_regression_loss,
            task_smooth_l1_beta=float(task_smooth_l1_beta),
            postgraph_sequence_mode=postgraph_sequence_mode,
            jepa_rate_weighting=jepa_rate_weighting,
            graph_message_calibration=graph_message_calibration,
        )
        for seed, gpu in zip(seeds, gpus)
    ]


def build_fresh_legacy_controls(
    candidates: Sequence[ReadoutJob],
    output_root: Path,
) -> list[ReadoutJob]:
    if not candidates:
        raise ValueError("packed candidates are required")
    if any(
        job.readout_type != "shared"
        or job.jepa_regression_aggregation != "target"
        or job.recurrent_padding_mode != "packed"
        or job.task_regression_loss != "mse"
        or job.task_smooth_l1_beta != 1.0
        or job.postgraph_sequence_mode != "independent"
        or job.jepa_rate_weighting != "uniform"
        or job.graph_message_calibration != "none"
        for job in candidates
    ):
        raise ValueError("fresh legacy controls require packed-only candidates")
    return [
        ReadoutJob(
            seed=job.seed,
            gpu=job.gpu,
            readout_type="shared",
            rank=job.rank,
            output_dir=(
                Path(output_root)
                / "deterministic-legacy"
                / "seed_{}".format(job.seed)
            ),
            jepa_regression_aggregation="target",
            recurrent_padding_mode="legacy",
            task_regression_loss="mse",
            task_smooth_l1_beta=1.0,
            postgraph_sequence_mode="independent",
            jepa_rate_weighting="uniform",
            graph_message_calibration="none",
        )
        for job in candidates
    ]


def build_confirmation_legacy_controls(
    candidates: Sequence[ReadoutJob],
    control_root: Path,
) -> list[ReadoutJob]:
    """Build only the missing direct Legacy controls for 5-seed confirmation."""
    if not candidates:
        raise ValueError("shared-postgraph confirmation candidates are required")
    if any(
        job.readout_type != "shared"
        or job.jepa_regression_aggregation != "target"
        or job.recurrent_padding_mode != "legacy"
        or job.task_regression_loss != "mse"
        or job.task_smooth_l1_beta != 1.0
        or job.postgraph_sequence_mode != "shared-bilstm"
        or job.jepa_rate_weighting != "uniform"
        or job.graph_message_calibration != "none"
        for job in candidates
    ):
        raise ValueError(
            "confirmation controls require shared-postgraph-only candidates"
        )
    return [
        ReadoutJob(
            seed=job.seed,
            gpu=job.gpu,
            readout_type="shared",
            rank=job.rank,
            output_dir=Path(control_root) / "seed_{}".format(job.seed),
            jepa_regression_aggregation="target",
            recurrent_padding_mode="legacy",
            task_regression_loss="mse",
            task_smooth_l1_beta=1.0,
            postgraph_sequence_mode="independent",
            jepa_rate_weighting="uniform",
            graph_message_calibration="none",
        )
        for job in candidates
    ]


def _legacy_control_job(
    reference: ReadoutJob,
    output_dir: Path,
) -> ReadoutJob:
    return ReadoutJob(
        seed=reference.seed,
        gpu=reference.gpu,
        readout_type="shared",
        rank=reference.rank,
        output_dir=Path(output_dir),
        jepa_regression_aggregation="target",
        recurrent_padding_mode="legacy",
        task_regression_loss="mse",
        task_smooth_l1_beta=1.0,
        postgraph_sequence_mode="independent",
        jepa_rate_weighting="uniform",
        graph_message_calibration="none",
    )


def build_command(job: ReadoutJob, python_executable: Path) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "gcnet_missing_m3.train_gcnet",
        "--dataset",
        "CMUMOSI",
        "--feature-root",
        str(FEATURE_ROOT),
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
        "all",
        "--hidden",
        "100",
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
        "1e-05",
        "--jepa-weight",
        "0.1",
        "--jepa-regression-aggregation",
        job.jepa_regression_aggregation,
        "--windowp",
        "1",
        "--windowf",
        "1",
        "--readout-type",
        job.readout_type,
        "--readout-rank",
        str(job.rank),
        "--recurrent-padding-mode",
        job.recurrent_padding_mode,
        "--task-regression-loss",
        job.task_regression_loss,
        "--task-smooth-l1-beta",
        str(job.task_smooth_l1_beta),
        "--postgraph-sequence-mode",
        job.postgraph_sequence_mode,
        "--jepa-rate-weighting",
        job.jepa_rate_weighting,
        "--graph-message-calibration",
        job.graph_message_calibration,
        "--skip-test-evaluation",
        "--num-threads",
        "2",
    ]


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _control_runner_sha256(
    control_root: Path,
    repo_root: Path,
    *,
    fresh_controls: bool,
) -> str:
    if fresh_controls:
        return _sha256(Path(__file__).resolve())
    packed_manifest = Path(control_root).parent / "packed-recurrent" / "MANIFEST.json"
    if packed_manifest.is_file():
        payload = json.loads(packed_manifest.read_text(encoding="utf-8"))
        source_hashes = payload.get("source_sha256", {})
        recorded = source_hashes.get("scripts/run_mosi_conditioned_readout.py")
        if recorded:
            return str(recorded)
    return _sha256(repo_root / "scripts" / "run_mosi_hidden_window_sweep.py")


def _manifest_without_artifact_hashes(
    payload: Mapping[str, object],
) -> dict[str, object]:
    control = payload.get("control")
    if not isinstance(control, Mapping):
        raise ValueError("experiment manifest control must be a mapping")
    return {
        **payload,
        "control": {
            key: value
            for key, value in control.items()
            if key != "artifact_sha256"
        },
    }


def _artifact_hashes_are_monotonic(
    existing: Mapping[str, object],
    updated: Mapping[str, object],
) -> bool:
    existing_control = existing.get("control")
    updated_control = updated.get("control")
    if not isinstance(existing_control, Mapping) or not isinstance(
        updated_control, Mapping
    ):
        return False
    existing_hashes = existing_control.get("artifact_sha256", {})
    updated_hashes = updated_control.get("artifact_sha256", {})
    if not isinstance(existing_hashes, Mapping) or not isinstance(
        updated_hashes, Mapping
    ):
        return False
    for seed, recorded_files in existing_hashes.items():
        current_files = updated_hashes.get(seed)
        if not isinstance(recorded_files, Mapping) or not isinstance(
            current_files, Mapping
        ):
            return False
        for name, recorded_sha in recorded_files.items():
            if current_files.get(name) != recorded_sha:
                return False
    return True


def write_manifest(
    path: Path,
    jobs: Sequence[ReadoutJob],
    *,
    control_root: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fresh_controls = [
        job
        for job in jobs
        if job.recurrent_padding_mode == "legacy"
        and job.readout_type == "shared"
        and job.jepa_regression_aggregation == "target"
        and job.task_regression_loss == "mse"
        and job.postgraph_sequence_mode == "independent"
        and job.jepa_rate_weighting == "uniform"
        and job.graph_message_calibration == "none"
    ]
    inherited_runs = {
        job.seed: _control_run(control_root, job.seed) for job in jobs
    }
    reference_jobs = {}
    for job in jobs:
        reference_jobs.setdefault(job.seed, job)
    direct_deterministic_control = bool(jobs) and all(
        run.parent == Path(control_root) for run in inherited_runs.values()
    )
    source_paths = (
        repo_root / "gcnet_modality_jepa" / "graph.py",
        repo_root / "gcnet_modality_jepa" / "model.py",
        repo_root / "gcnet_missing_m3" / "model.py",
        repo_root / "gcnet_missing_m3" / "loss.py",
        repo_root / "gcnet_missing_m3" / "train_gcnet.py",
        Path(__file__).resolve(),
    )
    payload = {
            "condition": {
                "dataset": "CMUMOSI",
                "features": [
                    "wav2vec-large-c-UTT",
                    "deberta-large-4-UTT",
                    "manet_UTT",
                ],
                "feature_root": str(FEATURE_ROOT),
                "hidden": 100,
                "window_past": 1,
                "window_future": 1,
                "time_attention": False,
                "train_rate_mode": "all",
                "learning_rate": 0.0005,
                "weight_decay": 0.00001,
                "jepa_weight": 0.1,
                "jepa_regression_aggregation": (
                    jobs[0].jepa_regression_aggregation
                ),
                "recurrent_padding_mode": jobs[0].recurrent_padding_mode,
                "task_regression_loss": jobs[0].task_regression_loss,
                "task_smooth_l1_beta": jobs[0].task_smooth_l1_beta,
                "postgraph_sequence_mode": jobs[0].postgraph_sequence_mode,
                "jepa_rate_weighting": jobs[0].jepa_rate_weighting,
                "graph_message_calibration": jobs[0].graph_message_calibration,
                "task_loss_reduction": "mean_valid_utterances",
                "zero_label_training_policy": "include",
                "binary_metric_zero_label_policy": "exclude",
                "python_hash_seed": "0",
                "relation_mapping": {
                    "temporal": {"past": 0, "now": 1, "future": 2},
                    "speaker": {"00": 0},
                },
                "selection": "validation_8rate_mean_weighted_f1",
                "test_policy": "not-computed-before-gate",
            },
            "control": {
                "policy": (
                    "fresh-deterministic-paired"
                    if fresh_controls
                    else "inherit-no-retrain"
                ),
                "root": str(
                    fresh_controls[0].output_dir.parent
                    if fresh_controls
                    else control_root
                ),
                "configuration": (
                    "deterministic-legacy"
                    if fresh_controls or direct_deterministic_control
                    else "hidden_100_window_1"
                ),
                "runner_sha256": _control_runner_sha256(
                    control_root,
                    repo_root,
                    fresh_controls=bool(fresh_controls),
                ),
                "artifact_sha256": {
                    str(seed): (
                        {
                            name: _sha256(run / name)
                            for name in ("config.json", "history.json")
                        }
                        if completed_job_is_compatible(
                            _legacy_control_job(reference_jobs[seed], run)
                        )
                        else {}
                    )
                    for seed, run in inherited_runs.items()
                },
            },
            "source_sha256": {
                str(source.relative_to(repo_root)): _sha256(source)
                for source in source_paths
            },
            "jobs": [
                {
                    "seed": job.seed,
                    "gpu": job.gpu,
                    "readout_type": job.readout_type,
                    "rank": job.rank,
                    "jepa_regression_aggregation": (
                        job.jepa_regression_aggregation
                    ),
                    "recurrent_padding_mode": job.recurrent_padding_mode,
                    "task_regression_loss": job.task_regression_loss,
                    "task_smooth_l1_beta": job.task_smooth_l1_beta,
                    "postgraph_sequence_mode": job.postgraph_sequence_mode,
                    "jepa_rate_weighting": job.jepa_rate_weighting,
                    "graph_message_calibration": job.graph_message_calibration,
                    "output_dir": str(job.output_dir),
                }
                for job in jobs
            ],
        }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        same_contract = _manifest_without_artifact_hashes(
            existing
        ) == _manifest_without_artifact_hashes(payload)
        if not same_contract or not _artifact_hashes_are_monotonic(
            existing, payload
        ):
            raise ValueError("experiment manifest is immutable once written")
        if existing != payload:
            _atomic_json(path, payload)
        return
    _atomic_json(path, payload)


def _rate_record(validation: Mapping[str, object], rate: float) -> Mapping[str, object]:
    for key in (format(rate, ".1f"), rate):
        if key in validation:
            value = validation[key]
            if not isinstance(value, Mapping):
                raise ValueError("validation rate record must be a mapping")
            return value
    raise ValueError("validation history is missing rate {}".format(rate))


def validation_snapshot(run_dir: Path) -> dict[str, object]:
    history = json.loads((Path(run_dir) / "history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError("history must contain at least one epoch")
    best = None
    best_score = -math.inf
    for record in history:
        score = float(record["validation_mean_weighted_f1"])
        if score > best_score:
            best = record
            best_score = score
    if best is None:
        raise ValueError("history has no selectable epoch")
    validation = best["validation"]
    per_rate = {
        rate: float(_rate_record(validation, rate)["weighted_f1"])
        for rate in VALID_RATES
    }
    collapsed = False
    for rate in VALID_RATES:
        record = _rate_record(validation, rate)
        score = per_rate[rate]
        if not math.isfinite(score) or score <= 0.55:
            collapsed = True
        if "prediction_std" in record:
            prediction_std = float(record["prediction_std"])
            if not math.isfinite(prediction_std) or prediction_std <= 1e-5:
                collapsed = True
        if "predicted_sign_count" in record:
            if int(record["predicted_sign_count"]) < 2:
                collapsed = True
    return {
        "best_epoch": int(best["epoch"]),
        "overall": sum(per_rate.values()) / len(VALID_RATES),
        "miss0": per_rate[0.0],
        "high_missing": sum(per_rate[rate] for rate in HIGH_MISSING_RATES)
        / len(HIGH_MISSING_RATES),
        "per_rate": {format(rate, ".1f"): per_rate[rate] for rate in VALID_RATES},
        "collapsed": collapsed,
    }


def paired_validation_gate(
    candidate: Mapping[int, Mapping[str, object]],
    control: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    seeds = sorted(candidate)
    if not seeds or seeds != sorted(control):
        raise ValueError("candidate and control must contain the same seeds")
    overall_by_seed = {
        seed: float(candidate[seed]["overall"]) - float(control[seed]["overall"])
        for seed in seeds
    }
    overall_delta = sum(overall_by_seed.values()) / len(seeds)
    high_delta = sum(
        float(candidate[seed]["high_missing"])
        - float(control[seed]["high_missing"])
        for seed in seeds
    ) / len(seeds)
    miss0_delta = sum(
        float(candidate[seed]["miss0"]) - float(control[seed]["miss0"])
        for seed in seeds
    ) / len(seeds)
    positive = sum(delta > 0 for delta in overall_by_seed.values())
    required_positive = 2 if len(seeds) == 3 else math.ceil(0.8 * len(seeds))
    collapsed = any(bool(candidate[seed]["collapsed"]) for seed in seeds)
    passed = (
        overall_delta >= 0.004
        and positive >= required_positive
        and high_delta >= 0
        and miss0_delta >= -0.003
        and min(overall_by_seed.values()) >= -0.01
        and not collapsed
    )
    return {
        "selection_source": "validation-only",
        "seeds": seeds,
        "overall_delta": overall_delta,
        "overall_delta_by_seed": overall_by_seed,
        "positive_seeds": positive,
        "required_positive_seeds": required_positive,
        "high_missing_delta": high_delta,
        "miss0_delta": miss0_delta,
        "collapsed": collapsed,
        "passed": passed,
    }


def paired_confirmation_gate(
    candidate: Mapping[int, Mapping[str, object]],
    control: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the predeclared five-seed confirmation gate."""
    if len(candidate) != 5:
        raise ValueError("five-seed confirmation requires exactly five seeds")
    summary = paired_validation_gate(candidate, control)
    summary["stage"] = "five-seed-confirmation"
    summary["required_positive_seeds"] = 4
    summary["passed"] = (
        float(summary["overall_delta"]) > 0
        and int(summary["positive_seeds"]) >= 4
        and float(summary["high_missing_delta"]) >= 0
        and float(summary["miss0_delta"]) >= -0.003
        and not bool(summary["collapsed"])
    )
    return summary


def _control_run(control_root: Path, seed: int) -> Path:
    direct = Path(control_root) / "seed_{}".format(seed)
    if (direct / "config.json").is_file():
        return direct
    return direct / "hidden_100_window_1"


def audit_inherited_control(
    candidate_run: Path,
    control_run: Path,
) -> dict[str, object]:
    defaults = {
        "readout_type": "shared",
        "readout_rank": 8,
        "jepa_regression_aggregation": "target",
        "recurrent_padding_mode": "legacy",
        "task_regression_loss": "mse",
        "task_smooth_l1_beta": 1.0,
        "postgraph_sequence_mode": "independent",
        "jepa_rate_weighting": "uniform",
        "graph_message_calibration": "none",
        "evaluate_test": True,
    }
    candidate = {
        **defaults,
        **json.loads(
            (Path(candidate_run) / "config.json").read_text(encoding="utf-8")
        ),
    }
    control = {
        **defaults,
        **json.loads(
            (Path(control_run) / "config.json").read_text(encoding="utf-8")
        ),
    }
    ignored = {
        "readout_type",
        "readout_rank",
        "jepa_regression_aggregation",
        "recurrent_padding_mode",
        "task_regression_loss",
        "postgraph_sequence_mode",
        "jepa_rate_weighting",
        "graph_message_calibration",
        "evaluate_test",
    }
    candidate_common = {
        key: value for key, value in candidate.items() if key not in ignored
    }
    control_common = {
        key: value for key, value in control.items() if key not in ignored
    }
    all_keys = sorted(set(candidate_common) | set(control_common))
    mismatches = [
        key
        for key in all_keys
        if candidate_common.get(key) != control_common.get(key)
    ]
    if mismatches:
        raise ValueError(
            "inherited control mismatch: {}".format(", ".join(mismatches))
        )
    candidate_type = candidate.get("readout_type", "shared")
    control_type = control.get("readout_type", "shared")
    candidate_aggregation = candidate.get(
        "jepa_regression_aggregation", "target"
    )
    control_aggregation = control.get("jepa_regression_aggregation", "target")
    candidate_padding = candidate.get("recurrent_padding_mode", "legacy")
    control_padding = control.get("recurrent_padding_mode", "legacy")
    candidate_task_loss = candidate.get("task_regression_loss", "mse")
    control_task_loss = control.get("task_regression_loss", "mse")
    candidate_postgraph = candidate.get(
        "postgraph_sequence_mode", "independent"
    )
    control_postgraph = control.get(
        "postgraph_sequence_mode", "independent"
    )
    candidate_rate_weighting = candidate.get(
        "jepa_rate_weighting", "uniform"
    )
    control_rate_weighting = control.get(
        "jepa_rate_weighting", "uniform"
    )
    candidate_message_calibration = candidate.get(
        "graph_message_calibration", "none"
    )
    control_message_calibration = control.get(
        "graph_message_calibration", "none"
    )
    readout_treatments = {
        "availability-low-rank",
        "shared-low-rank-parammatch",
        "availability-affine",
    }
    if candidate_type != "shared" and candidate_type not in readout_treatments:
        raise ValueError("candidate readout_type is not a registered treatment")
    candidate_changes = sum(
        (
            candidate_type != "shared",
            candidate_aggregation == "utterance",
            candidate_padding == "packed",
            candidate_task_loss == "smooth-l1",
            candidate_postgraph == "shared-bilstm",
            candidate_rate_weighting == "sparsity-budget",
            candidate_message_calibration == "branch-layernorm-residual",
        )
    )
    if candidate_changes != 1:
        raise ValueError("candidate must change exactly one registered treatment")
    if int(candidate.get("readout_rank", 8)) != 8:
        raise ValueError("candidate readout_rank must equal 8")
    if control_type != "shared":
        raise ValueError("inherited control readout_type must be shared")
    if control_aggregation != "target":
        raise ValueError("inherited control must use target aggregation")
    if control_padding != "legacy":
        raise ValueError("control recurrent padding mode must be legacy")
    if control_task_loss != "mse":
        raise ValueError("inherited control task regression loss must be MSE")
    if control_postgraph != "independent":
        raise ValueError(
            "inherited control postgraph sequence mode must be independent"
        )
    if control_rate_weighting != "uniform":
        raise ValueError("inherited control JEPA rate weighting must be uniform")
    if control_message_calibration != "none":
        raise ValueError(
            "inherited control graph message calibration must be none"
        )
    if candidate_task_loss not in {"mse", "smooth-l1"}:
        raise ValueError("candidate task regression loss is not registered")
    if candidate_postgraph not in {"independent", "shared-bilstm"}:
        raise ValueError("candidate postgraph sequence mode is not registered")
    if candidate_rate_weighting not in {"uniform", "sparsity-budget"}:
        raise ValueError("candidate JEPA rate weighting is not registered")
    if candidate_message_calibration not in {
        "none",
        "branch-layernorm-residual",
    }:
        raise ValueError("candidate graph message calibration is not registered")
    beta = float(candidate.get("task_smooth_l1_beta", 1.0))
    if beta != 1.0 or float(control.get("task_smooth_l1_beta", 1.0)) != 1.0:
        raise ValueError("task SmoothL1 beta must equal the locked value 1.0")
    return {
        "compatible": True,
        "matching_field_count": len(all_keys),
        "candidate_readout_type": candidate_type,
        "candidate_readout_rank": int(candidate.get("readout_rank", 8)),
        "candidate_jepa_regression_aggregation": candidate_aggregation,
        "candidate_recurrent_padding_mode": candidate_padding,
        "candidate_task_regression_loss": candidate_task_loss,
        "candidate_postgraph_sequence_mode": candidate_postgraph,
        "candidate_jepa_rate_weighting": candidate_rate_weighting,
        "candidate_graph_message_calibration": candidate_message_calibration,
        "control_readout_type": control_type,
        "control_readout_rank": int(control.get("readout_rank", 8)),
        "control_jepa_regression_aggregation": control_aggregation,
        "control_recurrent_padding_mode": control_padding,
        "control_task_regression_loss": control_task_loss,
        "control_postgraph_sequence_mode": control_postgraph,
        "control_jepa_rate_weighting": control_rate_weighting,
        "control_graph_message_calibration": control_message_calibration,
        "task_smooth_l1_beta": beta,
    }


def write_validation_summary(
    path: Path,
    jobs: Sequence[ReadoutJob],
    *,
    control_root: Path,
    control_jobs: Sequence[ReadoutJob] | None = None,
) -> dict[str, object]:
    if control_jobs is None:
        control_runs = {
            job.seed: _control_run(control_root, job.seed) for job in jobs
        }
        control_specs = {
            job.seed: _legacy_control_job(job, control_runs[job.seed])
            for job in jobs
        }
    else:
        control_runs = {job.seed: job.output_dir for job in control_jobs}
        control_specs = {job.seed: job for job in control_jobs}
        if sorted(control_runs) != sorted(job.seed for job in jobs):
            raise ValueError("fresh controls must match candidate seeds")
    for job in jobs:
        _require_completed_job(
            job,
            "validation summary requires completed candidate seed {}".format(
                job.seed
            ),
        )
        _require_completed_job(
            control_specs[job.seed],
            "validation summary requires completed control seed {}".format(
                job.seed
            ),
        )
    control_audit = {
        job.seed: audit_inherited_control(
            job.output_dir,
            control_runs[job.seed],
        )
        for job in jobs
    }
    candidate = {job.seed: validation_snapshot(job.output_dir) for job in jobs}
    control = {
        job.seed: validation_snapshot(control_runs[job.seed])
        for job in jobs
    }
    summary = {
        "control_audit": control_audit,
        "candidate": candidate,
        "control": control,
        "gate": paired_validation_gate(candidate, control),
    }
    _atomic_json(path, summary)
    return summary


def write_confirmation_summary(
    path: Path,
    jobs: Sequence[ReadoutJob],
    *,
    control_root: Path,
) -> dict[str, object]:
    if sorted(job.seed for job in jobs) != [66, 67, 68, 69, 70]:
        raise ValueError("confirmation summary requires seeds 66 through 70")
    control_runs = {
        job.seed: _control_run(control_root, job.seed) for job in jobs
    }
    for job in jobs:
        _require_completed_job(
            job,
            "confirmation summary requires completed candidate seed {}".format(
                job.seed
            ),
        )
        _require_completed_job(
            _legacy_control_job(job, control_runs[job.seed]),
            "confirmation summary requires completed control seed {}".format(
                job.seed
            ),
        )
    control_audit = {
        job.seed: audit_inherited_control(
            job.output_dir,
            control_runs[job.seed],
        )
        for job in jobs
    }
    candidate = {job.seed: validation_snapshot(job.output_dir) for job in jobs}
    control = {
        job.seed: validation_snapshot(control_runs[job.seed])
        for job in jobs
    }
    artifact_sha256 = {
        str(job.seed): {
            "candidate": {
                name: _sha256(job.output_dir / name)
                for name in ("config.json", "history.json")
            },
            "control": {
                name: _sha256(control_runs[job.seed] / name)
                for name in ("config.json", "history.json")
            },
        }
        for job in jobs
    }
    summary = {
        "control_audit": control_audit,
        "candidate": candidate,
        "control": control,
        "artifact_sha256": artifact_sha256,
        "gate": paired_confirmation_gate(candidate, control),
    }
    _atomic_json(path, summary)
    return summary


def validate_confirmation_prerequisites(
    variant_root: Path,
    control_root: Path,
) -> None:
    screen_summary = Path(variant_root) / "VALIDATION_SUMMARY.json"
    if not screen_summary.is_file():
        raise ValueError("confirmation requires the completed screen summary")
    screen = json.loads(screen_summary.read_text(encoding="utf-8"))
    if not bool(screen.get("gate", {}).get("passed")):
        raise ValueError("confirmation requires a passed three-seed screen")
    for seed in (66, 67, 68):
        candidate = ReadoutJob(
            seed=seed,
            gpu=7,
            readout_type="shared",
            rank=8,
            output_dir=Path(variant_root) / "seed_{}".format(seed),
            postgraph_sequence_mode="shared-bilstm",
        )
        control = _legacy_control_job(
            candidate,
            Path(control_root) / "seed_{}".format(seed),
        )
        _require_completed_job(
            candidate,
            "confirmation requires completed candidate seed {}".format(seed),
        )
        _require_completed_job(
            control,
            "confirmation requires completed control seed {}".format(seed),
        )


def validate_inherited_control_layout(
    jobs: Sequence[ReadoutJob],
    control_root: Path,
) -> None:
    requires_repaired_control = any(
        job.task_regression_loss == "smooth-l1"
        or job.postgraph_sequence_mode == "shared-bilstm"
        or job.jepa_rate_weighting == "sparsity-budget"
        or job.graph_message_calibration == "branch-layernorm-residual"
        for job in jobs
    )
    if not requires_repaired_control:
        return
    root = Path(control_root)
    for job in jobs:
        direct = root / "seed_{}".format(job.seed)
        try:
            complete = completed_job_is_compatible(
                _legacy_control_job(job, direct)
            )
        except (ValueError, json.JSONDecodeError, OSError):
            complete = False
        if not complete or _control_run(root, job.seed) != direct:
            raise ValueError(
                "new candidates require a completed direct deterministic "
                "Legacy control root with compatible config, metrics, and "
                "100-epoch history artifacts"
            )


def _waves(
    jobs: Sequence[ReadoutJob], max_concurrent_per_gpu: int
) -> list[list[ReadoutJob]]:
    if max_concurrent_per_gpu < 1:
        raise ValueError("max_concurrent_per_gpu must be positive")
    grouped: dict[int, list[ReadoutJob]] = {}
    for job in jobs:
        grouped.setdefault(job.gpu, []).append(job)
    count = max(
        (
            math.ceil(len(gpu_jobs) / max_concurrent_per_gpu)
            for gpu_jobs in grouped.values()
        ),
        default=0,
    )
    waves = []
    for index in range(count):
        start = index * max_concurrent_per_gpu
        stop = start + max_concurrent_per_gpu
        waves.append(
            [
                job
                for gpu_jobs in grouped.values()
                for job in gpu_jobs[start:stop]
            ]
        )
    return waves


def completed_job_is_compatible(job: ReadoutJob) -> bool:
    metrics_path = job.output_dir / "metrics.json"
    if not metrics_path.is_file():
        return False
    config_path = job.output_dir / "config.json"
    if not config_path.is_file():
        raise ValueError("completed job is missing config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_config = {
        "dataset": "CMUMOSI",
        "fold": 1,
        "seed": job.seed,
        "window_past": 1,
        "window_future": 1,
        "hidden": 100,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.0005,
        "weight_decay": 0.00001,
        "fusion_type": "slot",
        "representation_type": "slot",
        "train_rate_mode": "all",
        "mosi_task_mode": "regression",
        "graph_branch_mode": "both",
        "jepa_weight": 0.1,
        "jepa_regression_aggregation": job.jepa_regression_aggregation,
        "time_attention": False,
        "readout_type": job.readout_type,
        "readout_rank": job.rank,
        "recurrent_padding_mode": job.recurrent_padding_mode,
        "task_regression_loss": job.task_regression_loss,
        "task_smooth_l1_beta": job.task_smooth_l1_beta,
        "postgraph_sequence_mode": job.postgraph_sequence_mode,
        "jepa_rate_weighting": job.jepa_rate_weighting,
        "graph_message_calibration": job.graph_message_calibration,
        "evaluate_test": False,
    }
    for key, expected in expected_config.items():
        defaults = {
            "jepa_regression_aggregation": "target",
            "recurrent_padding_mode": "legacy",
            "task_regression_loss": "mse",
            "task_smooth_l1_beta": 1.0,
            "postgraph_sequence_mode": "independent",
            "jepa_rate_weighting": "uniform",
            "graph_message_calibration": "none",
        }
        default = defaults.get(key)
        if config.get(key, default) != expected:
            raise ValueError("completed job config mismatch: {}".format(key))
    history_path = job.output_dir / "history.json"
    if not history_path.is_file():
        raise ValueError("completed job is missing history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    expected_epochs = int(expected_config["epochs"])
    if (
        not isinstance(history, list)
        or len(history) != expected_epochs
        or any(
            not isinstance(record, Mapping)
            or int(record.get("epoch", -1)) != epoch
            for epoch, record in enumerate(history, start=1)
        )
    ):
        raise ValueError(
            "completed job history must contain epochs 1 through {}".format(
                expected_epochs
            )
        )
    expected_metrics = {
        "evaluation_stage": "train-validation-only",
        "readout_type": job.readout_type,
        "readout_rank": job.rank,
        "jepa_regression_aggregation": job.jepa_regression_aggregation,
        "recurrent_padding_mode": job.recurrent_padding_mode,
        "task_regression_loss": job.task_regression_loss,
        "task_smooth_l1_beta": job.task_smooth_l1_beta,
        "postgraph_sequence_mode": job.postgraph_sequence_mode,
        "jepa_rate_weighting": job.jepa_rate_weighting,
        "graph_message_calibration": job.graph_message_calibration,
    }
    for key, expected in expected_metrics.items():
        defaults = {
            "jepa_regression_aggregation": "target",
            "recurrent_padding_mode": "legacy",
            "task_regression_loss": "mse",
            "task_smooth_l1_beta": 1.0,
            "postgraph_sequence_mode": "independent",
            "jepa_rate_weighting": "uniform",
            "graph_message_calibration": "none",
        }
        default = defaults.get(key)
        if metrics.get(key, default) != expected:
            raise ValueError("completed job metrics mismatch: {}".format(key))
    return True


def _require_completed_job(job: ReadoutJob, message: str) -> None:
    try:
        complete = completed_job_is_compatible(job)
    except (ValueError, json.JSONDecodeError, OSError) as error:
        raise ValueError("{}: {}".format(message, error)) from error
    if not complete:
        raise ValueError(message)


def run_jobs(
    jobs: Sequence[ReadoutJob],
    *,
    python_executable: Path,
    repo_root: Path,
    max_concurrent_per_gpu: int,
) -> int:
    pending = [job for job in jobs if not completed_job_is_compatible(job)]
    failures = 0
    for wave_index, wave in enumerate(_waves(pending, max_concurrent_per_gpu)):
        processes = []
        for job in wave:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(job, python_executable)
            log = (job.output_dir / "train.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(job.gpu),
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "PYTHONHASHSEED": "0",
                }
            )
            started = time.time()
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            _atomic_json(
                job.output_dir / "status.json",
                {
                    "state": "running",
                    "pid": process.pid,
                    "gpu": job.gpu,
                    "wave": wave_index,
                    "started_at_unix": started,
                    "command": command,
                },
            )
            processes.append((job, process, log, started))
        for job, process, log, started in processes:
            returncode = process.wait()
            log.close()
            complete = (job.output_dir / "metrics.json").is_file()
            if not complete:
                failures += 1
            _atomic_json(
                job.output_dir / "status.json",
                {
                    "state": "complete" if complete else "failed",
                    "pid": process.pid,
                    "returncode": returncode,
                    "gpu": job.gpu,
                    "wave": wave_index,
                    "started_at_unix": started,
                    "finished_at_unix": time.time(),
                    "metrics_complete": complete,
                },
            )
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("screen", "validation-audit", "confirm"),
        default="screen",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(66, 67, 68))
    parser.add_argument("--gpus", type=int, nargs="+", default=(7, 7, 7))
    parser.add_argument(
        "--readout-type",
        choices=(
            "shared",
            "availability-low-rank",
            "shared-low-rank-parammatch",
            "availability-affine",
        ),
        default="availability-low-rank",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--jepa-regression-aggregation",
        choices=("target", "utterance"),
        default="target",
    )
    parser.add_argument(
        "--recurrent-padding-mode",
        choices=("legacy", "packed"),
        default="legacy",
    )
    parser.add_argument(
        "--task-regression-loss",
        choices=("mse", "smooth-l1"),
        default="mse",
    )
    parser.add_argument("--task-smooth-l1-beta", type=float, default=1.0)
    parser.add_argument(
        "--postgraph-sequence-mode",
        choices=("independent", "shared-bilstm"),
        default="independent",
    )
    parser.add_argument(
        "--jepa-rate-weighting",
        choices=("uniform", "sparsity-budget"),
        default="uniform",
    )
    parser.add_argument(
        "--graph-message-calibration",
        choices=("none", "branch-layernorm-residual"),
        default="none",
    )
    parser.add_argument("--fresh-legacy-control", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path("/data2/yb/reproduction_envs/gcnet-official/bin/python"),
    )
    parser.add_argument("--max-concurrent-per-gpu", type=int, default=3)
    return parser


def validate_run_arguments(args: argparse.Namespace) -> None:
    packed = args.recurrent_padding_mode == "packed"
    exact_packed_candidate = (
        packed
        and args.readout_type == "shared"
        and args.jepa_regression_aggregation == "target"
        and args.task_regression_loss == "mse"
        and args.task_smooth_l1_beta == 1.0
        and args.postgraph_sequence_mode == "independent"
        and args.jepa_rate_weighting == "uniform"
        and args.graph_message_calibration == "none"
    )
    exact_confirmation_candidate = (
        args.readout_type == "shared"
        and args.jepa_regression_aggregation == "target"
        and args.recurrent_padding_mode == "legacy"
        and args.task_regression_loss == "mse"
        and args.task_smooth_l1_beta == 1.0
        and args.postgraph_sequence_mode == "shared-bilstm"
        and args.jepa_rate_weighting == "uniform"
        and args.graph_message_calibration == "none"
    )
    if args.stage == "confirm":
        if tuple(args.seeds) != (69, 70):
            raise ValueError("confirmation is locked to seeds 69 and 70")
        if not exact_confirmation_candidate:
            raise ValueError(
                "confirmation is valid for the shared-postgraph-only candidate"
            )
        if args.fresh_legacy_control:
            raise ValueError(
                "confirmation creates only its missing controls automatically"
            )
        return
    if packed and not args.fresh_legacy_control:
        raise ValueError(
            "packed recurrent requires fresh deterministic legacy controls"
        )
    if args.fresh_legacy_control and not exact_packed_candidate:
        raise ValueError(
            "fresh legacy control is only valid for the packed-only candidate"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_run_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    repo_root = Path(__file__).resolve().parents[1]
    jobs = build_jobs(
        seeds=args.seeds,
        gpus=args.gpus,
        output_root=args.output_root,
        readout_type=args.readout_type,
        rank=args.rank,
        jepa_regression_aggregation=args.jepa_regression_aggregation,
        recurrent_padding_mode=args.recurrent_padding_mode,
        task_regression_loss=args.task_regression_loss,
        task_smooth_l1_beta=args.task_smooth_l1_beta,
        postgraph_sequence_mode=args.postgraph_sequence_mode,
        jepa_rate_weighting=args.jepa_rate_weighting,
        graph_message_calibration=args.graph_message_calibration,
    )
    control_jobs = None
    run_list = jobs
    variant_root = jobs[0].output_dir.parent
    if args.stage == "confirm":
        validate_confirmation_prerequisites(variant_root, args.control_root)
        control_jobs = build_confirmation_legacy_controls(
            jobs, args.control_root
        )
        run_list = [*jobs, *control_jobs]
    elif args.fresh_legacy_control:
        control_jobs = build_fresh_legacy_controls(jobs, args.output_root)
        run_list = [*jobs, *control_jobs]
    else:
        validate_inherited_control_layout(jobs, args.control_root)
    manifest = variant_root / (
        "CONFIRMATION_MANIFEST.json"
        if args.stage == "confirm"
        else "MANIFEST.json"
    )
    write_manifest(manifest, run_list, control_root=args.control_root)
    if args.stage in {"screen", "confirm"}:
        failures = run_jobs(
            run_list,
            python_executable=args.python_executable,
            repo_root=repo_root,
            max_concurrent_per_gpu=args.max_concurrent_per_gpu,
        )
        # A fresh-control manifest is written before launch so the experiment
        # contract exists even if a process is interrupted. Enrich it after
        # the run with newly available control hashes; existing hashes remain
        # immutable and are verified by write_manifest.
        write_manifest(manifest, run_list, control_root=args.control_root)
        if failures:
            return 1
    if args.stage == "confirm":
        all_jobs = build_jobs(
            seeds=(66, 67, 68, 69, 70),
            gpus=(7, 7, 7, 7, 7),
            output_root=args.output_root,
            readout_type=args.readout_type,
            rank=args.rank,
            jepa_regression_aggregation=args.jepa_regression_aggregation,
            recurrent_padding_mode=args.recurrent_padding_mode,
            task_regression_loss=args.task_regression_loss,
            task_smooth_l1_beta=args.task_smooth_l1_beta,
            postgraph_sequence_mode=args.postgraph_sequence_mode,
            jepa_rate_weighting=args.jepa_rate_weighting,
            graph_message_calibration=args.graph_message_calibration,
        )
        validate_inherited_control_layout(all_jobs, args.control_root)
        write_confirmation_summary(
            variant_root / "FIVE_SEED_VALIDATION_SUMMARY.json",
            all_jobs,
            control_root=args.control_root,
        )
        return 0
    write_validation_summary(
        variant_root / "VALIDATION_SUMMARY.json",
        jobs,
        control_root=args.control_root,
        control_jobs=control_jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
