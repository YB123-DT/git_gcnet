#!/usr/bin/env python3
"""Run the frozen CMU-MOSI missing-latent oracle diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

import config
import gcnet_modality_jepa.model as gcnet_graph_model
from gcnet_modality_jepa.protocol import SeedBundle
from gcnet_modality_jepa.train_gcnet import get_loaders, set_random_seed
from gcnet_missing_m3 import oracle_diagnostic as oracle
from gcnet_missing_m3.mixed_rate import MISSING_RATES
from gcnet_missing_m3.model import MODALITIES, MissingM3GraphModel
from gcnet_missing_m3.train_gcnet import (
    TrainConfig,
    _move_batch,
    _prepare_view,
    _resolve_task_contract,
    _schedules,
)


FEATURE_NAMES = {
    "audio": "wav2vec-large-c-UTT",
    "text": "deberta-large-4-UTT",
    "visual": "manet_UTT",
}
LOCKED_CONFIG = {
    "dataset": "CMUMOSI",
    "fold": 1,
    "fusion_type": "slot",
    "local_context_residual": False,
    "mosi_task_mode": "regression",
    "evaluation_protocol": "official",
    "mmoe_variant": "dual-gate",
    "classification_completion": True,
}
METRIC_NAMES = (
    "weighted_f1",
    "macro_f1",
    "accuracy",
    "mae",
    "correlation",
)
TEMPORAL_RELATIONS = ("past", "now", "future")


@contextmanager
def temporal_relation_order(order: Sequence[str]):
    """Temporarily recover a checkpoint's temporal relation-row semantics."""
    requested = tuple(order)
    if len(requested) != len(TEMPORAL_RELATIONS) or set(requested) != set(
        TEMPORAL_RELATIONS
    ):
        raise ValueError("order must be a permutation of past, now, future")
    desired = {name: index for index, name in enumerate(requested)}
    original = gcnet_graph_model.batch_graphify

    def remapped_batch_graphify(*args: Any, **kwargs: Any):
        node_features, edge_index, edge_type, mapping = original(*args, **kwargs)
        graph_type = kwargs.get("graph_type", args[6] if len(args) > 6 else None)
        if graph_type != "temporal":
            return node_features, edge_index, edge_type, mapping
        if set(mapping) != set(TEMPORAL_RELATIONS):
            raise ValueError("unexpected temporal relation mapping")
        original_edge_type = edge_type.clone()
        remapped = edge_type.clone()
        for relation, old_index in mapping.items():
            remapped[original_edge_type == int(old_index)] = desired[relation]
        return node_features, edge_index, remapped, dict(desired)

    gcnet_graph_model.batch_graphify = remapped_batch_graphify
    try:
        yield dict(desired)
    finally:
        gcnet_graph_model.batch_graphify = original


def validate_run_config(config_value: TrainConfig) -> None:
    """Reject checkpoints outside the pre-registered diagnostic contract."""
    for field, expected in LOCKED_CONFIG.items():
        actual = getattr(config_value, field)
        if actual != expected:
            raise ValueError(
                "{} must be {!r}, got {!r}".format(field, expected, actual)
            )


def prepare_output_directory(path: Path) -> None:
    """Create an output directory but never overwrite prior evidence."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("output directory is non-empty: {}".format(path))
    path.mkdir(parents=True, exist_ok=True)


def find_history_record(
    history: Sequence[Mapping[str, Any]], epoch: int, saved_score: float
) -> Mapping[str, Any]:
    """Select the unique checkpoint epoch and verify its saved selection score."""
    matches = [record for record in history if int(record.get("epoch", -1)) == epoch]
    if len(matches) != 1:
        raise ValueError("history must contain exactly one record for checkpoint epoch")
    record = matches[0]
    history_score = float(record["validation_mean_weighted_f1"])
    if not math.isclose(history_score, float(saved_score), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("history saved score does not match checkpoint")
    return record


def parse_sample_keys(sample_keys: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Split keys at their final colon so conversation IDs may contain colons."""
    conversations = []
    utterance_indices = []
    for key in sample_keys:
        if ":" not in key:
            raise ValueError("sample key must end in ':utterance_index'")
        conversation, raw_index = key.rsplit(":", 1)
        if not conversation:
            raise ValueError("sample key must contain a conversation ID")
        try:
            utterance_index = int(raw_index)
        except ValueError:
            raise ValueError("utterance index must be an integer")
        if utterance_index < 0:
            raise ValueError("utterance index must be non-negative")
        conversations.append(conversation)
        utterance_indices.append(utterance_index)
    return tuple(conversations), tuple(utterance_indices)


def sample_order_sha256(sample_keys: Sequence[str]) -> str:
    """Hash ordered, length-framed UTF-8 sample keys."""
    digest = hashlib.sha256()
    for key in sample_keys:
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_tensor_sha256(value: torch.Tensor) -> str:
    """Match the historical raw-byte-only tensor hash."""
    array = value.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def _rate_key(rate: float) -> str:
    return format(float(rate), ".1f")


def _rate_filename(rate: float) -> str:
    return "rate_{}".format(_rate_key(rate).replace(".", "p"))


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _buffer_snapshot_equal(
    module: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> bool:
    current = dict(module.named_buffers())
    return set(current) == set(snapshot) and all(
        current[name].shape == saved.shape
        and current[name].dtype == saved.dtype
        and torch.equal(current[name], saved.to(device=current[name].device))
        for name, saved in snapshot.items()
    )


def _build_model(
    config_value: TrainConfig,
    dimensions: Tuple[int, int, int],
    device: torch.device,
) -> MissingM3GraphModel:
    shape = _resolve_task_contract(
        config_value.dataset, config_value.mosi_task_mode
    )
    adim, tdim, vdim = dimensions
    # This literal is part of the historical checkpoint initialization contract.
    model_seed = SeedBundle(config_value.seed).derive(
        "missing_m3_model_init:fold:5"
    )
    set_random_seed(model_seed)
    return MissingM3GraphModel(
        config_value.base_model,
        adim,
        tdim,
        vdim,
        config_value.hidden,
        config_value.hidden // 2,
        n_speakers=int(shape["num_speakers"]),
        window_past=config_value.window_past,
        window_future=config_value.window_future,
        n_classes=int(shape["num_classes"]),
        dropout=config_value.dropout,
        time_attn=config_value.time_attention,
        no_cuda=device.type != "cuda",
        latent_dim=config_value.latent_dim,
        num_experts=config_value.num_experts,
        top_k=config_value.top_k,
        projector_dropout=config_value.projector_dropout,
        predictor_dropout=config_value.predictor_dropout,
        fusion_type=config_value.fusion_type,
        local_context_residual=config_value.local_context_residual,
        local_fusion_hidden_dim=config_value.local_fusion_hidden_dim,
        local_fusion_dropout=config_value.local_fusion_dropout,
        graph_branch_mode=config_value.graph_branch_mode,
        mmoe_variant=config_value.mmoe_variant,
        classification_completion=config_value.classification_completion,
    ).to(device)


def _build_validation_runtime(
    config_value: TrainConfig,
    feature_root: Path,
    device: torch.device,
) -> Tuple[Iterable[Sequence[object]], Tuple[int, int, int], MissingM3GraphModel]:
    label_path = feature_root.parent / "CMUMOSI_features_raw_2way.pkl"
    if not label_path.is_file():
        raise FileNotFoundError("missing MOSI labels: {}".format(label_path))
    feature_paths = {
        name: feature_root / feature_name
        for name, feature_name in FEATURE_NAMES.items()
    }
    for name, path in feature_paths.items():
        if not path.is_dir():
            raise FileNotFoundError("missing {} features: {}".format(name, path))
    config.PATH_TO_LABEL["CMUMOSI"] = str(label_path)
    set_random_seed(config_value.seed)
    loaders = get_loaders(
        audio_root=str(feature_paths["audio"]),
        text_root=str(feature_paths["text"]),
        video_root=str(feature_paths["visual"]),
        num_folder=1,
        dataset="CMUMOSI",
        batch_size=config_value.batch_size,
        num_workers=0,
        seed=config_value.seed,
        validation_fraction=config_value.validation_fraction,
        evaluation_protocol=config_value.evaluation_protocol,
    )
    _, validation_loaders, _, adim, tdim, vdim = loaders
    dimensions = (int(adim), int(tdim), int(vdim))
    model = _build_model(config_value, dimensions, device)
    return validation_loaders[config_value.fold - 1], dimensions, model


def _predictions(logits: torch.Tensor) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError("MOSI regression logits must have shape [N, 1]")
    return logits[:, 0].detach().to(device="cpu").numpy()


def _metric_delta(
    left: Mapping[str, float], right: Mapping[str, float]
) -> Dict[str, float]:
    return {name: float(left[name]) - float(right[name]) for name in METRIC_NAMES}


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def _collect_rate_state(
    model: MissingM3GraphModel,
    loader: Iterable[Sequence[object]],
    schedule: Any,
    dimensions: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[oracle.OracleState, Dict[str, float], torch.Tensor]:
    states = []
    audits = []
    legacy_availability = []
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)
    for raw in loader:
        data = _move_batch(raw, device)
        view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
        valid_time_major = view["umask"].T.bool()
        legacy_availability.append(
            view["availability"][valid_time_major].detach().to(device="cpu")
        )
        state, audit = oracle.extract_oracle_batch(model, view)
        states.append(state)
        audits.append(audit)
    combined = oracle.concatenate_oracle_states(states)
    audit_summary = {
        "predicted_hidden_max_abs_error": max(
            item["predicted_hidden_max_abs_error"] for item in audits
        ),
        "predicted_logits_max_abs_error": max(
            item["predicted_logits_max_abs_error"] for item in audits
        ),
    }
    return combined, audit_summary, torch.cat(legacy_availability, dim=0)


@torch.no_grad()
def _quick_validation_metrics(
    model: MissingM3GraphModel,
    loader: Iterable[Sequence[object]],
    schedule: Any,
    dimensions: Tuple[int, int, int],
    device: torch.device,
) -> Dict[str, float]:
    """Recompute only native MOSI predictions for relation-order recovery."""
    buffers = oracle.snapshot_named_buffers(model)
    all_labels = []
    all_predictions = []
    try:
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        for raw in loader:
            data = _move_batch(raw, device)
            view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
            logits, _, _, returned_predictions = model(
                [view["incomplete"]],
                view["availability"],
                view["qmask"],
                view["umask"],
                view["lengths"],
                predict_missing=False,
            )
            if returned_predictions is not None:
                raise RuntimeError("native inference unexpectedly returned predictions")
            flat_logits = oracle.flatten_valid_lbd(logits, view["umask"])
            flat_labels = oracle.flatten_valid_lbd(
                view["labels"].transpose(0, 1), view["umask"]
            )
            all_predictions.append(_predictions(flat_logits))
            all_labels.append(flat_labels.detach().to(device="cpu").numpy())
        return oracle.regression_metrics(
            np.concatenate(all_labels), np.concatenate(all_predictions)
        )
    finally:
        oracle.restore_named_buffers(model, buffers)


def _discover_temporal_relation_order(
    model: MissingM3GraphModel,
    loader: Iterable[Sequence[object]],
    schedules: Mapping[float, Any],
    dimensions: Tuple[int, int, int],
    device: torch.device,
    history_record: Mapping[str, Any],
    requested_rates: Sequence[float],
) -> Tuple[Tuple[str, ...], Sequence[Mapping[str, Any]]]:
    """Recover the relation-row order omitted from historical checkpoints."""
    preferred = [0.0, 0.4, 0.7]
    calibration_rates = [rate for rate in preferred if rate in requested_rates]
    if not calibration_rates:
        calibration_rates = [float(requested_rates[0])]
    candidates = []
    for order in itertools.permutations(TEMPORAL_RELATIONS):
        rate_errors = {}
        with temporal_relation_order(order):
            for rate in calibration_rates:
                actual = _quick_validation_metrics(
                    model, loader, schedules[rate], dimensions, device
                )
                expected = history_record["validation"][_rate_key(rate)]
                errors = {
                    name: abs(float(actual[name]) - float(expected[name]))
                    for name in METRIC_NAMES
                }
                rate_errors[_rate_key(rate)] = {
                    "max_abs_error": float(max(errors.values())),
                    "per_metric_abs_error": errors,
                }
        candidates.append(
            {
                "order": list(order),
                "calibration_rates": list(calibration_rates),
                "max_abs_error": float(
                    max(item["max_abs_error"] for item in rate_errors.values())
                ),
                "rate_errors": rate_errors,
            }
        )
    matches = [item for item in candidates if item["max_abs_error"] < 1e-6]
    if len(matches) != 1:
        raise RuntimeError(
            "historical temporal relation order is not uniquely recoverable: {}".format(
                candidates
            )
        )
    return tuple(matches[0]["order"]), candidates


def _path_bundle(
    model: MissingM3GraphModel,
    state: oracle.OracleState,
    seed: int,
    rate: float,
    requested_shuffle_count: int,
) -> Dict[str, Any]:
    labels = state.labels.detach().to(device="cpu").numpy()
    graph_logits, graph_residual = oracle.compute_path_output(
        state.graph_hidden,
        None,
        state.target_mask,
        model.missing_latent_fusion,
        model.smax_fc,
    )
    predicted_logits, predicted_residual = oracle.compute_path_output(
        state.graph_hidden,
        state.predicted_latents,
        state.target_mask,
        model.missing_latent_fusion,
        model.smax_fc,
    )
    teacher_logits, teacher_residual = oracle.compute_path_output(
        state.graph_hidden,
        state.teacher_latents,
        state.target_mask,
        model.missing_latent_fusion,
        model.smax_fc,
    )
    graph_predictions = _predictions(graph_logits)
    predicted_predictions = _predictions(predicted_logits)
    teacher_predictions = _predictions(teacher_logits)
    metrics = {
        "graph_only": oracle.regression_metrics(labels, graph_predictions),
        "predicted": oracle.regression_metrics(labels, predicted_predictions),
        "real_teacher": oracle.regression_metrics(labels, teacher_predictions),
    }

    shuffle_predictions = []
    shuffle_metrics = []
    shuffle_metadata = []

    def append_shuffle(index: int) -> None:
        shuffled_latents, metadata = oracle.shuffle_targets_by_modality(
            state.teacher_latents,
            state.target_mask,
            master_seed=seed,
            rate=rate,
            shuffle_index=index,
        )
        logits, _ = oracle.compute_path_output(
            state.graph_hidden,
            shuffled_latents,
            state.target_mask,
            model.missing_latent_fusion,
            model.smax_fc,
        )
        prediction = _predictions(logits)
        shuffle_predictions.append(prediction)
        shuffle_metrics.append(oracle.regression_metrics(labels, prediction))
        shuffle_metadata.append(metadata)

    for shuffle_index in range(requested_shuffle_count):
        append_shuffle(shuffle_index)
    shuffle_wf1 = np.asarray(
        [item["weighted_f1"] for item in shuffle_metrics], dtype=np.float64
    )
    mcse = (
        float(shuffle_wf1.std(ddof=1) / math.sqrt(shuffle_wf1.size))
        if shuffle_wf1.size > 1
        else 0.0
    )
    if mcse > 0.001 and requested_shuffle_count < 32:
        for shuffle_index in range(requested_shuffle_count, 32):
            append_shuffle(shuffle_index)
        shuffle_wf1 = np.asarray(
            [item["weighted_f1"] for item in shuffle_metrics], dtype=np.float64
        )
        mcse = float(shuffle_wf1.std(ddof=1) / math.sqrt(shuffle_wf1.size))

    shuffle_metric_summary = oracle.metric_mean_std(shuffle_metrics)
    shuffle_matrix = np.stack(shuffle_predictions, axis=0)
    metrics["shuffled_teacher_mean"] = {
        name: float(shuffle_metric_summary[name]["mean"])
        for name in METRIC_NAMES
    }
    bootstrap_seed = oracle.stable_seed(seed, rate, "conversation-bootstrap")
    bootstrap = {
        "real_minus_predicted_wf1": oracle.conversation_cluster_bootstrap(
            labels,
            teacher_predictions,
            predicted_predictions,
            state.sample_keys,
            seed=bootstrap_seed,
        ),
        "real_minus_shuffled_wf1": oracle.conversation_cluster_bootstrap(
            labels,
            teacher_predictions,
            shuffle_matrix,
            state.sample_keys,
            seed=oracle.stable_seed(bootstrap_seed, "shuffle"),
        ),
    }
    diagnostics = {
        "predicted": oracle.fusion_path_diagnostics(
            state.predicted_latents,
            state.target_mask,
            model.missing_latent_fusion,
            state.graph_hidden,
            graph_logits,
            model.smax_fc,
        ),
        "real_teacher": oracle.fusion_path_diagnostics(
            state.teacher_latents,
            state.target_mask,
            model.missing_latent_fusion,
            state.graph_hidden,
            graph_logits,
            model.smax_fc,
        ),
    }
    return {
        "logits": {
            "graph_only": graph_logits,
            "predicted": predicted_logits,
            "real_teacher": teacher_logits,
        },
        "residuals": {
            "graph_only": graph_residual,
            "predicted": predicted_residual,
            "real_teacher": teacher_residual,
        },
        "predictions": {
            "graph_only": graph_predictions,
            "predicted": predicted_predictions,
            "real_teacher": teacher_predictions,
            "shuffled": shuffle_matrix,
        },
        "metrics": metrics,
        "shuffle": {
            "requested_count": int(requested_shuffle_count),
            "effective_count": int(len(shuffle_metrics)),
            "metrics_per_run": shuffle_metrics,
            "metric_summary": shuffle_metric_summary,
            "weighted_f1_mcse": mcse,
            "permutations": shuffle_metadata,
        },
        "deltas": {
            "real_minus_predicted": _metric_delta(
                metrics["real_teacher"], metrics["predicted"]
            ),
            "real_minus_shuffled_mean": _metric_delta(
                metrics["real_teacher"], metrics["shuffled_teacher_mean"]
            ),
        },
        "bootstrap": bootstrap,
        "ood_diagnostics": diagnostics,
    }


def _rate_zero_error(paths: Mapping[str, Any]) -> float:
    reference = paths["predictions"]["graph_only"]
    candidates = [
        paths["predictions"]["predicted"],
        paths["predictions"]["real_teacher"],
    ]
    candidates.extend(paths["predictions"]["shuffled"])
    return max(
        [0.0]
        + [float(np.max(np.abs(reference - candidate))) for candidate in candidates]
    )


def _history_parity(
    expected: Mapping[str, Any], actual: Mapping[str, float]
) -> Dict[str, Any]:
    errors = {
        name: abs(float(actual[name]) - float(expected[name]))
        for name in METRIC_NAMES
    }
    maximum = max(errors.values())
    if maximum >= 1e-6:
        raise RuntimeError(
            "predicted validation metrics do not reproduce history; "
            "max error={}; errors={}; actual={}; expected={}".format(
                maximum,
                errors,
                {name: float(actual[name]) for name in METRIC_NAMES},
                {name: float(expected[name]) for name in METRIC_NAMES},
            )
        )
    return {
        "expected_metrics": {name: float(expected[name]) for name in METRIC_NAMES},
        "recomputed_metrics": {name: float(actual[name]) for name in METRIC_NAMES},
        "per_metric_abs_error": errors,
        "max_abs_error": float(maximum),
    }


def _evaluate_seed(
    seed: int,
    rates: Sequence[float],
    feature_root: Path,
    checkpoint_root: Path,
    history_root: Path,
    output_dir: Path,
    device: torch.device,
    code_commit: str,
    shuffle_count: int,
) -> Sequence[Mapping[str, Any]]:
    checkpoint_path = checkpoint_root / "seed_{}".format(seed) / "best.pt"
    history_dir = history_root / "seed_{}".format(seed)
    history_path = history_dir / "history.json"
    config_path = history_dir / "config.json"
    for path in (checkpoint_path, history_path, config_path):
        if not path.is_file():
            raise FileNotFoundError("required evidence file is missing: {}".format(path))
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    required_checkpoint_keys = {
        "model",
        "config",
        "epoch",
        "validation_mean_weighted_f1",
    }
    if set(checkpoint) != required_checkpoint_keys:
        raise ValueError("checkpoint keys do not match the formal contract")
    config_value = TrainConfig(**checkpoint["config"])
    validate_run_config(config_value)
    if config_value.seed != seed:
        raise ValueError("checkpoint seed does not match requested seed")
    if _load_json(config_path) != checkpoint["config"]:
        raise ValueError("history config.json does not match checkpoint config")
    history_record = find_history_record(
        _load_json(history_path),
        epoch=int(checkpoint["epoch"]),
        saved_score=float(checkpoint["validation_mean_weighted_f1"]),
    )
    loader, dimensions, model = _build_validation_runtime(
        config_value, feature_root, device
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint["model"]
    model.eval()
    state_hash_before = oracle.state_dict_sha256(model)
    seed_records = []
    schedules = _schedules(config_value, "validation")
    relation_order, relation_candidates = _discover_temporal_relation_order(
        model,
        loader,
        schedules,
        dimensions,
        device,
        history_record,
        rates,
    )
    for rate in rates:
        rate_buffers = oracle.snapshot_named_buffers(model)
        with temporal_relation_order(relation_order):
            state, extraction_audit, legacy_availability = _collect_rate_state(
                model,
                loader,
                schedules[float(rate)],
                dimensions,
                device,
            )
        history_expected = history_record["validation"][_rate_key(rate)]
        quick_predicted_metrics = oracle.regression_metrics(
            state.labels,
            _predictions(state.native_logits),
        )
        history_parity = _history_parity(
            history_expected, quick_predicted_metrics
        )
        paths = _path_bundle(model, state, seed, rate, shuffle_count)
        buffers_restored = _buffer_snapshot_equal(model, rate_buffers)
        state_hash_after = oracle.state_dict_sha256(model)
        if state_hash_after != state_hash_before:
            raise RuntimeError("model state changed during frozen diagnostic")
        if not buffers_restored:
            raise RuntimeError("named buffers changed during frozen diagnostic")
        if extraction_audit["predicted_hidden_max_abs_error"] >= 1e-6:
            raise RuntimeError("manual predicted hidden does not match native forward")
        if extraction_audit["predicted_logits_max_abs_error"] >= 1e-6:
            raise RuntimeError("manual predicted logits do not match native forward")
        native_manual_metric_error = max(
            abs(
                float(paths["metrics"]["predicted"][name])
                - float(quick_predicted_metrics[name])
            )
            for name in METRIC_NAMES
        )
        if native_manual_metric_error >= 1e-6:
            raise RuntimeError("manual predicted metrics differ from native metrics")
        target_count = int(state.target_mask.sum().item())
        rate0_error = _rate_zero_error(paths) if float(rate) == 0.0 else 0.0
        if float(rate) == 0.0 and (target_count != 0 or rate0_error >= 1e-6):
            raise RuntimeError("rate 0.0 does not preserve all four paths")
        tensors_to_check = [
            state.labels,
            state.graph_hidden,
            state.predicted_latents,
            state.teacher_latents,
        ] + list(paths["logits"].values())
        finite = all(bool(torch.isfinite(value).all()) for value in tensors_to_check)
        finite = finite and bool(
            np.isfinite(paths["predictions"]["shuffled"]).all()
        )
        if not finite:
            raise RuntimeError("non-finite oracle output")
        conversations, utterance_indices = parse_sample_keys(state.sample_keys)
        target_pool_counts = {
            name: int(state.target_mask[:, index].sum().item())
            for index, name in enumerate(MODALITIES)
        }
        record = {
            "seed": int(seed),
            "rate": float(rate),
            "split": "validation",
            "sample_count": len(state.sample_keys),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "epoch": int(checkpoint["epoch"]),
                "saved_validation_score": float(
                    checkpoint["validation_mean_weighted_f1"]
                ),
            },
            "provenance": {
                "code_commit": code_commit,
                "config": asdict(config_value),
                "feature_root": str(feature_root),
                "feature_names": FEATURE_NAMES,
                "dimensions": list(dimensions),
                "device": str(device),
                "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                "temporal_relation_order": list(relation_order),
                "relation_order_recovery": relation_candidates,
            },
            "history_parity": history_parity,
            "hashes": {
                "model_state_before": state_hash_before,
                "model_state_after": state_hash_after,
                "legacy_time_major_mask": _legacy_tensor_sha256(
                    legacy_availability
                ),
                "aligned_availability": oracle.tensor_sha256(state.availability),
                "target_mask": oracle.tensor_sha256(state.target_mask),
                "sample_order": sample_order_sha256(state.sample_keys),
            },
            "audit": {
                **extraction_audit,
                "native_manual_metric_max_abs_error": float(
                    native_manual_metric_error
                ),
                "buffers_restored": buffers_restored,
                "finite": finite,
                "rate0_max_path_error": rate0_error,
            },
            "target_pools": target_pool_counts,
            "metrics": paths["metrics"],
            "shuffle": paths["shuffle"],
            "deltas": paths["deltas"],
            "bootstrap": paths["bootstrap"],
            "ood_diagnostics": paths["ood_diagnostics"],
        }
        seed_dir = output_dir / "seed_{}".format(seed)
        stem = _rate_filename(rate)
        _write_json(seed_dir / (stem + ".json"), record)
        _write_npz(
            seed_dir / (stem + ".npz"),
            sample_keys=np.asarray(state.sample_keys, dtype=np.str_),
            conversation_ids=np.asarray(conversations, dtype=np.str_),
            utterance_indices=np.asarray(utterance_indices, dtype=np.int64),
            labels=state.labels.detach().to(device="cpu").numpy(),
            availability=state.availability.detach()
            .to(device="cpu", dtype=torch.uint8)
            .numpy(),
            target_mask=state.target_mask.detach()
            .to(device="cpu", dtype=torch.uint8)
            .numpy(),
            graph_only_predictions=paths["predictions"]["graph_only"],
            predicted_predictions=paths["predictions"]["predicted"],
            real_teacher_predictions=paths["predictions"]["real_teacher"],
            shuffled_teacher_predictions=paths["predictions"]["shuffled"],
        )
        seed_records.append(record)
        print(
            "seed={} rate={:.1f} pred={:.4f} real={:.4f} shuffle={:.4f} K={}".format(
                seed,
                rate,
                paths["metrics"]["predicted"]["weighted_f1"],
                paths["metrics"]["real_teacher"]["weighted_f1"],
                paths["metrics"]["shuffled_teacher_mean"]["weighted_f1"],
                paths["shuffle"]["effective_count"],
            ),
            flush=True,
        )
    return seed_records


def _aggregate_summary(
    records: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    rates: Sequence[float],
    code_commit: str,
) -> Dict[str, Any]:
    per_rate = {}
    for rate in rates:
        selected = [record for record in records if record["rate"] == float(rate)]
        path_summary = {}
        for path in (
            "graph_only",
            "predicted",
            "real_teacher",
            "shuffled_teacher_mean",
        ):
            path_summary[path] = oracle.metric_mean_std(
                [record["metrics"][path] for record in selected]
            )
        per_rate[_rate_key(rate)] = {
            "paths": path_summary,
            "real_minus_predicted_wf1": _mean_std(
                [
                    record["deltas"]["real_minus_predicted"]["weighted_f1"]
                    for record in selected
                ]
            ),
            "real_minus_shuffled_wf1": _mean_std(
                [
                    record["deltas"]["real_minus_shuffled_mean"]["weighted_f1"]
                    for record in selected
                ]
            ),
        }

    def aggregate_rate_group(group_rates: Sequence[float]) -> Dict[str, Any]:
        seed_values = {"real_minus_predicted": [], "real_minus_shuffled": []}
        if not group_rates:
            return {
                "rates": [],
                "per_seed": seed_values,
                "across_seed": {},
            }
        for seed in seeds:
            selected = [
                record
                for record in records
                if record["seed"] == int(seed) and record["rate"] in group_rates
            ]
            if len(selected) != len(group_rates):
                raise RuntimeError("incomplete seed/rate result grid")
            seed_values["real_minus_predicted"].append(
                float(
                    np.mean(
                        [
                            record["deltas"]["real_minus_predicted"][
                                "weighted_f1"
                            ]
                            for record in selected
                        ]
                    )
                )
            )
            seed_values["real_minus_shuffled"].append(
                float(
                    np.mean(
                        [
                            record["deltas"]["real_minus_shuffled_mean"][
                                "weighted_f1"
                            ]
                            for record in selected
                        ]
                    )
                )
            )
        return {
            "rates": [float(rate) for rate in group_rates],
            "per_seed": seed_values,
            "across_seed": {
                name: _mean_std(values) for name, values in seed_values.items()
            },
        }

    nonzero = [float(rate) for rate in rates if float(rate) > 0.0]
    high_missing = [float(rate) for rate in rates if float(rate) >= 0.5]
    return {
        "experiment": "missing_m3_mosi_latent_oracle_20260903",
        "split": "validation",
        "code_commit": code_commit,
        "seeds": [int(seed) for seed in seeds],
        "rates": [float(rate) for rate in rates],
        "record_count": len(records),
        "per_rate": per_rate,
        "nonzero_rates": aggregate_rate_group(nonzero),
        "high_missing_rates": aggregate_rate_group(high_missing),
        "test_split_evaluated": False,
    }


def _resolve_code_commit(explicit: str) -> str:
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPOSITORY_ROOT),
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("--code-commit is required outside a Git worktree")


def _parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[66, 67, 68, 69, 70])
    parser.add_argument(
        "--rates", type=float, nargs="+", default=list(MISSING_RATES)
    )
    parser.add_argument("--shuffle-count", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument("--code-commit", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = _parse_args(argv)
    seeds = tuple(args.seeds)
    rates = tuple(round(float(rate), 1) for rate in args.rates)
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    if len(rates) != len(set(rates)) or any(rate not in MISSING_RATES for rate in rates):
        raise ValueError("rates must be unique members of the formal 0.0--0.7 grid")
    if args.shuffle_count < 8 or args.shuffle_count > 32:
        raise ValueError("shuffle-count must be between 8 and 32")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    code_commit = _resolve_code_commit(args.code_commit)
    prepare_output_directory(args.output_dir)
    records = []
    for seed in seeds:
        records.extend(
            _evaluate_seed(
                seed,
                rates,
                args.feature_root,
                args.checkpoint_root,
                args.history_root,
                args.output_dir,
                device,
                code_commit,
                args.shuffle_count,
            )
        )
    expected_count = len(seeds) * len(rates)
    if len(records) != expected_count:
        raise RuntimeError("result grid is incomplete")
    summary = _aggregate_summary(records, seeds, rates, code_commit)
    _write_json(args.output_dir / "summary.json", summary)
    print("wrote {} seed-rate records".format(len(records)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
