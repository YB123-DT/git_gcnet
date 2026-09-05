#!/usr/bin/env python3
"""Training-free audit for a global state plus asynchronous modality states.

The audit freezes the already-trained Missing-M3 student projectors and asks a
small, fixed-capacity ridge probe whether modality-specific temporal history
adds validation information beyond the current student node and a generic
conversation pool.  It never performs an optimizer step and never iterates a
test loader.

The three primary representations are:

``local``
    Current fused student node ``e_t``.
``generic``
    ``[e_t; mean(e_{!=t})]`` within the same conversation.
``asynchronous``
    ``[e_t; zbar_A, delta_A; zbar_T, delta_T; zbar_V, delta_V]``.  ``zbar``
    is an exponential-distance weighted mean of valid observations of one
    modality; ``delta`` is its weighted signed temporal offset.  Both are
    computed from frozen student latents and the fixed availability mask.

Two controls are also recorded: a same-width history with deterministic
random values and a same-modality history whose observed latent values are
permuted within each conversation.  They are controls, not candidate models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


MODALITIES = ("audio", "text", "visual")
RATES = (0.0, 0.5, 0.7)
PROBE_SEEDS = (66, 67, 68)
DEFAULT_DECAY = 1.0
RIDGE_ALPHA = 10.0


def _stable_int(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _as_bool_mask(valid: np.ndarray, length: int, batch: int) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (batch, length):
        raise ValueError("valid must have shape [B, L]")
    return valid


def asynchronous_state_features(
    student_latents: np.ndarray,
    availability: np.ndarray,
    valid: np.ndarray,
    decay: float = DEFAULT_DECAY,
) -> np.ndarray:
    """Build modality-specific asynchronous states.

    Parameters
    ----------
    student_latents:
        Frozen per-modality student latents with shape ``[L, B, 3, D]``.
    availability:
        Binary observed-modality mask with shape ``[L, B, 3]``.
    valid:
        Conversation padding mask with shape ``[B, L]``.
    decay:
        Fixed positive exponential distance coefficient.  The contribution of
        an observation at offset ``d`` is ``exp(-decay * abs(d))``.

    Returns
    -------
    np.ndarray
        ``[L, B, 3 * (D + 1)]``.  For each modality, the block is
        ``[zbar_t^m, delta_t^m]``.  ``zbar`` uses the nearest valid
        observation on each side of the current time (and the current value
        when observed), while ``delta`` is their weighted signed offset.
    """

    latents = np.asarray(student_latents, dtype=np.float64)
    mask = np.asarray(availability)
    if latents.ndim != 4:
        raise ValueError("student_latents must have shape [L, B, 3, D]")
    length, batch, modality_count, latent_dim = latents.shape
    if modality_count != 3:
        raise ValueError("student_latents must contain three modalities")
    if mask.shape != (length, batch, 3):
        raise ValueError("availability must have shape [L, B, 3]")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("availability must be binary")
    valid_mask = _as_bool_mask(valid, length, batch)
    result = np.zeros(
        (length, batch, modality_count * (latent_dim + 1)),
        dtype=np.float64,
    )
    for item in range(batch):
        valid_positions = np.flatnonzero(valid_mask[item])
        for modality in range(modality_count):
            observed = np.asarray(
                [
                    time
                    for time in valid_positions
                    if bool(mask[time, item, modality])
                ],
                dtype=np.int64,
            )
            if observed.size == 0:
                continue
            for time in valid_positions:
                # This is a local asynchronous state, not a second global
                # pooling operation: only the nearest valid observation on
                # each side contributes at a given time.
                insertion = int(np.searchsorted(observed, time))
                candidates: list[int] = []
                if insertion < observed.size and int(observed[insertion]) == int(time):
                    candidates.append(int(observed[insertion]))
                if insertion > 0:
                    candidates.append(int(observed[insertion - 1]))
                if insertion < observed.size and int(observed[insertion]) != int(time):
                    candidates.append(int(observed[insertion]))
                candidate_array = np.asarray(sorted(set(candidates)), dtype=np.int64)
                if candidate_array.size == 0:
                    continue
                offsets = candidate_array - int(time)
                weights = np.exp(-float(decay) * np.abs(offsets))
                normalizer = float(weights.sum())
                if normalizer <= 0.0 or not math.isfinite(normalizer):
                    continue
                weighted = (
                    weights[:, None]
                    * latents[candidate_array, item, modality]
                ).sum(axis=0) / normalizer
                signed_offset = float((weights * offsets).sum() / normalizer)
                start = modality * (latent_dim + 1)
                result[time, item, start : start + latent_dim] = weighted
                result[time, item, start + latent_dim] = signed_offset
    return result


def generic_context_features(
    student_nodes: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return ``[e_t; mean(e_{!=t})]`` without crossing conversations."""

    nodes = np.asarray(student_nodes, dtype=np.float64)
    if nodes.ndim != 3:
        raise ValueError("student_nodes must have shape [L, B, D]")
    length, batch, width = nodes.shape
    valid_mask = _as_bool_mask(valid, length, batch)
    pooled = np.zeros_like(nodes)
    for item in range(batch):
        positions = np.flatnonzero(valid_mask[item])
        if positions.size <= 1:
            continue
        total = nodes[positions, item].sum(axis=0)
        for time in positions:
            pooled[time, item] = (total - nodes[time, item]) / float(
                positions.size - 1
            )
    return np.concatenate([nodes, pooled], axis=-1)


def shuffle_modality_history(
    student_latents: np.ndarray,
    availability: np.ndarray,
    valid: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Permute each modality's observed values within each conversation.

    Availability and current ``e_t`` are untouched by this control.  Only the
    values used to construct the asynchronous history are permuted.
    """

    latents = np.asarray(student_latents, dtype=np.float64).copy()
    mask = np.asarray(availability)
    if latents.ndim != 4 or latents.shape[2] != 3:
        raise ValueError("student_latents must have shape [L, B, 3, D]")
    length, batch = latents.shape[:2]
    if mask.shape != (length, batch, 3):
        raise ValueError("availability must have shape [L, B, 3]")
    valid_mask = _as_bool_mask(valid, length, batch)
    rng = np.random.default_rng(seed)
    for item in range(batch):
        positions = np.flatnonzero(valid_mask[item])
        for modality in range(3):
            observed = positions[mask[positions, item, modality].astype(bool)]
            if observed.size > 1:
                latents[observed, item, modality] = latents[
                    rng.permutation(observed), item, modality
                ]
    return latents


def dimension_matched_random_history(
    asynchronous_features: np.ndarray,
    seed: int,
    base_dim: int = 0,
) -> np.ndarray:
    """Replace history with random values while preserving exact width.

    ``base_dim`` columns at the front (the current ``e_t`` block) can be kept
    intact.  The generated history has the same shape and therefore exactly
    the same ridge input width as the asynchronous candidate.
    """

    features = np.asarray(asynchronous_features, dtype=np.float64)
    if features.ndim < 1 or not 0 <= int(base_dim) <= features.shape[-1]:
        raise ValueError("base_dim must lie within the feature width")
    result = features.copy()
    rng = np.random.default_rng(seed)
    history_width = features.shape[-1] - int(base_dim)
    if history_width:
        result[..., int(base_dim) :] = rng.standard_normal(
            result[..., int(base_dim) :].shape
        )
    return result


def _flatten_valid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    valid_mask = np.asarray(valid, dtype=bool).T
    if values.shape[:2] != valid_mask.shape:
        raise ValueError("values and valid leading dimensions differ")
    return values[valid_mask]


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    from sklearn.metrics import mean_absolute_error, r2_score

    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    correlation: float | None
    if target.size < 2 or np.std(target) <= 1e-12 or np.std(prediction) <= 1e-12:
        correlation = None
    else:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "r2": float(r2_score(target, prediction)),
        "correlation": correlation,
    }


def fit_ridge_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, object]:
    """Fit one fixed-capacity standardized ridge probe."""

    if float(alpha) <= 0 or not math.isfinite(float(alpha)):
        raise ValueError("ridge alpha must be finite and positive")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    validation_scaled = scaler.transform(validation_features)
    probe = Ridge(alpha=float(alpha))
    probe.fit(train_scaled, np.asarray(train_labels, dtype=np.float64))
    prediction = probe.predict(validation_scaled)
    return {
        "feature_dim": int(train_features.shape[-1]),
        "ridge_alpha": float(alpha),
        "parameter_count": int(train_features.shape[-1] + 1),
        "metrics": _regression_metrics(validation_labels, prediction),
    }


def _compatibility_state(state: Mapping[str, object]) -> dict[str, object]:
    return {
        name.replace(".conv2.lin_l.", ".conv2.lin_rel.").replace(
            ".conv2.lin_r.", ".conv2.lin_root."
        ): value
        for name, value in state.items()
    }


def _build_frozen_model(checkpoint: Mapping[str, object], dimensions: tuple[int, int, int]):
    # Heavy imports are kept inside the audit path so pure helper tests run in
    # environments without PyTorch.
    from gcnet_missing_m3.model import MissingM3GraphModel
    from gcnet_missing_m3.train_gcnet import _resolve_task_contract

    config = dict(checkpoint["config"])
    shape = _resolve_task_contract(
        str(config["dataset"]), str(config.get("mosi_task_mode", "regression"))
    )
    model = MissingM3GraphModel(
        config.get("base_model", "LSTM"),
        *dimensions,
        int(config.get("hidden", 200)),
        int(config.get("hidden", 200)) // 2,
        n_speakers=int(shape["num_speakers"]),
        window_past=int(config.get("window_past", 2)),
        window_future=int(config.get("window_future", 2)),
        n_classes=int(shape["num_classes"]),
        dropout=float(config.get("dropout", 0.5)),
        time_attn=bool(config.get("time_attention", False)),
        no_cuda=True,
        latent_dim=int(config.get("latent_dim", 256)),
        num_experts=int(config.get("num_experts", 4)),
        top_k=int(config.get("top_k", 2)),
        projector_dropout=float(config.get("projector_dropout", 0.1)),
        predictor_dropout=float(config.get("predictor_dropout", 0.1)),
        fusion_type=str(config.get("fusion_type", "slot")),
        local_context_residual=bool(config.get("local_context_residual", False)),
        local_fusion_hidden_dim=int(config.get("local_fusion_hidden_dim", 256)),
        local_fusion_dropout=float(config.get("local_fusion_dropout", 0.2)),
        graph_branch_mode=str(config.get("graph_branch_mode", "both")),
        mmoe_variant=str(config.get("mmoe_variant", "dual-gate")),
        target_private_rank=int(config.get("target_private_rank", 0)),
        classification_completion=bool(config.get("classification_completion", False)),
        representation_type=str(config.get("representation_type", "slot")),
        node_interaction_residual=bool(config.get("node_interaction_residual", False)),
        readout_type=str(config.get("readout_type", "shared")),
        readout_rank=int(config.get("readout_rank", 8)),
        recurrent_padding_mode=str(config.get("recurrent_padding_mode", "legacy")),
        postgraph_sequence_mode=str(config.get("postgraph_sequence_mode", "independent")),
        graph_message_calibration=str(config.get("graph_message_calibration", "none")),
        graph_second_layer=str(config.get("graph_second_layer", "graphconv")),
        postgraph_bilstm_ablation=str(config.get("postgraph_bilstm_ablation", "none")),
    )
    model.load_state_dict(_compatibility_state(checkpoint["model"]), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _collect_split(
    model,
    loader,
    schedule,
    dimensions: tuple[int, int, int],
    split: str,
    seed: int,
    rate: float,
    decay: float,
) -> tuple[dict[str, np.ndarray], int]:
    from gcnet_missing_m3.train_gcnet import _prepare_view

    local_parts: list[np.ndarray] = []
    generic_parts: list[np.ndarray] = []
    asynchronous_parts: list[np.ndarray] = []
    shuffled_parts: list[np.ndarray] = []
    random_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    conversation_count = 0
    for data in loader:
        view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
        with __import__("torch").no_grad():
            encoded, latents = model.observed_set(
                view["incomplete"], view["availability"], view["umask"]
            )
        stacked_latents = __import__("torch").stack(
            [latents[name] for name in MODALITIES], dim=2
        )
        encoded_np = encoded.detach().cpu().numpy()
        latent_np = stacked_latents.detach().cpu().numpy()
        availability_np = view["availability"].detach().cpu().numpy()
        valid_np = view["umask"].detach().cpu().numpy().astype(bool)
        labels_np = view["labels"].detach().cpu().numpy()
        for item in range(encoded_np.shape[1]):
            length = int(valid_np[item].sum())
            if length <= 0:
                continue
            nodes = encoded_np[:length, item][..., None, :]
            nodes = nodes[:, 0, :]  # [L, D], explicit for readability
            conv_latents = latent_np[:length, item][:, None, :, :]
            conv_availability = availability_np[:length, item][:, None, :]
            conv_valid = np.ones((1, length), dtype=bool)
            local = nodes[:, None, :]
            generic = generic_context_features(local, conv_valid)[:, 0]
            asynchronous = asynchronous_state_features(
                conv_latents, conv_availability, conv_valid, decay=decay
            )[:, 0]
            asynchronous_full = np.concatenate([nodes, asynchronous], axis=-1)
            shuffle_latents = shuffle_modality_history(
                conv_latents,
                conv_availability,
                conv_valid,
                seed=_stable_int(seed, split, rate, item, conversation_count),
            )
            shuffled_state = asynchronous_state_features(
                shuffle_latents,
                conv_availability,
                conv_valid,
                decay=decay,
            )[:, 0]
            shuffled_full = np.concatenate([nodes, shuffled_state], axis=-1)
            random_full = dimension_matched_random_history(
                asynchronous_full,
                seed=_stable_int("random-history", seed, split, rate, item),
                base_dim=nodes.shape[-1],
            )
            labels = labels_np[item, :length].astype(np.float64, copy=False)
            local_parts.append(nodes)
            generic_parts.append(generic)
            asynchronous_parts.append(asynchronous_full)
            shuffled_parts.append(shuffled_full)
            random_parts.append(random_full)
            label_parts.append(labels)
            conversation_count += 1
    if not label_parts:
        raise RuntimeError(f"{split} split produced no valid utterances")
    return {
        "local": np.concatenate(local_parts, axis=0),
        "generic": np.concatenate(generic_parts, axis=0),
        "asynchronous": np.concatenate(asynchronous_parts, axis=0),
        "asynchronous_shuffled": np.concatenate(shuffled_parts, axis=0),
        "asynchronous_random": np.concatenate(random_parts, axis=0),
        "labels": np.concatenate(label_parts, axis=0),
    }, conversation_count


def _schedule(dataset: str, split: str, fold: int, seed: int, rate: float):
    from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule
    from gcnet_modality_jepa.protocol import SeedBundle

    return ConversationMaskSchedule(
        dataset=dataset,
        split=split,
        fold=fold,
        requested_missing_rate=rate,
        mask_seed=SeedBundle(seed).derive("missing_mask"),
        freeze_evaluation=True,
    )


def _run_audit(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from gcnet_missing_m3.train_gcnet import get_loaders

    checkpoint_root = Path(args.checkpoint_root)
    feature_root = Path(args.feature_root)
    results: list[dict[str, object]] = []
    feature_names = ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
    roots = [feature_root / name for name in feature_names]
    for seed in args.probe_seeds:
        checkpoint_path = checkpoint_root / f"seed_{seed}" / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = dict(checkpoint["config"])
        if str(config.get("train_rate_mode", "cyclic")) != "cyclic":
            raise ValueError("asynchronous audit requires a cyclic checkpoint")
        loaders = get_loaders(
            *(str(root) for root in roots),
            num_folder=1,
            dataset="CMUMOSI",
            batch_size=int(config.get("batch_size", 32)),
            num_workers=0,
            seed=int(seed),
            validation_fraction=float(config.get("validation_fraction", 0.1)),
            evaluation_protocol=str(config.get("evaluation_protocol", "official")),
        )
        train_loaders, validation_loaders, _test_loaders, adim, tdim, vdim = loaders
        dimensions = (int(adim), int(tdim), int(vdim))
        model = _build_frozen_model(checkpoint, dimensions)
        for rate in args.rates:
            train_values, train_conversations = _collect_split(
                model,
                train_loaders[0],
                _schedule("CMUMOSI", "train", 1, seed, rate),
                dimensions,
                "train",
                seed,
                rate,
                args.decay,
            )
            validation_values, validation_conversations = _collect_split(
                model,
                validation_loaders[0],
                _schedule("CMUMOSI", "validation", 1, seed, rate),
                dimensions,
                "validation",
                seed,
                rate,
                args.decay,
            )
            labels = train_values.pop("labels")
            validation_labels = validation_values.pop("labels")
            probes = {
                name: fit_ridge_probe(
                    train_values[name],
                    labels,
                    validation_values[name],
                    validation_labels,
                    alpha=args.ridge_alpha,
                )
                for name in (
                    "local",
                    "generic",
                    "asynchronous",
                    "asynchronous_shuffled",
                    "asynchronous_random",
                )
            }
            results.append(
                {
                    "seed": int(seed),
                    "rate": float(rate),
                    "train_nodes": int(len(labels)),
                    "validation_nodes": int(len(validation_labels)),
                    "train_conversations": int(train_conversations),
                    "validation_conversations": int(validation_conversations),
                    "probes": probes,
                    "deltas": {
                        "asynchronous_minus_generic_correlation": _delta_metric(
                            probes, "asynchronous", "generic", "correlation"
                        ),
                        "asynchronous_minus_generic_mae": _delta_metric(
                            probes, "asynchronous", "generic", "mae"
                        ),
                        "shuffled_minus_generic_correlation": _delta_metric(
                            probes, "asynchronous_shuffled", "generic", "correlation"
                        ),
                        "random_minus_generic_correlation": _delta_metric(
                            probes, "asynchronous_random", "generic", "correlation"
                        ),
                    },
                }
            )
    summary = _summarize(results, args.rates, args.probe_seeds)
    return {
        "dataset": "CMUMOSI",
        "fold": 1,
        "checkpoint_root": str(checkpoint_root),
        "feature_root": str(feature_root),
        "checkpoint_selection_provenance": "cyclic test-oracle checkpoint; audit metrics are validation-only",
        "probe_seeds": [int(value) for value in args.probe_seeds],
        "rates": [float(value) for value in args.rates],
        "decay": float(args.decay),
        "ridge_alpha": float(args.ridge_alpha),
        "test_iterated": False,
        "results": results,
        "summary": summary,
        "gate": _gate(summary),
    }


def _delta_metric(
    probes: Mapping[str, Mapping[str, object]],
    left: str,
    right: str,
    metric: str,
) -> float | None:
    left_value = probes[left]["metrics"].get(metric)
    right_value = probes[right]["metrics"].get(metric)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _summarize(
    results: Sequence[Mapping[str, object]],
    rates: Sequence[float],
    seeds: Sequence[int],
) -> dict[str, object]:
    by_rate: dict[str, object] = {}
    for rate in rates:
        selected = [item for item in results if float(item["rate"]) == float(rate)]
        probe_summary: dict[str, object] = {}
        for name in (
            "local",
            "generic",
            "asynchronous",
            "asynchronous_shuffled",
            "asynchronous_random",
        ):
            metrics: dict[str, float | None] = {}
            for metric in ("mae", "r2", "correlation"):
                values = [
                    item["probes"][name]["metrics"][metric]
                    for item in selected
                    if item["probes"][name]["metrics"][metric] is not None
                ]
                metrics[metric] = float(np.mean(values)) if values else None
            probe_summary[name] = metrics
        corr_deltas = [
            item["deltas"]["asynchronous_minus_generic_correlation"]
            for item in selected
            if item["deltas"]["asynchronous_minus_generic_correlation"] is not None
        ]
        mae_deltas = [
            item["deltas"]["asynchronous_minus_generic_mae"]
            for item in selected
            if item["deltas"]["asynchronous_minus_generic_mae"] is not None
        ]
        shuffled_deltas = [
            item["deltas"]["shuffled_minus_generic_correlation"]
            for item in selected
            if item["deltas"]["shuffled_minus_generic_correlation"] is not None
        ]
        random_deltas = [
            item["deltas"]["random_minus_generic_correlation"]
            for item in selected
            if item["deltas"]["random_minus_generic_correlation"] is not None
        ]
        probe_summary["deltas"] = {
            "asynchronous_minus_generic_correlation_mean": _mean_or_none(corr_deltas),
            "asynchronous_minus_generic_mae_mean": _mean_or_none(mae_deltas),
            "shuffled_minus_generic_correlation_mean": _mean_or_none(shuffled_deltas),
            "random_minus_generic_correlation_mean": _mean_or_none(random_deltas),
            "asynchronous_positive_correlation_seeds": int(
                sum(value > 0 for value in corr_deltas)
            ),
            "seed_count": int(len(seeds)),
        }
        by_rate[str(rate)] = probe_summary
    return by_rate


def _mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _gate(summary: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    required_rates = ("0.5", "0.7")
    stable_gain: dict[str, bool] = {}
    for rate in required_rates:
        deltas = summary[rate]["deltas"]
        stable_gain[rate] = bool(
            deltas["asynchronous_minus_generic_correlation_mean"] is not None
            and deltas["asynchronous_minus_generic_correlation_mean"] > 0.0
            and deltas["asynchronous_positive_correlation_seeds"] >= 2
        )
    miss0_delta = summary["0.0"]["deltas"][
        "asynchronous_minus_generic_correlation_mean"
    ]
    miss0_not_obviously_lower = bool(miss0_delta is not None and miss0_delta > -0.02)
    shuffle_values = [
        summary[rate]["deltas"]["shuffled_minus_generic_correlation_mean"]
        for rate in required_rates
    ]
    async_values = [
        summary[rate]["deltas"]["asynchronous_minus_generic_correlation_mean"]
        for rate in required_rates
    ]
    shuffle_breaks_gain = bool(
        all(value is not None for value in shuffle_values)
        and all(value <= 0.0 for value in shuffle_values)
        and all(value is not None and value > 0.0 for value in async_values)
    )
    random_values = [
        summary[rate]["deltas"]["random_minus_generic_correlation_mean"]
        for rate in required_rates
    ]
    dimension_control_pass = bool(
        all(value is not None and value < 0.0 for value in random_values)
    )
    passed = bool(
        all(stable_gain.values())
        and miss0_not_obviously_lower
        and shuffle_breaks_gain
        and dimension_control_pass
    )
    return {
        "stable_gain_at_0.5_and_0.7": stable_gain,
        "miss0_not_obviously_lower": miss0_not_obviously_lower,
        "shuffle_removes_gain": shuffle_breaks_gain,
        "dimension_matched_random_history_rejected": dimension_control_pass,
        "passed": passed,
        "decision": (
            "implement full asynchronous-state SSM candidate"
            if passed
            else "do not implement/train full asynchronous-state SSM"
        ),
    }


def _run_self_test() -> None:
    latents = np.zeros((3, 1, 3, 2), dtype=np.float64)
    latents[:, 0, 0] = [[1.0, 0.0], [0.0, 0.0], [3.0, 0.0]]
    availability = np.asarray(
        [[[1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]], dtype=np.int64
    )
    valid = np.ones((1, 3), dtype=bool)
    state = asynchronous_state_features(latents, availability, valid)
    if state.shape != (3, 1, 9):
        raise AssertionError("unexpected asynchronous state shape")
    if not np.isclose(state[1, 0, 0], 2.0):
        raise AssertionError("symmetric observations were not averaged")
    if not np.isclose(state[1, 0, 2], 0.0):
        raise AssertionError("symmetric signed offset was not zero")
    nodes = np.arange(6, dtype=np.float64).reshape(3, 1, 2)
    generic = generic_context_features(nodes, valid)
    if generic.shape != (3, 1, 4):
        raise AssertionError("unexpected generic context shape")
    random = dimension_matched_random_history(generic, seed=66, base_dim=2)
    if random.shape != generic.shape or not np.array_equal(random[..., :2], nodes):
        raise AssertionError("random-history control changed current node")
    print("self-test: PASS")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-seeds", default="66,67,68")
    parser.add_argument("--rates", default="0.0,0.5,0.7")
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _run_self_test()
        return
    if args.checkpoint_root is None or args.feature_root is None or args.output is None:
        raise ValueError("checkpoint-root, feature-root, and output are required")
    args.probe_seeds = tuple(int(value) for value in str(args.probe_seeds).split(","))
    args.rates = tuple(float(value) for value in str(args.rates).split(","))
    if tuple(args.rates) != RATES:
        raise ValueError("audit rates must be exactly 0.0,0.5,0.7")
    if not args.probe_seeds:
        raise ValueError("probe-seeds cannot be empty")
    if args.decay <= 0 or not math.isfinite(args.decay):
        raise ValueError("decay must be finite and positive")
    if args.ridge_alpha <= 0 or not math.isfinite(args.ridge_alpha):
        raise ValueError("ridge-alpha must be finite and positive")
    output = _run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
