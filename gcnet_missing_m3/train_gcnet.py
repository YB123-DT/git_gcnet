"""Focused mixed-rate IEMOCAP trainer for Single-View Missing-M3 GCNet."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import config
from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule
from gcnet_modality_jepa.protocol import SeedBundle
from gcnet_modality_jepa.train_gcnet import (
    build_primary_mask_tensors,
    generate_inputs,
    get_loaders,
    set_random_seed,
)

from .loss import MissingM3Loss, missing_m3_loss
from .mixed_rate import (
    MISSING_RATES,
    STRATIFIED_RATE_ALGORITHM,
    BalancedBatchRateSchedule,
    mean_validation_weighted_f1,
    stratified_rates_for_batch,
)
from .model import MissingM3GraphModel


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "IEMOCAPSix"
    fold: int = 5
    seed: int = 66
    base_model: str = "LSTM"
    window_past: int = 2
    window_future: int = 2
    hidden: int = 200
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    latent_dim: int = 256
    num_experts: int = 4
    top_k: int = 2
    projector_dropout: float = 0.1
    predictor_dropout: float = 0.1
    fusion_type: str = "mean"
    local_context_residual: bool = False
    local_fusion_hidden_dim: int = 256
    local_fusion_dropout: float = 0.2
    jepa_weight: float = 0.1
    temperature: float = 0.03
    ema_tau: float = 0.996
    gradient_clip_norm: float = 1.0
    time_attention: bool = False
    evaluation_protocol: str = "official"
    validation_fraction: float = 0.1
    device: str = "cuda"
    train_rate_mode: str = "cyclic"
    mosi_task_mode: str = "regression"
    graph_branch_mode: str = "both"
    mmoe_variant: str = "dual-gate"
    target_private_rank: int = 0
    classification_completion: bool = False
    representation_type: str = "slot"
    node_interaction_residual: bool = False
    readout_type: str = "shared"
    readout_rank: int = 8
    evaluate_test: bool = True
    jepa_regression_aggregation: str = "target"
    recurrent_padding_mode: str = "legacy"
    task_regression_loss: str = "mse"
    task_smooth_l1_beta: float = 1.0
    postgraph_sequence_mode: str = "independent"
    jepa_rate_weighting: str = "uniform"
    graph_message_calibration: str = "none"
    graph_second_layer: str = "graphconv"
    fixed_missing_rate: float | None = None
    checkpoint_selection: str = "validation"
    jepa_contrastive_source: str = "contrastive"
    training_objective: str = "joint"
    initial_backbone_checkpoint: str | None = None
    pretrained_learning_rate: float | None = None


_TRAINING_OBJECTIVES = {
    "joint",
    "jepa-only",
    "emotion-only",
    "frozen-completion",
}
_STAGE2_EXCLUDED_PREFIXES = (
    "smax_fc.",
    "conditioned_readout.",
    "affine_readout.",
    "missing_predictor.",
    "missing_latent_fusion.",
    "teacher.",
)
_JOINT_FINETUNE_EXCLUDED_PREFIXES = (
    "smax_fc.",
    "conditioned_readout.",
    "affine_readout.",
    "missing_latent_fusion.",
)
_FROZEN_COMPLETION_TRAINABLE_PREFIXES = (
    "smax_fc.",
    "conditioned_readout.",
    "affine_readout.",
    "missing_latent_fusion.",
)


def _dataset_shape(dataset: str) -> Dict[str, object]:
    contracts = {
        "IEMOCAPFour": {
            "num_folds": 5,
            "num_classes": 4,
            "num_speakers": 2,
            "task": "classification",
        },
        "IEMOCAPSix": {
            "num_folds": 5,
            "num_classes": 6,
            "num_speakers": 2,
            "task": "classification",
        },
        "CMUMOSI": {
            "num_folds": 1,
            "num_classes": 1,
            "num_speakers": 1,
            "task": "regression",
        },
        "CMUMOSEI": {
            "num_folds": 1,
            "num_classes": 1,
            "num_speakers": 1,
            "task": "regression",
        },
    }
    try:
        return dict(contracts[dataset])
    except KeyError:
        raise ValueError("unsupported dataset: {}".format(dataset))


def _resolve_task_contract(dataset: str, mode: str) -> Dict[str, object]:
    if mode not in ("regression", "binary", "soft-ordinal"):
        raise ValueError("unsupported MOSI task mode: {}".format(mode))
    contract = _dataset_shape(dataset)
    if mode in ("binary", "soft-ordinal"):
        if dataset != "CMUMOSI":
            raise ValueError(
                "{} task mode is only supported for CMUMOSI".format(mode)
            )
    if mode == "binary":
        contract.update(task="binary", num_classes=2)
    elif mode == "soft-ordinal":
        contract.update(task="soft-ordinal", num_classes=1)
    return contract


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_run_config(path: Path, config_value: TrainConfig) -> None:
    _write_json(path, asdict(config_value))


def _save_best_checkpoint(
    path: Path,
    model_state: Mapping[str, torch.Tensor],
    config_value: TrainConfig,
    epoch: int,
    validation_mean_weighted_f1: float | None,
    selection_split: str = "validation",
) -> None:
    validation_score = (
        validation_mean_weighted_f1
        if selection_split == "validation"
        else None
    )
    torch.save(
        {
            "model": model_state,
            "config": asdict(config_value),
            "epoch": epoch,
            "validation_mean_weighted_f1": validation_score,
            "selection_split": selection_split,
            "selection_mean_weighted_f1": validation_mean_weighted_f1,
        },
        path,
    )


def _state_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inference_backbone_checkpoint(
    model: MissingM3GraphModel,
    checkpoint_path: str | Path,
    include_jepa_modules: bool = False,
) -> Dict[str, object]:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu")
    source_config = checkpoint.get("config", {})
    source_objective = source_config.get("training_objective")
    if source_objective != "jepa-only":
        raise ValueError(
            "initial backbone checkpoint must come from jepa-only pretraining"
        )
    source_state = checkpoint.get("model")
    if not isinstance(source_state, Mapping):
        raise ValueError("initial backbone checkpoint has no model state")
    target_state = model.state_dict()
    excluded_prefixes = (
        _JOINT_FINETUNE_EXCLUDED_PREFIXES
        if include_jepa_modules
        else _STAGE2_EXCLUDED_PREFIXES
    )
    loaded_keys = []
    for key, target_value in target_state.items():
        if key.startswith(excluded_prefixes):
            continue
        if key not in source_state:
            raise ValueError("pretrained backbone is missing key: " + key)
        source_value = source_state[key]
        if source_value.shape != target_value.shape:
            raise ValueError("pretrained backbone shape mismatch: " + key)
        target_state[key] = source_value.to(dtype=target_value.dtype)
        loaded_keys.append(key)
    model.load_state_dict(target_state, strict=True)
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "source_training_objective": source_objective,
        "source_epoch": checkpoint.get("epoch"),
        "loaded_key_count": len(loaded_keys),
        "included_jepa_modules": include_jepa_modules,
    }


def _configure_frozen_completion_probe(
    model: MissingM3GraphModel,
) -> Dict[str, object]:
    trainable_names = []
    frozen_names = []
    trainable_count = 0
    frozen_count = 0
    for name, parameter in model.named_parameters():
        trainable = name.startswith(_FROZEN_COMPLETION_TRAINABLE_PREFIXES)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_count += parameter.numel()
        else:
            frozen_names.append(name)
            frozen_count += parameter.numel()
    if not trainable_names:
        raise ValueError("frozen completion has no trainable parameters")
    return {
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": frozen_count,
    }


def _parameter_subset_sha256(
    model: MissingM3GraphModel,
    parameter_names: Sequence[str],
) -> str:
    parameters = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in sorted(parameter_names):
        if name not in parameters:
            raise ValueError("unknown parameter in hash subset: " + name)
        tensor = parameters[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _optimizer_parameter_groups(
    model: MissingM3GraphModel,
    config_value: TrainConfig,
) -> tuple[list[Dict[str, object]], Dict[str, object]]:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if config_value.pretrained_learning_rate is None:
        parameters = [parameter for _, parameter in trainable]
        return [
            {"params": parameters, "lr": config_value.learning_rate}
        ], {
            "default": {
                "learning_rate": config_value.learning_rate,
                "parameter_count": sum(value.numel() for value in parameters),
            }
        }

    fresh = [
        (name, parameter)
        for name, parameter in trainable
        if name.startswith(_JOINT_FINETUNE_EXCLUDED_PREFIXES)
    ]
    pretrained = [
        (name, parameter)
        for name, parameter in trainable
        if not name.startswith(_JOINT_FINETUNE_EXCLUDED_PREFIXES)
    ]
    if not fresh or not pretrained:
        raise ValueError("differential optimizer requires both parameter groups")
    groups = [
        {
            "params": [parameter for _, parameter in pretrained],
            "lr": config_value.pretrained_learning_rate,
        },
        {
            "params": [parameter for _, parameter in fresh],
            "lr": config_value.learning_rate,
        },
    ]
    provenance = {
        "pretrained": {
            "learning_rate": config_value.pretrained_learning_rate,
            "parameter_count": sum(
                parameter.numel() for _, parameter in pretrained
            ),
        },
        "fresh": {
            "learning_rate": config_value.learning_rate,
            "parameter_count": sum(parameter.numel() for _, parameter in fresh),
        },
    }
    return groups, provenance


def _readout_provenance(model: MissingM3GraphModel) -> Dict[str, object]:
    module = getattr(model, "conditioned_readout", None)
    if module is None:
        module = getattr(model, "affine_readout", None)
    return {
        "readout_type": model.readout_type,
        "readout_rank": model.readout_rank,
        "readout_parameter_count": (
            0
            if module is None
            else sum(parameter.numel() for parameter in module.parameters())
        ),
    }


def _sha256_tensor(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _build_schedule(
    config: TrainConfig,
    split: str,
    rate: float,
) -> ConversationMaskSchedule:
    return ConversationMaskSchedule(
        dataset=config.dataset,
        split=split,
        fold=config.fold,
        requested_missing_rate=rate,
        mask_seed=SeedBundle(config.seed).derive("missing_mask"),
        freeze_evaluation=True,
    )


def _schedules(config: TrainConfig, split: str) -> Dict[float, ConversationMaskSchedule]:
    return {rate: _build_schedule(config, split, rate) for rate in MISSING_RATES}


def _fixed_missing_rate(config: TrainConfig) -> float | None:
    rate = config.fixed_missing_rate
    if config.train_rate_mode == "fixed":
        if rate is None:
            raise ValueError(
                "fixed_missing_rate is required when train_rate_mode='fixed'"
            )
        normalized = float(rate)
        if not math.isfinite(normalized) or normalized not in MISSING_RATES:
            raise ValueError("fixed_missing_rate must be one of the official missing rates")
        return normalized
    if rate is not None:
        raise ValueError(
            "fixed_missing_rate is only valid when train_rate_mode='fixed'"
        )
    return None


def _protocol_rates(config: TrainConfig) -> tuple[float, ...]:
    fixed_rate = _fixed_missing_rate(config)
    if config.train_rate_mode == "fixed":
        return (fixed_rate,)
    if config.train_rate_mode in {"cyclic", "all", "stratified"}:
        return MISSING_RATES
    raise ValueError(
        "train_rate_mode must be 'cyclic', 'all', 'fixed', or 'stratified'"
    )


def _move_batch(data: Sequence[object], device: torch.device) -> list[object]:
    moved: list[object] = []
    for value in data:
        moved.append(value.to(device) if torch.is_tensor(value) else value)
    return moved


def _lengths(umask: torch.Tensor) -> list[int]:
    result = umask.sum(dim=1).long().tolist()
    if any(value < 1 for value in result):
        raise ValueError("every conversation must contain a valid utterance")
    return [int(value) for value in result]


def _build_stratified_mask_tensors(
    schedules: Mapping[float, ConversationMaskSchedule],
    conversation_rates: Sequence[float],
    conversation_ids: Sequence[str],
    umask: torch.Tensor,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if umask.ndim != 2:
        raise ValueError("umask must have shape [batch, sequence]")
    batch_size, sequence_length = umask.shape
    if len(conversation_rates) != batch_size:
        raise ValueError("conversation rates must match the umask batch size")
    if len(conversation_ids) != batch_size:
        raise ValueError("conversation IDs must match the umask batch size")
    if any(rate not in schedules for rate in conversation_rates):
        raise ValueError("every conversation rate must have a mask schedule")
    valid_lengths = [
        int(umask[batch_index].sum().item())
        for batch_index in range(batch_size)
    ]
    for conversation_id, valid_length in zip(conversation_ids, valid_lengths):
        if valid_length < 1:
            raise ValueError(
                "conversation {!r} has no real utterances".format(
                    conversation_id
                )
            )

    side_tensors = []
    for side in ("host", "guest"):
        conversations = []
        for batch_index in range(batch_size):
            generated = schedules[conversation_rates[batch_index]].generate(
                str(conversation_ids[batch_index]),
                length=sequence_length,
                valid_length=valid_lengths[batch_index],
                side=side,
                epoch=epoch,
            )
            conversations.append(torch.as_tensor(generated.availability))
        side_tensors.append(
            torch.stack(conversations, dim=1).to(device=umask.device)
        )
    return tuple(side_tensors)


def _prepare_view_from_primary_masks(
    data: Sequence[object],
    host_availability: torch.Tensor,
    guest_availability: torch.Tensor,
    dimensions: tuple[int, int, int],
) -> dict[str, object]:
    audio_host, text_host, visual_host = data[0], data[1], data[2]
    audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
    qmask, umask, labels = data[6], data[7], data[8]
    conversation_ids = data[-1]
    full = generate_inputs(
        audio_host,
        text_host,
        visual_host,
        audio_guest,
        text_guest,
        visual_guest,
        qmask,
    )[0]
    availability = generate_inputs(
        host_availability[..., 0:1],
        host_availability[..., 1:2],
        host_availability[..., 2:3],
        guest_availability[..., 0:1],
        guest_availability[..., 1:2],
        guest_availability[..., 2:3],
        qmask,
    )[0].to(dtype=full.dtype)
    expanded = torch.repeat_interleave(
        availability,
        torch.tensor(dimensions, device=availability.device),
        dim=-1,
    )
    return {
        "complete": full,
        "incomplete": full * expanded,
        "availability": availability,
        "qmask": qmask,
        "umask": umask,
        "labels": labels,
        "lengths": _lengths(umask),
        "conversation_ids": list(conversation_ids),
    }


def _prepare_stratified_view(
    data: Sequence[object],
    schedules: Mapping[float, ConversationMaskSchedule],
    conversation_rates: Sequence[float],
    epoch: int,
    dimensions: tuple[int, int, int],
) -> dict[str, object]:
    host_availability, guest_availability = _build_stratified_mask_tensors(
        schedules=schedules,
        conversation_rates=conversation_rates,
        conversation_ids=data[-1],
        umask=data[7],
        epoch=epoch,
    )
    return _prepare_view_from_primary_masks(
        data,
        host_availability,
        guest_availability,
        dimensions,
    )


def _prepare_view(
    data: Sequence[object],
    schedule: ConversationMaskSchedule,
    epoch: int,
    dimensions: tuple[int, int, int],
) -> dict[str, object]:
    host_availability, guest_availability = build_primary_mask_tensors(
        schedule,
        conversation_ids=data[-1],
        umask=data[7],
        epoch=epoch,
    )
    return _prepare_view_from_primary_masks(
        data,
        host_availability,
        guest_availability,
        dimensions,
    )


def _mosi_soft_targets(labels: torch.Tensor) -> torch.Tensor:
    """Map continuous MOSI labels to ordered binary probabilities."""

    return (labels.clamp(min=-3.0, max=3.0) + 3.0) / 6.0


def _task_loss(
    dataset: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    umask: torch.Tensor,
    mosi_task_mode: str = "regression",
    task_regression_loss: str = "mse",
    task_smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    if task_regression_loss not in {"mse", "smooth-l1"}:
        raise ValueError("task_regression_loss must be 'mse' or 'smooth-l1'")
    if task_regression_loss == "smooth-l1" and (
        not math.isfinite(task_smooth_l1_beta) or task_smooth_l1_beta <= 0
    ):
        raise ValueError("task SmoothL1 beta must be finite and positive")
    selected = umask.reshape(-1).bool()
    task = _resolve_task_contract(dataset, mosi_task_mode)["task"]
    if task == "soft-ordinal":
        if task_regression_loss != "mse":
            raise ValueError(
                "task SmoothL1 is only valid for continuous regression"
            )
        prediction = logits.transpose(0, 1).reshape(-1)
        target = _mosi_soft_targets(
            labels.reshape(-1).to(dtype=prediction.dtype)
        )
        if not bool(selected.any()):
            return prediction.sum() * 0.0
        return torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[selected], target[selected]
        )
    if task in ("classification", "binary"):
        if task_regression_loss != "mse":
            raise ValueError(
                "task SmoothL1 is only valid for continuous regression"
            )
        flat_logits = logits.transpose(0, 1).reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1).long()
        if task == "binary":
            continuous_labels = labels.reshape(-1)
            selected = selected & continuous_labels.ne(0)
            if not bool(selected.any()):
                return flat_logits.sum() * 0.0
            flat_labels = continuous_labels.gt(0).long()
        return torch.nn.functional.cross_entropy(
            flat_logits[selected], flat_labels[selected]
        )
    prediction = logits.transpose(0, 1).reshape(-1)
    target = labels.reshape(-1).to(dtype=prediction.dtype)
    if task_regression_loss == "mse":
        return torch.nn.functional.mse_loss(
            prediction[selected], target[selected]
        )
    return torch.nn.functional.smooth_l1_loss(
        prediction[selected],
        target[selected],
        beta=task_smooth_l1_beta,
    )


def _jepa_rate_weight(rate: float, mode: str) -> float:
    if mode == "uniform":
        return 1.0
    if mode == "sparsity-budget":
        active_rates = tuple(value for value in MISSING_RATES if value > 0)
        normalizer = sum(1.0 + value for value in active_rates) / len(
            active_rates
        )
        return (1.0 + float(rate)) / normalizer
    raise ValueError("unsupported jepa_rate_weighting: {}".format(mode))


def _metrics(
    dataset: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    mosi_task_mode: str = "regression",
) -> Dict[str, float]:
    task = _resolve_task_contract(dataset, mosi_task_mode)["task"]
    if task in ("classification", "binary", "soft-ordinal"):
        return {
            "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "accuracy": float(accuracy_score(labels, predictions)),
        }
    nonzero = labels != 0
    binary_labels = labels[nonzero] > 0
    binary_predictions = predictions[nonzero] > 0
    correlation = (
        float(np.corrcoef(labels, predictions)[0, 1])
        if labels.size >= 2 and np.std(labels) > 0 and np.std(predictions) > 0
        else 0.0
    )
    return {
        "weighted_f1": float(
            f1_score(binary_labels, binary_predictions, average="weighted")
        ),
        "macro_f1": float(
            f1_score(binary_labels, binary_predictions, average="macro")
        ),
        "accuracy": float(accuracy_score(binary_labels, binary_predictions)),
        "mae": float(np.mean(np.abs(labels - predictions))),
        "correlation": correlation,
        "prediction_std": float(np.std(predictions)),
        "predicted_sign_count": int(np.unique(binary_predictions).size),
    }


def _collect_predictions(
    dataset: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    umask: torch.Tensor,
    mosi_task_mode: str = "regression",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task = _resolve_task_contract(dataset, mosi_task_mode)["task"]
    if task in ("classification", "binary"):
        predicted = logits.argmax(dim=-1).transpose(0, 1)
    elif task == "soft-ordinal":
        predicted = logits.squeeze(-1).transpose(0, 1).gt(0).long()
    else:
        predicted = logits.squeeze(-1).transpose(0, 1)
    selected = umask.bool()
    if task in ("binary", "soft-ordinal"):
        selected = selected & labels.ne(0)
        metric_labels = labels.gt(0).long()
    else:
        metric_labels = labels
    return (
        predicted[selected].detach().cpu().numpy(),
        metric_labels[selected].detach().cpu().numpy(),
        labels[selected].detach().cpu().numpy(),
    )


def train_epoch(
    model: MissingM3GraphModel,
    loader: Iterable[Sequence[object]],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    schedules: Mapping[float, ConversationMaskSchedule],
    epoch: int,
    dimensions: tuple[int, int, int],
    device: torch.device,
) -> Dict[str, float]:
    if (
        config.train_rate_mode == "stratified"
        and config.jepa_rate_weighting != "uniform"
    ):
        raise ValueError(
            "stratified train_rate_mode requires uniform jepa_rate_weighting"
        )
    if config.training_objective not in _TRAINING_OBJECTIVES:
        raise ValueError("unsupported training_objective")
    train_emotion = config.training_objective in {
        "joint",
        "emotion-only",
        "frozen-completion",
    }
    train_jepa = config.training_objective in {"joint", "jepa-only"}
    model.train()
    predictor = getattr(model, "missing_predictor", None)
    mmoe = getattr(predictor, "mmoe", None)
    if mmoe is not None and train_jepa:
        mmoe.reset_routing_statistics()
    rate_schedule = BalancedBatchRateSchedule()
    fixed_rate = _fixed_missing_rate(config)
    epoch_size = None
    if config.train_rate_mode == "stratified":
        sampler = getattr(loader, "sampler", None)
        if sampler is not None:
            try:
                epoch_size = len(sampler)
            except TypeError:
                epoch_size = None
        if epoch_size is None:
            epoch_size = len(loader.dataset)
    conversations_seen = 0
    losses: list[float] = []
    cls_losses: list[float] = []
    jepa_losses: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    rate_counts = {rate: 0 for rate in MISSING_RATES}
    source_conversation_count = 0
    masked_view_count = 0
    model_forward_count = 0
    rate_conversation_counts = {rate: 0 for rate in MISSING_RATES}
    realized_missing = {rate: [0, 0] for rate in MISSING_RATES}
    rate_valid_utterance_counts = {rate: 0 for rate in MISSING_RATES}
    rate_jepa_target_counts = {rate: 0 for rate in MISSING_RATES}
    assignment_digest = hashlib.sha256()
    target_count = 0
    optimizer_steps = 0
    skipped_optimizer_batches = 0
    for batch_index, raw in enumerate(loader):
        data = _move_batch(raw, device)
        batch_size = len(data[-1])
        conversation_rates = None
        if config.train_rate_mode == "all":
            rates = MISSING_RATES
            optimizer.zero_grad(set_to_none=True)
            rate_views = (
                (rate, _prepare_view(data, schedules[rate], epoch, dimensions))
                for rate in rates
            )
        elif config.train_rate_mode == "cyclic":
            rate = rate_schedule.rate_for(epoch, batch_index)
            view = _prepare_view(data, schedules[rate], epoch, dimensions)
            optimizer.zero_grad(set_to_none=True)
            rate_views = ((rate, view),)
        elif config.train_rate_mode == "fixed":
            rate = fixed_rate
            view = _prepare_view(data, schedules[rate], epoch, dimensions)
            optimizer.zero_grad(set_to_none=True)
            rate_views = ((rate, view),)
        elif config.train_rate_mode == "stratified":
            conversation_ids = tuple(str(value) for value in data[-1])
            assignment = stratified_rates_for_batch(
                MISSING_RATES,
                master_seed=config.seed,
                dataset=config.dataset,
                fold=config.fold,
                epoch=epoch,
                batch_index=batch_index,
                epoch_size=epoch_size,
                conversations_seen=conversations_seen,
                conversation_ids=conversation_ids,
            )
            view = _prepare_stratified_view(
                data,
                schedules,
                assignment.rates,
                epoch,
                dimensions,
            )
            optimizer.zero_grad(set_to_none=True)
            rate_views = ((None, view),)
            conversation_rates = assignment.rates
            assignment_digest.update(b"\0")
            assignment_digest.update(assignment.assignment_hash.encode("ascii"))
            for conversation_index, rate in enumerate(assignment.rates):
                rate_conversation_counts[rate] += 1
                valid_availability = view["availability"][:, conversation_index][
                    view["umask"][conversation_index].bool()
                ]
                rate_valid_utterance_counts[rate] += int(
                    view["umask"][conversation_index].bool().sum().item()
                )
                realized_missing[rate][0] += int(
                    valid_availability.eq(0).sum().item()
                )
                realized_missing[rate][1] += int(valid_availability.numel())
            conversations_seen += batch_size
        else:
            raise ValueError(
                "train_rate_mode must be 'cyclic', 'all', 'fixed', or 'stratified'"
            )
        teacher = None
        batch_has_backward = False
        for rate, view in rate_views:
            if rate is not None:
                rate_counts[rate] += 1
                rate_conversation_counts[rate] += batch_size
                valid_rows = view["umask"].transpose(0, 1).bool()
                valid_availability = view["availability"][valid_rows]
                rate_valid_utterance_counts[rate] += int(
                    valid_rows.sum().item()
                )
                realized_missing[rate][0] += int(
                    valid_availability.eq(0).sum().item()
                )
                realized_missing[rate][1] += int(valid_availability.numel())
            if config.train_rate_mode == "all" and train_jepa and teacher is None:
                with torch.no_grad():
                    teacher = model.encode_teacher_targets([view["complete"]])
            logits, _, _, predictions = model(
                [view["incomplete"]],
                view["availability"],
                view["qmask"],
                view["umask"],
                view["lengths"],
                predict_missing=train_jepa,
            )
            model_forward_count += 1
            zero = logits.sum() * 0.0
            cls = (
                _task_loss(
                    config.dataset,
                    logits,
                    view["labels"],
                    view["umask"],
                    config.mosi_task_mode,
                    config.task_regression_loss,
                    config.task_smooth_l1_beta,
                )
                if train_emotion
                else zero
            )
            if train_jepa:
                valid_target_mask = (
                    predictions.target_mask
                    & view["umask"].transpose(0, 1).bool().unsqueeze(-1)
                )
                if rate is None:
                    for conversation_index, conversation_rate in enumerate(
                        conversation_rates
                    ):
                        rate_jepa_target_counts[conversation_rate] += int(
                            valid_target_mask[:, conversation_index].sum().item()
                        )
                else:
                    rate_jepa_target_counts[rate] += int(
                        valid_target_mask.sum().item()
                    )
                if teacher is None:
                    with torch.no_grad():
                        teacher = model.encode_teacher_targets([view["complete"]])
                jepa = missing_m3_loss(
                    predictions,
                    teacher,
                    temperature=config.temperature,
                    regression_aggregation=config.jepa_regression_aggregation,
                    contrastive_prediction_source=config.jepa_contrastive_source,
                )
                jepa_rate_weight = (
                    1.0
                    if rate is None
                    else _jepa_rate_weight(rate, config.jepa_rate_weighting)
                )
            else:
                jepa = MissingM3Loss(zero, zero, zero, 0)
                jepa_rate_weight = 0.0
            if config.training_objective == "joint":
                loss = cls + config.jepa_weight * jepa_rate_weight * jepa.total
            elif config.training_objective == "jepa-only":
                loss = jepa_rate_weight * jepa.total
            elif config.training_objective in {
                "emotion-only",
                "frozen-completion",
            }:
                loss = cls
            else:
                raise ValueError("unsupported training_objective")
            if not bool(torch.isfinite(loss.detach())):
                raise ValueError("training loss must be finite")
            has_supervision = not (
                config.training_objective == "jepa-only"
                and jepa.target_count == 0
            )
            if has_supervision:
                if config.train_rate_mode == "all":
                    (loss / len(MISSING_RATES)).backward()
                else:
                    loss.backward()
                batch_has_backward = True
            predicted, expected, _ = _collect_predictions(
                config.dataset,
                logits,
                view["labels"],
                view["umask"],
                config.mosi_task_mode,
            )
            all_predictions.append(predicted)
            all_labels.append(expected)
            losses.append(float(loss.detach()))
            cls_losses.append(float(cls.detach()))
            jepa_losses.append(float(jepa.total.detach()))
            target_count += jepa.target_count
        source_conversation_count += batch_size
        masked_view_count += batch_size * (
            len(MISSING_RATES) if config.train_rate_mode == "all" else 1
        )
        if not batch_has_backward:
            skipped_optimizer_batches += 1
            continue
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                config.gradient_clip_norm,
            )
        optimizer.step()
        optimizer_steps += 1
        if train_jepa:
            model.update_teacher(config.ema_tau)
    if (
        config.train_rate_mode == "stratified"
        and conversations_seen != epoch_size
    ):
        raise RuntimeError(
            "stratified epoch expected {} conversations but observed {}".format(
                epoch_size,
                conversations_seen,
            )
        )
    metrics = _metrics(
        config.dataset,
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
        config.mosi_task_mode,
    )
    routing_record = {}
    if mmoe is not None and train_jepa:
        routing = mmoe.routing_statistics()
        for branch_index, branch_name in enumerate(("regression", "contrastive")):
            routing_record[branch_name] = {
                "selection_count": routing["selection_count"][branch_index]
                .cpu()
                .tolist(),
                "probability_mass": routing["probability_mass"][branch_index]
                .cpu()
                .tolist(),
                "usage": routing["usage"][branch_index].cpu().tolist(),
                "entropy": float(routing["entropy"][branch_index].cpu()),
                "token_count": int(routing["token_count"][branch_index].cpu()),
            }
    return {
        **metrics,
        "loss": float(np.mean(losses)),
        "classification_loss": float(np.mean(cls_losses)),
        "jepa_loss": float(np.mean(jepa_losses)),
        "jepa_target_count": int(target_count),
        "rate_batch_counts": {str(rate): count for rate, count in rate_counts.items()},
        "source_conversation_count": source_conversation_count,
        "masked_view_count": masked_view_count,
        "model_forward_count": model_forward_count,
        "rate_conversation_counts": {
            str(rate): rate_conversation_counts[rate] for rate in MISSING_RATES
        },
        "rate_realized_missing_fraction": {
            str(rate): (
                realized_missing[rate][0] / realized_missing[rate][1]
                if realized_missing[rate][1]
                else None
            )
            for rate in MISSING_RATES
        },
        "rate_valid_utterance_counts": {
            str(rate): rate_valid_utterance_counts[rate]
            for rate in MISSING_RATES
        },
        "rate_missing_modality_counts": {
            str(rate): realized_missing[rate][0] for rate in MISSING_RATES
        },
        "rate_modality_element_counts": {
            str(rate): realized_missing[rate][1] for rate in MISSING_RATES
        },
        "rate_jepa_target_counts": {
            str(rate): rate_jepa_target_counts[rate]
            for rate in MISSING_RATES
        },
        "stratified_assignment_hash": (
            assignment_digest.hexdigest()
            if config.train_rate_mode == "stratified"
            else None
        ),
        "stratified_rate_algorithm": (
            STRATIFIED_RATE_ALGORITHM
            if config.train_rate_mode == "stratified"
            else None
        ),
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_batches": skipped_optimizer_batches,
        "routing": routing_record,
        "training_objective": config.training_objective,
    }


@torch.no_grad()
def evaluate_rate(
    model: MissingM3GraphModel,
    loader: Iterable[Sequence[object]],
    schedule: ConversationMaskSchedule,
    dataset: str,
    dimensions: tuple[int, int, int],
    device: torch.device,
    collect: bool,
    mosi_task_mode: str = "regression",
    task_regression_loss: str = "mse",
    task_smooth_l1_beta: float = 1.0,
) -> tuple[Dict[str, float], Dict[str, np.ndarray] | None]:
    model.eval()
    task = _resolve_task_contract(dataset, mosi_task_mode)["task"]
    losses: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_continuous_labels: list[np.ndarray] = []
    all_signed_logits: list[np.ndarray] = []
    all_availability: list[np.ndarray] = []
    all_full_availability: list[np.ndarray] = []
    for raw in loader:
        data = _move_batch(raw, device)
        view = _prepare_view(data, schedule, epoch=0, dimensions=dimensions)
        logits, _, _, predictions = model(
            [view["incomplete"]],
            view["availability"],
            view["qmask"],
            view["umask"],
            view["lengths"],
            predict_missing=False,
        )
        if predictions is not None:
            raise RuntimeError("inference path must not return missing predictions")
        loss = _task_loss(
            dataset,
            logits,
            view["labels"],
            view["umask"],
            mosi_task_mode,
            task_regression_loss,
            task_smooth_l1_beta,
        )
        predicted, expected, continuous = _collect_predictions(
            dataset,
            logits,
            view["labels"],
            view["umask"],
            mosi_task_mode,
        )
        all_predictions.append(predicted)
        all_labels.append(expected)
        all_continuous_labels.append(continuous)
        if task == "soft-ordinal" and collect:
            metric_selected = view["umask"].bool() & view["labels"].ne(0)
            signed_logits = logits.squeeze(-1).transpose(0, 1)
            all_signed_logits.append(
                signed_logits[metric_selected].cpu().numpy()
            )
        losses.append(float(loss))
        if collect:
            valid = view["umask"].T.bool()
            all_full_availability.append(
                view["availability"][valid].cpu().numpy()
            )
            metric_availability = view["availability"].transpose(0, 1)
            selected = view["umask"].bool()
            if task in ("binary", "soft-ordinal"):
                selected = selected & view["labels"].ne(0)
            all_availability.append(
                metric_availability[selected].cpu().numpy()
            )
    predictions_array = np.concatenate(all_predictions)
    labels_array = np.concatenate(all_labels)
    continuous_labels_array = np.concatenate(all_continuous_labels)
    metrics = {
        **_metrics(dataset, labels_array, predictions_array, mosi_task_mode),
        "loss": float(np.mean(losses)),
    }
    artifacts = None
    if collect:
        availability_array = np.concatenate(all_availability)
        artifacts = {
            "predictions": predictions_array,
            "labels": labels_array,
            "availability": availability_array,
        }
        if task in ("binary", "soft-ordinal"):
            artifacts["continuous_labels"] = continuous_labels_array
        if task == "soft-ordinal":
            artifacts["signed_logits"] = np.concatenate(all_signed_logits)
        full_availability_array = np.concatenate(all_full_availability)
        metrics["mask_sha256"] = _sha256_tensor(
            torch.from_numpy(full_availability_array)
        )
    return metrics, artifacts


def run_experiment(
    config_value: TrainConfig,
    audio_root: str,
    text_root: str,
    visual_root: str,
    output_dir: str | Path,
) -> Dict[str, object]:
    protocol_rates = _protocol_rates(config_value)
    if config_value.training_objective not in _TRAINING_OBJECTIVES:
        raise ValueError("unsupported training_objective")
    if config_value.pretrained_learning_rate is not None:
        if config_value.initial_backbone_checkpoint is None:
            raise ValueError(
                "pretrained_learning_rate requires initial_backbone_checkpoint"
            )
        if config_value.training_objective != "joint":
            raise ValueError(
                "pretrained_learning_rate is only valid for joint training"
            )
        if not 0.0 < config_value.pretrained_learning_rate < config_value.learning_rate:
            raise ValueError(
                "pretrained_learning_rate must be lower than learning_rate"
            )
    frozen_completion = config_value.training_objective == "frozen-completion"
    if frozen_completion and config_value.initial_backbone_checkpoint is None:
        raise ValueError(
            "frozen-completion requires initial_backbone_checkpoint"
        )
    if frozen_completion and not config_value.classification_completion:
        raise ValueError("frozen-completion requires classification_completion")
    if (
        config_value.initial_backbone_checkpoint is not None
        and config_value.training_objective
        not in {"joint", "emotion-only", "frozen-completion"}
    ):
        raise ValueError(
            "initial_backbone_checkpoint is only valid for joint, emotion-only, "
            "or frozen-completion training"
        )
    if config_value.training_objective == "jepa-only":
        if config_value.evaluate_test:
            raise ValueError(
                "jepa-only pretraining requires skip-test-evaluation"
            )
        if config_value.checkpoint_selection != "validation":
            raise ValueError(
                "jepa-only pretraining uses a fixed-final checkpoint"
            )
    if config_value.checkpoint_selection not in ("validation", "test-oracle"):
        raise ValueError(
            "checkpoint_selection must be 'validation' or 'test-oracle'"
        )
    shape = _resolve_task_contract(
        config_value.dataset, config_value.mosi_task_mode
    )
    if not 1 <= config_value.fold <= int(shape["num_folds"]):
        raise ValueError("fold is outside the dataset fold range")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_run_config(output / "config.json", config_value)
    set_random_seed(config_value.seed)
    device = torch.device(config_value.device)
    loaders = get_loaders(
        audio_root=audio_root,
        text_root=text_root,
        video_root=visual_root,
        num_folder=int(shape["num_folds"]),
        dataset=config_value.dataset,
        batch_size=config_value.batch_size,
        num_workers=0,
        seed=config_value.seed,
        validation_fraction=config_value.validation_fraction,
        evaluation_protocol=config_value.evaluation_protocol,
    )
    train_loaders, validation_loaders, test_loaders, adim, tdim, vdim = loaders
    fold_index = config_value.fold - 1
    train_loader = train_loaders[fold_index]
    validation_loader = validation_loaders[fold_index]
    test_loader = test_loaders[fold_index]
    dimensions = (adim, tdim, vdim)
    model_seed = SeedBundle(config_value.seed).derive(
        "missing_m3_model_init:fold:5"
    )
    set_random_seed(model_seed)
    model = MissingM3GraphModel(
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
        target_private_rank=config_value.target_private_rank,
        classification_completion=config_value.classification_completion,
        representation_type=config_value.representation_type,
        node_interaction_residual=config_value.node_interaction_residual,
        readout_type=config_value.readout_type,
        readout_rank=config_value.readout_rank,
        recurrent_padding_mode=config_value.recurrent_padding_mode,
        postgraph_sequence_mode=config_value.postgraph_sequence_mode,
        graph_message_calibration=config_value.graph_message_calibration,
        graph_second_layer=config_value.graph_second_layer,
    ).to(device)
    initialization = None
    frozen_probe = None
    frozen_hash_before = None
    if config_value.initial_backbone_checkpoint is not None:
        initialization = _load_inference_backbone_checkpoint(
            model,
            config_value.initial_backbone_checkpoint,
            include_jepa_modules=config_value.training_objective
            in {"joint", "frozen-completion"},
        )
    if frozen_completion:
        frozen_probe = _configure_frozen_completion_probe(model)
        frozen_hash_before = _parameter_subset_sha256(
            model,
            frozen_probe["frozen_parameter_names"],
        )
    optimizer_groups, optimizer_group_provenance = _optimizer_parameter_groups(
        model, config_value
    )
    optimizer = torch.optim.Adam(
        optimizer_groups,
        weight_decay=config_value.weight_decay,
    )
    train_schedules = _schedules(config_value, "train")
    validation_schedules = (
        _schedules(config_value, "validation")
        if config_value.checkpoint_selection == "validation"
        else None
    )
    test_schedules = _schedules(config_value, "test")
    history: list[Dict[str, object]] = []
    jepa_pretraining = config_value.training_objective == "jepa-only"
    best_score: float | None = None if jepa_pretraining else -math.inf
    best_epoch = 0
    best_state = None
    for epoch in range(config_value.epochs):
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            config_value,
            train_schedules,
            epoch,
            dimensions,
            device,
        )
        if jepa_pretraining:
            history.append({"epoch": epoch + 1, "train": train_metrics})
            _write_json(output / "history.json", history)
            print(
                "epoch={:03d} jepa={:.4f}".format(
                    epoch + 1, train_metrics["jepa_loss"]
                ),
                flush=True,
            )
            continue
        selection_loader = (
            validation_loader
            if config_value.checkpoint_selection == "validation"
            else test_loader
        )
        selection_schedules = (
            validation_schedules
            if config_value.checkpoint_selection == "validation"
            else test_schedules
        )
        if selection_schedules is None:
            raise RuntimeError("selection schedules were not initialized")
        selection_metrics: Dict[float, Dict[str, float]] = {}
        for rate in protocol_rates:
            selection_metrics[rate], _ = evaluate_rate(
                model,
                selection_loader,
                selection_schedules[rate],
                config_value.dataset,
                dimensions,
                device,
                collect=False,
                mosi_task_mode=config_value.mosi_task_mode,
                task_regression_loss=config_value.task_regression_loss,
                task_smooth_l1_beta=config_value.task_smooth_l1_beta,
            )
        selection_mean = sum(
            float(selection_metrics[rate]["weighted_f1"])
            for rate in protocol_rates
        ) / len(protocol_rates)
        selection_key = (
            "validation"
            if config_value.checkpoint_selection == "validation"
            else "test_oracle"
        )
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            selection_key: {
                str(rate): value for rate, value in selection_metrics.items()
            },
            selection_key + "_mean_weighted_f1": selection_mean,
        }
        history.append(record)
        _write_json(output / "history.json", history)
        print(
            "epoch={:03d} train_wf1={:.4f} {}_wf1={:.4f} cls={:.4f} jepa={:.4f}".format(
                epoch + 1,
                train_metrics["weighted_f1"],
                selection_key,
                selection_mean,
                train_metrics["classification_loss"],
                train_metrics["jepa_loss"],
            ),
            flush=True,
        )
        if best_score is None:
            raise RuntimeError("emotion checkpoint score was not initialized")
        if selection_mean > best_score:
            best_score = selection_mean
            best_epoch = epoch + 1
            best_state = _state_to_cpu(model)
            _save_best_checkpoint(
                output / "best.pt",
                model_state=best_state,
                config_value=config_value,
                epoch=best_epoch,
                validation_mean_weighted_f1=best_score,
                selection_split=config_value.checkpoint_selection,
            )
    if jepa_pretraining and config_value.epochs > 0:
        best_epoch = config_value.epochs
        best_state = _state_to_cpu(model)
        _save_best_checkpoint(
            output / "best.pt",
            model_state=best_state,
            config_value=config_value,
            epoch=best_epoch,
            validation_mean_weighted_f1=None,
            selection_split="fixed-final",
        )
    if best_state is None:
        raise RuntimeError("no best checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    frozen_integrity = None
    if frozen_probe is not None:
        frozen_hash_after = _parameter_subset_sha256(
            model,
            frozen_probe["frozen_parameter_names"],
        )
        if frozen_hash_after != frozen_hash_before:
            raise RuntimeError("frozen completion backbone changed during training")
        frozen_integrity = {
            "trainable_parameter_count": frozen_probe[
                "trainable_parameter_count"
            ],
            "frozen_parameter_count": frozen_probe["frozen_parameter_count"],
            "frozen_parameter_sha256_before": frozen_hash_before,
            "frozen_parameter_sha256_after": frozen_hash_after,
        }
    test_metrics: Dict[str, Dict[str, float]] = {}
    mask_hashes: Dict[str, str] = {}
    if config_value.evaluate_test:
        for rate in protocol_rates:
            metrics, artifacts = evaluate_rate(
                model,
                test_loader,
                test_schedules[rate],
                config_value.dataset,
                dimensions,
                device,
                collect=True,
                mosi_task_mode=config_value.mosi_task_mode,
                task_regression_loss=config_value.task_regression_loss,
                task_smooth_l1_beta=config_value.task_smooth_l1_beta,
            )
            if artifacts is None:
                raise RuntimeError("test artifacts were not collected")
            rate_key = format(rate, ".1f")
            test_metrics[rate_key] = metrics
            mask_hashes[rate_key] = str(metrics["mask_sha256"])
            np.savez_compressed(
                output
                / ("predictions_miss_" + rate_key.replace(".", "p") + ".npz"),
                **artifacts,
            )
    selection_split = (
        "fixed-final" if jepa_pretraining else config_value.checkpoint_selection
    )
    result: Dict[str, object] = {
        "best_epoch": best_epoch,
        "selection_split": selection_split,
        "best_selection_mean_weighted_f1": best_score,
        "best_validation_mean_weighted_f1": (
            best_score
            if selection_split == "validation"
            else None
        ),
        "test": test_metrics,
        "mask_sha256": mask_hashes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "ema_steps": model.ema_step,
        "evaluation_stage": (
            "jepa-pretrain-only"
            if jepa_pretraining
            else (
                "train-test-oracle"
                if config_value.checkpoint_selection == "test-oracle"
                else (
                    "train-validation-test"
                    if config_value.evaluate_test
                    else "train-validation-only"
                )
            )
        ),
        "training_objective": config_value.training_objective,
        "backbone_initialization": initialization,
        "frozen_completion_integrity": frozen_integrity,
        "optimizer_parameter_groups": optimizer_group_provenance,
        "jepa_regression_aggregation": (
            config_value.jepa_regression_aggregation
        ),
        "jepa_contrastive_source": config_value.jepa_contrastive_source,
        "recurrent_padding_mode": config_value.recurrent_padding_mode,
        "task_regression_loss": config_value.task_regression_loss,
        "task_smooth_l1_beta": config_value.task_smooth_l1_beta,
        "postgraph_sequence_mode": config_value.postgraph_sequence_mode,
        "jepa_rate_weighting": config_value.jepa_rate_weighting,
        "graph_message_calibration": config_value.graph_message_calibration,
        "graph_second_layer": config_value.graph_second_layer,
        "train_missing_rate": _fixed_missing_rate(config_value),
        "selection_missing_rates": list(protocol_rates),
        **_readout_provenance(model),
    }
    _write_json(output / "metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("IEMOCAPFour", "IEMOCAPSix", "CMUMOSI", "CMUMOSEI"),
        default="IEMOCAPSix",
    )
    parser.add_argument(
        "--mosi-task-mode",
        choices=("regression", "binary", "soft-ordinal"),
        default="regression",
    )
    parser.add_argument(
        "--graph-branch-mode",
        choices=("both", "temporal-only", "speaker-only"),
        default="both",
    )
    parser.add_argument("--audio-feature", required=True)
    parser.add_argument("--text-feature", required=True)
    parser.add_argument("--video-feature", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--train-rate-mode",
        choices=("cyclic", "all", "fixed", "stratified"),
        default="cyclic",
    )
    parser.add_argument("--train-missing-rate", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--mmoe-variant",
        choices=("dual-gate", "paper-faithful"),
        default="dual-gate",
    )
    parser.add_argument("--target-private-rank", type=int, default=0)
    parser.add_argument("--classification-completion", action="store_true")
    parser.add_argument(
        "--representation-type",
        choices=("slot", "track"),
        default="slot",
    )
    parser.add_argument("--node-interaction-residual", action="store_true")
    parser.add_argument(
        "--readout-type",
        choices=(
            "shared",
            "availability-low-rank",
            "shared-low-rank-parammatch",
            "availability-affine",
        ),
        default="shared",
    )
    parser.add_argument("--readout-rank", type=int, default=8)
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
    parser.add_argument(
        "--graph-second-layer",
        choices=("graphconv", "identity"),
        default="graphconv",
    )
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation", "test-oracle"),
        default="validation",
    )
    parser.add_argument(
        "--training-objective",
        choices=(
            "joint",
            "jepa-only",
            "emotion-only",
            "frozen-completion",
        ),
        default="joint",
    )
    parser.add_argument("--initial-backbone-checkpoint", default=None)
    parser.add_argument(
        "--fusion-type",
        choices=("mean", "slot", "raw-residual", "text-anchor-residual"),
        default="mean",
    )
    parser.add_argument("--local-context-residual", action="store_true")
    parser.add_argument("--local-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--local-fusion-dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained-lr", type=float, default=None)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument(
        "--jepa-regression-aggregation",
        choices=("target", "utterance"),
        default="target",
    )
    parser.add_argument(
        "--jepa-contrastive-source",
        choices=("contrastive", "regression"),
        default="contrastive",
    )
    parser.add_argument("--ema-tau", type=float, default=0.996)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--windowp", type=int, default=2)
    parser.add_argument("--windowf", type=int, default=2)
    parser.add_argument("--time-attn", action="store_true")
    parser.add_argument("--evaluation-protocol", choices=("official", "strict"), default="official")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-threads", type=int, default=6)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.num_threads)
    config_value = TrainConfig(
        dataset=args.dataset,
        fold=args.fold,
        seed=args.seed,
        window_past=args.windowp,
        window_future=args.windowf,
        hidden=args.hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        train_rate_mode=args.train_rate_mode,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.l2,
        latent_dim=args.latent_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        fusion_type=args.fusion_type,
        local_context_residual=args.local_context_residual,
        local_fusion_hidden_dim=args.local_fusion_hidden_dim,
        local_fusion_dropout=args.local_fusion_dropout,
        jepa_weight=args.jepa_weight,
        temperature=args.temperature,
        jepa_regression_aggregation=args.jepa_regression_aggregation,
        jepa_contrastive_source=args.jepa_contrastive_source,
        ema_tau=args.ema_tau,
        gradient_clip_norm=args.gradient_clip_norm,
        time_attention=args.time_attn,
        evaluation_protocol=args.evaluation_protocol,
        validation_fraction=args.validation_fraction,
        device=args.device,
        mosi_task_mode=args.mosi_task_mode,
        graph_branch_mode=args.graph_branch_mode,
        mmoe_variant=args.mmoe_variant,
        target_private_rank=args.target_private_rank,
        classification_completion=args.classification_completion,
        representation_type=args.representation_type,
        node_interaction_residual=args.node_interaction_residual,
        readout_type=args.readout_type,
        readout_rank=args.readout_rank,
        evaluate_test=not args.skip_test_evaluation,
        recurrent_padding_mode=args.recurrent_padding_mode,
        task_regression_loss=args.task_regression_loss,
        task_smooth_l1_beta=args.task_smooth_l1_beta,
        postgraph_sequence_mode=args.postgraph_sequence_mode,
        jepa_rate_weighting=args.jepa_rate_weighting,
        graph_message_calibration=args.graph_message_calibration,
        graph_second_layer=args.graph_second_layer,
        fixed_missing_rate=args.train_missing_rate,
        checkpoint_selection=args.checkpoint_selection,
        training_objective=args.training_objective,
        initial_backbone_checkpoint=args.initial_backbone_checkpoint,
        pretrained_learning_rate=args.pretrained_lr,
    )
    feature_root = args.feature_root or config.PATH_TO_FEATURES[config_value.dataset]
    roots = [
        os.path.join(feature_root, name)
        for name in (args.audio_feature, args.text_feature, args.video_feature)
    ]
    if not all(os.path.exists(root) for root in roots):
        raise FileNotFoundError("one or more feature roots do not exist")
    run_experiment(config_value, *roots, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
