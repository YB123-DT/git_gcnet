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


_OBSERVED_PATTERN_IDS = (1, 2, 3, 4, 5, 6, 7)


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
    emotion_loss_mode: str = "sample-mean"
    group_dro_eta: float = 0.1
    graph_message_calibration: str = "none"
    graph_second_layer: str = "graphconv"
    postgraph_bilstm_ablation: str = "none"
    fixed_missing_rate: float | None = None
    checkpoint_selection: str = "validation"
    jepa_contrastive_source: str = "contrastive"
    training_objective: str = "joint"
    initial_backbone_checkpoint: str | None = None
    pretrained_learning_rate: float | None = None
    backbone_type: str = "gcnet"
    osram_output_dim: int = 500
    osram_num_heads: int = 4
    osram_key_dim: int = 32
    osram_value_dim: int = 32
    osram_read_ridge: float = 1e-3
    osram_write_ridge: float = 1e-3
    osram_predictor_mode: str = "structured"
    osram_ablation: str = "full"
    osram_emotion_ablation: str = "full"
    osram_query_availability: bool = True
    osram_bidirectional: bool = True
    osram_forward_slot_reuse: bool = False
    osram_write_step: float = 1.0
    osram_readout_fusion: str = "flat"
    completion_path: str = "none"
    b2_base_checkpoint: str | None = None
    b2_pretrain_checkpoint: str | None = None
    teacher_mode: str = "ema"
    teacher_checkpoint: str | None = None
    target_space: str = "all-modalities"
    text_subspace_checkpoint: str | None = None
    text_core: bool = False
    uniform_forced_text_probability: float = 0.25

    def __post_init__(self) -> None:
        if self.target_space not in {"all-modalities", "full-text", "predictable-subspace"}:
            raise ValueError("unsupported target_space")
        if self.emotion_loss_mode not in {"sample-mean", "pattern-balanced", "pattern-groupdro"}:
            raise ValueError("emotion_loss_mode must be sample-mean, pattern-balanced, or pattern-groupdro")
        if not math.isfinite(float(self.group_dro_eta)) or float(self.group_dro_eta) < 0:
            raise ValueError("group_dro_eta must be finite and nonnegative")
        forced_text_probability = float(self.uniform_forced_text_probability)
        if (not math.isfinite(forced_text_probability)
                or not 0.0 <= forced_text_probability <= 1.0):
            raise ValueError("uniform_forced_text_probability must be between zero and one")
        if (self.target_space == "predictable-subspace") != (self.text_subspace_checkpoint is not None):
            raise ValueError("text_subspace_checkpoint is required only for predictable-subspace")
        if self.target_space != "all-modalities" and (
                self.training_objective != "joint" or self.teacher_mode != "pretrained-frozen"
                or self.jepa_weight != .1 or self.temperature != .03
                or self.jepa_rate_weighting != "uniform"
                or self.jepa_regression_aggregation != "target"
                or self.jepa_contrastive_source != "contrastive"
                or self.b2_base_checkpoint is not None or self.b2_pretrain_checkpoint is not None):
            raise ValueError("text target spaces require frozen-teacher joint training with unchanged JEPA weights and no B2")
        if self.text_core and (
            self.training_objective != "emotion-only"
            or self.backbone_type != "osram"
            or self.osram_bidirectional
            or self.osram_forward_slot_reuse
            or self.osram_write_step != 0.6
            or self.osram_readout_fusion != "flat"
            or self.fusion_type != "mean"
            or self.completion_path != "none"
            or self.classification_completion
            or self.target_space != "all-modalities"
            or self.teacher_mode != "ema"
            or self.teacher_checkpoint is not None
            or self.emotion_loss_mode != "sample-mean"
        ):
            raise ValueError(
                "text-core requires emotion-only causal eta=.6 mean/Flat OSRAM "
                "with no JEPA, completion, teacher transfer, or group weighting"
            )
        if self.teacher_mode not in {"ema", "pretrained-frozen"}:
            raise ValueError("teacher_mode must be ema or pretrained-frozen")
        if self.teacher_mode == "ema" and self.teacher_checkpoint is not None:
            raise ValueError("teacher_checkpoint requires pretrained-frozen teacher_mode")
        if self.training_objective == "joint-reg-only":
            if (self.teacher_mode != "pretrained-frozen"
                    or self.jepa_weight != 0.1
                    or self.target_space != "all-modalities"):
                raise ValueError(
                    "joint-reg-only requires pretrained-frozen teacher, "
                    "all-modalities targets, and jepa_weight=0.1"
                )
        if self.teacher_mode == "pretrained-frozen":
            if (not self.teacher_checkpoint
                    or self.training_objective not in {
                        "joint", "joint-reg-only", "emotion-only"
                    }
                    or self.backbone_type != "osram" or self.osram_bidirectional
                    or self.osram_write_step != .6 or self.osram_forward_slot_reuse
                    or self.fusion_type != "mean" or self.osram_readout_fusion != "flat"
                    or self.osram_predictor_mode != "structured"
                    or self.osram_ablation != "full" or self.osram_emotion_ablation != "full"
                    or self.local_context_residual or self.node_interaction_residual
                    or self.readout_type != "shared" or self.classification_completion
                    or self.completion_path != "none" or self.initial_backbone_checkpoint is not None):
                raise ValueError("pretrained-frozen requires a teacher checkpoint and joint/joint-reg-only/emotion-only causal mean/flat structured OSRAM at write step 0.6 without other interventions")
        if self.training_objective == "future-state":
            if (self.backbone_type != "osram" or self.osram_bidirectional
                    or self.osram_forward_slot_reuse or self.osram_write_step != 0.6
                    or self.osram_readout_fusion != "flat" or self.fusion_type != "mean"
                    or self.osram_ablation != "full" or self.osram_emotion_ablation != "full"
                    or not self.osram_query_availability or self.local_context_residual
                    or self.node_interaction_residual or self.readout_type != "shared"
                    or self.completion_path != "none" or self.classification_completion
                    or self.initial_backbone_checkpoint is not None
                    or self.jepa_rate_weighting != "uniform" or self.jepa_weight != 0.1):
                raise ValueError("future-state requires causal mean/flat OSRAM with write step 0.6, no ablation/completion/legacy transfer, and uniform 0.1 loss weighting")
        if self.training_objective == "write-state":
            if (self.backbone_type != "osram" or self.osram_bidirectional
                    or self.osram_forward_slot_reuse or self.osram_write_step != 0.6
                    or self.osram_readout_fusion != "flat"
                    or self.completion_path != "none" or self.classification_completion
                    or self.initial_backbone_checkpoint is not None
                    or self.jepa_rate_weighting != "uniform"):
                raise ValueError("write-state requires causal flat OSRAM with write step 0.6, no completion or legacy transfer, and uniform loss weighting")
        if self.training_objective == "complete-state":
            if (self.backbone_type != "osram" or self.osram_bidirectional
                    or self.osram_readout_fusion != "flat"
                    or self.completion_path != "none" or self.classification_completion
                    or self.initial_backbone_checkpoint is not None
                    or self.jepa_rate_weighting != "uniform"):
                raise ValueError("complete-state requires causal flat OSRAM without completion or legacy transfer, and uniform loss weighting")
        if self.osram_readout_fusion not in {
            "flat", "local-gated", "local-cross-attn", "modality-tracks",
            "modality-track-residual"
        }:
            raise ValueError(
                "osram_readout_fusion must be flat, local-gated, local-cross-attn, modality-tracks, or modality-track-residual"
            )
        if self.osram_readout_fusion != "flat":
            if self.backbone_type != "osram":
                raise ValueError(f"{self.osram_readout_fusion} requires the osram backbone")
        if self.osram_readout_fusion == "modality-track-residual":
            if (
                self.training_objective != "emotion-only"
                or self.initial_backbone_checkpoint is None
                or self.osram_bidirectional
                or self.osram_write_step != 0.6
                or self.osram_forward_slot_reuse
                or self.fusion_type != "mean"
                or self.completion_path != "none"
                or self.classification_completion
                or self.osram_ablation != "full"
                or self.osram_emotion_ablation != "full"
            ):
                raise ValueError(
                    "modality-track-residual requires a frozen no-JEPA emotion-only "
                    "causal mean/Flat OSRAM checkpoint at write step 0.6"
                )
        if self.checkpoint_selection == "test-oracle-per-rate":
            if self.train_rate_mode == "fixed" or not self.evaluate_test:
                raise ValueError("test-oracle-per-rate requires all eight rates and test evaluation")
        step = float(self.osram_write_step)
        if not math.isfinite(step) or not 0.0 <= step <= 1.0:
            raise ValueError("osram_write_step must be finite and between zero and one")


_TRAINING_OBJECTIVES = {
    "joint",
    "joint-reg-only",
    "complete-state",
    "write-state",
    "future-state",
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
_FROZEN_MODALITY_TRACK_RESIDUAL_TRAINABLE_PREFIXES = (
    "osram.modality_track_residual.",
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
    if mode not in ("regression", "binary", "three-class", "dual", "soft-ordinal"):
        raise ValueError("unsupported MOSI task mode: {}".format(mode))
    contract = _dataset_shape(dataset)
    if mode in ("binary", "three-class", "dual", "soft-ordinal"):
        if dataset != "CMUMOSI":
            raise ValueError(
                "{} task mode is only supported for CMUMOSI".format(mode)
            )
    if mode == "binary":
        contract.update(task="binary", num_classes=2)
    elif mode == "three-class":
        contract.update(task="three-class", num_classes=3)
    elif mode == "dual":
        contract.update(task="dual", num_classes=1)
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
    selection_protocol: str | None = None,
    text_subspace_provenance: Mapping[str, object] | None = None,
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
            "text_subspace_provenance": text_subspace_provenance,
            "epoch": epoch,
            "validation_mean_weighted_f1": validation_score,
            "selection_split": selection_split,
            "selection_mean_weighted_f1": validation_mean_weighted_f1,
            "selection_protocol": selection_protocol
            or (
                "8-rate-mean-test-oracle"
                if selection_split == "test-oracle"
                else "8-rate-mean-validation"
            ),
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
    load_all_shared_modules: bool = False,
    allow_missing_prefixes: Sequence[str] = (),
) -> Dict[str, object]:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu")
    source_config = checkpoint.get("config", {})
    source_objective = source_config.get("training_objective")
    if source_objective not in {"jepa-only", "emotion-only"}:
        raise ValueError(
            "initial backbone checkpoint must come from jepa-only or emotion-only training"
        )
    source_state = checkpoint.get("model")
    if not isinstance(source_state, Mapping):
        raise ValueError("initial backbone checkpoint has no model state")
    target_state = model.state_dict()
    excluded_prefixes = () if load_all_shared_modules else (
        _JOINT_FINETUNE_EXCLUDED_PREFIXES
        if include_jepa_modules
        else _STAGE2_EXCLUDED_PREFIXES
    )
    loaded_keys = []
    for key, target_value in target_state.items():
        if key.startswith(excluded_prefixes):
            continue
        if key not in source_state:
            if key.startswith(tuple(allow_missing_prefixes)):
                continue
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
        "load_all_shared_modules": load_all_shared_modules,
        "allowed_missing_prefixes": list(allow_missing_prefixes),
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


def _configure_frozen_modality_track_residual_probe(
    model: MissingM3GraphModel,
) -> Dict[str, object]:
    trainable_names = []
    frozen_names = []
    trainable_count = 0
    frozen_count = 0
    for name, parameter in model.named_parameters():
        trainable = name.startswith(_FROZEN_MODALITY_TRACK_RESIDUAL_TRAINABLE_PREFIXES)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_count += parameter.numel()
        else:
            frozen_names.append(name)
            frozen_count += parameter.numel()
    if not trainable_names:
        raise ValueError("modality-track-residual has no trainable parameters")
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
    if config.train_rate_mode in {"cyclic", "all", "stratified", "uniform-forced-text"}:
        return MISSING_RATES
    raise ValueError(
        "train_rate_mode must be 'cyclic', 'all', 'fixed', 'stratified', or 'uniform-forced-text'"
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


def _uniform_forced_text_mask_tensors(
    config: TrainConfig,
    data: Sequence[object],
    epoch: int,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Sample continuous per-utterance rates with an optional Text force.

    Each valid utterance receives an independent ``r ~ U(0, 1)`` and each
    modality is retained with probability ``1-r``.  Text is then forced
    missing with the configured probability.  Empty rows are repaired by
    retaining one modality, while preserving a forced Text omission whenever
    possible.  The host and guest masks are generated independently, just as
    the official schedule generates side-specific masks.
    """
    umask = data[7]
    conversation_ids = [str(value) for value in data[-1]]
    qmask = data[6]
    if umask.ndim != 2 or qmask.shape != umask.shape:
        raise ValueError("umask and qmask must both have shape [batch, sequence]")
    batch_size, sequence_length = umask.shape
    if len(conversation_ids) != batch_size:
        raise ValueError("conversation IDs must match the umask batch size")
    forced_probability = float(config.uniform_forced_text_probability)
    root_seed = SeedBundle(config.seed).derive("uniform_forced_text_mask")
    digest = hashlib.sha256()
    side_tensors: list[torch.Tensor] = []
    forced_tensors: list[torch.Tensor] = []
    sampled_rate_sum = 0.0
    sampled_rate_count = 0
    for side in ("host", "guest"):
        conversations = []
        forced_conversations = []
        for batch_index_in_batch, conversation_id in enumerate(conversation_ids):
            valid_length = int(umask[batch_index_in_batch].sum().item())
            if valid_length < 1:
                raise ValueError("every conversation must contain a valid utterance")
            payload = json.dumps(
                {
                    "algorithm": "uniform-missing-rate-forced-text-v1",
                    "dataset": config.dataset,
                    "fold": config.fold,
                    "seed": root_seed,
                    "epoch": int(epoch),
                    "batch_index": int(batch_index),
                    "conversation_id": conversation_id,
                    "side": side,
                    "forced_text_probability": format(forced_probability, ".17g"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            sampled_rates = rng.random(valid_length)
            availability = rng.random((valid_length, 3)) >= sampled_rates[:, None]
            forced_text = rng.random(valid_length) < forced_probability
            availability[forced_text, 1] = False
            empty_rows = np.flatnonzero(~availability.any(axis=1))
            if empty_rows.size:
                for row in empty_rows.tolist():
                    choices = (0, 2) if forced_text[row] else (0, 1, 2)
                    retained = int(rng.choice(choices))
                    availability[row, retained] = True
            padded = np.zeros((sequence_length, 3), dtype=np.uint8)
            padded[:valid_length] = availability.astype(np.uint8, copy=False)
            forced_padded = np.zeros(sequence_length, dtype=np.bool_)
            forced_padded[:valid_length] = forced_text
            conversations.append(torch.as_tensor(padded))
            forced_conversations.append(torch.as_tensor(forced_padded))
            sampled_rate_sum += float(sampled_rates.sum())
            sampled_rate_count += valid_length
            digest.update(payload)
            digest.update(b"\0")
            digest.update(padded.tobytes(order="C"))
        side_tensors.append(torch.stack(conversations, dim=1).to(device=umask.device))
        forced_tensors.append(torch.stack(forced_conversations, dim=1).to(device=umask.device))
    host_availability, guest_availability = side_tensors
    host_forced, guest_forced = forced_tensors
    selected_forced = torch.where(qmask.transpose(0, 1).bool(), guest_forced, host_forced)
    return (
        host_availability,
        guest_availability,
        {
            "sampled_rate_sum": sampled_rate_sum,
            "sampled_rate_count": sampled_rate_count,
            "selected_forced_text": selected_forced,
            "mask_hash": digest.hexdigest(),
        },
    )


def _prepare_uniform_forced_text_view(
    config: TrainConfig,
    data: Sequence[object],
    epoch: int,
    batch_index: int,
    dimensions: tuple[int, int, int],
) -> tuple[dict[str, object], dict[str, object]]:
    host_availability, guest_availability, audit = _uniform_forced_text_mask_tensors(
        config, data, epoch, batch_index
    )
    return (
        _prepare_view_from_primary_masks(
            data,
            host_availability,
            guest_availability,
            dimensions,
        ),
        audit,
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
    if task in ("classification", "binary", "three-class"):
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
        elif task == "three-class":
            continuous_labels = labels.reshape(-1)
            flat_labels = torch.where(
                continuous_labels < 0,
                torch.zeros_like(continuous_labels, dtype=torch.long),
                torch.where(
                    continuous_labels > 0,
                    torch.full_like(continuous_labels, 2, dtype=torch.long),
                    torch.ones_like(continuous_labels, dtype=torch.long),
                ),
            )
        return torch.nn.functional.cross_entropy(
            flat_logits[selected], flat_labels[selected]
        )
    prediction = logits.transpose(0, 1).reshape(-1)
    target = labels.reshape(-1).to(dtype=prediction.dtype)
    if not bool(selected.any()):
        return prediction.sum() * 0.0
    if task == "dual":
        if task_regression_loss == "mse":
            regression_loss = torch.nn.functional.mse_loss(
                prediction[selected], target[selected]
            )
        else:
            regression_loss = torch.nn.functional.smooth_l1_loss(
                prediction[selected],
                target[selected],
                beta=task_smooth_l1_beta,
            )
        sign_selected = selected & target.ne(0)
        if not bool(sign_selected.any()):
            classification_loss = prediction.sum() * 0.0
        else:
            classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction[sign_selected], target[sign_selected].gt(0).to(prediction.dtype)
            )
        return regression_loss + classification_loss
    if task_regression_loss == "mse":
        return torch.nn.functional.mse_loss(
            prediction[selected], target[selected]
        )
    return torch.nn.functional.smooth_l1_loss(
        prediction[selected],
        target[selected],
        beta=task_smooth_l1_beta,
    )


def _pattern_group_losses(
    dataset: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    umask: torch.Tensor,
    availability: torch.Tensor,
    mosi_task_mode: str = "regression",
    task_regression_loss: str = "mse",
    task_smooth_l1_beta: float = 1.0,
) -> tuple[tuple[int, ...], tuple[torch.Tensor, ...]]:
    """Return task losses grouped by the observed A/T/V pattern in this view.

    ``availability`` is [L, B, 3], while ``umask`` is [B, L].  Pattern ids
    use binary weights A=4, T=2, V=1, so the seven non-empty observed sets
    map to ids 1..7.  Only groups present among valid utterances are returned.
    JEPA targets are intentionally not involved here.
    """

    valid = umask.transpose(0, 1).bool()
    pattern_ids = (
        availability[..., 0].long() * 4
        + availability[..., 1].long() * 2
        + availability[..., 2].long()
    )
    group_ids: list[int] = []
    group_losses: list[torch.Tensor] = []
    for pattern_id in _OBSERVED_PATTERN_IDS:
        selected = valid & pattern_ids.eq(pattern_id)
        if bool(selected.any()):
            group_ids.append(pattern_id)
            group_losses.append(
                _task_loss(
                    dataset,
                    logits,
                    labels,
                    selected.T,
                    mosi_task_mode,
                    task_regression_loss,
                    task_smooth_l1_beta,
                )
            )
    return tuple(group_ids), tuple(group_losses)


def _emotion_loss(
    dataset: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    umask: torch.Tensor,
    availability: torch.Tensor,
    mode: str,
    mosi_task_mode: str = "regression",
    task_regression_loss: str = "mse",
    task_smooth_l1_beta: float = 1.0,
    group_dro_weights: torch.Tensor | None = None,
    group_dro_eta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute classification/emotion loss without changing JEPA weighting.

    ``pattern-balanced`` gives each observed-set group equal weight within the
    current view.  ``pattern-groupdro`` uses a persistent exponentiated-gradient
    weight over the seven groups, renormalized over groups present in the view.
    The state update is detached and therefore cannot alter model gradients.
    """

    if mode == "sample-mean":
        return (
            _task_loss(
                dataset,
                logits,
                labels,
                umask,
                mosi_task_mode,
                task_regression_loss,
                task_smooth_l1_beta,
            ),
            {},
        )
    if mode not in {"pattern-balanced", "pattern-groupdro"}:
        raise ValueError("unsupported emotion_loss_mode: {}".format(mode))
    group_ids, group_losses = _pattern_group_losses(
        dataset,
        logits,
        labels,
        umask,
        availability,
        mosi_task_mode,
        task_regression_loss,
        task_smooth_l1_beta,
    )
    if not group_losses:
        zero = logits.sum() * 0.0
        return zero, {}
    stacked = torch.stack(group_losses)
    if mode == "pattern-balanced":
        return stacked.mean(), {
            str(group_id): float(loss.detach().cpu())
            for group_id, loss in zip(group_ids, group_losses)
        }
    if group_dro_weights is None:
        raise ValueError("group_dro_weights is required for pattern-groupdro")
    if group_dro_weights.numel() != len(_OBSERVED_PATTERN_IDS):
        raise ValueError("group_dro_weights must contain seven pattern weights")
    active_indices = torch.tensor(
        [pattern_id - 1 for pattern_id in group_ids],
        dtype=torch.long,
    )
    with torch.no_grad():
        active_weights = group_dro_weights[active_indices]
        normalized = active_weights / active_weights.sum().clamp_min(1e-12)
        loss_value = stacked.detach().to(dtype=group_dro_weights.dtype, device="cpu")
        group_dro_weights[active_indices] *= torch.exp(
            float(group_dro_eta) * loss_value.clamp(max=20.0)
        )
        group_dro_weights.div_(group_dro_weights.sum().clamp_min(1e-12))
    return (normalized.to(device=stacked.device, dtype=stacked.dtype) * stacked).sum(), {
        str(group_id): float(loss.detach().cpu())
        for group_id, loss in zip(group_ids, group_losses)
    }


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
    if task in ("classification", "binary", "three-class", "dual", "soft-ordinal"):
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
    if task in ("classification", "binary", "three-class"):
        if task == "three-class":
            # Evaluate only the negative/positive logits on original nonzero
            # samples; the neutral class participates in training but is not
            # a class in the reported Non0 binary metric.
            nonzero_logits = torch.stack((logits[..., 0], logits[..., 2]), dim=-1)
            predicted = nonzero_logits.argmax(dim=-1).transpose(0, 1)
        else:
            predicted = logits.argmax(dim=-1).transpose(0, 1)
    elif task in ("dual", "soft-ordinal"):
        predicted = logits.squeeze(-1).transpose(0, 1).gt(0).long()
    else:
        predicted = logits.squeeze(-1).transpose(0, 1)
    selected = umask.bool()
    if task in ("binary", "three-class", "dual", "soft-ordinal"):
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
    group_dro_weights: torch.Tensor | None = None,
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
        "joint-reg-only",
        "complete-state",
        "write-state",
        "future-state",
        "emotion-only",
        "frozen-completion",
    }
    train_jepa = config.training_objective in {"joint", "joint-reg-only", "jepa-only"}
    train_state = config.training_objective == "complete-state"
    train_write_state = config.training_objective == "write-state"
    train_future_state = config.training_objective == "future-state"
    model.train()
    if config.osram_readout_fusion == "modality-track-residual":
        # The frozen no-JEPA anchor must be deterministic while the new
        # residual branch keeps its own dropout active during optimization.
        model.eval()
        model.osram.modality_track_residual.train()
    predictor = getattr(model, "source_only_predictor", None) if config.completion_path == "pre_osram_b2" else getattr(model, "missing_predictor", None)
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
    text_core_losses: list[float] = []
    text_core_self_losses: list[float] = []
    text_core_pred_losses: list[float] = []
    text_core_align_losses: list[float] = []
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
    uniform_mask_digest = hashlib.sha256()
    uniform_sampled_rate_sum = 0.0
    uniform_sampled_rate_count = 0
    uniform_forced_text_count = 0
    uniform_forced_text_valid_count = 0
    uniform_missing_count = 0
    uniform_modality_count = 0
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
        elif config.train_rate_mode == "uniform-forced-text":
            view, uniform_audit = _prepare_uniform_forced_text_view(
                config, data, epoch, batch_index, dimensions
            )
            optimizer.zero_grad(set_to_none=True)
            rate_views = ((None, view),)
            uniform_mask_digest.update(uniform_audit["mask_hash"].encode("ascii"))
            uniform_mask_digest.update(b"\0")
            uniform_sampled_rate_sum += float(uniform_audit["sampled_rate_sum"])
            uniform_sampled_rate_count += int(uniform_audit["sampled_rate_count"])
            valid_rows = view["umask"].transpose(0, 1).bool()
            uniform_missing_count += int((view["availability"].eq(0) & valid_rows.unsqueeze(-1)).sum().item())
            uniform_modality_count += int(valid_rows.sum().item() * 3)
            selected_forced = uniform_audit["selected_forced_text"]
            uniform_forced_text_count += int(selected_forced[valid_rows].sum().item())
            uniform_forced_text_valid_count += int(valid_rows.sum().item())
        else:
            raise ValueError(
                "train_rate_mode must be 'cyclic', 'all', 'fixed', 'stratified', or 'uniform-forced-text'"
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
            logits, hidden, _, predictions = model(
                [view["incomplete"]],
                view["availability"],
                view["qmask"],
                view["umask"],
                view["lengths"],
                predict_missing=train_jepa,
            )
            model_forward_count += 1
            zero = logits.sum() * 0.0
            cls, pattern_losses = (
                _emotion_loss(
                    config.dataset,
                    logits,
                    view["labels"],
                    view["umask"],
                    view["availability"],
                    config.emotion_loss_mode,
                    config.mosi_task_mode,
                    config.task_regression_loss,
                    config.task_smooth_l1_beta,
                    group_dro_weights,
                    config.group_dro_eta,
                )
                if train_emotion
                else (zero, {})
            )
            text_core_loss = zero
            text_core_self = zero
            text_core_pred = zero
            text_core_align = zero
            if config.text_core:
                text_core_output = getattr(model, "last_text_core", None)
                if text_core_output is None:
                    raise RuntimeError("text-core forward did not expose task-space outputs")
                valid_mask = view["umask"].bool()
                text_observed_mask = (
                    valid_mask & view["availability"][..., 1].bool().T
                )
                text_missing_mask = valid_mask & ~view["availability"][..., 1].bool().T
                source_exists_mask = (
                    view["availability"][..., (0, 2)].bool().any(dim=-1).T
                )
                align_mask = text_observed_mask & source_exists_mask
                text_core_self = _task_loss(
                    config.dataset,
                    text_core_output["real_logits"],
                    view["labels"],
                    text_observed_mask,
                    config.mosi_task_mode,
                    config.task_regression_loss,
                    config.task_smooth_l1_beta,
                )
                text_core_pred = _task_loss(
                    config.dataset,
                    text_core_output["pred_logits"],
                    view["labels"],
                    text_missing_mask,
                    config.mosi_task_mode,
                    config.task_regression_loss,
                    config.task_smooth_l1_beta,
                )
                align_positions = align_mask.T
                if bool(align_positions.any()):
                    text_core_align = torch.nn.functional.smooth_l1_loss(
                        text_core_output["u_pred"][align_positions],
                        text_core_output["u_true"][align_positions],
                    )
                text_core_loss = (
                    0.25 * text_core_self
                    + 0.5 * text_core_pred
                    + 0.1 * text_core_align
                )
            if train_future_state:
                future_loss, future_count = model.future_state_loss(
                    view["complete"], view["umask"]
                )
                # A transition is adjacent within one conversation, regardless
                # of availability; complete views have exactly the same targets.
                valid = view["umask"].bool()
                transitions = valid[:, :-1] & valid[:, 1:]
                if future_count != int(transitions.sum().item()):
                    raise RuntimeError("future-state target count must match adjacent valid transitions")
                if rate is None:
                    for conversation_index, conversation_rate in enumerate(conversation_rates):
                        rate_jepa_target_counts[conversation_rate] += int(
                            transitions[conversation_index].sum().item()
                        )
                else:
                    rate_jepa_target_counts[rate] += future_count
                jepa = MissingM3Loss(future_loss, future_loss, zero, future_count)
                jepa_rate_weight = 1.0
            elif train_write_state:
                write_loss, write_count = model.write_state_loss(
                    view["complete"], view["availability"], view["qmask"], view["umask"]
                )
                valid_write_mask = (
                    view["availability"].eq(0)
                    & view["umask"].transpose(0, 1).bool().unsqueeze(-1)
                )
                if rate is None:
                    for conversation_index, conversation_rate in enumerate(conversation_rates):
                        rate_jepa_target_counts[conversation_rate] += int(
                            valid_write_mask[:, conversation_index].sum().item()
                        )
                else:
                    rate_jepa_target_counts[rate] += write_count
                jepa = MissingM3Loss(write_loss, write_loss, zero, write_count)
                jepa_rate_weight = 1.0
            elif train_state:
                state_loss, state_count = model.complete_state_loss(
                    hidden, view["complete"], view["availability"], view["umask"]
                )
                # Counts are utterances, not modality targets, for this objective.
                valid_state_mask = (
                    view["availability"].eq(0).any(-1)
                    & view["umask"].transpose(0, 1).bool()
                )
                if rate is None:
                    for conversation_index, conversation_rate in enumerate(conversation_rates):
                        rate_jepa_target_counts[conversation_rate] += int(
                            valid_state_mask[:, conversation_index].sum().item()
                        )
                else:
                    rate_jepa_target_counts[rate] += state_count
                jepa = MissingM3Loss(state_loss, state_loss, zero, state_count)
                jepa_rate_weight = 1.0
            elif train_jepa:
                valid_target_mask = (
                    predictions.target_mask
                    & view["umask"].transpose(0, 1).bool().unsqueeze(-1)
                )
                if config.target_space != "all-modalities":
                    valid_target_mask = valid_target_mask[..., 1:2]
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
                if config.target_space == "all-modalities":
                    loss_kwargs = dict(
                        temperature=config.temperature,
                        regression_aggregation=config.jepa_regression_aggregation,
                        contrastive_prediction_source=config.jepa_contrastive_source,
                    )
                    if config.training_objective == "joint-reg-only":
                        loss_kwargs["include_contrastive"] = False
                    jepa = missing_m3_loss(predictions, teacher, **loss_kwargs)
                else:
                    from .text_subspace import text_jepa_loss
                    jepa = text_jepa_loss(predictions, teacher, model.text_subspace,
                                          temperature=config.temperature)
                jepa_rate_weight = (
                    1.0
                    if rate is None
                    else _jepa_rate_weight(rate, config.jepa_rate_weighting)
                )
            else:
                jepa = MissingM3Loss(zero, zero, zero, 0)
                jepa_rate_weight = 0.0
            if config.text_core:
                loss = cls + text_core_loss
            elif config.training_objective in {
                "joint",
                "joint-reg-only",
                "complete-state",
                "write-state",
                "future-state",
            }:
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
            if config.text_core:
                text_core_losses.append(float(text_core_loss.detach()))
                text_core_self_losses.append(float(text_core_self.detach()))
                text_core_pred_losses.append(float(text_core_pred.detach()))
                text_core_align_losses.append(float(text_core_align.detach()))
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
        if (train_jepa or train_state or train_write_state or train_future_state) and config.teacher_mode == "ema":
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
        **({
            "text_core_loss": float(np.mean(text_core_losses)),
            "text_core_self_loss": float(np.mean(text_core_self_losses)),
            "text_core_pred_loss": float(np.mean(text_core_pred_losses)),
            "text_core_align_loss": float(np.mean(text_core_align_losses)),
        } if config.text_core else {}),
        "jepa_target_count": int(target_count),
        **({"state_loss": float(np.mean(jepa_losses)),
            "state_target_count": int(target_count)} if train_state else {}),
        **({"write_state_loss": float(np.mean(jepa_losses)),
            "write_state_target_count": int(target_count)} if train_write_state else {}),
        **({"future_state_loss": float(np.mean(jepa_losses)),
            "future_state_transition_count": int(target_count),
            "rate_future_transition_counts": {
                str(rate): count for rate, count in rate_jepa_target_counts.items()
            }} if train_future_state else {}),
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
        "uniform_forced_text_probability": (
            float(config.uniform_forced_text_probability)
            if config.train_rate_mode == "uniform-forced-text"
            else None
        ),
        "uniform_sampled_rate_mean": (
            uniform_sampled_rate_sum / uniform_sampled_rate_count
            if uniform_sampled_rate_count else None
        ),
        "uniform_realized_missing_fraction": (
            uniform_missing_count / uniform_modality_count
            if uniform_modality_count else None
        ),
        "uniform_forced_text_fraction": (
            uniform_forced_text_count / uniform_forced_text_valid_count
            if uniform_forced_text_valid_count else None
        ),
        "uniform_mask_hash": (
            uniform_mask_digest.hexdigest()
            if config.train_rate_mode == "uniform-forced-text"
            else None
        ),
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_batches": skipped_optimizer_batches,
        "routing": routing_record,
        "training_objective": config.training_objective,
        "emotion_loss_mode": config.emotion_loss_mode,
        **({"pattern_group_losses": pattern_losses} if config.emotion_loss_mode != "sample-mean" else {}),
        **({
            "pattern_group_dro_weights": {
                str(pattern_id): float(group_dro_weights[index])
                for index, pattern_id in enumerate(_OBSERVED_PATTERN_IDS)
            }
        } if config.emotion_loss_mode == "pattern-groupdro" and group_dro_weights is not None else {}),
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
    all_text_core_pred_u: list[np.ndarray] = []
    all_text_core_target_u: list[np.ndarray] = []
    all_text_core_observed: list[np.ndarray] = []
    all_text_core_real_pred: list[np.ndarray] = []
    all_text_core_real_labels: list[np.ndarray] = []
    all_text_core_missing_pred: list[np.ndarray] = []
    all_text_core_missing_labels: list[np.ndarray] = []
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
            if task in ("binary", "three-class", "dual", "soft-ordinal"):
                selected = selected & view["labels"].ne(0)
            all_availability.append(
                metric_availability[selected].cpu().numpy()
            )
            text_core_module = getattr(model, "text_core", None)
            text_core_output = getattr(model, "last_text_core", None)
            if text_core_module is not None:
                if text_core_output is None:
                    raise RuntimeError("text-core evaluation did not expose task-space outputs")
                complete_parts = torch.split(view["complete"], dimensions, dim=-1)
                complete_text_latent = model.observed_set.projectors["text"](
                    complete_parts[1]
                )
                target_u = text_core_module.encoder(complete_text_latent)
                observed_mask = valid & text_core_output["text_observed"]
                missing_mask = valid & ~text_core_output["text_observed"]
                real_pred, real_labels, _ = _collect_predictions(
                    dataset,
                    text_core_output["real_logits"],
                    view["labels"],
                    observed_mask.T,
                    mosi_task_mode,
                )
                missing_pred, missing_labels, _ = _collect_predictions(
                    dataset,
                    text_core_output["pred_logits"],
                    view["labels"],
                    missing_mask.T,
                    mosi_task_mode,
                )
                all_text_core_pred_u.append(
                    text_core_output["u_pred"][valid].cpu().numpy()
                )
                all_text_core_target_u.append(target_u[valid].cpu().numpy())
                all_text_core_observed.append(
                    text_core_output["text_observed"][valid].cpu().numpy()
                )
                all_text_core_real_pred.append(real_pred)
                all_text_core_real_labels.append(real_labels)
                all_text_core_missing_pred.append(missing_pred)
                all_text_core_missing_labels.append(missing_labels)
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
        if task in ("binary", "three-class", "dual", "soft-ordinal"):
            artifacts["continuous_labels"] = continuous_labels_array
        if task == "soft-ordinal":
            artifacts["signed_logits"] = np.concatenate(all_signed_logits)
        full_availability_array = np.concatenate(all_full_availability)
        metrics["mask_sha256"] = _sha256_tensor(
            torch.from_numpy(full_availability_array)
        )
        if getattr(model, "text_core", None) is not None:
            predicted_u = np.concatenate(all_text_core_pred_u, axis=0)
            target_u = np.concatenate(all_text_core_target_u, axis=0)
            observed_flags = np.concatenate(all_text_core_observed, axis=0).astype(bool)
            missing_flags = ~observed_flags
            metrics["text_core_real_slot"] = (
                _metrics(
                    dataset,
                    np.concatenate(all_text_core_real_labels),
                    np.concatenate(all_text_core_real_pred),
                    mosi_task_mode,
                )
                if all_text_core_real_pred and sum(
                    values.size for values in all_text_core_real_pred
                )
                else None
            )
            metrics["text_core_predicted_slot"] = (
                _metrics(
                    dataset,
                    np.concatenate(all_text_core_missing_labels),
                    np.concatenate(all_text_core_missing_pred),
                    mosi_task_mode,
                )
                if all_text_core_missing_pred and sum(
                    values.size for values in all_text_core_missing_pred
                )
                else None
            )
            if int(missing_flags.sum()) >= 2:
                pred_missing = predicted_u[missing_flags]
                target_missing = target_u[missing_flags]
                pred_centered = pred_missing - pred_missing.mean(axis=0, keepdims=True)
                target_centered = target_missing - target_missing.mean(axis=0, keepdims=True)
                metrics["text_core_centered_cosine"] = float(
                    np.sum(pred_centered * target_centered)
                    / (np.linalg.norm(pred_centered) * np.linalg.norm(target_centered) + 1e-8)
                )
            else:
                metrics["text_core_centered_cosine"] = None
            metrics["text_core_observed_count"] = int(observed_flags.sum())
            metrics["text_core_missing_count"] = int(missing_flags.sum())
            metrics["text_core_pred_target_std_ratio"] = float(
                np.std(predicted_u) / (np.std(target_u) + 1e-8)
            )
            artifacts["text_core_pred_u"] = predicted_u
            artifacts["text_core_target_u"] = target_u
            artifacts["text_core_text_observed"] = observed_flags
    return metrics, artifacts


def run_experiment(
    config_value: TrainConfig,
    audio_root: str,
    text_root: str,
    visual_root: str,
    output_dir: str | Path,
) -> Dict[str, object]:
    if (config_value.training_objective == "future-state"
            and config_value.checkpoint_selection != "test-oracle-per-rate"):
        raise ValueError("future-state experiments require test-oracle-per-rate selection")
    if (config_value.osram_readout_fusion != "flat"
            and config_value.checkpoint_selection != "test-oracle-per-rate"):
        raise ValueError(f"{config_value.osram_readout_fusion} requires test-oracle-per-rate selection")
    if config_value.completion_path == "pre_osram_b2":
        if (not config_value.b2_base_checkpoint or not config_value.b2_pretrain_checkpoint
                or config_value.initial_backbone_checkpoint or config_value.pretrained_learning_rate is not None
                or config_value.training_objective != "joint" or config_value.train_rate_mode != "cyclic"):
            raise ValueError("B2 joint cyclic training requires its base and Stage1 checkpoints, no legacy transfer")
        from .b2_training import validate_stage2_config
        validate_stage2_config(config_value)
    protocol_rates = _protocol_rates(config_value)
    if config_value.training_objective not in _TRAINING_OBJECTIVES:
        raise ValueError("unsupported training_objective")
    if config_value.pretrained_learning_rate is not None:
        if config_value.initial_backbone_checkpoint is None:
            raise ValueError(
                "pretrained_learning_rate requires initial_backbone_checkpoint"
            )
        if config_value.training_objective not in {"joint", "joint-reg-only"}:
            raise ValueError(
                "pretrained_learning_rate is only valid for joint training"
            )
        if not 0.0 < config_value.pretrained_learning_rate < config_value.learning_rate:
            raise ValueError(
                "pretrained_learning_rate must be lower than learning_rate"
            )
    frozen_completion = config_value.training_objective == "frozen-completion"
    frozen_modality_track_residual = (
        config_value.osram_readout_fusion == "modality-track-residual"
    )
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
    if config_value.checkpoint_selection not in ("validation", "test-oracle", "test-oracle-per-rate"):
        raise ValueError(
            "checkpoint_selection must be validation, test-oracle, or test-oracle-per-rate"
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
        postgraph_bilstm_ablation=config_value.postgraph_bilstm_ablation,
        backbone_type=config_value.backbone_type,
        osram_output_dim=config_value.osram_output_dim,
        osram_num_heads=config_value.osram_num_heads,
        osram_key_dim=config_value.osram_key_dim,
        osram_value_dim=config_value.osram_value_dim,
        osram_read_ridge=config_value.osram_read_ridge,
        osram_write_ridge=config_value.osram_write_ridge,
        osram_write_step=config_value.osram_write_step,
        osram_predictor_mode=config_value.osram_predictor_mode,
        osram_ablation=config_value.osram_ablation,
        osram_emotion_ablation=config_value.osram_emotion_ablation,
        osram_query_availability=config_value.osram_query_availability,
        osram_bidirectional=config_value.osram_bidirectional,
        osram_forward_slot_reuse=config_value.osram_forward_slot_reuse,
        osram_readout_fusion=config_value.osram_readout_fusion,
        completion_path=config_value.completion_path,
        complete_state_jepa=config_value.training_objective == "complete-state",
        write_state_completion=config_value.training_objective == "write-state",
        future_state_jepa=config_value.training_objective == "future-state",
        teacher_mode=config_value.teacher_mode,
        teacher_checkpoint=config_value.teacher_checkpoint,
        target_space=config_value.target_space,
        text_subspace_checkpoint=config_value.text_subspace_checkpoint,
        training_objective=config_value.training_objective,
        text_core=config_value.text_core,
    ).to(device)
    text_subspace_hash_before = (
        model.text_subspace_integrity()
        if config_value.target_space == "predictable-subspace" else None
    )
    teacher_hash_before = (
        model.teacher_integrity() if config_value.teacher_mode == "pretrained-frozen" else None
    )
    if config_value.teacher_mode == "pretrained-frozen" and not teacher_hash_before:
        raise RuntimeError("pretrained-frozen teacher integrity hash is missing")
    if config_value.teacher_mode == "pretrained-frozen":
        source_config = model.teacher_provenance.get("source_config", {})
        for field in ("dataset", "fold", "seed", "latent_dim"):
            expected = getattr(config_value, field)
            actual = source_config.get(field)
            if actual != expected:
                raise ValueError(
                    f"pretrained-frozen teacher {field} mismatch: source={actual!r}, current={expected!r}"
                )
    initialization = None
    frozen_probe = None
    frozen_hash_before = None
    if config_value.completion_path == "pre_osram_b2":
        from .b2_training import load_b2_initialization
        initialization = load_b2_initialization(model,config_value.b2_base_checkpoint,
                                               config_value.b2_pretrain_checkpoint)
    if config_value.initial_backbone_checkpoint is not None:
        initialization = _load_inference_backbone_checkpoint(
            model,
            config_value.initial_backbone_checkpoint,
            include_jepa_modules=config_value.training_objective
            in {"joint", "frozen-completion"},
            load_all_shared_modules=frozen_modality_track_residual,
            allow_missing_prefixes=(
                _FROZEN_MODALITY_TRACK_RESIDUAL_TRAINABLE_PREFIXES
                if frozen_modality_track_residual else ()
            ),
        )
    if frozen_completion:
        frozen_probe = _configure_frozen_completion_probe(model)
        frozen_hash_before = _parameter_subset_sha256(
            model,
            frozen_probe["frozen_parameter_names"],
        )
    elif frozen_modality_track_residual:
        frozen_probe = _configure_frozen_modality_track_residual_probe(model)
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
    group_dro_weights = (
        torch.ones(len(_OBSERVED_PATTERN_IDS), dtype=torch.float64)
        if config_value.emotion_loss_mode == "pattern-groupdro"
        else None
    )
    history: list[Dict[str, object]] = []
    jepa_pretraining = config_value.training_objective == "jepa-only"
    per_rate_oracle = config_value.checkpoint_selection == "test-oracle-per-rate"
    best_score: float | None = None if jepa_pretraining or per_rate_oracle else -math.inf
    best_epoch = None if per_rate_oracle else 0
    best_state = None
    selected_epoch_by_rate: Dict[str, int] = {}
    selected_score_by_rate: Dict[str, float] = {}
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
            group_dro_weights,
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
        if per_rate_oracle:
            record["mean_is_descriptive_only"] = True
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
        if per_rate_oracle:
            for rate in protocol_rates:
                rate_key = format(rate, ".1f")
                score = float(selection_metrics[rate]["weighted_f1"])
                if score > selected_score_by_rate.get(rate_key, -math.inf):
                    selected_score_by_rate[rate_key] = score
                    selected_epoch_by_rate[rate_key] = epoch + 1
                    torch.save({
                        "model": _state_to_cpu(model),
                        "config": asdict(config_value),
                        "text_subspace_provenance": getattr(model, "text_subspace_provenance", None),
                        "epoch": epoch + 1,
                        "selection_split": "test",
                        "selection_protocol": "per-rate-test-oracle",
                        "selection_rate": rate,
                        "selection_weighted_f1": score,
                    }, output / ("best_miss_" + rate_key.replace(".", "p") + ".pt"))
            continue
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
                text_subspace_provenance=getattr(model, "text_subspace_provenance", None),
                selection_split=config_value.checkpoint_selection,
                selection_protocol=(
                    "8-rate-mean-test-oracle"
                    if config_value.checkpoint_selection == "test-oracle"
                    else "8-rate-mean-validation"
                ),
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
            selection_protocol="fixed-final",
        )
    # Check the final training state before a best-checkpoint restore could hide
    # accidental changes to the supposedly frozen target coordinate system.
    if (teacher_hash_before is not None
            and model.teacher_integrity() != teacher_hash_before):
        raise RuntimeError("pretrained-frozen teacher changed during training")
    if (text_subspace_hash_before is not None
            and model.text_subspace_integrity() != text_subspace_hash_before):
        raise RuntimeError("frozen text subspace changed during training")
    if per_rate_oracle:
        if len(selected_epoch_by_rate) != len(protocol_rates):
            raise RuntimeError("no best checkpoint was selected for every missing rate")
    elif best_state is None:
        raise RuntimeError("no best checkpoint was selected")
    else:
        model.load_state_dict(best_state, strict=True)
    model.to(device)
    frozen_integrity = None
    if frozen_probe is not None:
        frozen_hash_after = _parameter_subset_sha256(
            model,
            frozen_probe["frozen_parameter_names"],
        )
        if frozen_hash_after != frozen_hash_before:
            raise RuntimeError("frozen backbone parameters changed during training")
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
    selected_diagnostics_by_rate: Dict[str, object] = {}
    if config_value.evaluate_test:
        for rate in protocol_rates:
            if per_rate_oracle:
                checkpoint_path = output / (
                    "best_miss_" + format(rate, ".1f").replace(".", "p") + ".pt"
                )
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model"], strict=True)
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
            if config_value.osram_readout_fusion != "flat":
                selected_diagnostics_by_rate[rate_key] = {
                    "selected_epoch": selected_epoch_by_rate[rate_key],
                    "last_batch": copy.deepcopy(model.osram.last_diagnostics),
                }
            test_metrics[rate_key] = metrics
            mask_hashes[rate_key] = str(metrics["mask_sha256"])
            np.savez_compressed(
                output
                / ("predictions_miss_" + rate_key.replace(".", "p") + ".npz"),
                **artifacts,
            )
    teacher_integrity = None
    if teacher_hash_before is not None:
        teacher_hash_after = model.teacher_integrity()
        if teacher_hash_after != teacher_hash_before:
            raise RuntimeError("pretrained-frozen teacher changed during checkpoint selection/evaluation")
        teacher_integrity = {"before": teacher_hash_before, "after": teacher_hash_after, "unchanged": True}
    if (text_subspace_hash_before is not None
            and model.text_subspace_integrity() != text_subspace_hash_before):
        raise RuntimeError("frozen text subspace changed during checkpoint selection/evaluation")
    selection_split = (
        "test" if per_rate_oracle else (
            "fixed-final" if jepa_pretraining else config_value.checkpoint_selection
        )
    )
    result: Dict[str, object] = {
        "best_epoch": best_epoch,
        "selection_split": selection_split,
        "selection_protocol": (
            "per-rate-test-oracle" if per_rate_oracle else (
            "8-rate-mean-test-oracle"
            if selection_split == "test-oracle"
            else (
                "8-rate-mean-validation"
                if selection_split == "validation"
                else selection_split
            )
        )),
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
                if config_value.checkpoint_selection in {"test-oracle", "test-oracle-per-rate"}
                else (
                    "train-validation-test"
                    if config_value.evaluate_test
                    else "train-validation-only"
                )
            )
        ),
        "training_objective": config_value.training_objective,
        "teacher_mode": config_value.teacher_mode,
        "teacher_provenance": getattr(model, "teacher_provenance", None),
        "teacher_integrity": teacher_integrity,
        "target_space": config_value.target_space,
        "text_subspace_provenance": getattr(model, "text_subspace_provenance", None),
        "text_subspace_integrity": (
            {"before": text_subspace_hash_before, "after": model.text_subspace_integrity(),
             "unchanged": model.text_subspace_integrity() == text_subspace_hash_before}
            if text_subspace_hash_before is not None else None
        ),
        "backbone_initialization": initialization,
        "frozen_completion_integrity": frozen_integrity,
        "frozen_readout_integrity": frozen_integrity
        if frozen_modality_track_residual else None,
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
        "emotion_loss_mode": config_value.emotion_loss_mode,
        "group_dro_eta": config_value.group_dro_eta,
        "graph_message_calibration": config_value.graph_message_calibration,
        "graph_second_layer": config_value.graph_second_layer,
        "postgraph_bilstm_ablation": config_value.postgraph_bilstm_ablation,
        "backbone_type": config_value.backbone_type,
        "osram_predictor_mode": config_value.osram_predictor_mode,
        "osram_ablation": config_value.osram_ablation,
        "osram_emotion_ablation": config_value.osram_emotion_ablation,
        "osram_query_availability": config_value.osram_query_availability,
        "osram_bidirectional": config_value.osram_bidirectional,
        "osram_forward_slot_reuse": config_value.osram_forward_slot_reuse,
        "osram_write_step": config_value.osram_write_step,
        "osram_readout_fusion": config_value.osram_readout_fusion,
        "osram_dimensions": (
            {
                "latent_dim": config_value.latent_dim,
                "output_dim": config_value.osram_output_dim,
                "num_heads": config_value.osram_num_heads,
                "key_dim": config_value.osram_key_dim,
                "value_dim": config_value.osram_value_dim,
                "read_ridge": config_value.osram_read_ridge,
                "write_ridge": config_value.osram_write_ridge,
            }
            if config_value.backbone_type == "osram"
            else None
        ),
        "train_missing_rate": _fixed_missing_rate(config_value),
        "selection_missing_rates": list(protocol_rates),
        **_readout_provenance(model),
    }
    if per_rate_oracle:
        result["selected_epoch_by_rate"] = selected_epoch_by_rate
        result["selected_weighted_f1_by_rate"] = selected_score_by_rate
    _write_json(output / "metrics.json", result)
    if config_value.backbone_type == "osram":
        _write_json(
            output / "diagnostics.json",
            {
                "evaluation_stage": result["evaluation_stage"],
                "selection_protocol": result["selection_protocol"],
                "last_batch": getattr(model.osram, "last_diagnostics", {}),
                **({
                    "per_rate_scope": "last evaluation batch, not a dataset aggregate",
                    "selected_checkpoint_by_rate": selected_diagnostics_by_rate,
                } if config_value.osram_readout_fusion != "flat" else {}),
            },
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-mode", choices=("ema", "pretrained-frozen"), default="ema")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--target-space", choices=("all-modalities", "full-text", "predictable-subspace"), default="all-modalities")
    parser.add_argument("--text-subspace-checkpoint", default=None)
    parser.add_argument(
        "--text-core",
        action="store_true",
        help="Train the 64-d Text-Core task slot on the causal OSRAM read path.",
    )
    parser.add_argument("--completion-path",choices=("none","pre_osram_b2"),default="none")
    parser.add_argument("--b2-base-checkpoint",default=None)
    parser.add_argument("--b2-pretrain-checkpoint",default=None)
    parser.add_argument(
        "--dataset",
        choices=("IEMOCAPFour", "IEMOCAPSix", "CMUMOSI", "CMUMOSEI"),
        default="IEMOCAPSix",
    )
    parser.add_argument(
        "--mosi-task-mode",
        choices=("regression", "binary", "three-class", "dual", "soft-ordinal"),
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
        choices=("cyclic", "all", "fixed", "stratified", "uniform-forced-text"),
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
        "--emotion-loss-mode",
        choices=("sample-mean", "pattern-balanced", "pattern-groupdro"),
        default="sample-mean",
    )
    parser.add_argument("--group-dro-eta", type=float, default=0.1)
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
    parser.add_argument(
        "--postgraph-bilstm-ablation",
        choices=("none", "temporal", "speaker"),
        default="none",
    )
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation", "test-oracle", "test-oracle-per-rate"),
        default="validation",
    )
    parser.add_argument(
        "--training-objective",
        choices=(
            "joint",
            "joint-reg-only",
            "complete-state",
            "write-state",
            "future-state",
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
    parser.add_argument(
        "--backbone-type",
        choices=("gcnet", "osram"),
        default="gcnet",
    )
    parser.add_argument("--osram-output-dim", type=int, default=500)
    parser.add_argument("--osram-num-heads", type=int, default=4)
    parser.add_argument("--osram-key-dim", type=int, default=32)
    parser.add_argument("--osram-value-dim", type=int, default=32)
    parser.add_argument("--osram-read-ridge", type=float, default=1e-3)
    parser.add_argument("--osram-write-ridge", type=float, default=1e-3)
    parser.add_argument("--osram-write-step", type=float, default=1.0)
    parser.add_argument(
        "--osram-readout-fusion",
        choices=(
            "flat", "local-gated", "local-cross-attn", "modality-tracks",
            "modality-track-residual",
        ),
        default="flat",
    )
    parser.add_argument(
        "--osram-predictor-mode",
        choices=("structured", "legacy-hidden"),
        default="structured",
    )
    parser.add_argument(
        "--osram-ablation",
        choices=("full", "local-only", "local-base"),
        default="full",
    )
    parser.add_argument(
        "--osram-query-availability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition OSRAM queries on the explicit A/T/V availability vector.",
    )
    parser.add_argument("--osram-emotion-ablation",
                        choices=("full", "local-only", "local-base", "local-gap"),
                        default="full", help="Mask only classification readout slots; preserve predictor contexts.")
    parser.add_argument("--osram-bidirectional", action=argparse.BooleanOptionalAction,
                        default=True, help="Enable future-context backward memory scan.")
    parser.add_argument("--osram-forward-slot-reuse", action=argparse.BooleanOptionalAction,
                        default=False, help="Reuse past context in both slots; requires no bidirectional scan.")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.num_threads)
    config_value = TrainConfig(
        dataset=args.dataset,
        completion_path=args.completion_path,
        b2_base_checkpoint=args.b2_base_checkpoint,
        b2_pretrain_checkpoint=args.b2_pretrain_checkpoint,
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
        teacher_mode=args.teacher_mode,
        teacher_checkpoint=args.teacher_checkpoint,
        target_space=args.target_space,
        text_subspace_checkpoint=args.text_subspace_checkpoint,
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
        emotion_loss_mode=args.emotion_loss_mode,
        group_dro_eta=args.group_dro_eta,
        graph_message_calibration=args.graph_message_calibration,
        graph_second_layer=args.graph_second_layer,
        postgraph_bilstm_ablation=args.postgraph_bilstm_ablation,
        fixed_missing_rate=args.train_missing_rate,
        checkpoint_selection=args.checkpoint_selection,
        training_objective=args.training_objective,
        initial_backbone_checkpoint=args.initial_backbone_checkpoint,
        pretrained_learning_rate=args.pretrained_lr,
        backbone_type=args.backbone_type,
        osram_output_dim=args.osram_output_dim,
        osram_num_heads=args.osram_num_heads,
        osram_key_dim=args.osram_key_dim,
        osram_value_dim=args.osram_value_dim,
        osram_read_ridge=args.osram_read_ridge,
        osram_write_ridge=args.osram_write_ridge,
        osram_write_step=args.osram_write_step,
        osram_predictor_mode=args.osram_predictor_mode,
        osram_ablation=args.osram_ablation,
        osram_emotion_ablation=args.osram_emotion_ablation,
        osram_query_availability=args.osram_query_availability,
        osram_bidirectional=args.osram_bidirectional,
        osram_forward_slot_reuse=args.osram_forward_slot_reuse,
        osram_readout_fusion=args.osram_readout_fusion,
        text_core=args.text_core,
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
