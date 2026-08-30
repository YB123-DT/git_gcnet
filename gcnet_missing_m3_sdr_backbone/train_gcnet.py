"""Locked CMU-MOSI trainer for the Missing-M3 SDR backbones."""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, List, Union

import numpy as np
import torch

import config
from gcnet_missing_m3 import train_gcnet as base_train
from gcnet_missing_m3.mixed_rate import MISSING_RATES
from gcnet_modality_jepa.protocol import SeedBundle

from .model import MissingM3SDRModel, SDRConversationBackbone


# Preserve the registered control's data, mask, optimization, evaluation, and
# checkpoint primitives.  This module owns only candidate construction and the
# eight-rate validation orchestration needed to record treatment provenance.
_resolve_task_contract = base_train._resolve_task_contract
_save_best_checkpoint = base_train._save_best_checkpoint
_sha256_tensor = base_train._sha256_tensor
_schedules = base_train._schedules
_state_to_cpu = base_train._state_to_cpu
_write_json = base_train._write_json
_write_run_config = base_train._write_run_config
evaluate_rate = base_train.evaluate_rate
get_loaders = base_train.get_loaders
set_random_seed = base_train.set_random_seed
train_epoch = base_train.train_epoch


@dataclass(frozen=True)
class SDRTrainConfig(base_train.TrainConfig):
    """Registered SDR treatment; only identity/lifecycle fields are mutable."""

    dataset: str = "CMUMOSI"
    fold: int = 1
    base_model: str = "LSTM"
    window_past: int = 2
    window_future: int = 2
    hidden: int = 200
    dropout: float = 0.5
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    latent_dim: int = 256
    num_experts: int = 4
    top_k: int = 2
    projector_dropout: float = 0.1
    predictor_dropout: float = 0.1
    fusion_type: str = "slot"
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
    train_rate_mode: str = "all"
    mosi_task_mode: str = "regression"
    graph_branch_mode: str = "both"
    mmoe_variant: str = "dual-gate"
    classification_completion: bool = False
    representation_type: str = "slot"
    node_interaction_residual: bool = False
    readout_type: str = "shared"
    readout_rank: int = 8
    jepa_regression_aggregation: str = "target"
    recurrent_padding_mode: str = "legacy"
    task_regression_loss: str = "mse"
    task_smooth_l1_beta: float = 1.0
    postgraph_sequence_mode: str = "independent"
    jepa_rate_weighting: str = "uniform"
    graph_message_calibration: str = "none"
    fixed_missing_rate: None = None
    graph_hidden: int = 100
    sdr_variant: str = "sdr-public"

    _OPEN_FIELDS: ClassVar[FrozenSet[str]] = frozenset(
        {"seed", "device", "epochs", "evaluate_test", "sdr_variant"}
    )

    def __post_init__(self) -> None:
        if self.sdr_variant not in SDRConversationBackbone.VARIANTS:
            raise ValueError("sdr_variant must be 'sdr-public' or 'sdr-paper'")
        for name, field in self.__dataclass_fields__.items():
            if name in self._OPEN_FIELDS or name.startswith("_"):
                continue
            expected = field.default
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(
                    "{} is locked to {!r}, got {!r}".format(
                        name,
                        expected,
                        actual,
                    )
                )


def build_model(
    config_value: SDRTrainConfig,
    adim: int,
    tdim: int,
    vdim: int,
    device: torch.device,
) -> MissingM3SDRModel:
    """Construct exactly one registered SDR treatment on ``device``."""

    shape = _resolve_task_contract(
        config_value.dataset,
        config_value.mosi_task_mode,
    )
    return MissingM3SDRModel(
        config_value.base_model,
        adim,
        tdim,
        vdim,
        config_value.hidden,
        config_value.graph_hidden,
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
        representation_type=config_value.representation_type,
        node_interaction_residual=config_value.node_interaction_residual,
        readout_type=config_value.readout_type,
        readout_rank=config_value.readout_rank,
        recurrent_padding_mode=config_value.recurrent_padding_mode,
        postgraph_sequence_mode=config_value.postgraph_sequence_mode,
        graph_message_calibration=config_value.graph_message_calibration,
        sdr_variant=config_value.sdr_variant,
    ).to(device)


def _parameter_provenance(model: MissingM3SDRModel) -> Dict[str, int]:
    backbone = model.conversation_backbone
    return {
        "registered_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "registered_backbone_parameters": sum(
            parameter.numel() for parameter in backbone.parameters()
        ),
        "trainable_backbone_parameters": sum(
            parameter.numel()
            for parameter in backbone.parameters()
            if parameter.requires_grad
        ),
    }


def _peak_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def run_experiment(
    config_value: SDRTrainConfig,
    audio_root: str,
    text_root: str,
    visual_root: str,
    output_dir: Union[str, Path],
) -> Dict[str, object]:
    """Train and evaluate one validation-selected SDR treatment."""

    started_at = time.perf_counter()
    shape = _resolve_task_contract(
        config_value.dataset,
        config_value.mosi_task_mode,
    )
    if not 1 <= config_value.fold <= int(shape["num_folds"]):
        raise ValueError("fold is outside the dataset fold range")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_run_config(output / "config.json", config_value)

    set_random_seed(config_value.seed)
    device = torch.device(config_value.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
    model = build_model(config_value, adim, tdim, vdim, device)
    optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=config_value.learning_rate,
        weight_decay=config_value.weight_decay,
    )

    train_schedules = _schedules(config_value, "train")
    validation_schedules = _schedules(config_value, "validation")
    test_schedules = _schedules(config_value, "test")
    history: List[Dict[str, object]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    best_validation: Dict[str, Dict[str, float]] = {}

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
        validation: Dict[float, Dict[str, float]] = {}
        for rate in MISSING_RATES:
            validation[rate], _ = evaluate_rate(
                model,
                validation_loader,
                validation_schedules[rate],
                config_value.dataset,
                dimensions,
                device,
                collect=False,
                mosi_task_mode=config_value.mosi_task_mode,
                task_regression_loss=config_value.task_regression_loss,
                task_smooth_l1_beta=config_value.task_smooth_l1_beta,
            )
        validation_mean = sum(
            float(validation[rate]["weighted_f1"])
            for rate in MISSING_RATES
        ) / len(MISSING_RATES)
        serialized_validation = {
            format(rate, ".1f"): value
            for rate, value in validation.items()
        }
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": serialized_validation,
                "validation_mean_weighted_f1": validation_mean,
            }
        )
        _write_json(output / "history.json", history)
        print(
            (
                "epoch={:03d} train_wf1={:.4f} val8_wf1={:.4f} "
                "cls={:.4f} jepa={:.4f}"
            ).format(
                epoch + 1,
                train_metrics["weighted_f1"],
                validation_mean,
                train_metrics["classification_loss"],
                train_metrics["jepa_loss"],
            ),
            flush=True,
        )
        if validation_mean > best_score:
            best_score = validation_mean
            best_epoch = epoch + 1
            best_validation = copy.deepcopy(serialized_validation)
            best_state = _state_to_cpu(model)
            _save_best_checkpoint(
                output / "best.pt",
                model_state=best_state,
                config_value=config_value,
                epoch=best_epoch,
                validation_mean_weighted_f1=best_score,
            )

    if best_state is None:
        raise RuntimeError("no best checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    model.to(device)

    test_metrics: Dict[str, Dict[str, float]] = {}
    mask_hashes: Dict[str, str] = {}
    prediction_availability_hashes: Dict[str, str] = {}
    if config_value.evaluate_test:
        for rate in MISSING_RATES:
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
            prediction_availability_sha256 = _sha256_tensor(
                torch.from_numpy(
                    np.ascontiguousarray(artifacts["availability"])
                )
            )
            metrics[
                "prediction_availability_sha256"
            ] = prediction_availability_sha256
            test_metrics[rate_key] = metrics
            mask_hashes[rate_key] = str(metrics["mask_sha256"])
            prediction_availability_hashes[
                rate_key
            ] = prediction_availability_sha256
            np.savez_compressed(
                output
                / ("predictions_miss_" + rate_key.replace(".", "p") + ".npz"),
                **artifacts,
            )

    result: Dict[str, object] = {
        "best_epoch": best_epoch,
        "best_validation_mean_weighted_f1": best_score,
        "best_validation": best_validation,
        "test": test_metrics,
        "mask_sha256": mask_hashes,
        "prediction_availability_sha256": (
            prediction_availability_hashes
        ),
        "variant": config_value.sdr_variant,
        "sdr_variant": config_value.sdr_variant,
        "backbone": "sdr-gnn-whole-backbone",
        "ema_steps": model.ema_step,
        "selection_missing_rates": list(MISSING_RATES),
        "evaluation_stage": (
            "train-validation-test"
            if config_value.evaluate_test
            else "train-validation-only"
        ),
        "wall_time_seconds": float(time.perf_counter() - started_at),
        "peak_memory_bytes": _peak_memory(device),
        "jepa_regression_aggregation": config_value.jepa_regression_aggregation,
        "recurrent_padding_mode": config_value.recurrent_padding_mode,
        "task_regression_loss": config_value.task_regression_loss,
        "task_smooth_l1_beta": config_value.task_smooth_l1_beta,
        "postgraph_sequence_mode": config_value.postgraph_sequence_mode,
        "jepa_rate_weighting": config_value.jepa_rate_weighting,
        "graph_message_calibration": config_value.graph_message_calibration,
        "train_missing_rate": config_value.fixed_missing_rate,
        **_parameter_provenance(model),
    }
    _write_json(output / "metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Expose paths, lifecycle identity, and single-valued protocol markers."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("CMUMOSI",), default="CMUMOSI")
    parser.add_argument("--train-rate-mode", choices=("all",), default="all")
    parser.add_argument("--fusion-type", choices=("slot",), default="slot")
    parser.add_argument("--lr", type=float, choices=(5e-4,), default=5e-4)
    parser.add_argument(
        "--sdr-variant",
        choices=SDRConversationBackbone.VARIANTS,
        default="sdr-public",
    )
    parser.add_argument("--audio-feature", required=True)
    parser.add_argument("--text-feature", required=True)
    parser.add_argument("--video-feature", required=True)
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_value = SDRTrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        evaluate_test=not args.skip_test,
        sdr_variant=args.sdr_variant,
    )
    feature_root = args.feature_root or config.PATH_TO_FEATURES[
        config_value.dataset
    ]
    roots = [
        os.path.join(feature_root, name)
        for name in (args.audio_feature, args.text_feature, args.video_feature)
    ]
    if not all(os.path.exists(root) for root in roots):
        raise FileNotFoundError("one or more feature roots do not exist")
    run_experiment(config_value, *roots, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
