"""Locked CMU-MOSI trainer for the SDT-style Missing-M3 candidate."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict

import numpy as np
import torch

import config
from gcnet_missing_m3 import train_gcnet as base_train
from gcnet_missing_m3.mixed_rate import (
    MISSING_RATES,
    mean_validation_weighted_f1,
)
from gcnet_modality_jepa.protocol import SeedBundle

from .model import MissingM3SDTModel


# Keep the data, masking, loss, metric, and checkpoint primitives identical to
# the registered Missing-M3 control.  This module owns only the candidate
# construction and epoch orchestration.
_resolve_task_contract = base_train._resolve_task_contract
_save_best_checkpoint = base_train._save_best_checkpoint
_schedules = base_train._schedules
_state_to_cpu = base_train._state_to_cpu
_write_json = base_train._write_json
_write_run_config = base_train._write_run_config
evaluate_rate = base_train.evaluate_rate
get_loaders = base_train.get_loaders
set_random_seed = base_train.set_random_seed
train_epoch = base_train.train_epoch


@dataclass(frozen=True)
class SDTTrainConfig(base_train.TrainConfig):
    """Registered treatment with only run lifecycle fields left mutable."""

    dataset: str = "CMUMOSI"
    fold: int = 1
    window_past: int = 1
    window_future: int = 1
    hidden: int = 100
    learning_rate: float = 5e-4
    fusion_type: str = "slot"
    train_rate_mode: str = "all"
    transformer_dim: int = 384
    transformer_heads: int = 8
    transformer_layers: int = 5
    transformer_ff_dim: int = 704
    transformer_max_len: int = 512

    _OPEN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"seed", "device", "epochs", "evaluate_test"}
    )

    def __post_init__(self) -> None:
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
    config_value: SDTTrainConfig,
    adim: int,
    tdim: int,
    vdim: int,
    device: torch.device,
) -> MissingM3SDTModel:
    """Construct the single registered SDT candidate on ``device``."""

    shape = _resolve_task_contract(
        config_value.dataset,
        config_value.mosi_task_mode,
    )
    return MissingM3SDTModel(
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
        representation_type=config_value.representation_type,
        node_interaction_residual=config_value.node_interaction_residual,
        readout_type=config_value.readout_type,
        readout_rank=config_value.readout_rank,
        recurrent_padding_mode=config_value.recurrent_padding_mode,
        postgraph_sequence_mode=config_value.postgraph_sequence_mode,
        graph_message_calibration=config_value.graph_message_calibration,
    ).to(device)


def _backbone_parameter_provenance(
    model: MissingM3SDTModel,
) -> Dict[str, int]:
    backbone = model.conversation_backbone
    registered = sum(parameter.numel() for parameter in backbone.parameters())
    padding_row = backbone.speaker_embedding.embedding_dim
    return {
        "registered_backbone_parameters": registered,
        "active_backbone_parameters": registered - padding_row,
        # Registered control value from the parameter-matched experiment.  It
        # is deliberately metadata, not a reason to construct an Original.
        "control_active_backbone_parameters": 5_864_700,
    }


def run_experiment(
    config_value: SDTTrainConfig,
    audio_root: str,
    text_root: str,
    visual_root: str,
    output_dir: str | Path,
) -> Dict[str, object]:
    """Train and evaluate the locked candidate using control primitives."""

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
    history: list[Dict[str, object]] = []
    best_score = -math.inf
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
        validation_mean = mean_validation_weighted_f1(validation)
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": {
                    str(rate): value for rate, value in validation.items()
                },
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
            test_metrics[rate_key] = metrics
            mask_hashes[rate_key] = str(metrics["mask_sha256"])
            np.savez_compressed(
                output
                / ("predictions_miss_" + rate_key.replace(".", "p") + ".npz"),
                **artifacts,
            )

    result: Dict[str, object] = {
        "best_epoch": best_epoch,
        "best_validation_mean_weighted_f1": best_score,
        "test": test_metrics,
        "mask_sha256": mask_hashes,
        "registered_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "ema_steps": model.ema_step,
        "evaluation_stage": (
            "train-validation-test"
            if config_value.evaluate_test
            else "train-validation-only"
        ),
        "jepa_regression_aggregation": config_value.jepa_regression_aggregation,
        "recurrent_padding_mode": config_value.recurrent_padding_mode,
        "task_regression_loss": config_value.task_regression_loss,
        "task_smooth_l1_beta": config_value.task_smooth_l1_beta,
        "postgraph_sequence_mode": config_value.postgraph_sequence_mode,
        "jepa_rate_weighting": config_value.jepa_rate_weighting,
        "graph_message_calibration": config_value.graph_message_calibration,
        "backbone": "sdt-style-full-context",
        "transformer": {
            "d_model": config_value.transformer_dim,
            "heads": config_value.transformer_heads,
            "layers": config_value.transformer_layers,
            "ff_dim": config_value.transformer_ff_dim,
        },
        **_backbone_parameter_provenance(model),
    }
    _write_json(output / "metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Expose run identity plus explicit, single-valued protocol markers."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("CMUMOSI",), default="CMUMOSI")
    parser.add_argument("--train-rate-mode", choices=("all",), default="all")
    parser.add_argument("--fusion-type", choices=("slot",), default="slot")
    parser.add_argument("--lr", type=float, choices=(5e-4,), default=5e-4)
    parser.add_argument("--audio-feature", required=True)
    parser.add_argument("--text-feature", required=True)
    parser.add_argument("--video-feature", required=True)
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-test-evaluation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_value = SDTTrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        evaluate_test=not args.skip_test_evaluation,
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
