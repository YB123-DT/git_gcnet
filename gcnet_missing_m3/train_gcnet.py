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
    BalancedBatchRateSchedule,
    mean_validation_weighted_f1,
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
    jepa_weight: float = 0.1
    temperature: float = 0.03
    ema_tau: float = 0.996
    gradient_clip_norm: float = 1.0
    time_attention: bool = False
    evaluation_protocol: str = "official"
    validation_fraction: float = 0.1
    device: str = "cuda"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _state_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
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


def _prepare_view(
    data: Sequence[object],
    schedule: ConversationMaskSchedule,
    epoch: int,
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
    host_availability, guest_availability = build_primary_mask_tensors(
        schedule,
        conversation_ids=conversation_ids,
        umask=umask,
        epoch=epoch,
    )
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


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    umask: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.transpose(0, 1).reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1).long()
    selected = umask.reshape(-1).bool()
    return torch.nn.functional.cross_entropy(
        flat_logits[selected], flat_labels[selected]
    )


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def _collect_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    umask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = logits.argmax(dim=-1).transpose(0, 1)
    selected = umask.bool()
    return (
        predicted[selected].detach().cpu().numpy(),
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
    model.train()
    rate_schedule = BalancedBatchRateSchedule()
    losses: list[float] = []
    cls_losses: list[float] = []
    jepa_losses: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    rate_counts = {rate: 0 for rate in MISSING_RATES}
    target_count = 0
    for batch_index, raw in enumerate(loader):
        rate = rate_schedule.rate_for(epoch, batch_index)
        rate_counts[rate] += 1
        data = _move_batch(raw, device)
        view = _prepare_view(data, schedules[rate], epoch, dimensions)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _, predictions = model(
            [view["incomplete"]],
            view["availability"],
            view["qmask"],
            view["umask"],
            view["lengths"],
            predict_missing=True,
        )
        cls = _classification_loss(logits, view["labels"], view["umask"])
        with torch.no_grad():
            teacher = model.encode_teacher_targets([view["complete"]])
        jepa: MissingM3Loss = missing_m3_loss(
            predictions, teacher, temperature=config.temperature
        )
        loss = cls + config.jepa_weight * jepa.total
        if not bool(torch.isfinite(loss.detach())):
            raise ValueError("training loss must be finite")
        loss.backward()
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                config.gradient_clip_norm,
            )
        optimizer.step()
        model.update_teacher(config.ema_tau)
        predicted, expected = _collect_predictions(
            logits, view["labels"], view["umask"]
        )
        all_predictions.append(predicted)
        all_labels.append(expected)
        losses.append(float(loss.detach()))
        cls_losses.append(float(cls.detach()))
        jepa_losses.append(float(jepa.total.detach()))
        target_count += jepa.target_count
    metrics = _metrics(np.concatenate(all_labels), np.concatenate(all_predictions))
    return {
        **metrics,
        "loss": float(np.mean(losses)),
        "classification_loss": float(np.mean(cls_losses)),
        "jepa_loss": float(np.mean(jepa_losses)),
        "jepa_target_count": int(target_count),
        "rate_batch_counts": {str(rate): count for rate, count in rate_counts.items()},
    }


@torch.no_grad()
def evaluate_rate(
    model: MissingM3GraphModel,
    loader: Iterable[Sequence[object]],
    schedule: ConversationMaskSchedule,
    dimensions: tuple[int, int, int],
    device: torch.device,
    collect: bool,
) -> tuple[Dict[str, float], Dict[str, np.ndarray] | None]:
    model.eval()
    losses: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_availability: list[np.ndarray] = []
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
        loss = _classification_loss(logits, view["labels"], view["umask"])
        predicted, expected = _collect_predictions(
            logits, view["labels"], view["umask"]
        )
        all_predictions.append(predicted)
        all_labels.append(expected)
        losses.append(float(loss))
        if collect:
            all_availability.append(
                view["availability"][view["umask"].T.bool()].cpu().numpy()
            )
    predictions_array = np.concatenate(all_predictions)
    labels_array = np.concatenate(all_labels)
    metrics = {
        **_metrics(labels_array, predictions_array),
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
        metrics["mask_sha256"] = _sha256_tensor(
            torch.from_numpy(availability_array)
        )
    return metrics, artifacts


def run_experiment(
    config_value: TrainConfig,
    audio_root: str,
    text_root: str,
    visual_root: str,
    output_dir: str | Path,
) -> Dict[str, object]:
    if config_value.dataset != "IEMOCAPSix":
        raise ValueError("the first experiment is locked to IEMOCAPSix")
    if config_value.fold != 5:
        raise ValueError("the first experiment is locked to fold 5")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", asdict(config_value))
    set_random_seed(config_value.seed)
    device = torch.device(config_value.device)
    loaders = get_loaders(
        audio_root=audio_root,
        text_root=text_root,
        video_root=visual_root,
        num_folder=5,
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
        n_speakers=2,
        window_past=config_value.window_past,
        window_future=config_value.window_future,
        n_classes=6,
        dropout=config_value.dropout,
        time_attn=config_value.time_attention,
        no_cuda=device.type != "cuda",
        latent_dim=config_value.latent_dim,
        num_experts=config_value.num_experts,
        top_k=config_value.top_k,
        projector_dropout=config_value.projector_dropout,
        predictor_dropout=config_value.predictor_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
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
                dimensions,
                device,
                collect=False,
            )
        validation_mean = mean_validation_weighted_f1(validation)
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": {str(rate): value for rate, value in validation.items()},
            "validation_mean_weighted_f1": validation_mean,
        }
        history.append(record)
        _write_json(output / "history.json", history)
        print(
            "epoch={:03d} train_wf1={:.4f} val8_wf1={:.4f} cls={:.4f} jepa={:.4f}".format(
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
            torch.save(
                {
                    "model": best_state,
                    "config": asdict(config_value),
                    "epoch": best_epoch,
                    "validation_mean_weighted_f1": best_score,
                },
                output / "best.pt",
            )
    if best_state is None:
        raise RuntimeError("no best checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    test_metrics: Dict[str, Dict[str, float]] = {}
    mask_hashes: Dict[str, str] = {}
    for rate in MISSING_RATES:
        metrics, artifacts = evaluate_rate(
            model,
            test_loader,
            test_schedules[rate],
            dimensions,
            device,
            collect=True,
        )
        if artifacts is None:
            raise RuntimeError("test artifacts were not collected")
        rate_key = format(rate, ".1f")
        test_metrics[rate_key] = metrics
        mask_hashes[rate_key] = str(metrics["mask_sha256"])
        np.savez_compressed(
            output / ("predictions_miss_" + rate_key.replace(".", "p") + ".npz"),
            **artifacts,
        )
    result: Dict[str, object] = {
        "best_epoch": best_epoch,
        "best_validation_mean_weighted_f1": best_score,
        "test": test_metrics,
        "mask_sha256": mask_hashes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "ema_steps": model.ema_step,
    }
    _write_json(output / "metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-feature", required=True)
    parser.add_argument("--text-feature", required=True)
    parser.add_argument("--video-feature", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.03)
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


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(args.num_threads)
    config_value = TrainConfig(
        fold=args.fold,
        seed=args.seed,
        window_past=args.windowp,
        window_future=args.windowf,
        hidden=args.hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.l2,
        latent_dim=args.latent_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        jepa_weight=args.jepa_weight,
        temperature=args.temperature,
        ema_tau=args.ema_tau,
        gradient_clip_norm=args.gradient_clip_norm,
        time_attention=args.time_attn,
        evaluation_protocol=args.evaluation_protocol,
        validation_fraction=args.validation_fraction,
        device=args.device,
    )
    feature_root = config.PATH_TO_FEATURES[config_value.dataset]
    roots = [
        os.path.join(feature_root, name)
        for name in (args.audio_feature, args.text_feature, args.video_feature)
    ]
    if not all(os.path.exists(root) for root in roots):
        raise FileNotFoundError("one or more feature roots do not exist")
    run_experiment(config_value, *roots, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
