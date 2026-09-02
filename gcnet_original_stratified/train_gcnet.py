"""Equal-budget stratified trainer for the matched Original GCNet control."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

import config
from gcnet_missing_m3.mixed_rate import (
    MISSING_RATES,
    STRATIFIED_RATE_ALGORITHM,
    stratified_rates_for_batch,
)
from gcnet_missing_m3.train_gcnet import (
    _collect_predictions,
    _metrics,
    _move_batch,
    _prepare_stratified_view,
    _resolve_task_contract,
    _schedules,
    _save_best_checkpoint,
    _state_to_cpu,
    _task_loss,
    _write_json,
    evaluate_rate,
    get_loaders,
    set_random_seed,
)
from gcnet_modality_jepa.loss import MaskedReconLoss
from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule
from gcnet_modality_jepa.protocol import SeedBundle

from .model import OriginalGCNetControl


@dataclass(frozen=True)
class OriginalTrainConfig:
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
    gradient_clip_norm: float = 1.0
    time_attention: bool = False
    evaluation_protocol: str = "official"
    validation_fraction: float = 0.1
    device: str = "cuda"
    train_rate_mode: str = "stratified"
    evaluate_test: bool = True

    def __post_init__(self) -> None:
        if self.train_rate_mode != "stratified":
            raise ValueError(
                "Original matched control requires train_rate_mode='stratified'"
            )


def original_control_loss(
    *,
    logits: torch.Tensor,
    reconstruction: Sequence[torch.Tensor],
    complete_features: torch.Tensor,
    availability: torch.Tensor,
    labels: torch.Tensor,
    umask: torch.Tensor,
    dataset: str,
    dimensions: tuple[int, int, int],
    reconstruction_criterion: MaskedReconLoss | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the formal Original task plus corrected missing-only MSE."""

    task = _task_loss(dataset, logits, labels, umask)
    criterion = reconstruction_criterion or MaskedReconLoss()
    reconstruction_loss = criterion(
        list(reconstruction),
        [complete_features],
        [availability],
        umask,
        *dimensions,
    )
    return task + reconstruction_loss, task, reconstruction_loss


def _epoch_size(loader: Iterable[Sequence[object]]) -> int:
    for value in (getattr(loader, "sampler", None), getattr(loader, "dataset", None)):
        if value is None:
            continue
        try:
            size = len(value)
        except TypeError:
            continue
        if size <= 0:
            raise ValueError("stratified loader must contain conversations")
        return int(size)
    raise TypeError(
        "stratified loader requires a sized sampler or sized dataset"
    )


def train_epoch(
    model: OriginalGCNetControl,
    loader: Iterable[Sequence[object]],
    optimizer: torch.optim.Optimizer,
    config: OriginalTrainConfig,
    schedules: Mapping[float, ConversationMaskSchedule],
    epoch: int,
    dimensions: tuple[int, int, int],
    device: torch.device,
) -> Dict[str, object]:
    """Train one equal-budget epoch and retain raw protocol audit counts."""

    if config.train_rate_mode != "stratified":
        raise ValueError("Original matched control only supports stratified training")
    model.train()
    reconstruction_criterion = MaskedReconLoss()
    epoch_size = _epoch_size(loader)
    conversations_seen = 0
    optimizer_steps = 0
    source_conversation_count = 0
    masked_view_count = 0
    model_forward_count = 0
    total_losses: list[float] = []
    task_losses: list[float] = []
    reconstruction_losses: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    assignment_digest = hashlib.sha256()
    rate_conversation_counts = {rate: 0 for rate in MISSING_RATES}
    rate_valid_utterance_counts = {rate: 0 for rate in MISSING_RATES}
    rate_missing_modality_counts = {rate: 0 for rate in MISSING_RATES}
    rate_modality_element_counts = {rate: 0 for rate in MISSING_RATES}

    for batch_index, raw in enumerate(loader):
        data = _move_batch(raw, device)
        conversation_ids = tuple(str(value) for value in data[-1])
        batch_size = len(conversation_ids)
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
        logits, reconstruction, _, predictions = model(
            [view["incomplete"]],
            view["availability"],
            view["qmask"],
            view["umask"],
            view["lengths"],
            predict_missing=False,
        )
        model_forward_count += 1
        if predictions is not None:
            raise RuntimeError("Original inference path returned missing predictions")
        total, task, reconstruction_loss = original_control_loss(
            logits=logits,
            reconstruction=reconstruction,
            complete_features=view["complete"],
            availability=view["availability"],
            labels=view["labels"],
            umask=view["umask"],
            dataset=config.dataset,
            dimensions=dimensions,
            reconstruction_criterion=reconstruction_criterion,
        )
        if not bool(torch.isfinite(total.detach())):
            raise ValueError("training loss must be finite")
        total.backward()
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
        optimizer.step()
        optimizer_steps += 1

        predicted, expected, _ = _collect_predictions(
            config.dataset,
            logits,
            view["labels"],
            view["umask"],
        )
        all_predictions.append(predicted)
        all_labels.append(expected)
        total_losses.append(float(total.detach()))
        task_losses.append(float(task.detach()))
        reconstruction_losses.append(float(reconstruction_loss.detach()))

        assignment_digest.update(b"\0")
        assignment_digest.update(assignment.assignment_hash.encode("ascii"))
        for conversation_index, rate in enumerate(assignment.rates):
            valid = view["umask"][conversation_index].bool()
            valid_availability = view["availability"][:, conversation_index][valid]
            valid_utterances = int(valid.sum().item())
            missing_modalities = int(valid_availability.eq(0).sum().item())
            rate_conversation_counts[rate] += 1
            rate_valid_utterance_counts[rate] += valid_utterances
            rate_missing_modality_counts[rate] += missing_modalities
            rate_modality_element_counts[rate] += int(valid_availability.numel())

        conversations_seen += batch_size
        source_conversation_count += batch_size
        masked_view_count += batch_size

    if conversations_seen != epoch_size:
        raise RuntimeError(
            "stratified epoch expected {} conversations but observed {}".format(
                epoch_size, conversations_seen
            )
        )
    if not all_predictions:
        raise RuntimeError("training loader produced no batches")

    metrics = _metrics(
        config.dataset,
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
    )
    reconstruction_target_count = sum(rate_missing_modality_counts.values())
    return {
        **metrics,
        "loss": float(np.mean(total_losses)),
        "classification_loss": float(np.mean(task_losses)),
        "task_loss": float(np.mean(task_losses)),
        "reconstruction_loss": float(np.mean(reconstruction_losses)),
        "jepa_loss": 0.0,
        "jepa_target_count": 0,
        "reconstruction_target_count": reconstruction_target_count,
        "source_conversation_count": source_conversation_count,
        "masked_view_count": masked_view_count,
        "model_forward_count": model_forward_count,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_batches": 0,
        "rate_conversation_counts": {
            str(rate): rate_conversation_counts[rate] for rate in MISSING_RATES
        },
        "rate_valid_utterance_counts": {
            str(rate): rate_valid_utterance_counts[rate] for rate in MISSING_RATES
        },
        "rate_missing_modality_counts": {
            str(rate): rate_missing_modality_counts[rate] for rate in MISSING_RATES
        },
        "rate_modality_element_counts": {
            str(rate): rate_modality_element_counts[rate] for rate in MISSING_RATES
        },
        "rate_realized_missing_fraction": {
            str(rate): (
                rate_missing_modality_counts[rate]
                / rate_modality_element_counts[rate]
                if rate_modality_element_counts[rate]
                else None
            )
            for rate in MISSING_RATES
        },
        "rate_reconstruction_target_counts": {
            str(rate): rate_missing_modality_counts[rate] for rate in MISSING_RATES
        },
        "stratified_assignment_hash": assignment_digest.hexdigest(),
        "stratified_rate_algorithm": STRATIFIED_RATE_ALGORITHM,
        "model_arm": "original-gcnet",
        "training_objective": "classification-plus-masked-reconstruction",
        "reconstruction_loss_variant": "corrected-formal-repo",
        "reconstruction_weight": 1.0,
    }


def _build_model(
    config_value: OriginalTrainConfig,
    *,
    adim: int,
    tdim: int,
    vdim: int,
    n_speakers: int,
    n_classes: int,
    device: torch.device,
) -> OriginalGCNetControl:
    return OriginalGCNetControl(
        config_value.base_model,
        adim,
        tdim,
        vdim,
        config_value.hidden,
        config_value.hidden // 2,
        n_speakers=n_speakers,
        window_past=config_value.window_past,
        window_future=config_value.window_future,
        n_classes=n_classes,
        dropout=config_value.dropout,
        time_attn=config_value.time_attention,
        no_cuda=device.type != "cuda",
    ).to(device)


def run_experiment(
    config_value: OriginalTrainConfig,
    audio_root: str,
    text_root: str,
    visual_root: str,
    output_dir: str | Path,
) -> Dict[str, object]:
    """Train one matched Original control and evaluate one checkpoint at 8 rates."""

    if config_value.train_rate_mode != "stratified":
        raise ValueError("Original matched control only supports stratified training")
    shape = _resolve_task_contract(config_value.dataset, "regression")
    if not 1 <= config_value.fold <= int(shape["num_folds"]):
        raise ValueError("fold is outside the dataset fold range")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", asdict(config_value))

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
    model = _build_model(
        config_value,
        adim=adim,
        tdim=tdim,
        vdim=vdim,
        n_speakers=int(shape["num_speakers"]),
        n_classes=int(shape["num_classes"]),
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_value.learning_rate,
        weight_decay=config_value.weight_decay,
    )
    train_schedules = _schedules(config_value, "train")
    validation_schedules = _schedules(config_value, "validation")
    test_schedules = _schedules(config_value, "test")

    history: list[Dict[str, object]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    validation_mask_hashes: Dict[str, str] = {}
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
        validation_metrics: Dict[float, Dict[str, float]] = {}
        for rate in MISSING_RATES:
            metrics, _ = evaluate_rate(
                model,
                validation_loader,
                validation_schedules[rate],
                config_value.dataset,
                dimensions,
                device,
                collect=True,
            )
            validation_metrics[rate] = metrics
            validation_mask_hashes[format(rate, ".1f")] = str(
                metrics["mask_sha256"]
            )
        validation_mean = sum(
            float(validation_metrics[rate]["weighted_f1"])
            for rate in MISSING_RATES
        ) / len(MISSING_RATES)
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": {
                str(rate): value for rate, value in validation_metrics.items()
            },
            "validation_mean_weighted_f1": validation_mean,
        }
        history.append(record)
        _write_json(output / "history.json", history)
        print(
            "epoch={:03d} train_wf1={:.4f} validation_wf1={:.4f} "
            "task={:.4f} recon={:.4f}".format(
                epoch + 1,
                train_metrics["weighted_f1"],
                validation_mean,
                train_metrics["task_loss"],
                train_metrics["reconstruction_loss"],
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
                selection_split="validation",
            )

    if best_state is None:
        raise RuntimeError("no best checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    model.to(device)

    test_metrics: Dict[str, Dict[str, float]] = {}
    test_mask_hashes: Dict[str, str] = {}
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
            )
            if artifacts is None:
                raise RuntimeError("test artifacts were not collected")
            rate_key = format(rate, ".1f")
            test_metrics[rate_key] = metrics
            test_mask_hashes[rate_key] = str(metrics["mask_sha256"])
            np.savez_compressed(
                output
                / ("predictions_miss_" + rate_key.replace(".", "p") + ".npz"),
                **artifacts,
            )

    result: Dict[str, object] = {
        "best_epoch": best_epoch,
        "selection_split": "validation",
        "best_selection_mean_weighted_f1": best_score,
        "best_validation_mean_weighted_f1": best_score,
        "test": test_metrics,
        "mask_sha256": test_mask_hashes,
        "validation_mask_sha256": validation_mask_hashes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "ema_steps": 0,
        "evaluation_stage": (
            "train-validation-test"
            if config_value.evaluate_test
            else "train-validation-only"
        ),
        "model_arm": "original-gcnet",
        "training_objective": "classification-plus-masked-reconstruction",
        "reconstruction_loss_variant": "corrected-formal-repo",
        "reconstruction_weight": 1.0,
        "reccls_flag": False,
        "train_rate_mode": "stratified",
        "selection_missing_rates": list(MISSING_RATES),
        "model_initialization_seed": model_seed,
    }
    _write_json(output / "metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("IEMOCAPFour", "IEMOCAPSix", "CMUMOSI", "CMUMOSEI"),
        default="IEMOCAPSix",
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
        "--train-rate-mode", choices=("stratified",), default="stratified"
    )
    parser.add_argument("--base-model", choices=("LSTM", "GRU"), default="LSTM")
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--windowp", type=int, default=2)
    parser.add_argument("--windowf", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--time-attn", action="store_true")
    parser.add_argument(
        "--evaluation-protocol", choices=("official", "strict"), default="official"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.num_threads)
    config_value = OriginalTrainConfig(
        dataset=args.dataset,
        fold=args.fold,
        seed=args.seed,
        base_model=args.base_model,
        window_past=args.windowp,
        window_future=args.windowf,
        hidden=args.hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.l2,
        gradient_clip_norm=args.gradient_clip_norm,
        time_attention=args.time_attn,
        evaluation_protocol=args.evaluation_protocol,
        validation_fraction=args.validation_fraction,
        device=args.device,
        train_rate_mode=args.train_rate_mode,
        evaluate_test=not args.skip_test_evaluation,
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
