"""Complete-modality CMU-MOSI training gate for the SAM-style backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor

import config

from .model import MaskAwareSAMModel


@dataclass(frozen=True)
class SAMTrainConfig:
    dataset: str = "CMUMOSI"
    missing_rate: float = 0.0
    evaluation_protocol: str = "official"
    checkpoint_selection: str = "validation_loss"
    seed: int = 66
    width: int = 120
    heads: int = 4
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.1
    gradient_clip_norm: float = 1.0
    device: str = "cuda"
    evaluate_test: bool = True

    def __post_init__(self) -> None:
        if self.dataset != "CMUMOSI":
            raise ValueError("SAM Stage 1 is locked to CMUMOSI")
        if self.missing_rate != 0.0:
            raise ValueError("SAM Stage 1 is locked to missing_rate=0.0")
        if self.evaluation_protocol != "official":
            raise ValueError("SAM Stage 1 requires the official split")
        if self.checkpoint_selection != "validation_loss":
            raise ValueError("checkpoint selection must use validation loss")
        if self.width <= 0 or self.heads <= 0 or self.width % self.heads:
            raise ValueError("width must be positive and divisible by heads")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not 0.0 < self.learning_rate or not 0.0 <= self.weight_decay:
            raise ValueError("optimizer values must be non-negative")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _stable_seed(master_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(
        "{}:{}".format(master_seed, namespace).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complete_batch(
    raw: Sequence[object],
    device: torch.device,
) -> Dict[str, object]:
    if len(raw) < 10:
        raise ValueError("CMU-MOSI batch must contain ten fields")
    moved = [
        value.to(device) if torch.is_tensor(value) else value
        for value in raw
    ]
    features = moved[:3]
    umask = moved[7]
    labels = moved[8]
    if not torch.is_tensor(umask) or not torch.is_tensor(labels):
        raise TypeError("umask and labels must be tensors")
    availability = umask.transpose(0, 1).unsqueeze(-1).repeat(1, 1, 3)
    return {
        "features": features,
        "availability": availability,
        "umask": umask,
        "labels": labels,
        "conversation_ids": list(moved[9]),
    }


def regression_loss(
    prediction: Tensor,
    labels: Tensor,
    umask: Tensor,
) -> Tensor:
    predicted = prediction.squeeze(-1).transpose(0, 1)
    selected = umask.bool()
    if not bool(selected.any()):
        raise ValueError("batch has no valid utterance")
    return torch.nn.functional.mse_loss(
        predicted[selected],
        labels.to(predicted.dtype)[selected],
    )


def regression_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    if labels.shape != predictions.shape or labels.size == 0:
        raise ValueError("labels and predictions must be nonempty and aligned")
    nonzero = labels != 0
    if not bool(nonzero.any()):
        raise ValueError("MOSI metrics require at least one non-zero label")
    binary_labels = labels[nonzero] > 0
    binary_predictions = predictions[nonzero] > 0
    correlation = (
        float(np.corrcoef(labels, predictions)[0, 1])
        if labels.size >= 2
        and np.std(labels) > 0
        and np.std(predictions) > 0
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
        "sample_count": int(nonzero.sum()),
    }


def _collect(
    prediction: Tensor,
    labels: Tensor,
    umask: Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    predicted = prediction.squeeze(-1).transpose(0, 1)
    selected = umask.bool()
    return (
        predicted[selected].detach().cpu().numpy(),
        labels[selected].detach().cpu().numpy(),
    )


def train_epoch(
    model: MaskAwareSAMModel,
    loader: Iterable[Sequence[object]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
) -> Dict[str, object]:
    model.train()
    losses: List[float] = []
    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for raw in loader:
        view = complete_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        output, _, _ = model(
            view["features"],
            view["availability"],
            view["umask"],
        )
        loss = regression_loss(output, view["labels"], view["umask"])
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError("training loss is not finite")
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip_norm,
            )
        optimizer.step()
        predicted, expected = _collect(output, view["labels"], view["umask"])
        predictions.append(predicted)
        labels.append(expected)
        losses.append(float(loss.detach()))
    if not losses:
        raise RuntimeError("training loader produced no batches")
    return {
        **regression_metrics(np.concatenate(labels), np.concatenate(predictions)),
        "loss": float(np.mean(losses)),
    }


@torch.no_grad()
def evaluate_model(
    model: MaskAwareSAMModel,
    loader: Iterable[Sequence[object]],
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    model.eval()
    losses: List[float] = []
    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    conversation_ids: List[str] = []
    for raw in loader:
        view = complete_batch(raw, device)
        output, _, _ = model(
            view["features"],
            view["availability"],
            view["umask"],
        )
        loss = regression_loss(output, view["labels"], view["umask"])
        predicted, expected = _collect(output, view["labels"], view["umask"])
        predictions.append(predicted)
        labels.append(expected)
        conversation_ids.extend(view["conversation_ids"])
        losses.append(float(loss))
    if not losses:
        raise RuntimeError("evaluation loader produced no batches")
    prediction_array = np.concatenate(predictions)
    label_array = np.concatenate(labels)
    return (
        {
            **regression_metrics(label_array, prediction_array),
            "loss": float(np.mean(losses)),
        },
        {
            "predictions": prediction_array,
            "labels": label_array,
            "conversation_ids": np.asarray(conversation_ids),
        },
    )


def select_best_epoch(records: Sequence[Mapping[str, object]]) -> int:
    if not records:
        raise ValueError("records must not be empty")
    best = min(
        records,
        key=lambda record: float(record["validation"]["loss"]),
    )
    return int(best["epoch"])


def _state_to_cpu(model: torch.nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_model(
    model: MaskAwareSAMModel,
    train_loader: Iterable[Sequence[object]],
    validation_loader: Iterable[Sequence[object]],
    optimizer: torch.optim.Optimizer,
    config_value: SAMTrainConfig,
    device: torch.device,
    history_path: Path = None,
) -> Tuple[List[Dict[str, object]], Dict[str, Tensor], int, Dict[str, object]]:
    history: List[Dict[str, object]] = []
    best_state: Dict[str, Tensor] = {}
    best_epoch = 0
    best_validation: Dict[str, object] = {}
    best_loss = math.inf
    for epoch_index in range(config_value.epochs):
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_index)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config_value.gradient_clip_norm,
        )
        validation_metrics, _ = evaluate_model(
            model,
            validation_loader,
            device,
        )
        record = {
            "epoch": epoch_index + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        if history_path is not None:
            write_json(history_path, history)
        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch_index + 1
            best_state = _state_to_cpu(model)
            best_validation = dict(validation_metrics)
        print(
            "epoch={:03d} train_loss={:.4f} val_loss={:.4f} val_f1={:.2f}".format(
                epoch_index + 1,
                float(train_metrics["loss"]),
                validation_loss,
                100.0 * float(validation_metrics["weighted_f1"]),
            ),
            flush=True,
        )
    if not best_state:
        raise RuntimeError("training did not produce a best checkpoint")
    if select_best_epoch(history) != best_epoch:
        raise RuntimeError("best epoch provenance is inconsistent")
    return history, best_state, best_epoch, best_validation


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    os.replace(str(temporary), str(path))


def run_experiment(
    config_value: SAMTrainConfig,
    audio_root: str,
    text_root: str,
    visual_root: str,
    output_dir: str,
) -> Dict[str, object]:
    # Lazy import keeps the standalone SAM unit tests independent of PyG.
    from gcnet_modality_jepa.train_gcnet import get_loaders

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", asdict(config_value))
    set_random_seed(config_value.seed)
    device = torch.device(config_value.device)
    loaders = get_loaders(
        audio_root=audio_root,
        text_root=text_root,
        video_root=visual_root,
        num_folder=1,
        dataset="CMUMOSI",
        batch_size=config_value.batch_size,
        num_workers=0,
        seed=config_value.seed,
        validation_fraction=config_value.validation_fraction,
        evaluation_protocol=config_value.evaluation_protocol,
    )
    train_loader = loaders[0][0]
    validation_loader = loaders[1][0]
    test_loader = loaders[2][0]
    adim, tdim, vdim = loaders[3:]
    set_random_seed(_stable_seed(config_value.seed, "sam_model_init"))
    model = MaskAwareSAMModel(
        adim,
        tdim,
        vdim,
        width=config_value.width,
        heads=config_value.heads,
        dropout=config_value.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_value.learning_rate,
        weight_decay=config_value.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    history, best_state, best_epoch, best_validation = train_model(
        model,
        train_loader,
        validation_loader,
        optimizer,
        config_value,
        device,
        history_path=output / "history.json",
    )
    checkpoint_path = output / "best_checkpoint.pt"
    torch.save(
        {
            "model": best_state,
            "config": asdict(config_value),
            "epoch": best_epoch,
            "validation": best_validation,
            "selection_split": "validation",
        },
        str(checkpoint_path),
    )
    model.load_state_dict(best_state, strict=True)
    test_metrics = None
    artifacts = None
    if config_value.evaluate_test:
        test_metrics, artifacts = evaluate_model(model, test_loader, device)
        _atomic_save_npz(output / "predictions.npz", artifacts)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "variant": "mask-aware-sam-backbone",
        "dataset": "CMUMOSI",
        "missing_rate": 0.0,
        "seed": config_value.seed,
        "best_epoch": best_epoch,
        "selection_split": "validation",
        "validation": best_validation,
        "test": test_metrics,
        "registered_parameters": parameter_count,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "runtime_seconds": time.time() - started,
        "history_epochs": len(history),
        "collapsed": (
            None
            if test_metrics is None
            else bool(
                float(test_metrics["prediction_std"]) < 1e-8
                or int(test_metrics["predicted_sign_count"]) < 2
            )
        ),
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    feature_root = Path(args.feature_root).expanduser().resolve()
    names = (
        "wav2vec-large-c-UTT",
        "deberta-large-4-UTT",
        "manet_UTT",
    )
    roots = [feature_root / name for name in names]
    if not all(root.is_dir() for root in roots):
        raise FileNotFoundError("one or more official feature roots are missing")
    run_experiment(
        SAMTrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            device=args.device,
            evaluate_test=not args.skip_test,
        ),
        *(str(root) for root in roots),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SAMTrainConfig",
    "complete_batch",
    "evaluate_model",
    "regression_metrics",
    "run_experiment",
    "select_best_epoch",
    "train_model",
    "write_json",
]
