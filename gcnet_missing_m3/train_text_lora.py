"""Train and export leakage-safe MOSI RoBERTa-large LoRA text features."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .text_lora import (
    MosiRecord,
    SentimentRegressor,
    build_lora_encoder,
    canonical_feature_hash,
    canonical_uid_hash,
    load_mosi_records,
    masked_mean_pool,
    sha256_path,
    trainable_state_dict,
)


DEFAULT_LABEL_PICKLE = "dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl"
DEFAULT_BASE_MODEL = "/data2/yb/pretrained/roberta-large-722cf37"
DEFAULT_OUTPUT_DIR = "dataset/CMUMOSI/features/roberta-large-lora-r8-mean-UTT"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 66
    batch_size: int = 16
    max_length: int = 192
    max_epochs: int = 20
    patience: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    amp: bool = True


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_metrics: Dict[str, float]
    best_state: Dict[str, torch.Tensor]
    history: list[Dict[str, float]]


class _RecordDataset(Dataset):
    def __init__(self, records: Sequence[MosiRecord]) -> None:
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MosiRecord:
        return self.records[index]


def _collator(tokenizer, max_length: int, include_labels: bool):
    def collate(records: Sequence[MosiRecord]) -> Dict[str, object]:
        encoded = dict(
            tokenizer(
                [record.text for record in records],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        )
        encoded["uids"] = [record.uid for record in records]
        if include_labels:
            labels = [record.label for record in records]
            if any(label is None for label in labels):
                raise ValueError("supervised loader received a record without a label")
            encoded["labels"] = torch.tensor(labels, dtype=torch.float32)
        return encoded

    return collate


def build_training_loaders(
    splits: Mapping[str, Sequence[MosiRecord]],
    tokenizer,
    *,
    batch_size: int = 16,
    seed: int = 66,
    max_length: int = 192,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build only train and validation loaders; the test key is never read."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        _RecordDataset(splits["train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=_collator(tokenizer, max_length, True),
    )
    validation_loader = DataLoader(
        _RecordDataset(splits["validation"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collator(tokenizer, max_length, True),
    )
    return train_loader, validation_loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _move_model_inputs(batch: Mapping[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        name: value.to(device)
        for name, value in batch.items()
        if name not in {"labels", "uids"} and torch.is_tensor(value)
    }


def _weighted_binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    actual = labels >= 0
    predicted = predictions >= 0
    total = len(actual)
    if total == 0:
        return 0.0
    result = 0.0
    for positive in (False, True):
        support = int(np.sum(actual == positive))
        true_positive = int(np.sum((actual == positive) & (predicted == positive)))
        false_positive = int(np.sum((actual != positive) & (predicted == positive)))
        false_negative = int(np.sum((actual == positive) & (predicted != positive)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = (2 * true_positive / denominator) if denominator else 0.0
        result += support / total * f1
    return float(result)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        prediction = model(**_move_model_inputs(batch, device))
        predictions.append(prediction.detach().float().cpu().numpy().reshape(-1))
        labels.append(batch["labels"].detach().float().cpu().numpy().reshape(-1))
    prediction_array = np.concatenate(predictions)
    label_array = np.concatenate(labels)
    mae = float(np.mean(np.abs(prediction_array - label_array)))
    if np.std(prediction_array) == 0 or np.std(label_array) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(prediction_array, label_array)[0, 1])
    return {
        "mae": mae,
        "correlation": correlation,
        "weighted_f1": _weighted_binary_f1(label_array, prediction_array),
    }


def fit_model(
    model: nn.Module,
    train_loader: Iterable[Mapping[str, object]],
    validation_loader: Iterable[Mapping[str, object]],
    config: TrainingConfig,
    *,
    device: torch.device,
    evaluate_fn: Callable[..., Dict[str, float]] = evaluate_model,
) -> FitResult:
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.SmoothL1Loss()
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_mae = math.inf
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    best_state: Dict[str, torch.Tensor] = {}
    history: list[Dict[str, float]] = []
    stale_epochs = 0

    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predictions = model(**_move_model_inputs(batch, device))
                loss = loss_function(predictions.reshape(-1), labels.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            count = int(labels.numel())
            total_loss += float(loss.detach()) * count
            sample_count += count
        metrics = evaluate_fn(model, validation_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / max(sample_count, 1),
            **{name: float(value) for name, value in metrics.items()},
        }
        history.append(entry)
        validation_mae = float(metrics["mae"])
        if not math.isfinite(validation_mae):
            raise ValueError("validation MAE is not finite")
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_epoch = epoch
            best_metrics = {name: float(value) for name, value in metrics.items()}
            best_state = trainable_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state, strict=False)
    return FitResult(best_epoch, best_metrics, best_state, history)


@torch.no_grad()
def export_feature_bank(
    encoder: nn.Module,
    tokenizer,
    records: Sequence[object],
    output_dir: str | Path,
    *,
    batch_size: int = 16,
    max_length: int = 192,
    device: torch.device,
    expected_count: int = 2199,
    hidden_size: int = 1024,
) -> Dict[str, str]:
    if len(records) != expected_count:
        raise ValueError(f"expected {expected_count} export records, got {len(records)}")
    uids = [str(record.uid) for record in records]
    uid_set = set(uids)
    if len(uid_set) != len(uids):
        raise ValueError("export records contain duplicate UIDs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unexpected_existing = {
        path.stem for path in output.glob("*.npy") if path.stem not in uid_set
    }
    if unexpected_existing:
        raise ValueError(
            f"output contains feature files for unknown UIDs: {sorted(unexpected_existing)}"
        )
    loader = DataLoader(
        _RecordDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collator(tokenizer, max_length, False),
    )
    special_token_ids = {
        int(token_id)
        for token_id in (
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        if token_id is not None
    }
    encoder.to(device)
    encoder.eval()
    feature_hashes: Dict[str, str] = {}
    for batch in loader:
        inputs = _move_model_inputs(batch, device)
        result = encoder(**inputs)
        hidden = result.last_hidden_state if hasattr(result, "last_hidden_state") else result[0]
        pooled = masked_mean_pool(
            hidden, inputs["input_ids"], inputs["attention_mask"], special_token_ids
        )
        values = pooled.detach().float().cpu().numpy()
        for uid, value in zip(batch["uids"], values):
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (hidden_size,):
                raise ValueError(f"feature {uid} has shape {value.shape}, expected {(hidden_size,)}")
            if not np.isfinite(value).all():
                raise ValueError(f"feature {uid} contains a non-finite value")
            path = output / f"{uid}.npy"
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
            os.replace(temporary, path)
            feature_hashes[uid] = sha256_path(path)
    if len(feature_hashes) != expected_count:
        raise RuntimeError("feature export count changed while writing")
    return feature_hashes


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-pickle", default=DEFAULT_LABEL_PICKLE)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    final_output = Path(args.output_dir)
    if final_output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing feature bank: {final_output}"
        )
    config = TrainingConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        amp=not args.no_amp,
    )
    set_seed(config.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    splits = load_mosi_records(
        args.label_pickle, label_splits=("train", "validation")
    )
    encoder = build_lora_encoder(args.base_model)
    special_token_ids = {
        token_id
        for token_id in (
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        if token_id is not None
    }
    model = SentimentRegressor(encoder, special_token_ids)
    train_loader, validation_loader = build_training_loaders(
        splits,
        tokenizer,
        batch_size=config.batch_size,
        seed=config.seed,
        max_length=config.max_length,
        num_workers=args.num_workers,
    )
    result = fit_model(
        model, train_loader, validation_loader, config, device=device
    )

    output = final_output.with_name(final_output.name + f".staging-{os.getpid()}")
    output.mkdir(parents=True, exist_ok=True)
    adapter_dir = output / "adapter"
    model.encoder.save_pretrained(adapter_dir)
    head_path = output / "sentiment_head.pt"
    head_temporary = head_path.with_suffix(head_path.suffix + ".tmp")
    torch.save(model.head.state_dict(), head_temporary)
    os.replace(head_temporary, head_path)
    _write_json(output / "history.json", result.history)

    label_free = load_mosi_records(args.label_pickle, label_splits=())
    all_records = tuple(
        record
        for name in ("train", "validation", "test")
        for record in label_free[name]
    )
    feature_hashes = export_feature_bank(
        model.encoder,
        tokenizer,
        all_records,
        output,
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=device,
    )
    split_uids = {
        name: [record.uid for record in label_free[name]]
        for name in ("train", "validation", "test")
    }
    base_model_hash = sha256_path(args.base_model)
    weight_path = Path(args.base_model) / "model.safetensors"
    manifest = {
        "base_model_path": str(Path(args.base_model).resolve()),
        "base_model_sha256": base_model_hash,
        "base_model_weight_sha256": sha256_path(weight_path),
        "label_pickle_sha256": sha256_path(args.label_pickle),
        "lora": {
            "target_modules": ["query", "value"],
            "target_count": 48,
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
        },
        "pooling": "attention-masked mean excluding BOS/EOS/PAD",
        "training": asdict(config),
        "best_epoch": result.best_epoch,
        "best_validation_metrics": result.best_metrics,
        "uids": [record.uid for record in all_records],
        "split_counts": {name: len(uids) for name, uids in split_uids.items()},
        "split_uid_sha256": {
            name: canonical_uid_hash(uids) for name, uids in split_uids.items()
        },
        "feature_count": len(feature_hashes),
        "feature_aggregate_sha256": canonical_feature_hash(feature_hashes),
        "adapter_sha256": sha256_path(adapter_dir),
        "sentiment_head_sha256": sha256_path(output / "sentiment_head.pt"),
        "source_file_sha256": {
            path.name: sha256_path(path)
            for path in (
                Path(__file__).resolve().parent / "__init__.py",
                Path(__file__).resolve().parent / "text_lora.py",
                Path(__file__).resolve(),
            )
        },
        "git_commit": _git_commit(),
    }
    _write_json(output / "manifest.json", manifest)
    os.replace(output, final_output)


if __name__ == "__main__":
    main()
