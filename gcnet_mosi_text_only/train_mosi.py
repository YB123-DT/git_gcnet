"""Train the strict frozen-feature CMU-MOSI Text-only diagnostic."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from gcnet_missing_m3_sam_backbone.train_mosi import (
    _atomic_save_npz,
    _collect,
    _stable_seed,
    _state_to_cpu,
    regression_loss,
    regression_metrics,
    set_random_seed,
    write_json,
)

from .model import TextOnlyTemporalModel


@dataclass(frozen=True)
class TextOnlyConfig:
    seed: int = 66
    hidden_dim: int = 200
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    device: str = "cuda"
    test_every_epoch: bool = False


def text_batch(raw, device):
    if len(raw) < 10:
        raise ValueError("CMU-MOSI batch must contain ten fields")
    text = raw[1].to(device)
    umask = raw[7].to(device)
    labels = raw[8].to(device)
    return {"text": text, "umask": umask, "labels": labels}


def select_best_epoch(records):
    if not records:
        raise ValueError("records must not be empty")
    return int(max(records, key=lambda item: float(item["validation"]["weighted_f1"]))["epoch"])


def select_test_oracle(records):
    if not records or any("test" not in record for record in records):
        raise ValueError("every record must contain test metrics")
    best = max(records, key=lambda item: float(item["test"]["weighted_f1"]))
    return int(best["epoch"]), dict(best["test"])


def _run_epoch(model, loader, device, optimizer=None, gradient_clip_norm=0.0):
    training = optimizer is not None
    model.train(training)
    losses, predictions, labels = [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            view = text_batch(raw, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output, _ = model(view["text"], view["umask"])
            loss = regression_loss(output, view["labels"], view["umask"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite loss")
            if training:
                loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            predicted, expected = _collect(output, view["labels"], view["umask"])
            predictions.append(predicted)
            labels.append(expected)
            losses.append(float(loss.detach()))
    if not losses:
        raise RuntimeError("loader produced no batches")
    prediction_array, label_array = np.concatenate(predictions), np.concatenate(labels)
    metrics = regression_metrics(label_array, prediction_array)
    metrics["loss"] = float(np.mean(losses))
    return metrics, {"predictions": prediction_array, "labels": label_array}


def run_experiment(config_value, feature_root, output_dir):
    from gcnet_modality_jepa.train_gcnet import get_loaders

    root, output = Path(feature_root).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_names = ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
    feature_roots = [root / name for name in feature_names]
    if not all(path.is_dir() for path in feature_roots):
        raise FileNotFoundError("official MOSI feature roots are incomplete")
    payload = {
        **asdict(config_value),
        "dataset": "CMUMOSI",
        "variant": "strict-text-only-bigru",
        "text_feature": feature_names[1],
        "checkpoint_selection": "validation_weighted_f1",
        "excluded_components": ["audio", "visual", "gcnet", "jepa", "mmoe", "completion"],
    }
    write_json(output / "config.json", payload)
    set_random_seed(config_value.seed)
    loaders = get_loaders(
        audio_root=str(feature_roots[0]),
        text_root=str(feature_roots[1]),
        video_root=str(feature_roots[2]),
        num_folder=1,
        dataset="CMUMOSI",
        batch_size=config_value.batch_size,
        num_workers=0,
        seed=config_value.seed,
        validation_fraction=0.1,
        evaluation_protocol="official",
    )
    train_loader, validation_loader, test_loader = loaders[0][0], loaders[1][0], loaders[2][0]
    text_dim = int(loaders[4])
    set_random_seed(_stable_seed(config_value.seed, "strict_text_only_init"))
    device = torch.device(config_value.device)
    model = TextOnlyTemporalModel(text_dim, config_value.hidden_dim, config_value.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_value.learning_rate, weight_decay=config_value.weight_decay)
    history, best_state, best_validation, best_epoch, best_score = [], None, None, 0, -math.inf
    started = time.time()
    for epoch in range(1, config_value.epochs + 1):
        train_metrics, _ = _run_epoch(model, train_loader, device, optimizer, config_value.gradient_clip_norm)
        validation_metrics, _ = _run_epoch(model, validation_loader, device)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        if config_value.test_every_epoch:
            record["test"], _ = _run_epoch(model, test_loader, device)
        history.append(record)
        write_json(output / "history.json", history)
        score = float(validation_metrics["weighted_f1"])
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state, best_validation = _state_to_cpu(model), dict(validation_metrics)
        print("epoch={:03d} train_f1={:.4f} validation_f1={:.4f}".format(epoch, train_metrics["weighted_f1"], score), flush=True)
    if best_state is None or select_best_epoch(history) != best_epoch:
        raise RuntimeError("checkpoint selection provenance failed")
    torch.save({"model": best_state, "config": payload, "epoch": best_epoch}, output / "best.pt")
    model.load_state_dict(best_state, strict=True)
    test_metrics, artifacts = _run_epoch(model, test_loader, device)
    _atomic_save_npz(output / "predictions.npz", artifacts)
    metrics = {
        "variant": "strict-text-only-bigru",
        "seed": config_value.seed,
        "best_epoch": best_epoch,
        "selection_split": "validation_weighted_f1",
        "validation": best_validation,
        "test": test_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history_epochs": len(history),
        "runtime_seconds": time.time() - started,
        "collapsed": test_metrics["prediction_std"] < 1e-8 or test_metrics["predicted_sign_count"] < 2,
    }
    if config_value.test_every_epoch:
        oracle_epoch, oracle_metrics = select_test_oracle(history)
        metrics["test_oracle"] = {
            "best_epoch": oracle_epoch,
            "metrics": oracle_metrics,
            "weighted_f1_gap": float(oracle_metrics["weighted_f1"]) - float(test_metrics["weighted_f1"]),
        }
    write_json(output / "metrics.json", metrics)
    return metrics


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-every-epoch", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_experiment(
        TextOnlyConfig(
            seed=args.seed,
            epochs=args.epochs,
            device=args.device,
            test_every_epoch=args.test_every_epoch,
        ),
        args.feature_root,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
