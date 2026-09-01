"""Fine-tune the locked complete-M3 projectors with a temporal residual."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gcnet_missing_m3_sam_backbone.train_mosi import regression_metrics, write_json

from .model import CompleteM3Regressor


FEATURE_NAMES = ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")


@dataclass(frozen=True)
class Config:
    seed: int
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    latent_dim: int = 256
    projector_dropout: float = 0.1
    fusion_dropout: float = 0.2
    selection_split: str = "validation_weighted_f1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_standardizer(loader) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    sampler_indices = getattr(loader.sampler, "indices", None)
    if sampler_indices is None:
        raise ValueError("training loader must expose its split indices")
    deterministic = DataLoader(
        Subset(loader.dataset, sorted(int(index) for index in sampler_indices)),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=loader.collate_fn,
    )
    values = [[], [], []]
    for raw in deterministic:
        valid = raw[7].transpose(0, 1).bool()
        for index in range(3):
            values[index].append(raw[index][valid])
    stacked = [torch.cat(parts, dim=0) for parts in values]
    means = [value.mean(dim=0) for value in stacked]
    stds = [value.std(dim=0).clamp_min(1e-6) for value in stacked]
    return means, stds


def prepare(raw, means, stds, device):
    features = [
        (raw[index].to(device) - means[index]) / stds[index]
        for index in range(3)
    ]
    return features, raw[7].to(device), raw[8].to(device)


def loss_and_arrays(model, raw, means, stds, device):
    features, umask, labels = prepare(raw, means, stds, device)
    prediction = model(features, umask).squeeze(-1).transpose(0, 1)
    valid = umask.bool()
    loss = torch.nn.functional.mse_loss(prediction[valid], labels[valid])
    return loss, prediction[valid].detach().cpu().numpy(), labels[valid].cpu().numpy()


def run_epoch(model, loader, means, stds, device, optimizer=None):
    model.train(optimizer is not None)
    losses, predictions, labels = [], [], []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for raw in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, predicted, expected = loss_and_arrays(
                model, raw, means, stds, device
            )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach()))
            predictions.append(predicted)
            labels.append(expected)
    metrics = regression_metrics(np.concatenate(labels), np.concatenate(predictions))
    metrics["loss"] = float(np.mean(losses))
    return metrics


def _load_projectors(model, checkpoint: Path) -> None:
    state = torch.load(str(checkpoint), map_location="cpu")
    expected = model.projectors.state_dict()
    if set(state) != set(expected):
        raise ValueError("stage1 projector checkpoint key mismatch")
    model.projectors.load_state_dict(state, strict=True)


def run(config: Config, feature_root: Path, stage1_checkpoint: Path, output: Path):
    from gcnet_modality_jepa.train_gcnet import get_loaders

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", asdict(config))
    roots = [str(feature_root / name) for name in FEATURE_NAMES]
    loaders = get_loaders(
        audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset="CMUMOSI", batch_size=config.batch_size,
        num_workers=0, seed=config.seed, validation_fraction=0.1,
        evaluation_protocol="official",
    )
    train_loader, validation_loader, test_loader = loaders[0][0], loaders[1][0], loaders[2][0]
    means, stds = compute_standardizer(train_loader)
    set_seed(config.seed)
    model = CompleteM3Regressor(
        tuple(loaders[3:]), latent_dim=config.latent_dim,
        projector_dropout=config.projector_dropout,
        dropout=config.fusion_dropout, temporal_context=True,
    )
    _load_projectors(model, stage1_checkpoint)
    device = torch.device("cuda")
    model.to(device)
    means = [value.to(device) for value in means]
    stds = [value.to(device) for value in stds]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, means, stds, device, optimizer)
        validation = run_epoch(model, validation_loader, means, stds, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
        write_json(output / "history.json", history)
        score = float(validation["weighted_f1"])
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        print("epoch={:03d} train={:.4f} val_f1={:.2f}".format(epoch, train_metrics["loss"], 100 * score), flush=True)
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    test = run_epoch(model, test_loader, means, stds, device)
    torch.save({"model": best_state, "epoch": best_epoch, "config": asdict(config)}, str(output / "best.pt"))
    result = {
        "variant": "complete-m3-temporal-residual", "seed": config.seed,
        "best_epoch": best_epoch, "selection_split": config.selection_split,
        "validation": history[best_epoch - 1]["validation"], "test": test,
        "history_epochs": len(history), "collapsed": bool(test["predicted_sign_count"] < 2),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "runtime_seconds": time.time() - started,
    }
    write_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run(Config(seed=args.seed, epochs=args.epochs), Path(args.feature_root), Path(args.stage1_checkpoint), Path(args.output_dir))


if __name__ == "__main__":
    main()
