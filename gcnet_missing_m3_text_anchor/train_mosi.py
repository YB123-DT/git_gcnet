"""Train one complete-modality CMU-MOSI Text-Anchored model."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from gcnet_missing_m3_sam_backbone.train_mosi import (
    SAMTrainConfig,
    _atomic_save_npz,
    _state_to_cpu,
    _stable_seed,
    evaluate_model,
    set_random_seed,
    train_epoch,
    write_json,
)

from .model import TextAnchoredResidualModel


def select_best_epoch(records: Sequence[Mapping[str, object]]) -> int:
    if not records:
        raise ValueError("records must not be empty")
    best = max(
        records,
        key=lambda record: float(record["validation"]["weighted_f1"]),
    )
    return int(best["epoch"])


def train_model(
    model: TextAnchoredResidualModel,
    train_loader: Iterable[Sequence[object]],
    validation_loader: Iterable[Sequence[object]],
    optimizer: torch.optim.Optimizer,
    config_value: SAMTrainConfig,
    device: torch.device,
    history_path: Path = None,
) -> Tuple[List[Dict[str, object]], Dict[str, torch.Tensor], int, Dict[str, object]]:
    history: List[Dict[str, object]] = []
    best_state: Dict[str, torch.Tensor] = {}
    best_epoch = 0
    best_validation: Dict[str, object] = {}
    best_score = -math.inf
    for epoch_index in range(config_value.epochs):
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
        score = float(validation_metrics["weighted_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch_index + 1
            best_state = _state_to_cpu(model)
            best_validation = dict(validation_metrics)
        print(
            "epoch={:03d} train_loss={:.4f} val_loss={:.4f} val_f1={:.2f}".format(
                epoch_index + 1,
                float(train_metrics["loss"]),
                float(validation_metrics["loss"]),
                100.0 * score,
            ),
            flush=True,
        )
    if not best_state or select_best_epoch(history) != best_epoch:
        raise RuntimeError("validation W-F1 checkpoint provenance is inconsistent")
    return history, best_state, best_epoch, best_validation


def run_experiment(
    config_value: SAMTrainConfig,
    audio_root: str,
    text_root: str,
    visual_root: str,
    output_dir: str,
):
    from gcnet_modality_jepa.train_gcnet import get_loaders

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(config_value)
    config_payload["checkpoint_selection"] = "validation_weighted_f1"
    write_json(output / "config.json", config_payload)
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
    train_loader, validation_loader, test_loader = (
        loaders[0][0],
        loaders[1][0],
        loaders[2][0],
    )
    adim, tdim, vdim = loaders[3:]
    set_random_seed(_stable_seed(config_value.seed, "text_anchor_model_init"))
    model = TextAnchoredResidualModel(
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
    torch.save(
        {
            "model": best_state,
            "config": config_payload,
            "epoch": best_epoch,
            "validation": best_validation,
            "selection_split": "validation_weighted_f1",
        },
        str(output / "best_checkpoint.pt"),
    )
    model.load_state_dict(best_state, strict=True)
    test_metrics, artifacts = evaluate_model(model, test_loader, device)
    _atomic_save_npz(output / "predictions.npz", artifacts)
    metrics = {
        "variant": "text-anchored-residual-backbone",
        "dataset": "CMUMOSI",
        "missing_rate": 0.0,
        "seed": config_value.seed,
        "best_epoch": best_epoch,
        "selection_split": "validation_weighted_f1",
        "validation": best_validation,
        "test": test_metrics,
        "registered_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "runtime_seconds": time.time() - started,
        "history_epochs": len(history),
        "collapsed": bool(
            float(test_metrics["prediction_std"]) < 1e-8
            or int(test_metrics["predicted_sign_count"]) < 2
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    feature_root = Path(args.feature_root).expanduser().resolve()
    roots = [
        feature_root / name
        for name in (
            "wav2vec-large-c-UTT",
            "deberta-large-4-UTT",
            "manet_UTT",
        )
    ]
    if not all(root.is_dir() for root in roots):
        raise FileNotFoundError("one or more official feature roots are missing")
    run_experiment(
        SAMTrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            device=args.device,
        ),
        *(str(root) for root in roots),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


__all__ = ["run_experiment", "select_best_epoch", "train_model"]
