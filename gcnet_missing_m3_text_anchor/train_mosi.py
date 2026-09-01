"""Train one complete-modality CMU-MOSI Text-Anchored model."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from gcnet_missing_m3_sam_backbone.train_mosi import (
    SAMTrainConfig,
    _atomic_save_npz,
    _stable_seed,
    evaluate_model,
    set_random_seed,
    train_model,
    write_json,
)

from .model import TextAnchoredResidualModel


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
            "config": asdict(config_value),
            "epoch": best_epoch,
            "validation": best_validation,
            "selection_split": "validation",
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
        "selection_split": "validation",
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


__all__ = ["run_experiment"]
