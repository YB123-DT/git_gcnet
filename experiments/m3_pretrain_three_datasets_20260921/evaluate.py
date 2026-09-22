"""Evaluation entrypoint for a saved three-dataset M3 pretraining checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .run import (
    ThreeDatasetJEPA,
    build_joint_loaders,
    evaluate_train_validation,
    resolve_dataset_names,
)


def load_checkpoint(path: str | Path, device: torch.device) -> ThreeDatasetJEPA:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ThreeDatasetJEPA(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--iemocap-fold", type=int, default=5)
    parser.add_argument("--evaluation-protocol", choices=("strict", "official"), default="strict")
    parser.add_argument("--output", default="evaluation.json")
    parser.add_argument("--feature-root", action="append", nargs=4, metavar=("DATASET", "AUDIO", "TEXT", "VISUAL"), required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    feature_roots = {entry[0]: entry[1:] for entry in args.feature_root}
    try:
        dataset_names = resolve_dataset_names(feature_roots)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    model = load_checkpoint(args.checkpoint, device)
    loaders, dimensions = build_joint_loaders(
        feature_roots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        iemocap_fold=args.iemocap_fold,
        evaluation_protocol=args.evaluation_protocol,
    )
    results = {
        "checkpoint": str(args.checkpoint),
        "dataset_names": dataset_names,
        "dimensions": dimensions,
        "metrics": evaluate_train_validation(
            model, loaders, seed=args.seed, device=device
        ),
    }
    Path(args.output).write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
