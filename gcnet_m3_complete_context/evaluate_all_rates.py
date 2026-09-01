"""Evaluate the five locked complete-M3 checkpoints at all missing rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule
from gcnet_modality_jepa.protocol import SeedBundle
from gcnet_modality_jepa.train_gcnet import (
    build_primary_mask_tensors,
    generate_inputs,
    get_loaders,
)
from gcnet_missing_m3.mixed_rate import MISSING_RATES
from gcnet_missing_m3_sam_backbone.train_mosi import regression_metrics, write_json

from .model import CompleteM3Regressor
from .train_mosi import FEATURE_NAMES, compute_standardizer


@torch.no_grad()
def evaluate_seed(seed: int, feature_root: Path, checkpoint: Path):
    roots = [str(feature_root / name) for name in FEATURE_NAMES]
    loaders = get_loaders(
        audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset="CMUMOSI", batch_size=128, num_workers=0,
        seed=seed, validation_fraction=0.1, evaluation_protocol="official",
    )
    means, stds = compute_standardizer(loaders[0][0])
    device = torch.device("cuda")
    model = CompleteM3Regressor(tuple(loaders[3:])).to(device).eval()
    payload = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    means = [value.to(device) for value in means]
    stds = [value.to(device) for value in stds]
    result = {}
    for rate in MISSING_RATES:
        schedule = ConversationMaskSchedule(
            dataset="CMUMOSI", split="test", fold=5,
            requested_missing_rate=rate,
            mask_seed=SeedBundle(seed).derive("missing_mask"),
            freeze_evaluation=True,
        )
        predictions, labels = [], []
        for raw in loaders[2][0]:
            host, guest = build_primary_mask_tensors(
                schedule, conversation_ids=raw[9], umask=raw[7], epoch=0
            )
            availability = generate_inputs(
                host[..., 0:1], host[..., 1:2], host[..., 2:3],
                guest[..., 0:1], guest[..., 1:2], guest[..., 2:3], raw[6]
            )[0].to(device)
            features = [
                ((raw[index].to(device) - means[index]) / stds[index])
                * availability[..., index : index + 1]
                for index in range(3)
            ]
            output = model(features, raw[7].to(device)).squeeze(-1).transpose(0, 1)
            valid = raw[7].bool()
            predictions.append(output.cpu()[valid].numpy())
            labels.append(raw[8][valid].numpy())
        result[format(rate, ".1f")] = regression_metrics(
            np.concatenate(labels), np.concatenate(predictions)
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = {}
    for seed in range(66, 71):
        checkpoint = Path(args.checkpoint_root) / "seed_{}".format(seed) / "best.pt"
        records[str(seed)] = evaluate_seed(seed, Path(args.feature_root), checkpoint)
        print("done seed={}".format(seed), flush=True)
    means = {
        key: float(np.mean([records[str(seed)][key]["weighted_f1"] for seed in range(66, 71)]))
        for key in (format(rate, ".1f") for rate in MISSING_RATES)
    }
    summary = {
        "variant": "locked-complete-m3-eval-only",
        "training_rates": [0.0],
        "test_rates": list(MISSING_RATES),
        "seeds": list(range(66, 71)),
        "weighted_f1_by_seed": records,
        "mean_weighted_f1_by_rate": means,
        "overall_rate_seed_mean_weighted_f1": float(np.mean(list(means.values()))),
    }
    write_json(Path(args.output), summary)
    print(json.dumps(summary["mean_weighted_f1_by_rate"], indent=2), flush=True)
    print("overall={:.6f}".format(summary["overall_rate_seed_mean_weighted_f1"]), flush=True)


if __name__ == "__main__":
    main()
