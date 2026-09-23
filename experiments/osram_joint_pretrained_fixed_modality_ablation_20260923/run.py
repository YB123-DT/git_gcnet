"""Fixed-modality evaluation of the joint-pretrained dual-projector model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import (  # noqa: E402
    CHECKPOINT_RATE,
    PATTERNS,
    _build_model,
    _evaluate_pattern,
    _roots,
)

SOURCE_ROOT = Path(
    "/data2/yb/remote_experiments/osram_joint_dual_projector_completion_20260922"
)
OUTPUT_ROOT = Path(
    "/data2/yb/remote_experiments/osram_joint_pretrained_fixed_modality_ablation_20260923"
)
SEEDS = (66, 67, 68)
GPUS = (4, 5, 7)


def evaluate_seed(seed: int) -> None:
    import torch

    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig

    source = SOURCE_ROOT / f"seed_{seed}"
    output = OUTPUT_ROOT / f"seed_{seed}.json"
    if output.exists():
        print(f"SKIP seed={seed}", flush=True)
        return
    config = TrainConfig(**json.loads((source / "config.json").read_text()))
    if config.completion_path != "pre_osram_joint_dual_projector":
        raise ValueError("source is not the joint-pretrained dual-projector model")
    if config.train_rate_mode != "cyclic":
        raise ValueError("source did not use the cyclic mixed-rate protocol")

    loaders = get_loaders(
        audio_root=_roots()[0],
        text_root=_roots()[1],
        video_root=_roots()[2],
        num_folder=1,
        dataset="CMUMOSI",
        batch_size=config.batch_size,
        num_workers=0,
        seed=seed,
        validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol,
    )
    _, _, test_loaders, adim, tdim, vdim = loaders
    dimensions = (adim, tdim, vdim)
    device = torch.device("cuda:0")
    model = _build_model(config, dimensions).to(device)
    patterns = {}
    for name, observed in PATTERNS.items():
        rate = CHECKPOINT_RATE[name]
        checkpoint = source / f"best_miss_{rate.replace('.', 'p')}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        metrics = _evaluate_pattern(
            model, test_loaders[0], observed, dimensions, device
        )
        patterns[name] = {
            **metrics,
            "checkpoint_rate": rate,
            "checkpoint_epoch": int(state["epoch"]),
        }
        print(
            f"seed={seed} pattern={name} wf1={100 * metrics['weighted_f1']:.3f}",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
                "seed": seed,
                "model": "joint-pretrained dual-projector completion",
                "protocol": "evaluation-only fixed observed-set ablation",
                "checkpoint_rule": "nearest official rate to exact fixed missing fraction",
                "source": str(source),
                "patterns": patterns,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def summarize() -> None:
    rows = []
    for seed in SEEDS:
        payload = json.loads((OUTPUT_ROOT / f"seed_{seed}.json").read_text())
        for pattern, metrics in payload["patterns"].items():
            rows.append(
                {
                    "seed": seed,
                    "pattern": pattern,
                    "observed_count": sum(PATTERNS[pattern]),
                    "checkpoint_rate": metrics["checkpoint_rate"],
                    "checkpoint_epoch": metrics["checkpoint_epoch"],
                    "weighted_f1": 100 * metrics["weighted_f1"],
                    "accuracy": 100 * metrics["accuracy"],
                    "mae": metrics["mae"],
                    "correlation": metrics["correlation"],
                }
            )
    with (OUTPUT_ROOT / "per_seed_pattern.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    full_mean = float(
        np.mean([row["weighted_f1"] for row in rows if row["pattern"] == "ALV"])
    )
    summary = []
    for pattern in PATTERNS:
        selected = [row for row in rows if row["pattern"] == pattern]
        values = np.asarray([row["weighted_f1"] for row in selected])
        summary.append(
            {
                "pattern": pattern,
                "observed_count": sum(PATTERNS[pattern]),
                "mean_wf1": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
                "drop_from_alv": float(values.mean() - full_mean),
                "mean_accuracy": float(
                    np.mean([row["accuracy"] for row in selected])
                ),
                "mean_mae": float(np.mean([row["mae"] for row in selected])),
            }
        )
    with (OUTPUT_ROOT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


def launch() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    children = []
    for gpu, seed in zip(GPUS, SEEDS):
        log = (OUTPUT_ROOT / f"seed_{seed}.log").open("w")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTHONPATH=str(REPO),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        )
        child = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__)), "--seed", str(seed)],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        children.append((child, log, seed))
    failures = []
    for child, log, seed in children:
        if child.wait():
            failures.append(seed)
        log.close()
    if failures:
        raise RuntimeError(f"failed seeds: {failures}")
    summarize()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--seed", type=int, choices=SEEDS)
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.summarize:
        summarize()
    else:
        evaluate_seed(args.seed)


if __name__ == "__main__":
    main()
