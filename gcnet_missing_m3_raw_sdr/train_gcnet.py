"""Locked CMU-MOSI trainer for the formal raw-residual SDR treatment."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, Union

import torch

import config
from gcnet_missing_m3_sdr_backbone import train_gcnet as shared_train
from gcnet_missing_m3_sdr_backbone.train_gcnet import SDRTrainConfig

from .model import MissingM3RawSDRModel


@dataclass(frozen=True)
class RawSDRTrainConfig(SDRTrainConfig):
    """Formal raw-public SDR protocol with only lifecycle fields mutable."""

    fusion_type: str = "raw-residual"
    sdr_variant: str = "sdr-public"
    sdr_input_type: str = "raw-residual"

    _OPEN_FIELDS: ClassVar[FrozenSet[str]] = frozenset(
        {"seed", "device", "epochs", "evaluate_test"}
    )


def build_model(
    config_value: RawSDRTrainConfig,
    adim: int,
    tdim: int,
    vdim: int,
    device: torch.device,
) -> MissingM3RawSDRModel:
    """Construct the one registered raw-residual public SDR model."""

    return shared_train.build_model(
        config_value,
        adim,
        tdim,
        vdim,
        device,
        model_type=MissingM3RawSDRModel,
    )


def run_experiment(
    config_value: RawSDRTrainConfig,
    audio_root: str,
    text_root: str,
    visual_root: str,
    output_dir: Union[str, Path],
) -> Dict[str, object]:
    """Run the shared SDR lifecycle with fixed raw-public identity."""

    return shared_train.run_experiment(
        config_value,
        audio_root,
        text_root,
        visual_root,
        output_dir,
        model_builder=build_model,
        result_identity={
            "variant": "raw-residual-sdr-public",
            "sdr_variant": "sdr-public",
            "sdr_input_type": "raw-residual",
            "backbone": "raw-residual-sdr-public",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Expose only feature paths and lifecycle controls."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-feature", required=True)
    parser.add_argument("--text-feature", required=True)
    parser.add_argument("--video-feature", required=True)
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-rate-mode",
        choices=("all",),
        default="all",
    )
    parser.add_argument("--lr", type=float, choices=(5e-4,), default=5e-4)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_value = RawSDRTrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        evaluate_test=not args.skip_test,
        train_rate_mode=args.train_rate_mode,
        learning_rate=args.lr,
    )
    feature_root = args.feature_root or config.PATH_TO_FEATURES[
        config_value.dataset
    ]
    roots = [
        os.path.join(feature_root, name)
        for name in (args.audio_feature, args.text_feature, args.video_feature)
    ]
    if not all(os.path.exists(root) for root in roots):
        raise FileNotFoundError("one or more feature roots do not exist")
    run_experiment(config_value, *roots, output_dir=args.output_dir)


__all__ = [
    "MissingM3RawSDRModel",
    "RawSDRTrainConfig",
    "build_model",
    "run_experiment",
]


if __name__ == "__main__":
    main()
