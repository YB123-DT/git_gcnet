"""Formal raw-residual Missing-M3 treatment with the public SDR backbone."""

from .model import MissingM3RawSDRModel
from .train_gcnet import RawSDRTrainConfig, build_model, run_experiment

__all__ = [
    "MissingM3RawSDRModel",
    "RawSDRTrainConfig",
    "build_model",
    "run_experiment",
]
