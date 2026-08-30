"""Formal raw-residual Missing-M3 treatment with the public SDR backbone."""

from importlib import import_module

from .model import MissingM3RawSDRModel


_TRAINING_EXPORTS = frozenset(
    {"RawSDRTrainConfig", "build_model", "run_experiment"}
)


def __getattr__(name):
    if name not in _TRAINING_EXPORTS:
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(__name__, name)
        )
    value = getattr(import_module(".train_gcnet", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "MissingM3RawSDRModel",
    "RawSDRTrainConfig",
    "build_model",
    "run_experiment",
]
