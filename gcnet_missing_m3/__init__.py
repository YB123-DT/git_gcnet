"""Single-view Missing-M3 adaptation for GCNet.

The graph model is imported lazily so feature-preparation utilities can run in
the lightweight Transformers environment, which intentionally has no PyG.
"""

from typing import Any


__all__ = ["MissingM3GraphModel"]


def __getattr__(name: str) -> Any:
    if name == "MissingM3GraphModel":
        from .model import MissingM3GraphModel

        return MissingM3GraphModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
