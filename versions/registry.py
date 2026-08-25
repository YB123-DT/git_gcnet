"""Load the locked switches for published GCNet versions."""

import json
from pathlib import Path


VERSION_NAMES = ("original", "mpfilm", "cp_lecc", "sequence_aff")
_ROOT = Path(__file__).resolve().parent


def resolve(name):
    if name not in VERSION_NAMES:
        raise ValueError("unknown completed version: {!r}".format(name))
    path = _ROOT / name / "config.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if set(payload) != {"graph_conv_variant", "branch_fusion"}:
        raise ValueError("invalid locked configuration for {!r}".format(name))
    return payload
