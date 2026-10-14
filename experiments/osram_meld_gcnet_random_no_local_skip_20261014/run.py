"""Retrain MELD Flat without Local skip under the canonical GCNet mask."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket


CANONICAL_MASK_PROTOCOL = "gcnet-random-mask-v1"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def corrected_settings(source: dict) -> dict:
    """Apply only the no-Local-skip intervention and reject mixed protocols."""
    settings = dict(source)
    if settings.get("missing_mask_protocol") != CANONICAL_MASK_PROTOCOL:
        raise ValueError(
            "source config must use the canonical original-GCNet sampler "
            f"{CANONICAL_MASK_PROTOCOL!r}; got "
            f"{settings.get('missing_mask_protocol')!r}"
        )
    settings["train_missing_rates"] = tuple(settings["train_missing_rates"])
    settings["osram_disable_local_skip"] = True
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, choices=(66, 67, 68), required=True)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal --gpu-uuid")

    # Dataset paths are consumed while loader modules are imported.
    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
    os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)
    os.environ["GCNET_CACHE_ROOT"] = str(args.output.parent / "cache")
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment

    source = json.loads(args.source_config.read_text())
    config = TrainConfig(**corrected_settings(source))
    required = {
        "dataset": "MELD",
        "seed": args.seed,
        "epochs": 50,
        "missing_mask_protocol": CANONICAL_MASK_PROTOCOL,
        "checkpoint_selection": "test-oracle-per-rate",
        "backbone_type": "osram",
        "osram_readout_fusion": "flat",
        "training_objective": "emotion-only",
        "osram_bidirectional": False,
        "osram_disable_local_skip": True,
    }
    for key, expected in required.items():
        actual = getattr(config, key)
        if actual != expected:
            raise ValueError(f"{key} must be {expected!r}, got {actual!r}")

    import torch

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(min(args.num_threads, 4))

    args.output.mkdir(parents=True)
    provenance = {
        "status": "running",
        "started_utc": _now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "commit": args.code_commit,
        "source_config": str(args.source_config),
        "dataset_root": str(args.dataset_root),
        "meld_pickle": str(args.meld_pickle),
        "gpu_uuid": args.gpu_uuid,
        "protocol": "original GCNet batch random_mask; internal Test-oracle diagnostic",
        "intervention": "retrain from scratch without final local_skip(Local); adapter Local retained",
        "config": asdict(config),
    }
    _write(args.output / "PROVENANCE.json", provenance)
    try:
        metrics = run_experiment(
            config, "MELD-pickle", "MELD-pickle", "MELD-pickle", args.output
        )
        provenance.update(
            status="complete",
            finished_utc=_now(),
            selection_metric=metrics.get("selection_metric"),
        )
        _write(args.output / "PROVENANCE.json", provenance)
    except BaseException as error:
        provenance.update(status="failed", finished_utc=_now(), error=repr(error))
        _write(args.output / "PROVENANCE.json", provenance)
        raise


if __name__ == "__main__":
    main()
