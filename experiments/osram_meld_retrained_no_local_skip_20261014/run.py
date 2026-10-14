"""Retrain MELD Flat while removing only its final Local shortcut."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must equal --gpu-uuid")

    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment

    source = json.loads(args.source_config.read_text())
    source["train_missing_rates"] = tuple(source["train_missing_rates"])
    source["osram_disable_local_skip"] = True
    config = TrainConfig(**source)
    required = {
        "dataset": "MELD",
        "seed": 66,
        "epochs": 50,
        "missing_mask_protocol": "gcnet-legacy-fixed-v1",
        "checkpoint_selection": "test-oracle-per-rate",
        "backbone_type": "osram",
        "osram_readout_fusion": "flat",
        "training_objective": "emotion-only",
        "osram_bidirectional": False,
        "osram_disable_local_skip": True,
    }
    for key, expected in required.items():
        if getattr(config, key) != expected:
            raise ValueError(f"{key} must be {expected!r}, got {getattr(config, key)!r}")

    os.environ["GCNET_DATASET_ROOT"] = str(args.dataset_root)
    os.environ["GCNET_MELD_PICKLE"] = str(args.meld_pickle)
    os.environ["GCNET_CACHE_ROOT"] = str(args.output.parent / "cache")
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
        "intervention": "retrain from scratch without final local_skip(Local); adapter Local retained",
        "config": asdict(config),
    }
    _write(args.output / "PROVENANCE.json", provenance)
    try:
        metrics = run_experiment(
            config, "MELD-pickle", "MELD-pickle", "MELD-pickle", args.output,
        )
        provenance.update(
            status="complete", finished_utc=_now(),
            selection_metric=metrics.get("selection_metric"),
        )
        _write(args.output / "PROVENANCE.json", provenance)
    except BaseException as error:
        provenance.update(status="failed", finished_utc=_now(), error=repr(error))
        _write(args.output / "PROVENANCE.json", provenance)
        raise


if __name__ == "__main__":
    main()
