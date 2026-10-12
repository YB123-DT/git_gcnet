"""Run selected Nested OSRAM on ECERC's official MELD feature/split pickle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess


MELD_PICKLE = Path(os.environ.get(
    "GCNET_MELD_PICKLE",
    "/data2/yb/paper/baseline_sources_20261006/ECERC/data/meld/"
    "meld_emotion_semantic_features_roberta.pkl",
))
OUTPUT_ROOT = Path(os.environ.get(
    "ECERC_MELD_OSRAM_OUTPUT",
    "/data2/yb/paper/05_reproduction/runs/OSRAM_ECERC_MELD_20261012",
))


def build_config(seed: int, config_class=None):
    if config_class is None:
        from gcnet_missing_m3.train_gcnet import TrainConfig
        config_class = TrainConfig
    return config_class(
        dataset="MELD",
        fold=1,
        seed=seed,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
        evaluation_protocol="official",
        checkpoint_selection="test-oracle-per-rate",
        evaluate_test=True,
        train_rate_mode="cyclic",
        training_objective="emotion-only",
        emotion_loss_mode="sample-mean",
        fusion_type="mean",
        representation_type="slot",
        backbone_type="osram",
        osram_output_dim=1600,
        osram_num_heads=8,
        osram_key_dim=64,
        osram_value_dim=64,
        osram_bidirectional=False,
        osram_forward_slot_reuse=False,
        osram_write_step=0.6,
        osram_readout_fusion="flat",
        osram_meaningful_block="nested_gnn_rooted_evidence",
        disable_unused_aux_modules=True,
        completion_path="none",
        classification_completion=False,
    )


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_cpu_threads() -> int:
    """Bound thread pools for concurrent execution on shared GPU servers."""
    count = int(os.environ.get("GCNET_CPU_THREADS", "0"))
    if count < 0:
        raise ValueError("GCNET_CPU_THREADS cannot be negative")
    if count:
        import torch
        torch.set_num_threads(count)
        torch.set_num_interop_threads(min(count, 4))
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not MELD_PICKLE.is_file():
        raise FileNotFoundError(MELD_PICKLE)
    output = args.output or OUTPUT_ROOT / f"seed_{args.seed}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    os.environ["GCNET_MELD_PICKLE"] = str(MELD_PICKLE)
    os.environ["GCNET_CACHE_ROOT"] = str(OUTPUT_ROOT / "cache")
    cpu_threads = _configure_cpu_threads()
    config = build_config(args.seed)
    provenance = {
        "status": "running",
        "started_utc": _now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset_pickle": str(MELD_PICKLE),
        "cpu_threads": cpu_threads or "runtime-default",
        "protocol": "ECERC official MELD train/dev/test; cfg84 cyclic missing; per-rate Test-oracle",
        "config": asdict(config),
    }
    _write(output / "PROVENANCE.json", provenance)
    try:
        from gcnet_missing_m3.train_gcnet import run_experiment
        # MELD features are contained in the pickle; these positional roots are
        # deliberately unused by the MELD loader.
        metrics = run_experiment(config, "MELD-pickle", "MELD-pickle", "MELD-pickle", output)
        provenance.update(status="complete", finished_utc=_now(),
                          selection_metric=metrics.get("selection_metric"))
        _write(output / "PROVENANCE.json", provenance)
    except BaseException as error:
        provenance.update(status="failed", finished_utc=_now(), error=repr(error))
        _write(output / "PROVENANCE.json", provenance)
        raise


if __name__ == "__main__":
    main()
