"""Run cfg84 no-JEPA causal OSRAM on CMU-MOSEI for three seeds.

This is an internal Test-oracle diagnostic.  The model and protocol match the
current cfg84 IEMOCAP runner: causal OSRAM, eta=.6, Flat/mean readout,
cyclic mixed-rate training, and independent per-rate Test-W-F1 selection.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment


REMOTE_ROOT = Path("/data2/yb/remote_experiments")
OUTPUT_ROOT = REMOTE_ROOT / "osram_mosei_cfg84_nojepa_20260919"
SOURCE_ROOT = REMOTE_ROOT / "osram_mosei_20260906"
FEATURE_ROOT = Path(
    "/data2/yb/paper/GCNet_repro_cmumosei_10seed_20260819/dataset/CMUMOSEI/features"
)
DATASET = "CMUMOSEI"
SEEDS = (66, 67, 68)
LABEL = "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _source_config(seed: int) -> dict:
    path = SOURCE_ROOT / f"seed_{seed}" / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"MOSEI source config not found: {path}")
    return json.loads(path.read_text())


def configuration(seed: int) -> TrainConfig:
    settings = _source_config(seed)
    settings.update(
        {
            "seed": seed,
            "dataset": DATASET,
            "fold": 1,
            "epochs": 100,
            "train_rate_mode": "cyclic",
            "training_objective": "emotion-only",
            "checkpoint_selection": "test-oracle-per-rate",
            "evaluate_test": True,
            "osram_bidirectional": False,
            "osram_write_step": 0.6,
            "osram_readout_fusion": "flat",
            "fusion_type": "mean",
            "osram_output_dim": 1600,
            "osram_num_heads": 8,
            "osram_key_dim": 64,
            "osram_value_dim": 64,
            "disable_unused_aux_modules": True,
            "initial_backbone_checkpoint": None,
            "classification_completion": False,
            "completion_path": "none",
            "target_space": "all-modalities",
            "teacher_mode": "ema",
            "teacher_checkpoint": None,
        }
    )
    return TrainConfig(**settings)


def train(seed: int) -> None:
    output = OUTPUT_ROOT / f"seed_{seed}"
    if output.exists():
        provenance = output / "PROVENANCE.json"
        metrics = output / "metrics.json"
        if provenance.exists() and json.loads(provenance.read_text()).get("status") == "complete":
            print(f"SKIP complete MOSEI seed={seed}", flush=True)
            return
        if metrics.exists():
            data = json.loads(metrics.read_text())
            if data.get("selection_protocol") == "per-rate-test-oracle":
                payload = json.loads(provenance.read_text()) if provenance.exists() else {}
                payload.update(
                    status="complete",
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    selection_protocol="per-rate-test-oracle",
                )
                _write(provenance, payload)
                print(f"RECOVER complete MOSEI seed={seed}", flush=True)
                return
        raise FileExistsError(f"refusing to overwrite {output}")

    output.mkdir(parents=True)
    cfg = configuration(seed)
    _write(output / "config.json", asdict(cfg))
    provenance = {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": LABEL,
        "dataset": DATASET,
        "seed": seed,
        "fold": 1,
        "selection_protocol": "per-rate-test-oracle",
        "selection_split": "test",
        "training_objective": "emotion-only",
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "configuration_delta": {
            "osram_output_dim": 1600,
            "osram_num_heads": 8,
            "osram_key_dim": 64,
            "osram_value_dim": 64,
            "osram_bidirectional": False,
            "osram_write_step": 0.6,
            "osram_readout_fusion": "flat",
            "fusion_type": "mean",
            "disable_unused_aux_modules": True,
            "train_rate_mode": "cyclic",
            "checkpoint_selection": "per-rate-test-oracle",
        },
        "feature_root": str(FEATURE_ROOT),
    }
    _write(output / "PROVENANCE.json", provenance)
    roots = [
        str(FEATURE_ROOT / name)
        for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
    ]
    print(
        f"START MOSEI seed={seed} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        "cfg84 no-JEPA cyclic per-rate-test-oracle",
        flush=True,
    )
    try:
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise RuntimeError("unexpected checkpoint selection protocol")
        provenance.update(
            status="complete",
            completed_utc=datetime.now(timezone.utc).isoformat(),
            selected_epoch_by_rate=metrics.get("selected_epoch_by_rate"),
            selected_weighted_f1_by_rate=metrics.get("selected_weighted_f1_by_rate"),
        )
        _write(output / "PROVENANCE.json", provenance)
        print(f"COMPLETE MOSEI seed={seed}", flush=True)
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write(output / "PROVENANCE.json", provenance)
        raise


def launch() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "seeds": list(SEEDS),
        "selection_protocol": "per-rate-test-oracle",
        "label": LABEL,
        "tasks": [],
    }
    _write(OUTPUT_ROOT / "QUEUE.json", queue)
    for seed in SEEDS:
        started = time.time()
        row = {"seed": seed, "status": "running"}
        queue["tasks"].append(row)
        _write(OUTPUT_ROOT / "QUEUE.json", queue)
        train(seed)
        row.update(status="complete", elapsed_seconds=time.time() - started)
        _write(OUTPUT_ROOT / "QUEUE.json", queue)
    queue.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    _write(OUTPUT_ROOT / "QUEUE.json", queue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.seed is not None:
        train(args.seed)
    else:
        parser.error("use --launch or --seed")


if __name__ == "__main__":
    main()
