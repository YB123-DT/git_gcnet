"""MOSI causal OSRAM hyper-parameter screening queue.

The queue runs one seed-66 configuration per process, with ten configurations
serially assigned to each of three GPUs.  It intentionally keeps the protocol
fixed (cyclic masks and per-rate Test-oracle checkpoint selection) and changes
only the declared training/capacity fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
REMOTE = Path("/data2/yb/remote_experiments")
SOURCE_ROOT = REMOTE / "osram_causal_nojepa_20260910/mosi"
ROOT = REMOTE / "osram_mosi_hparam_sweep_20260918"
FEATURES = Path("/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features")
SEED = 66
GPU_ASSIGNMENT = (1, 2, 3)
RATES = tuple(f"{i / 10:.1f}" for i in range(8))


def _spec(identifier: str, group: str, **overrides: Any) -> dict[str, Any]:
    return {"id": identifier, "group": group, "overrides": overrides}


# Thirty intentionally coarse configurations.  The first ten probe optimizer,
# regularization and batch size; the second ten probe OSRAM capacity; the last
# ten probe optimizer/schedule and the longer training budget.  This is a
# screening grid, not a claim that these values are already optimal.
SPECS: tuple[dict[str, Any], ...] = (
    _spec("cfg01_baseline", "optimization"),
    _spec("cfg02_lr3e4", "optimization", learning_rate=3e-4),
    _spec("cfg03_lr3e3", "optimization", learning_rate=3e-3),
    _spec("cfg04_lr1e4_b16_d3", "optimization", learning_rate=1e-4, batch_size=16, dropout=.3),
    _spec("cfg05_lr3e4_b16_d2", "optimization", learning_rate=3e-4, batch_size=16, dropout=.2),
    _spec("cfg06_lr1e3_b16_d2", "optimization", learning_rate=1e-3, batch_size=16, dropout=.2),
    _spec("cfg07_lr3e4_b8_d1", "optimization", learning_rate=3e-4, batch_size=8, dropout=.1),
    _spec("cfg08_lr1e3_b8_d1", "optimization", learning_rate=1e-3, batch_size=8, dropout=.1),
    _spec("cfg09_lr3e4_wd1e3_d3", "optimization", learning_rate=3e-4, weight_decay=1e-3, dropout=.3),
    _spec("cfg10_lr1e3_wd1e4_d3_proj0_bb05", "optimization", learning_rate=1e-3,
          weight_decay=1e-4, dropout=.3, projector_dropout=0.0,
          backbone_lr_multiplier=.5, projector_lr_multiplier=2.0),

    _spec("cfg11_capacity_baseline", "capacity"),
    _spec("cfg12_out1400", "capacity", osram_output_dim=1400),
    _spec("cfg13_kv64", "capacity", osram_key_dim=64, osram_value_dim=64),
    _spec("cfg14_out1400_kv64", "capacity", osram_output_dim=1400,
          osram_key_dim=64, osram_value_dim=64),
    _spec("cfg15_lat512_out1024_kv64", "capacity", latent_dim=512,
          osram_output_dim=1024, osram_key_dim=64, osram_value_dim=64),
    _spec("cfg16_heads16_kv32", "capacity", osram_num_heads=16),
    _spec("cfg17_heads4_kv64", "capacity", osram_num_heads=4,
          osram_key_dim=64, osram_value_dim=64),
    _spec("cfg18_out1024_kv48", "capacity", osram_output_dim=1024,
          osram_key_dim=48, osram_value_dim=48),
    _spec("cfg19_lat384_out1024_kv48", "capacity", latent_dim=384,
          osram_output_dim=1024, osram_key_dim=48, osram_value_dim=48),
    _spec("cfg20_out1400_kv48", "capacity", osram_output_dim=1400,
          osram_key_dim=48, osram_value_dim=48),

    _spec("cfg21_adamw_lr1e3_wd1e4_d3", "schedule", optimizer="adamw",
          learning_rate=1e-3, weight_decay=1e-4, dropout=.3),
    _spec("cfg22_adamw_lr3e4_wd1e4_d2", "schedule", optimizer="adamw",
          learning_rate=3e-4, weight_decay=1e-4, dropout=.2),
    _spec("cfg23_adamw_lr1e3_wd1e3_d3", "schedule", optimizer="adamw",
          learning_rate=1e-3, weight_decay=1e-3, dropout=.3),
    _spec("cfg24_adamw_lr3e4_wd1e3_d2", "schedule", optimizer="adamw",
          learning_rate=3e-4, weight_decay=1e-3, dropout=.2),
    _spec("cfg25_adam_cosine_lr1e3", "schedule", lr_schedule="cosine",
          warmup_ratio=.05, learning_rate=1e-3),
    _spec("cfg26_adam_cosine_lr3e4", "schedule", lr_schedule="cosine",
          warmup_ratio=.05, learning_rate=3e-4),
    _spec("cfg27_adamw_cosine_lr1e3", "schedule", optimizer="adamw",
          lr_schedule="cosine", warmup_ratio=.05, learning_rate=1e-3,
          weight_decay=1e-4),
    _spec("cfg28_adamw_cosine_lr3e4", "schedule", optimizer="adamw",
          lr_schedule="cosine", warmup_ratio=.05, learning_rate=3e-4,
          weight_decay=1e-4),
    _spec("cfg29_long200_clip05", "schedule", learning_rate=1e-3,
          epochs=200, gradient_clip_norm=.5, backbone_lr_multiplier=.5,
          projector_lr_multiplier=2.0),
    _spec("cfg30_long400_adamw_cosine", "schedule", optimizer="adamw",
          lr_schedule="cosine", warmup_ratio=.05, learning_rate=3e-4,
          weight_decay=1e-4, epochs=400, gradient_clip_norm=5.0),
)

SPEC_BY_ID = {item["id"]: item for item in SPECS}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _reference_config():
    from gcnet_missing_m3.train_gcnet import TrainConfig

    source = SOURCE_ROOT / f"seed_{SEED}"
    old = json.loads((source / "config.json").read_text())
    required = {
        "dataset": "CMUMOSI",
        "seed": SEED,
        "training_objective": "emotion-only",
        "backbone_type": "osram",
        "fusion_type": "mean",
        "osram_bidirectional": False,
        "osram_forward_slot_reuse": False,
        "osram_write_step": .6,
        "osram_readout_fusion": "flat",
        "train_rate_mode": "cyclic",
        "checkpoint_selection": "test-oracle-per-rate",
        "evaluate_test": True,
        "completion_path": "none",
        "classification_completion": False,
        "initial_backbone_checkpoint": None,
    }
    mismatch = {key: old.get(key) for key, expected in required.items()
                if old.get(key) != expected}
    if mismatch:
        raise ValueError(f"reference no-JEPA config is not locked: {mismatch}")
    return TrainConfig(**old), source


def configuration(item: dict[str, Any]):
    base, source = _reference_config()
    from gcnet_missing_m3.train_gcnet import TrainConfig

    overrides = dict(item["overrides"])
    overrides.update({
        "seed": SEED,
        "training_objective": "emotion-only",
        "backbone_type": "osram",
        "fusion_type": "mean",
        "osram_bidirectional": False,
        "osram_forward_slot_reuse": False,
        "osram_write_step": .6,
        "osram_readout_fusion": "flat",
        "train_rate_mode": "cyclic",
        "checkpoint_selection": "test-oracle-per-rate",
        "evaluate_test": True,
        "completion_path": "none",
        "classification_completion": False,
        "initial_backbone_checkpoint": None,
        "teacher_mode": "ema",
        "teacher_checkpoint": None,
        "target_space": "all-modalities",
        "text_subspace_checkpoint": None,
        "text_core": False,
    })
    cfg = replace(base, **overrides)
    # Deliberately reject accidental method changes in future grid edits.
    for field in ("training_objective", "backbone_type", "fusion_type",
                  "osram_bidirectional", "osram_forward_slot_reuse",
                  "osram_write_step", "osram_readout_fusion", "train_rate_mode",
                  "checkpoint_selection", "evaluate_test", "completion_path",
                  "classification_completion", "initial_backbone_checkpoint",
                  "teacher_mode", "teacher_checkpoint", "target_space",
                  "text_subspace_checkpoint", "text_core"):
        if getattr(cfg, field) != getattr(base, field):
            raise ValueError(f"grid changed locked field {field}: {getattr(cfg, field)!r}")
    return cfg, source


def _provenance(item: dict[str, Any], cfg, source: Path) -> dict[str, Any]:
    return {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "experiment": "MOSI causal OSRAM hyperparameter/capacity screening",
        "spec": item,
        "seed": SEED,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection_protocol": "per-rate-test-oracle",
        "reference": str(source),
        "reference_sha256": {
            name: sha(source / name) for name in ("config.json", "history.json", "metrics.json")
        },
        "config": asdict(cfg),
        "features": str(FEATURES),
    }


def train(spec_id: str) -> None:
    item = SPEC_BY_ID[spec_id]
    cfg, source = configuration(item)
    output = ROOT / "screen_seed66" / spec_id
    if output.exists():
        provenance_path = output / "PROVENANCE.json"
        if provenance_path.exists():
            state = json.loads(provenance_path.read_text())
            if state.get("status") == "complete":
                print(f"SKIP complete {spec_id}", flush=True)
                return
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    provenance = _provenance(item, cfg, source)
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "PROVENANCE.json", provenance)
    try:
        import torch
        from gcnet_missing_m3.train_gcnet import run_experiment

        torch.set_num_threads(6)
        roots = [str(FEATURES / name) for name in
                 ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")]
        print(f"TRAIN {spec_id} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} overrides={item['overrides']}", flush=True)
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise ValueError("unexpected selection protocol")
        reference_metrics = json.loads((source / "metrics.json").read_text())
        if metrics.get("mask_sha256") != reference_metrics.get("mask_sha256"):
            raise ValueError("evaluation mask hashes differ from no-JEPA reference")
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "PROVENANCE.json", provenance)
        raise
    provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE {spec_id}", flush=True)


def _worker(spec_ids: list[str], gpu: int) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="6",
               MKL_NUM_THREADS="6", PYTHONPATH=str(REPO))
    os.environ.update(env)
    for spec_id in spec_ids:
        try:
            train(spec_id)
        except BaseException as error:
            print(f"FAILED {spec_id}: {type(error).__name__}: {error}", flush=True)


def launch() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "gpus": list(GPU_ASSIGNMENT),
        "configs": [item for item in SPECS],
        "selection_protocol": "per-rate-test-oracle",
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "workers": [],
    }
    write_json(ROOT / "QUEUE.json", queue)
    buckets = [list() for _ in GPU_ASSIGNMENT]
    for index, item in enumerate(SPECS):
        buckets[index % len(buckets)].append(item["id"])
    children = []
    for gpu, spec_ids in zip(GPU_ASSIGNMENT, buckets):
        log_path = ROOT / f"gpu{gpu}.log"
        log = log_path.open("a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="6",
                   MKL_NUM_THREADS="6", PYTHONPATH=str(REPO))
        child = subprocess.Popen(
            [sys.executable, "-u", __file__, "--worker", *spec_ids], cwd=REPO,
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        row = {"gpu": gpu, "pid": child.pid, "spec_ids": spec_ids,
               "log": str(log_path), "status": "running"}
        queue["workers"].append(row)
        children.append((child, log, row))
        print(f"START GPU={gpu} PID={child.pid} specs={spec_ids}", flush=True)
    queue["status"] = "running"
    write_json(ROOT / "QUEUE.json", queue)
    for child, log, row in children:
        row["exit_code"] = child.wait()
        row["status"] = "complete" if row["exit_code"] == 0 else "failed"
        log.close()
        write_json(ROOT / "QUEUE.json", queue)
    queue["status"] = "complete" if all(row["exit_code"] == 0 for _, _, row in children) else "failed"
    queue["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / "QUEUE.json", queue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--worker", nargs="+", metavar="SPEC_ID")
    group.add_argument("--train", metavar="SPEC_ID", choices=tuple(SPEC_BY_ID))
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.worker:
        gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
        _worker(args.worker, gpu)
    else:
        train(args.train)


if __name__ == "__main__":
    main()
