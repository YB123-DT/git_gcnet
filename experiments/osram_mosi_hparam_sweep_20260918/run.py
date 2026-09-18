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

import numpy as np


REPO = Path(__file__).resolve().parents[2]
REMOTE = Path("/data2/yb/remote_experiments")
SOURCE_ROOT = REMOTE / "osram_causal_nojepa_20260910/mosi"
ROOT = REMOTE / "osram_mosi_hparam_sweep_20260918_parallel"
FEATURES = Path("/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features")
SEED = 66
GPU_ASSIGNMENT = (1, 2, 3)
RATES = tuple(f"{i / 10:.1f}" for i in range(8))


def _spec(identifier: str, group: str, **overrides: Any) -> dict[str, Any]:
    return {"id": identifier, "group": group, "overrides": overrides}


# Ninety intentionally coarse configurations.  They cover the requested
# learning-rate, batch, optimizer, regularization, schedule, clipping, epoch,
# and capacity ranges without attempting the infeasible full Cartesian product.
# The launcher runs ten configurations concurrently on each GPU, in waves.
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

    _spec("cfg31_lr3e5_b32_d5", "optimization", learning_rate=3e-5),
    _spec("cfg32_lr3e5_b16_d3", "optimization", learning_rate=3e-5,
          batch_size=16, dropout=.3),
    _spec("cfg33_lr1e4_b32_d1", "optimization", learning_rate=1e-4,
          dropout=.1),
    _spec("cfg34_lr3e4_b32_d0", "optimization", learning_rate=3e-4,
          dropout=0.0),
    _spec("cfg35_lr1e3_b32_d0", "optimization", learning_rate=1e-3,
          dropout=0.0),
    _spec("cfg36_lr3e3_b16_d5", "optimization", learning_rate=3e-3,
          batch_size=16),
    _spec("cfg37_lr3e4_wd0_d3", "optimization", learning_rate=3e-4,
          weight_decay=0.0, dropout=.3),
    _spec("cfg38_lr3e4_wd1e6_d3", "optimization", learning_rate=3e-4,
          weight_decay=1e-6, dropout=.3),
    _spec("cfg39_lr3e4_wd1e2_d3", "optimization", learning_rate=3e-4,
          weight_decay=1e-2, dropout=.3),
    _spec("cfg40_lr1e3_proj2_bb05_cls1p5", "optimization", learning_rate=1e-3,
          backbone_lr_multiplier=.5, projector_lr_multiplier=2.0,
          classifier_lr_multiplier=1.5),

    _spec("cfg41_out500_kv32", "capacity", osram_output_dim=500),
    _spec("cfg42_out900_kv32", "capacity", osram_output_dim=900),
    _spec("cfg43_out1100_kv32", "capacity", osram_output_dim=1100),
    _spec("cfg44_out1400_kv48", "capacity", osram_output_dim=1400,
          osram_key_dim=48, osram_value_dim=48),
    _spec("cfg45_lat320_out900_kv40", "capacity", latent_dim=320,
          osram_output_dim=900, osram_key_dim=40, osram_value_dim=40),
    _spec("cfg46_lat384_out1400_kv64", "capacity", latent_dim=384,
          osram_output_dim=1400, osram_key_dim=64, osram_value_dim=64),
    _spec("cfg47_lat512_out1400_kv32", "capacity", latent_dim=512,
          osram_output_dim=1400),
    _spec("cfg48_heads4_out1400_kv32", "capacity", osram_num_heads=4,
          osram_output_dim=1400),
    _spec("cfg49_heads8_out1024_kv64", "capacity", osram_output_dim=1024,
          osram_key_dim=64, osram_value_dim=64),
    _spec("cfg50_heads16_out1024_kv64", "capacity", osram_num_heads=16,
          osram_output_dim=1024, osram_key_dim=64, osram_value_dim=64),

    _spec("cfg51_adamw_lr1e4_cosine200", "schedule", optimizer="adamw",
          learning_rate=1e-4, weight_decay=1e-4, lr_schedule="cosine",
          warmup_ratio=.05, epochs=200),
    _spec("cfg52_adam_lr1e4_cosine200", "schedule", learning_rate=1e-4,
          lr_schedule="cosine", warmup_ratio=.05, epochs=200),
    _spec("cfg53_adamw_lr3e4_cosine200", "schedule", optimizer="adamw",
          learning_rate=3e-4, weight_decay=1e-4, lr_schedule="cosine",
          warmup_ratio=.05, epochs=200),
    _spec("cfg54_adam_lr3e4_constant200", "schedule", learning_rate=3e-4,
          epochs=200, gradient_clip_norm=1.0),
    _spec("cfg55_adamw_lr1e3_constant200", "schedule", optimizer="adamw",
          learning_rate=1e-3, weight_decay=1e-4, epochs=200),
    _spec("cfg56_adam_lr1e3_constant400", "schedule", learning_rate=1e-3,
          epochs=400),
    _spec("cfg57_adamw_lr1e4_cosine400", "schedule", optimizer="adamw",
          learning_rate=1e-4, weight_decay=1e-4, lr_schedule="cosine",
          warmup_ratio=.05, epochs=400),
    _spec("cfg58_adam_lr3e3_clip05", "schedule", learning_rate=3e-3,
          gradient_clip_norm=.5),
    _spec("cfg59_adamw_lr3e4_cosine_clip05", "schedule", optimizer="adamw",
          learning_rate=3e-4, weight_decay=1e-4, lr_schedule="cosine",
          warmup_ratio=.05, gradient_clip_norm=.5,
          backbone_lr_multiplier=.5, projector_lr_multiplier=2.0),
    _spec("cfg60_adam_lr1e3_cosine_clip5_cls05", "schedule",
          learning_rate=1e-3, lr_schedule="cosine", warmup_ratio=.05,
          gradient_clip_norm=5.0, classifier_lr_multiplier=.5),

    # Follow-up coverage for axes that are intentionally sparse above:
    # batch_size=4, projector dropout, stronger block-LR ratios, and capacity
    # boundaries.  These are a third queue wave, not duplicates of cfg01-60.
    _spec("cfg61_lr3e4_b4_d3", "optimization", learning_rate=3e-4,
          batch_size=4, dropout=.3),
    _spec("cfg62_lr1e3_b4_d5", "optimization", learning_rate=1e-3,
          batch_size=4, dropout=.5),
    _spec("cfg63_adamw_lr3e4_b4_wd1e4", "optimization", optimizer="adamw",
          learning_rate=3e-4, batch_size=4, weight_decay=1e-4, dropout=.3),
    _spec("cfg64_adam_cosine_lr1e3_b4", "schedule", learning_rate=1e-3,
          batch_size=4, lr_schedule="cosine", warmup_ratio=.05),
    _spec("cfg65_lr3e4_projdrop1", "optimization", learning_rate=3e-4,
          projector_dropout=.1),
    _spec("cfg66_lr3e4_projdrop2", "optimization", learning_rate=3e-4,
          projector_dropout=.2),
    _spec("cfg67_adamw_cosine_projdrop1", "schedule", optimizer="adamw",
          learning_rate=3e-4, weight_decay=1e-4, lr_schedule="cosine",
          warmup_ratio=.05, projector_dropout=.1),
    _spec("cfg68_adamw_cosine_projdrop2", "schedule", optimizer="adamw",
          learning_rate=3e-4, weight_decay=1e-4, lr_schedule="cosine",
          warmup_ratio=.05, projector_dropout=.2),
    _spec("cfg69_lr1e4_b16_d1_proj1", "optimization", learning_rate=1e-4,
          batch_size=16, dropout=.1, projector_dropout=.1),
    _spec("cfg70_lr3e4_b16_d5_proj2", "optimization", learning_rate=3e-4,
          batch_size=16, dropout=.5, projector_dropout=.2),
    _spec("cfg71_lr1e3_wd1e6_d0", "optimization", learning_rate=1e-3,
          weight_decay=1e-6, dropout=0.0),
    _spec("cfg72_lr1e3_wd1e2_d0", "optimization", learning_rate=1e-3,
          weight_decay=1e-2, dropout=0.0),
    _spec("cfg73_lr3e5_b4_d1", "optimization", learning_rate=3e-5,
          batch_size=4, dropout=.1),
    _spec("cfg74_lr3e3_b4_clip05", "optimization", learning_rate=3e-3,
          batch_size=4, gradient_clip_norm=.5),
    _spec("cfg75_adam_cosine_lr1e4_b8_clip5", "schedule", learning_rate=1e-4,
          batch_size=8, lr_schedule="cosine", warmup_ratio=.05,
          gradient_clip_norm=5.0),
    _spec("cfg76_adamw_lr3e3_wd1e3_d1", "schedule", optimizer="adamw",
          learning_rate=3e-3, weight_decay=1e-3, dropout=.1),
    _spec("cfg77_adamw_lr3e5_wd0_d5", "optimization", optimizer="adamw",
          learning_rate=3e-5, weight_decay=0.0, dropout=.5),
    _spec("cfg78_lr3e4_block025_proj4", "optimization", learning_rate=3e-4,
          backbone_lr_multiplier=.25, projector_lr_multiplier=4.0),
    _spec("cfg79_lr3e4_block2_proj05", "optimization", learning_rate=3e-4,
          backbone_lr_multiplier=2.0, projector_lr_multiplier=.5),
    _spec("cfg80_lr1e3_block2_proj05_cls05", "optimization", learning_rate=1e-3,
          backbone_lr_multiplier=2.0, projector_lr_multiplier=.5,
          classifier_lr_multiplier=.5),
    _spec("cfg81_out700_kv16", "capacity", osram_output_dim=700,
          osram_key_dim=16, osram_value_dim=16),
    _spec("cfg82_out700_kv48", "capacity", osram_output_dim=700,
          osram_key_dim=48, osram_value_dim=48),
    _spec("cfg83_heads4_out1024_kv32", "capacity", osram_num_heads=4,
          osram_output_dim=1024),
    _spec("cfg84_out1600_kv64", "capacity", osram_output_dim=1600,
          osram_key_dim=64, osram_value_dim=64),
    _spec("cfg85_lat320_out700_kv32", "capacity", latent_dim=320,
          osram_output_dim=700),
    _spec("cfg86_lat512_out1024_kv48", "capacity", latent_dim=512,
          osram_output_dim=1024, osram_key_dim=48, osram_value_dim=48),
    _spec("cfg87_heads2_out700_kv32", "capacity", osram_num_heads=2,
          osram_output_dim=700),
    _spec("cfg88_heads8_out1600_kv64", "capacity", osram_output_dim=1600,
          osram_key_dim=64, osram_value_dim=64),
    _spec("cfg89_heads16_out1400_kv48", "capacity", osram_num_heads=16,
          osram_output_dim=1400, osram_key_dim=48, osram_value_dim=48),
    _spec("cfg90_lat512_out1600_kv64", "capacity", latent_dim=512,
          osram_output_dim=1600, osram_key_dim=64, osram_value_dim=64),
)

SPEC_BY_ID = {item["id"]: item for item in SPECS}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical_mask_hashes(output: Path) -> dict[str, str]:
    """Hash mask rows independent of DataLoader batch/order serialization."""
    result = {}
    for rate in RATES:
        key = rate.replace(".", "p")
        path = output / f"predictions_miss_{key}.npz"
        with np.load(path) as archive:
            availability = archive["availability"].astype(np.float32, copy=False)
        ordered = availability[np.lexsort((availability[:, 2], availability[:, 1], availability[:, 0]))]
        result[rate] = hashlib.sha256(ordered.tobytes()).hexdigest()
    return result


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
        reference_output = source
        if canonical_mask_hashes(output) != canonical_mask_hashes(reference_output):
            raise ValueError("canonical evaluation masks differ from no-JEPA reference")
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "PROVENANCE.json", provenance)
        raise
    provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE {spec_id}", flush=True)


def _worker(spec_ids: list[str], gpu: int) -> None:
    """Backward-compatible serial worker for manual recovery runs."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="6",
               MKL_NUM_THREADS="6", PYTHONPATH=str(REPO))
    os.environ.update(env)
    for spec_id in spec_ids:
        train(spec_id)


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
    queue["status"] = "running"
    wave_size = 10
    failed = False
    for wave_start in range(0, max(len(bucket) for bucket in buckets), wave_size):
        children = []
        wave = {"wave": wave_start // wave_size + 1, "tasks": []}
        for gpu, spec_ids in zip(GPU_ASSIGNMENT, buckets):
            for spec_id in spec_ids[wave_start:wave_start + wave_size]:
                log_path = ROOT / f"gpu{gpu}_{spec_id}.log"
                log = log_path.open("a")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                           MKL_NUM_THREADS="2", PYTHONPATH=str(REPO))
                child = subprocess.Popen(
                    [sys.executable, "-u", __file__, "--train", spec_id], cwd=REPO,
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                row = {"gpu": gpu, "pid": child.pid, "spec_id": spec_id,
                       "log": str(log_path), "status": "running", "wave": wave["wave"]}
                queue["workers"].append(row)
                wave["tasks"].append(row)
                children.append((child, log, row))
                print(f"START wave={wave['wave']} GPU={gpu} PID={child.pid} spec={spec_id}", flush=True)
        queue.setdefault("waves", []).append(wave)
        write_json(ROOT / "QUEUE.json", queue)
        for child, log, row in children:
            row["exit_code"] = child.wait()
            row["status"] = "complete" if row["exit_code"] == 0 else "failed"
            failed = failed or row["exit_code"] != 0
            log.close()
            write_json(ROOT / "QUEUE.json", queue)
    queue["status"] = "failed" if failed else "complete"
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
