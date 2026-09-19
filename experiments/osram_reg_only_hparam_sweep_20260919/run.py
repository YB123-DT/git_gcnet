"""MOSI full-MMoE regression-only hyper-parameter screening.

This reuses the existing 90-spec sweep, but keeps the fixed supervised Teacher
and the original MMoE predictor.  Eight latent-dimension variants are recorded
as skipped because the available supervised Teacher checkpoint is 256-d.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_mosi_hparam_sweep_20260918 import run as base  # noqa: E402

REMOTE = Path("/data2/yb/remote_experiments")
ROOT = Path(
    os.environ.get(
        "REG_ONLY_SWEEP_ROOT",
        str(REMOTE / "osram_reg_only_hparam_sweep_20260919"),
    )
)
TEACHER = REMOTE / "osram_supervised_teacher_20260914/teacher/seed_66/teacher_projectors.pt"
SEED = 66
GPU_ASSIGNMENT = (1, 2, 3)
WAVE_SIZE = 10
LABEL = "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"

# Full MMoE reg-only can use the existing 256-d supervised Teacher.  The four
# latent-dimension variants require separate Teacher pretraining and are not
# silently mixed into this sweep.
SKIPPED_LATENT_SPECS = tuple(
    item["id"] for item in base.SPECS if "latent_dim" in item["overrides"]
)
SPECS = tuple(
    item for item in base.SPECS if "latent_dim" not in item["overrides"]
)
MAX_EPOCHS = (
    int(os.environ["REG_ONLY_SWEEP_MAX_EPOCHS"])
    if os.environ.get("REG_ONLY_SWEEP_MAX_EPOCHS")
    else None
)
if MAX_EPOCHS is not None:
    SPECS = tuple(
        item for item in SPECS if item["overrides"].get("epochs", 100) <= MAX_EPOCHS
    )
SPEC_BY_ID = {item["id"]: item for item in SPECS}


def _reference_config():
    old = json.loads((base.SOURCE_ROOT / f"seed_{SEED}" / "config.json").read_text())
    required = {
        "dataset": "CMUMOSI",
        "seed": SEED,
        "training_objective": "emotion-only",
        "backbone_type": "osram",
        "fusion_type": "mean",
        "osram_bidirectional": False,
        "osram_forward_slot_reuse": False,
        "osram_write_step": 0.6,
        "osram_readout_fusion": "flat",
        "train_rate_mode": "cyclic",
        "checkpoint_selection": "test-oracle-per-rate",
        "evaluate_test": True,
        "completion_path": "none",
        "classification_completion": False,
        "initial_backbone_checkpoint": None,
    }
    mismatch = {
        key: old.get(key)
        for key, expected in required.items()
        if old.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"reference no-JEPA config is not locked: {mismatch}")
    from gcnet_missing_m3.train_gcnet import TrainConfig

    return TrainConfig(**old), base.SOURCE_ROOT / f"seed_{SEED}"


def configuration(item: dict[str, Any]):
    base_cfg, source = _reference_config()
    overrides = dict(item["overrides"])
    overrides.update(
        {
            "seed": SEED,
            "training_objective": "joint-reg-only",
            "backbone_type": "osram",
            "fusion_type": "mean",
            "osram_bidirectional": False,
            "osram_forward_slot_reuse": False,
            "osram_write_step": 0.6,
            "osram_readout_fusion": "flat",
            "train_rate_mode": "cyclic",
            "checkpoint_selection": "test-oracle-per-rate",
            "evaluate_test": True,
            "completion_path": "none",
            "classification_completion": False,
            "initial_backbone_checkpoint": None,
            "teacher_mode": "pretrained-frozen",
            "teacher_checkpoint": str(TEACHER),
            "target_space": "all-modalities",
            "text_subspace_checkpoint": None,
            "text_core": False,
            "simple_regression_predictor": False,
        }
    )
    cfg = replace(base_cfg, **overrides)
    if cfg.latent_dim != 256:
        raise ValueError("fixed supervised Teacher requires latent_dim=256")
    for field, expected in {
        "training_objective": "joint-reg-only",
        "backbone_type": "osram",
        "fusion_type": "mean",
        "osram_bidirectional": False,
        "osram_forward_slot_reuse": False,
        "osram_write_step": 0.6,
        "osram_readout_fusion": "flat",
        "train_rate_mode": "cyclic",
        "checkpoint_selection": "test-oracle-per-rate",
        "evaluate_test": True,
        "completion_path": "none",
        "classification_completion": False,
        "teacher_mode": "pretrained-frozen",
        "teacher_checkpoint": str(TEACHER),
        "target_space": "all-modalities",
        "text_core": False,
        "simple_regression_predictor": False,
    }.items():
        if getattr(cfg, field) != expected:
            raise ValueError(f"grid changed locked field {field}: {getattr(cfg, field)!r}")
    return cfg, source


def _provenance(item: dict[str, Any], cfg, source: Path) -> dict[str, Any]:
    return {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": LABEL,
        "experiment": "MOSI full-MMoE regression-only 90-spec screening",
        "spec": item,
        "executed_valid_spec_count": len(SPECS),
        "requested_spec_count": len(base.SPECS),
        "skipped_latent_specs": list(SKIPPED_LATENT_SPECS),
        "max_epochs_filter": MAX_EPOCHS,
        "seed": SEED,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection_protocol": "per-rate-test-oracle",
        "teacher_mode": "pretrained-frozen",
        "teacher_checkpoint": str(TEACHER),
        "reference": str(source),
        "config": asdict(cfg),
    }


def train(spec_id: str) -> None:
    item = SPEC_BY_ID[spec_id]
    cfg, source = configuration(item)
    output = ROOT / "screen_seed66" / spec_id
    if output.exists():
        provenance = output / "PROVENANCE.json"
        if provenance.exists() and json.loads(provenance.read_text()).get("status") == "complete":
            print(f"SKIP complete {spec_id}", flush=True)
            return
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    base.write_json(output / "config.json", asdict(cfg))
    provenance = _provenance(item, cfg, source)
    base.write_json(output / "PROVENANCE.json", provenance)
    try:
        import torch
        from gcnet_missing_m3.train_gcnet import run_experiment

        torch.set_num_threads(2)
        roots = [
            str(base.FEATURES / name)
            for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
        ]
        print(
            f"TRAIN {spec_id} full-MMoE reg-only GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"overrides={item['overrides']}",
            flush=True,
        )
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise ValueError("unexpected selection protocol")
        if base.canonical_mask_hashes(output) != base.canonical_mask_hashes(source):
            raise ValueError("canonical evaluation masks differ from no-JEPA reference")
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        base.write_json(output / "PROVENANCE.json", provenance)
        raise
    provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    base.write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE {spec_id}", flush=True)


def launch() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "gpus": list(GPU_ASSIGNMENT),
        "requested_spec_count": len(base.SPECS),
        "executed_spec_count": len(SPECS),
        "skipped_latent_specs": list(SKIPPED_LATENT_SPECS),
        "max_epochs_filter": MAX_EPOCHS,
        "configs": list(base.SPECS),
        "selection_protocol": "per-rate-test-oracle",
        "teacher_mode": "pretrained-frozen",
        "teacher_checkpoint": str(TEACHER),
        "label": LABEL,
        "workers": [],
        "waves": [],
    }
    base.write_json(ROOT / "QUEUE.json", queue)
    buckets = [[] for _ in GPU_ASSIGNMENT]
    for index, item in enumerate(SPECS):
        buckets[index % len(buckets)].append(item["id"])
    failed = False
    queue["status"] = "running"
    for wave_start in range(0, max(len(bucket) for bucket in buckets), WAVE_SIZE):
        children = []
        wave_number = wave_start // WAVE_SIZE + 1
        wave = {"wave": wave_number, "tasks": []}
        for gpu, spec_ids in zip(GPU_ASSIGNMENT, buckets):
            for spec_id in spec_ids[wave_start : wave_start + WAVE_SIZE]:
                log_path = ROOT / f"gpu{gpu}_{spec_id}.log"
                log = log_path.open("a")
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    OMP_NUM_THREADS="2",
                    MKL_NUM_THREADS="2",
                    PYTHONPATH=str(REPO),
                )
                child = subprocess.Popen(
                    [sys.executable, "-u", __file__, "--train", spec_id],
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                row = {
                    "gpu": gpu,
                    "pid": child.pid,
                    "spec_id": spec_id,
                    "log": str(log_path),
                    "status": "running",
                    "wave": wave_number,
                }
                queue["workers"].append(row)
                wave["tasks"].append(row)
                children.append((child, log, row))
        queue["waves"].append(wave)
        base.write_json(ROOT / "QUEUE.json", queue)
        for child, log, row in children:
            row["exit_code"] = child.wait()
            row["status"] = "complete" if row["exit_code"] == 0 else "failed"
            failed = failed or row["exit_code"] != 0
            log.close()
            base.write_json(ROOT / "QUEUE.json", queue)
    queue["status"] = "failed" if failed else "complete"
    queue["completed_utc"] = datetime.now(timezone.utc).isoformat()
    base.write_json(ROOT / "QUEUE.json", queue)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--train", metavar="SPEC_ID", choices=tuple(SPEC_BY_ID))
    args = parser.parse_args()
    if args.launch:
        launch()
    else:
        train(args.train)


if __name__ == "__main__":
    main()
