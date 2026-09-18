"""Train the unchanged causal no-JEPA OSRAM with binary MOSI supervision."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_causal_readout_20260910.run import (  # noqa: E402
    FEATURES,
    SEEDS,
    sha,
    write_json,
)
from experiments.osram_causal_readout_20260910.summarize import (  # noqa: E402
    extract_best,
    mask_hashes,
)

REMOTE = Path("/data2/yb/remote_experiments")
SOURCE_ROOT = REMOTE / "osram_causal_nojepa_20260910/mosi"
ROOT = REMOTE / "osram_mosi_binary_20260918"


def configuration(seed: int):
    from gcnet_missing_m3.train_gcnet import TrainConfig

    if seed not in SEEDS:
        raise ValueError(f"unsupported seed {seed}")
    source = SOURCE_ROOT / f"seed_{seed}"
    old = json.loads((source / "config.json").read_text())
    required = {
        "dataset": "CMUMOSI",
        "training_objective": "emotion-only",
        "train_rate_mode": "cyclic",
        "backbone_type": "osram",
        "fusion_type": "mean",
        "osram_bidirectional": False,
        "osram_forward_slot_reuse": False,
        "osram_write_step": 0.6,
        "osram_readout_fusion": "flat",
        "checkpoint_selection": "test-oracle-per-rate",
        "evaluate_test": True,
    }
    mismatch = {key: old.get(key) for key, value in required.items() if old.get(key) != value}
    if mismatch:
        raise ValueError(f"no-JEPA reference is not locked: {mismatch}")
    cfg = TrainConfig(
        **dict(
            old,
            mosi_task_mode="binary",
            checkpoint_selection="test-oracle-per-rate",
        )
    )
    before = asdict(TrainConfig(**old))
    after = asdict(cfg)
    delta = {key: [before[key], after[key]] for key in before if before[key] != after[key]}
    expected = {"mosi_task_mode": ["regression", "binary"]}
    if delta != expected:
        raise ValueError(f"unexpected configuration delta: {delta}")
    return cfg, source, delta


def train(seed: int) -> None:
    import torch

    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, source, delta = configuration(seed)
    output = ROOT / "mosi" / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "reference": str(source),
        "from_scratch": True,
        "source_checkpoint_loaded_into_model": False,
        "configuration_delta": delta,
        "selection_protocol": "per-rate-test-oracle",
        "mask_protocol": "inherited official cyclic train/test schedules; test masks must match no-JEPA",
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "reference_sha256": {
            name: sha(source / name) for name in ("config.json", "history.json", "metrics.json")
        },
        "source_sha256": {
            name: sha(REPO / name)
            for name in (
                "gcnet_missing_m3/train_gcnet.py",
                "gcnet_missing_m3/osram.py",
                "experiments/osram_mosi_binary_20260918/run.py",
            )
        },
    }
    write_json(output / "PROVENANCE.json", provenance)
    write_json(output / "config.json", asdict(cfg))
    try:
        torch.set_num_threads(6)
        roots = [
            str(FEATURES / name)
            for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
        ]
        print(
            f"TRAIN seed={seed} no-JEPA binary MOSI cyclic eta=.6 epochs={cfg.epochs}",
            flush=True,
        )
        run_experiment(cfg, *roots, output_dir=str(output))
        extract_best(json.loads((output / "history.json").read_text()))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics["selection_protocol"] != "per-rate-test-oracle":
            raise ValueError("unexpected selection protocol")
        if mask_hashes(metrics) != mask_hashes(json.loads((source / "metrics.json").read_text())):
            raise ValueError("evaluation mask hashes differ from no-JEPA reference")
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "PROVENANCE.json", provenance)
        raise
    provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE seed={seed}", flush=True)


def launch(gpus=(2, 2, 2, 3, 3)) -> None:
    if len(gpus) != len(SEEDS):
        raise ValueError("one GPU assignment is required per seed")
    for seed in SEEDS:
        configuration(seed)
        if (ROOT / "mosi" / f"seed_{seed}").exists():
            raise FileExistsError(f"seed {seed} already has output; refusing duplicate launch")
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "QUEUE.json"
    state = {"status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(), "tasks": []}
    with manifest.open("x") as handle:
        json.dump(state, handle)
    children = []
    for seed, gpu in zip(SEEDS, gpus):
        log_path = ROOT / f"seed{seed}.log"
        with log_path.open("x") as log:
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="6",
                MKL_NUM_THREADS="6",
                PYTHONPATH=str(REPO),
            )
            child = subprocess.Popen(
                [sys.executable, "-u", __file__, "--seed", str(seed)],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        row = {"seed": seed, "gpu": gpu, "pid": child.pid, "log": str(log_path), "status": "running"}
        children.append((child, row))
        state["tasks"].append(row)
        write_json(manifest, state)
        print(f"START seed={seed} GPU={gpu} PID={child.pid}", flush=True)
    state["status"] = "running"
    write_json(manifest, state)
    for child, row in children:
        row["exit_code"] = child.wait()
        row["status"] = "complete" if row["exit_code"] == 0 else "failed"
        write_json(manifest, state)
    state.update(
        status="complete" if all(row["exit_code"] == 0 for _, row in children) else "failed",
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(manifest, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    launch() if args.launch else train(args.seed)
