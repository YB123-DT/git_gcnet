"""Run the MOSI causal OSRAM Base--Gap Delta Fusion experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

REMOTE = Path("/data2/yb/remote_experiments")
ROOT = REMOTE / "osram_mosi_base_gap_delta_20260921"
REFERENCE_ROOT = REMOTE / "osram_no_aux_cfg84_20260919"
FEATURES = Path(
    "/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features"
)
SEEDS = (66, 67, 68, 69, 70)
GPU_IDS = (0, 1, 2, 4, 5)
LABEL = "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical_mask_hashes(output: Path) -> dict[str, str]:
    result = {}
    for index in range(8):
        rate = f"{index / 10:.1f}"
        key = rate.replace(".", "p")
        with np.load(output / f"predictions_miss_{key}.npz") as archive:
            availability = archive["availability"].astype(np.float32, copy=False)
        ordered = availability[
            np.lexsort((availability[:, 2], availability[:, 1], availability[:, 0]))
        ]
        result[rate] = hashlib.sha256(ordered.tobytes()).hexdigest()
    return result


def configuration(seed: int):
    from gcnet_missing_m3.train_gcnet import TrainConfig

    source = REFERENCE_ROOT / f"seed_{seed}"
    old = json.loads((source / "config.json").read_text())
    settings = dict(
        old,
        seed=seed,
        osram_readout_fusion="base-gap-delta",
        checkpoint_selection="test-oracle-per-rate",
    )
    cfg = TrainConfig(**settings)
    if cfg.osram_output_dim != 1600 or cfg.osram_num_heads != 8:
        raise ValueError("reference is not cfg84 output/head configuration")
    if cfg.osram_key_dim != 64 or cfg.osram_value_dim != 64 or cfg.osram_write_step != 0.6:
        raise ValueError("reference is not cfg84 key/value/write-step configuration")
    if cfg.osram_readout_fusion != "base-gap-delta":
        raise ValueError("delta readout was not enabled")
    return cfg, source


def _roots() -> list[str]:
    return [
        str(FEATURES / name)
        for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
    ]


def train(seed: int) -> None:
    import torch

    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, source = configuration(seed)
    output = ROOT / f"seed_{seed}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": LABEL,
        "experiment": "MOSI cfg84 causal OSRAM Base-Gap Delta Fusion",
        "seed": seed,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "reference": str(source),
        "selection_protocol": "per-rate-test-oracle",
        "from_scratch": True,
        "configuration_delta": {
            "osram_readout_fusion": "base-gap-delta",
            "checkpoint_selection": "test-oracle-per-rate",
        },
        "config": asdict(cfg),
        "reference_sha256": {
            name: sha(source / name) for name in ("config.json", "history.json", "metrics.json")
        },
        "source_sha256": {
            name: sha(REPO / name)
            for name in (
                "gcnet_missing_m3/osram.py",
                "gcnet_missing_m3/model.py",
                "gcnet_missing_m3/train_gcnet.py",
                "experiments/osram_mosi_base_gap_delta_20260921/run.py",
            )
        },
    }
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(2)
        print(
            f"TRAIN base-gap-delta seed={seed} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            "cfg84 causal eta=.6 cyclic per-rate-test-oracle",
            flush=True,
        )
        run_experiment(cfg, *_roots(), output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise ValueError("unexpected checkpoint selection protocol")
        if canonical_mask_hashes(output) != canonical_mask_hashes(source):
            raise ValueError("canonical evaluation masks differ from cfg84 reference")
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "PROVENANCE.json", provenance)
        raise
    provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE seed={seed}", flush=True)


def launch() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": LABEL,
        "selection_protocol": "per-rate-test-oracle",
        "reference_root": str(REFERENCE_ROOT),
        "readout_fusion": "base-gap-delta",
        "tasks": [],
    }
    write_json(ROOT / "QUEUE.json", queue)
    processes = []
    for seed, gpu in zip(SEEDS, GPU_IDS):
        log_path = ROOT / f"seed_{seed}.log"
        log = log_path.open("w")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            PYTHONPATH=str(REPO),
            GCNET_DATASET_ROOT="/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset",
        )
        child = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__)), "--seed", str(seed)],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        row = {"seed": seed, "gpu": gpu, "pid": child.pid, "status": "running", "log": str(log_path)}
        queue["tasks"].append(row)
        processes.append((child, log, row))
        print(f"START seed={seed} GPU={gpu} PID={child.pid}", flush=True)
    write_json(ROOT / "QUEUE.json", queue)
    while processes:
        remaining = []
        for child, log, row in processes:
            code = child.poll()
            if code is None:
                remaining.append((child, log, row))
                continue
            row["exit_code"] = int(code)
            row["status"] = "complete" if code == 0 else "failed"
            log.close()
        processes = remaining
        write_json(ROOT / "QUEUE.json", queue)
        if processes:
            time.sleep(10)
    queue["status"] = "failed" if any(row["status"] == "failed" for row in queue["tasks"]) else "complete"
    queue["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / "QUEUE.json", queue)


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
        parser.error("use --launch or --seed SEED")


if __name__ == "__main__":
    main()
