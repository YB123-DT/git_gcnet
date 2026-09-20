"""Run the MOSI Local/Base/Raw-Gap/Residual-Gap diagnostic.

Each variant is trained from scratch from the same cfg84 causal no-JEPA
configuration.  The queue uses one worker per available GPU and keeps the
official masks and per-rate Test-oracle selection unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_mosi_hparam_sweep_20260918.run import (  # noqa: E402
    canonical_mask_hashes,
    sha,
    write_json,
)

REMOTE = Path("/data2/yb/remote_experiments")
ROOT = REMOTE / "osram_mosi_memory_gap_ablation_20260920"
REFERENCE_ROOT = REMOTE / "osram_no_aux_cfg84_20260919"
FEATURES = Path(
    "/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset/CMUMOSI/features"
)
SEEDS = (66, 67, 68, 69, 70)
VARIANTS = ("local-only", "local-base", "raw-gap", "full")
# Three concurrent workers keep host RAM stable; the model is GPU-memory light
# but each Python process still holds the full feature cache in host memory.
GPU_IDS = (1, 2, 3)
RATES = tuple(f"{index / 10:.1f}" for index in range(8))
LABEL = "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"


def configuration(variant: str, seed: int):
    from gcnet_missing_m3.train_gcnet import TrainConfig

    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError(f"unsupported variant/seed: {variant}/{seed}")
    source = REFERENCE_ROOT / f"seed_{seed}"
    old = json.loads((source / "config.json").read_text())
    if old.get("dataset") != "CMUMOSI" or old.get("seed") != seed:
        raise ValueError(f"unexpected reference config: {source}")
    if old.get("training_objective") != "emotion-only":
        raise ValueError("memory-gap ablation requires the emotion-only cfg84 reference")
    gap_read = "raw" if variant == "raw-gap" else "residual"
    osram_ablation = {
        "local-only": "local-only",
        "local-base": "local-base",
        "raw-gap": "full",
        "full": "full",
    }[variant]
    settings = dict(
        old,
        seed=seed,
        osram_ablation=osram_ablation,
        osram_gap_read=gap_read,
        checkpoint_selection="test-oracle-per-rate",
    )
    cfg = TrainConfig(**settings)
    if cfg.osram_output_dim != 1600 or cfg.osram_key_dim != 64 or cfg.osram_value_dim != 64:
        raise ValueError("reference is not cfg84 (1600 output, 64/64 key/value)")
    if cfg.osram_num_heads != 8 or cfg.osram_write_step != 0.6:
        raise ValueError("reference is not the locked causal cfg84 setup")
    return cfg, source


def _roots() -> list[str]:
    return [str(FEATURES / name) for name in (
        "wav2vec-large-c-UTT",
        "deberta-large-4-UTT",
        "manet_UTT",
    )]


def train(variant: str, seed: int) -> None:
    import torch

    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, source = configuration(variant, seed)
    output = ROOT / variant / f"seed_{seed}"
    if output.exists():
        provenance_path = output / "PROVENANCE.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("status") == "complete":
                print(f"SKIP complete {variant} seed={seed}", flush=True)
                return
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": LABEL,
        "experiment": "MOSI cfg84 OSRAM memory/base/gap ablation",
        "variant": variant,
        "seed": seed,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "reference": str(source),
        "selection_protocol": "per-rate-test-oracle",
        "from_scratch": True,
        "configuration_delta": {
            "osram_ablation": cfg.osram_ablation,
            "osram_gap_read": cfg.osram_gap_read,
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
                "experiments/osram_mosi_memory_gap_ablation_20260920/run.py",
            )
        },
    }
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(2)
        print(
            f"TRAIN {variant} seed={seed} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} "
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
    provenance.update(
        status="complete",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        mask_validation="canonical_row_multiset",
    )
    write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE {variant} seed={seed}", flush=True)


def _write_queue(queue: dict) -> None:
    write_json(ROOT / "QUEUE.json", queue)


def launch() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [{"variant": variant, "seed": seed} for variant, seed in product(VARIANTS, SEEDS)]
    queue = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": LABEL,
        "selection_protocol": "per-rate-test-oracle",
        "reference_root": str(REFERENCE_ROOT),
        "tasks": [],
    }
    _write_queue(queue)
    pending = list(tasks)
    running: dict[int, tuple[subprocess.Popen, object, dict]] = {}
    while pending or running:
        free_gpus = [gpu for gpu in GPU_IDS if gpu not in running]
        while pending and free_gpus:
            task = pending.pop(0)
            gpu = free_gpus.pop(0)
            variant, seed = task["variant"], task["seed"]
            log_path = ROOT / f"{variant}_seed{seed}.log"
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
                [sys.executable, "-u", str(Path(__file__)), "--variant", variant, "--seed", str(seed)],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            row = {"variant": variant, "seed": seed, "gpu": gpu, "pid": child.pid,
                   "status": "running", "log": str(log_path)}
            queue["tasks"].append(row)
            running[gpu] = (child, log, row)
            print(f"START {variant} seed={seed} GPU={gpu} PID={child.pid}", flush=True)
            _write_queue(queue)
        finished = []
        for gpu, (child, log, row) in running.items():
            code = child.poll()
            if code is None:
                continue
            row["exit_code"] = int(code)
            row["status"] = "complete" if code == 0 else "failed"
            log.close()
            finished.append(gpu)
        for gpu in finished:
            del running[gpu]
        if finished:
            _write_queue(queue)
        if running:
            time.sleep(5)
    queue["status"] = "failed" if any(row["status"] == "failed" for row in queue["tasks"]) else "complete"
    queue["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_queue(queue)
    if queue["status"] == "complete":
        subprocess.run([sys.executable, str(Path(__file__).with_name("summarize.py")), "--root", str(ROOT)],
                       cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.variant is not None and args.seed is not None:
        train(args.variant, args.seed)
    else:
        parser.error("use --launch or --variant VARIANT --seed SEED")


if __name__ == "__main__":
    main()
