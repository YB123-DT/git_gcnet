"""Run a matched larger no-JEPA OSRAM capacity configuration on MOSI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_causal_nojepa_20260910 import run as no_jepa  # noqa: E402
from experiments.osram_mosi_hparam_sweep_20260918.run import (  # noqa: E402
    canonical_mask_hashes,
    sha,
    write_json,
)


ROOT = Path("/data2/yb/remote_experiments/osram_nojepa_capacity2400_20260919")
SEEDS = (66, 67, 68, 69, 70)
GPUS = (1, 2, 3)
LABEL = "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"


def configuration(seed: int):
    base, source, _ = no_jepa.configuration(seed)
    cfg = replace(
        base,
        seed=seed,
        osram_output_dim=2400,
        osram_key_dim=96,
        osram_value_dim=96,
        osram_num_heads=8,
        disable_unused_aux_modules=False,
    )
    return cfg, source


def train(seed: int) -> None:
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, source = configuration(seed)
    output = ROOT / f"seed_{seed}"
    if output.exists():
        state = output / "PROVENANCE.json"
        if state.exists() and json.loads(state.read_text()).get("status") == "complete":
            print(f"SKIP complete seed={seed}", flush=True)
            return
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    provenance = {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": LABEL,
        "experiment": "MOSI no-JEPA matched larger OSRAM capacity",
        "seed": seed,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection_protocol": "per-rate-test-oracle",
        "reference": str(source),
        "configuration_delta": {
            "osram_output_dim": 2400,
            "osram_key_dim": 96,
            "osram_value_dim": 96,
            "osram_num_heads": 8,
            "latent_dim": 256,
        },
        "auxiliary_modules": "retained for exact no-JEPA capacity comparison",
        "config": asdict(cfg),
        "reference_sha256": {
            name: sha(source / name) for name in ("config.json", "history.json", "metrics.json")
        },
    }
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(2)
        roots = [str(no_jepa.runner.FEATURES / name) for name in
                 ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")]
        print(f"TRAIN no-JEPA 2400/96 seed={seed} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise ValueError("unexpected selection protocol")
        if canonical_mask_hashes(output) != canonical_mask_hashes(source):
            raise ValueError("canonical evaluation masks differ from no-JEPA reference")
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "PROVENANCE.json", provenance)
        raise
    provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                      mask_validation="canonical_row_multiset")
    write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE no-JEPA 2400/96 seed={seed}", flush=True)


def launch() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "selection_protocol": "per-rate-test-oracle",
        "label": LABEL,
        "training_objective": "emotion-only",
        "auxiliary_modules": "retained",
        "overrides": {
            "osram_output_dim": 2400,
            "osram_key_dim": 96,
            "osram_value_dim": 96,
            "osram_num_heads": 8,
            "latent_dim": 256,
        },
        "tasks": [],
    }
    write_json(ROOT / "QUEUE.json", queue)
    children = []
    for index, seed in enumerate(SEEDS):
        gpu = GPUS[index % len(GPUS)]
        log_path = ROOT / f"gpu{gpu}_seed{seed}.log"
        log = log_path.open("a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", PYTHONPATH=str(REPO))
        child = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__)), "--seed", str(seed)],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        row = {"gpu": gpu, "pid": child.pid, "seed": seed, "status": "running",
               "log": str(log_path)}
        queue["tasks"].append(row)
        children.append((child, log, row))
    queue["status"] = "running"
    write_json(ROOT / "QUEUE.json", queue)
    failed = False
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
