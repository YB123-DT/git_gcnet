"""Run cfg84 no-JEPA MOSI with one independently trained model per rate.

This is an internal diagnostic: each official missing rate is used for both
training and test evaluation, with the best epoch selected by that rate's Test
weighted-F1.  It is intentionally separate from the cyclic mixed-rate runner.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_no_aux_cfg84_20260919 import run as no_aux  # noqa: E402
from experiments.osram_mosi_hparam_sweep_20260918.run import (  # noqa: E402
    sha,
    write_json,
)


ROOT = Path("/data2/yb/remote_experiments/osram_mosi_cfg84_nojepa_fixed_rate_20260920")
SEED = 66
RATES = tuple(round(index / 10, 1) for index in range(8))
GPUS = (1, 4, 6)
LABEL = "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"
SELECTION_METRIC = "weighted_f1"


def rate_key(rate: float) -> str:
    return format(rate, ".1f")


def rate_dir(rate: float) -> str:
    return rate_key(rate).replace(".", "p")


def configuration(rate: float):
    base, source = no_aux.configuration(SEED)
    return replace(
        base,
        seed=SEED,
        train_rate_mode="fixed",
        fixed_missing_rate=rate,
        checkpoint_selection="test-oracle",
        evaluate_test=True,
        training_objective="emotion-only",
        osram_output_dim=1600,
        osram_key_dim=64,
        osram_value_dim=64,
        osram_num_heads=8,
        disable_unused_aux_modules=True,
    ), source


def train(rate: float, gpu: int | None = None) -> None:
    import torch

    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, source = configuration(rate)
    output = ROOT / f"seed_{SEED}" / f"rate_{rate_dir(rate)}"
    if output.exists():
        metrics_path = output / "metrics.json"
        provenance_path = output / "PROVENANCE.json"
        if metrics_path.exists() and provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("status") == "complete":
                print(f"SKIP complete rate={rate_key(rate)}", flush=True)
                return
        raise FileExistsError(f"refusing to overwrite {output}")

    output.mkdir(parents=True)
    provenance = {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": LABEL,
        "experiment": "MOSI cfg84 no-JEPA fixed-rate train/test",
        "dataset": "CMUMOSI",
        "seed": SEED,
        "rate": rate,
        "train_rate_mode": "fixed",
        "fixed_missing_rate": rate,
        "selection_protocol": "single-rate-test-oracle",
        "selection_metric": SELECTION_METRIC,
        "selection_split": "test",
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", str(gpu) if gpu is not None else None),
        "configuration_delta": {
            "train_rate_mode": "fixed",
            "fixed_missing_rate": rate,
            "checkpoint_selection": "test-oracle",
            "osram_output_dim": 1600,
            "osram_num_heads": 8,
            "osram_key_dim": 64,
            "osram_value_dim": 64,
            "disable_unused_aux_modules": True,
        },
        "reference": str(source),
        "reference_sha256": {
            name: sha(source / name)
            for name in ("config.json", "history.json", "metrics.json")
        },
    }
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "PROVENANCE.json", provenance)

    roots = [
        str(no_aux.no_jepa.runner.FEATURES / name)
        for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
    ]
    try:
        torch.set_num_threads(2)
        print(
            f"TRAIN MOSI seed={SEED} rate={rate_key(rate)} fixed-rate "
            f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics.get("selection_metric") != SELECTION_METRIC:
            raise RuntimeError("MOSI fixed-rate selection must use weighted_f1")
        if metrics.get("selection_split") != "test-oracle":
            raise RuntimeError("fixed-rate run must use Test-oracle selection")
        key = rate_key(rate)
        expected_hash = json.loads((source / "metrics.json").read_text())["mask_sha256"][key]
        actual_hash = metrics["mask_sha256"][key]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"mask mismatch for rate={key}: expected {expected_hash}, got {actual_hash}"
            )
        provenance.update(
            status="complete",
            completed_utc=datetime.now(timezone.utc).isoformat(),
            selected_epoch=metrics.get("best_epoch"),
            selected_score=metrics.get("best_selection_mean_score"),
            mask_sha256=actual_hash,
        )
        write_json(output / "PROVENANCE.json", provenance)
        print(f"COMPLETE MOSI seed={SEED} rate={key}", flush=True)
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "PROVENANCE.json", provenance)
        raise


def launch() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CMUMOSI",
        "seed": SEED,
        "rates": list(RATES),
        "train_rate_mode": "fixed",
        "selection_protocol": "single-rate-test-oracle",
        "selection_metric": SELECTION_METRIC,
        "gpus": list(GPUS),
        "tasks": [],
    }
    write_json(ROOT / "QUEUE.json", queue)
    for start in range(0, len(RATES), len(GPUS)):
        children = []
        for gpu, rate in zip(GPUS, RATES[start : start + len(GPUS)]):
            log_path = ROOT / f"seed_{SEED}" / f"rate_{rate_dir(rate)}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("a")
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                PYTHONPATH=str(REPO),
            )
            child = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__)), "--rate", rate_key(rate)],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            task = {
                "rate": rate,
                "gpu": gpu,
                "pid": child.pid,
                "status": "running",
                "log": str(log_path),
            }
            queue["tasks"].append(task)
            children.append((child, log, task))
        queue["status"] = "running"
        write_json(ROOT / "QUEUE.json", queue)
        for child, log, task in children:
            task["exit_code"] = child.wait()
            task["status"] = "complete" if task["exit_code"] == 0 else "failed"
            log.close()
            write_json(ROOT / "QUEUE.json", queue)
            if task["exit_code"] != 0:
                queue["status"] = "failed"
                write_json(ROOT / "QUEUE.json", queue)
                raise RuntimeError(f"fixed-rate task failed: rate={task['rate']}")
    queue.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(ROOT / "QUEUE.json", queue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--rate", type=float, choices=RATES)
    args = parser.parse_args()
    if args.launch:
        launch()
    else:
        train(args.rate)


if __name__ == "__main__":
    main()
