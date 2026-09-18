"""Train the causal eta=.6 Flat Text-Core experiment on CMU-MOSI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_causal_nojepa_20260910 import run as nojepa

ROOT = Path("/data2/yb/remote_experiments/osram_text_core_20260918")
SEEDS = tuple(nojepa.runner.SEEDS)
FEATURES = nojepa.runner.FEATURES


def configuration(seed: int):
    from gcnet_missing_m3.train_gcnet import TrainConfig

    base, source, _ = nojepa.configuration(seed)
    cfg = replace(
        base,
        text_core=True,
        training_objective="emotion-only",
        checkpoint_selection="test-oracle-per-rate",
        evaluate_test=True,
        emotion_loss_mode="sample-mean",
    )
    assert cfg.backbone_type == "osram"
    assert not cfg.osram_bidirectional and cfg.osram_write_step == 0.6
    assert cfg.osram_readout_fusion == "flat"
    # Keep this explicit so a future baseline config change cannot silently
    # turn Text-Core into a different experiment.
    assert asdict(cfg)["text_core"] is True
    assert TrainConfig(**asdict(cfg)).text_core is True
    return cfg, source


def train(seed: int) -> None:
    import torch

    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, source = configuration(seed)
    output = ROOT / "mosi" / f"seed_{seed}"
    if (output / "metrics.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "variant": "text-core",
        "seed": seed,
        "reference_no_jepa": str(source),
        "config": asdict(cfg),
        "selection_protocol": "per-rate-test-oracle",
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "source_sha256": {
            name: nojepa.runner.sha(REPO / name)
            for name in (
                "gcnet_missing_m3/text_core.py",
                "gcnet_missing_m3/osram.py",
                "gcnet_missing_m3/model.py",
                "gcnet_missing_m3/train_gcnet.py",
                "experiments/osram_text_core_20260918/run.py",
            )
        },
    }
    nojepa.runner.write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(6)
        roots = [
            str(FEATURES / name)
            for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")
        ]
        print(
            f"TRAIN Text-Core seed={seed} causal eta=.6 Flat cyclic "
            "epochs=100 per-rate-test-oracle",
            flush=True,
        )
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics["selection_protocol"] != "per-rate-test-oracle":
            raise RuntimeError("Text-Core must use per-rate Test-oracle selection")
        reference_metrics = json.loads((source / "metrics.json").read_text())
        if nojepa.runner.mask_hashes(metrics) != nojepa.runner.mask_hashes(reference_metrics):
            raise RuntimeError("Text-Core mask protocol differs from no-JEPA reference")
        provenance.update(
            {
                "status": "complete",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "selection_protocol": metrics["selection_protocol"],
            }
        )
    except BaseException as error:
        provenance.update(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        )
        nojepa.runner.write_json(output / "PROVENANCE.json", provenance)
        raise
    nojepa.runner.write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE Text-Core seed={seed}", flush=True)


def launch(gpus=(5, 5, 6, 6, 7)) -> None:
    if len(gpus) != len(SEEDS):
        raise ValueError("one GPU assignment is required per seed")
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "QUEUE.json"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite launch manifest: {manifest}")
    state = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": [],
    }
    nojepa.runner.write_json(manifest, state)
    children = []
    for seed, gpu in zip(SEEDS, gpus):
        configuration(seed)
        output = ROOT / "mosi" / f"seed_{seed}"
        if output.exists():
            raise FileExistsError(f"seed output already exists: {output}")
        log_path = ROOT / f"seed{seed}.log"
        log = log_path.open("x")
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
        row = {
            "seed": seed,
            "gpu": gpu,
            "pid": child.pid,
            "log": str(log_path),
            "status": "running",
        }
        children.append((child, log, row))
        state["tasks"].append(row)
        nojepa.runner.write_json(manifest, state)
        print(f"START Text-Core seed={seed} GPU={gpu} PID={child.pid}", flush=True)
    state["status"] = "running"
    nojepa.runner.write_json(manifest, state)
    for child, log, row in children:
        row["exit_code"] = child.wait()
        log.close()
        row["status"] = "complete" if row["exit_code"] == 0 else "failed"
        nojepa.runner.write_json(manifest, state)
    state.update(
        {
            "status": "complete"
            if all(row["exit_code"] == 0 for _, _, row in children)
            else "failed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    nojepa.runner.write_json(manifest, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    launch() if args.launch else train(args.seed)
