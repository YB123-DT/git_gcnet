"""L1 control: same frozen joint projectors, no predicted-slot reinjection."""

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

from experiments.osram_joint_frozen_reinjection_20260922.run import (  # noqa: E402
    DATASET_ROOT,
    GPUS,
    JOINT_ROOT,
    LABEL,
    LOCAL_BASE,
    LOCAL_MASK_REFERENCE,
    ROOT,
    SEEDS,
    feature_roots,
)
from experiments.osram_mosi_hparam_sweep_20260918.run import sha, write_json  # noqa: E402


def configuration(seed: int):
    from gcnet_missing_m3.train_gcnet import TrainConfig

    base = TrainConfig(
        **json.loads((LOCAL_BASE / "seed_66" / "config.json").read_text())
    )
    source = LOCAL_MASK_REFERENCE / f"seed_{seed}"
    config = replace(
        base,
        seed=seed,
        training_objective="emotion-only",
        completion_path="none",
        joint_pretrain_checkpoint=str(
            (JOINT_ROOT / f"seed_{seed}" / "checkpoint.pt").resolve()
        ),
        checkpoint_selection="test-oracle-per-rate",
        evaluate_test=True,
        # Keep the unused auxiliary module construction identical to L2 so
        # the same seed produces the same OSRAM/classifier initialization.
        disable_unused_aux_modules=False,
    )
    return config, source


def train(seed: int) -> None:
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment

    config, source = configuration(seed)
    output = ROOT / "l1" / f"seed_{seed}"
    pretrained = Path(config.joint_pretrain_checkpoint)
    if not pretrained.exists():
        raise FileNotFoundError(pretrained)
    if output.exists():
        state = output / "PROVENANCE.json"
        if state.exists() and json.loads(state.read_text()).get("status") == "complete":
            print(f"SKIP complete L1 seed={seed}", flush=True)
            return
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    provenance = {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": LABEL,
        "experiment": "MOSI L1 frozen joint-pretrain representation control",
        "seed": seed,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection_protocol": "per-rate-test-oracle",
        "reference": str(source),
        "pretraining_checkpoint": str(pretrained),
        "pretraining_checkpoint_sha256": sha(pretrained),
        "frozen_components": ["observed_set.projectors.*", "teacher.*"],
        "reinjection": False,
        "config": asdict(config),
        "source_sha256": {
            name: sha(REPO / name)
            for name in (
                "gcnet_missing_m3/model.py",
                "gcnet_missing_m3/train_gcnet.py",
                "experiments/osram_joint_frozen_reinjection_20260922/run_l1_control.py",
            )
        },
    }
    write_json(output / "config.json", asdict(config))
    write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(2)
        print(
            f"TRAIN L1 frozen joint representation seed={seed} "
            f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} epochs={config.epochs}",
            flush=True,
        )
        run_experiment(config, *feature_roots(), output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        source_metrics = json.loads((source / "metrics.json").read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise ValueError("unexpected selection protocol")
        if metrics.get("mask_sha256") != source_metrics.get("mask_sha256"):
            raise ValueError("evaluation mask hashes differ from no-JEPA control")
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "PROVENANCE.json", provenance)
        raise
    provenance.update(
        status="complete",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        mask_validation="metric_mask_sha256_equal_control",
    )
    write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE L1 seed={seed}", flush=True)


def launch() -> None:
    (ROOT / "l1").mkdir(parents=True, exist_ok=True)
    children = []
    for index, seed in enumerate(SEEDS):
        gpu = GPUS[index]
        log_path = ROOT / "l1" / f"gpu{gpu}_seed{seed}.log"
        log = log_path.open("a")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            GCNET_DATASET_ROOT=str(DATASET_ROOT),
            PYTHONPATH=str(REPO),
        )
        child = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__)), "--seed", str(seed)],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        children.append((child, log))
    failed = False
    for child, log in children:
        failed = failed or child.wait() != 0
        log.close()
    (ROOT / "l1" / "STATUS").write_text("failed\n" if failed else "complete\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.launch:
        launch()
    else:
        train(args.seed)


if __name__ == "__main__":
    main()
