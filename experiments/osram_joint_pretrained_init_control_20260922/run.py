"""MOSI control: initialize projectors from joint JEPA, then fine-tune them."""

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

from experiments.osram_mosi_hparam_sweep_20260918.run import sha, write_json  # noqa: E402

ROOT = Path("/data2/yb/remote_experiments/osram_joint_pretrained_init_control_20260922")
JOINT_ROOT = REPO / "experiments/m3_pretrain_three_datasets_20260921"
LOCAL_BASE = REPO / "experiments/local_cfg84_nojepa_20260919"
LOCAL_MASK_REFERENCE = REPO / "experiments/osram_no_aux_cfg84_cyclic_no0_20260920/results"
DATASET_ROOT = Path("/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset")
SEEDS = (66, 67, 68)
GPUS = (1, 4, 5)
LABEL = "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT"


def feature_roots() -> tuple[str, str, str]:
    root = DATASET_ROOT / "CMUMOSI" / "features"
    return tuple(str(root / name) for name in (
        "wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT"
    ))


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
        joint_pretrain_freeze=False,
        checkpoint_selection="test-oracle-per-rate",
        evaluate_test=True,
        disable_unused_aux_modules=False,
    )
    return config, source


def train(seed: int) -> None:
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment

    config, source = configuration(seed)
    pretrained = Path(config.joint_pretrain_checkpoint)
    if not pretrained.exists():
        raise FileNotFoundError(pretrained)
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
        "experiment": "MOSI joint-pretrained projector initialization control",
        "seed": seed,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection_protocol": "per-rate-test-oracle",
        "reference": str(source),
        "pretraining_checkpoint": str(pretrained),
        "pretraining_checkpoint_sha256": sha(pretrained),
        "frozen_components": [],
        "trainable_components": [
            "observed_set.projectors.* (joint-pretrained initialization)",
            "ObservedSetEncoder fusion/availability embeddings",
            "OSRAM",
            "emotion classifier",
        ],
        "config": asdict(config),
        "reference_sha256": {
            name: sha(source / name)
            for name in ("config.json", "metrics.json")
        },
    }
    write_json(output / "config.json", asdict(config))
    write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(2)
        print(
            f"TRAIN joint-pretrained-init seed={seed} "
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
    print(f"COMPLETE joint-pretrained-init seed={seed}", flush=True)


def launch() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "gpus": list(GPUS),
        "selection_protocol": "per-rate-test-oracle",
        "label": LABEL,
        "tasks": [],
    }
    write_json(ROOT / "QUEUE.json", queue)
    children = []
    for gpu, seed in zip(GPUS, SEEDS):
        log_path = ROOT / f"gpu{gpu}_seed{seed}.log"
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
        queue["tasks"].append({"gpu": gpu, "pid": child.pid, "seed": seed,
                                "status": "running", "log": str(log_path)})
        children.append((child, log))
    queue["status"] = "running"
    write_json(ROOT / "QUEUE.json", queue)
    failed = False
    for child, log in children:
        failed = failed or child.wait() != 0
        log.close()
    queue["status"] = "failed" if failed else "complete"
    write_json(ROOT / "QUEUE.json", queue)


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
