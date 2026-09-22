"""MOSI dual-projector frozen completion on the cfg84 emotion backbone."""

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

ROOT = Path("/data2/yb/remote_experiments/osram_joint_dual_projector_completion_20260922")
JOINT_ROOT = REPO / "experiments/m3_pretrain_three_datasets_20260921"
LOCAL_BASE = REPO / "experiments/local_cfg84_nojepa_20260919"
MASK_REFERENCE = REPO / "experiments/osram_no_aux_cfg84_cyclic_no0_20260920/results"
DATASET_ROOT = Path("/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset")
SEEDS = (66, 67, 68)
GPUS = (4, 5, 7)
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
    config = replace(
        base,
        seed=seed,
        training_objective="emotion-only",
        completion_path="pre_osram_joint_dual_projector",
        joint_pretrain_checkpoint=str(
            (JOINT_ROOT / f"seed_{seed}" / "checkpoint.pt").resolve()
        ),
        joint_pretrain_freeze=True,
        checkpoint_selection="test-oracle-per-rate",
        evaluate_test=True,
        disable_unused_aux_modules=True,
    )
    return config, MASK_REFERENCE / f"seed_{seed}"


def train(seed: int) -> None:
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment

    config, reference = configuration(seed)
    checkpoint = Path(config.joint_pretrain_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
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
        "seed": seed,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection_protocol": "per-rate-test-oracle",
        "pretraining_checkpoint": str(checkpoint),
        "pretraining_checkpoint_sha256": sha(checkpoint),
        "online_branch": "fresh cfg84 projectors; trainable with OSRAM/classifier",
        "completion_branch": "joint-pretrained projectors + SourceOnlyM3Predictor; frozen",
        "leakage_guard": "completion branch receives incomplete features and projects observed slots only",
        "persistent_write": "online real-observed latents only",
        "config": asdict(config),
    }
    write_json(output / "config.json", asdict(config))
    write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(2)
        run_experiment(config, *feature_roots(), output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        control = json.loads((reference / "metrics.json").read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise ValueError("unexpected selection protocol")
        if metrics.get("mask_sha256") != control.get("mask_sha256"):
            raise ValueError("evaluation masks differ from cfg84 control")
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "PROVENANCE.json", provenance)
        raise
    provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE dual-projector completion seed={seed}", flush=True)


def launch() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    children = []
    for gpu, seed in zip(GPUS, SEEDS):
        log_path = ROOT / f"gpu{gpu}_seed{seed}.log"
        log = log_path.open("a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", GCNET_DATASET_ROOT=str(DATASET_ROOT),
                   PYTHONPATH=str(REPO))
        child = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__)), "--seed", str(seed)],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        children.append((child, log))
    failed = False
    for child, log in children:
        failed = child.wait() != 0 or failed
        log.close()
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    launch() if args.launch else train(args.seed)


if __name__ == "__main__":
    main()
