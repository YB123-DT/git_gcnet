"""MOSI frozen reinjection from three-dataset utterance-level pretraining."""

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

from experiments.osram_mosi_hparam_sweep_20260918.run import (  # noqa: E402
    canonical_mask_hashes,
    sha,
    write_json,
)


ROOT = Path("/data2/yb/remote_experiments/osram_joint_frozen_reinjection_20260922")
JOINT_ROOT = REPO / "experiments/m3_pretrain_three_datasets_20260921"
LOCAL_BASE = REPO / "experiments/local_cfg84_nojepa_20260919"
LOCAL_MASK_REFERENCE = REPO / "experiments/osram_no_aux_cfg84_cyclic_no0_20260920/results"
DATASET_ROOT = Path("/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset")
SEEDS = (66, 67, 68)
GPUS = (0, 2, 3)
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
    # cfg84 is the 1600/64 causal OSRAM family used for the current no-JEPA
    # control.  Each downstream seed gets its own fixed joint-pretrain bank.
    config = replace(
        base,
        seed=seed,
        training_objective="emotion-only",
        completion_path="pre_osram_joint_frozen",
        joint_pretrain_checkpoint=str(
            (JOINT_ROOT / f"seed_{seed}" / "checkpoint.pt").resolve()
        ),
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
        "experiment": "MOSI L2 frozen joint-pretrain completion reinjection",
        "seed": seed,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection_protocol": "per-rate-test-oracle",
        "reference": str(source),
        "pretraining_checkpoint": str(pretrained),
        "pretraining_checkpoint_sha256": sha(pretrained),
        "frozen_components": [
            "observed_set.projectors.*",
            "source_only_predictor.* (pretrained MMoE)",
            "teacher.*",
        ],
        "trainable_components": [
            "ObservedSetEncoder fusion/availability embeddings",
            "CompletedReadFusion",
            "OSRAM",
            "emotion classifier",
        ],
        "config": asdict(config),
        "reference_sha256": {
            name: sha(source / name)
            for name in ("config.json", "metrics.json")
        },
        "source_sha256": {
            name: sha(REPO / name)
            for name in (
                "gcnet_missing_m3/model.py",
                "gcnet_missing_m3/train_gcnet.py",
                "gcnet_missing_m3/b2.py",
                "experiments/osram_joint_frozen_reinjection_20260922/run.py",
            )
        },
    }
    write_json(output / "config.json", asdict(config))
    write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(2)
        print(
            f"TRAIN frozen joint reinjection seed={seed} "
            f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} epochs={config.epochs}",
            flush=True,
        )
        run_experiment(config, *feature_roots(), output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise ValueError("unexpected selection protocol")
        source_metrics = json.loads((source / "metrics.json").read_text())
        if metrics.get("mask_sha256") != source_metrics.get("mask_sha256"):
            raise ValueError("evaluation mask hashes differ from no-JEPA control")
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
    print(f"COMPLETE frozen joint reinjection seed={seed}", flush=True)


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
    for index, seed in enumerate(SEEDS):
        gpu = GPUS[index]
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
        row = {"gpu": gpu, "pid": child.pid, "seed": seed,
               "status": "running", "log": str(log_path)}
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
