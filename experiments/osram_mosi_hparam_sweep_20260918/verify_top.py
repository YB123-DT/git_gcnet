"""Automatically expand the best seed-66 screen configurations to five seeds."""

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

from experiments.osram_mosi_hparam_sweep_20260918.run import (
    FEATURES, SPECS, SPEC_BY_ID, SOURCE_ROOT, sha, write_json,
)

ROOT_SCREEN = Path("/data2/yb/remote_experiments/osram_mosi_hparam_sweep_20260918_parallel")
ROOT_VERIFY = Path("/data2/yb/remote_experiments/osram_mosi_hparam_verify_20260918")
SEEDS = (66, 67, 68, 69, 70)
GPUS = (1, 2, 3)


def configuration(spec_id: str, seed: int):
    from gcnet_missing_m3.train_gcnet import TrainConfig

    source = SOURCE_ROOT / f"seed_{seed}"
    old = json.loads((source / "config.json").read_text())
    cfg = replace(
        TrainConfig(**old),
        seed=seed,
        **SPEC_BY_ID[spec_id]["overrides"],
        training_objective="emotion-only",
        backbone_type="osram",
        fusion_type="mean",
        osram_bidirectional=False,
        osram_forward_slot_reuse=False,
        osram_write_step=.6,
        osram_readout_fusion="flat",
        train_rate_mode="cyclic",
        checkpoint_selection="test-oracle-per-rate",
        evaluate_test=True,
        completion_path="none",
        classification_completion=False,
        initial_backbone_checkpoint=None,
        teacher_mode="ema",
        teacher_checkpoint=None,
        target_space="all-modalities",
        text_subspace_checkpoint=None,
        text_core=False,
    )
    return cfg, source


def train(spec_id: str, seed: int) -> None:
    import torch
    from gcnet_missing_m3.train_gcnet import run_experiment

    cfg, source = configuration(spec_id, seed)
    output = ROOT_VERIFY / spec_id / f"seed_{seed}"
    if output.exists():
        state_path = output / "PROVENANCE.json"
        if state_path.exists() and json.loads(state_path.read_text()).get("status") == "complete":
            print(f"SKIP complete {spec_id} seed={seed}", flush=True)
            return
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    provenance = {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "spec_id": spec_id,
        "seed": seed,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection_protocol": "per-rate-test-oracle",
        "reference": str(source),
        "config": asdict(cfg),
        "reference_sha256": {name: sha(source / name) for name in ("config.json", "history.json", "metrics.json")},
    }
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "PROVENANCE.json", provenance)
    try:
        torch.set_num_threads(2)
        roots = [str(FEATURES / name) for name in ("wav2vec-large-c-UTT", "deberta-large-4-UTT", "manet_UTT")]
        print(f"TRAIN verify {spec_id} seed={seed} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        run_experiment(cfg, *roots, output_dir=str(output))
        metrics = json.loads((output / "metrics.json").read_text())
        reference_metrics = json.loads((source / "metrics.json").read_text())
        if metrics.get("selection_protocol") != "per-rate-test-oracle":
            raise ValueError("unexpected selection protocol")
        if metrics.get("mask_sha256") != reference_metrics.get("mask_sha256"):
            raise ValueError("test mask hashes differ from same-seed baseline")
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "PROVENANCE.json", provenance)
        raise
    provenance.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "PROVENANCE.json", provenance)
    print(f"COMPLETE verify {spec_id} seed={seed}", flush=True)


def launch(top_n: int = 5) -> None:
    summary = json.loads((ROOT_SCREEN / "summary" / "summary.json").read_text())
    rows = summary.get("rows", [])
    if len(rows) < top_n:
        raise RuntimeError(f"only {len(rows)} completed screen configurations; need {top_n}")
    top_ids = [row["spec_id"] for row in rows[:top_n]]
    tasks = [(spec_id, seed) for spec_id in top_ids for seed in SEEDS]
    ROOT_VERIFY.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "top_n": top_n,
        "top_spec_ids": top_ids,
        "seeds": list(SEEDS),
        "selection_protocol": "per-rate-test-oracle",
        "label": "INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT",
        "tasks": [],
    }
    write_json(ROOT_VERIFY / "QUEUE.json", queue)
    buckets = [[] for _ in GPUS]
    for index, task in enumerate(tasks):
        buckets[index % len(buckets)].append(task)
    failed = False
    wave_size = 10
    for start in range(0, max(len(bucket) for bucket in buckets), wave_size):
        children = []
        for gpu, bucket in zip(GPUS, buckets):
            for spec_id, seed in bucket[start:start + wave_size]:
                log_path = ROOT_VERIFY / f"gpu{gpu}_{spec_id}_seed{seed}.log"
                log = log_path.open("a")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                           MKL_NUM_THREADS="2", PYTHONPATH=str(REPO))
                child = subprocess.Popen(
                    [sys.executable, "-u", __file__, "--task", spec_id, "--seed", str(seed)],
                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                row = {"gpu": gpu, "pid": child.pid, "spec_id": spec_id,
                       "seed": seed, "status": "running", "log": str(log_path)}
                queue["tasks"].append(row)
                children.append((child, log, row))
        queue["status"] = "running"
        write_json(ROOT_VERIFY / "QUEUE.json", queue)
        for child, log, row in children:
            row["exit_code"] = child.wait()
            row["status"] = "complete" if row["exit_code"] == 0 else "failed"
            failed = failed or row["exit_code"] != 0
            log.close()
            write_json(ROOT_VERIFY / "QUEUE.json", queue)
    queue["status"] = "failed" if failed else "complete"
    queue["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT_VERIFY / "QUEUE.json", queue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--task", choices=tuple(SPEC_BY_ID))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.launch:
        launch(args.top)
    elif args.task is not None and args.seed is not None:
        train(args.task, args.seed)
    else:
        parser.error("use --launch or --task SPEC --seed SEED")


if __name__ == "__main__":
    main()
