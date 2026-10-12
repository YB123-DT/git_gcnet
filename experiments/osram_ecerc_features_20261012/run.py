"""Train the selected Nested OSRAM with the feature bundle distributed by ECERC."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


OUTPUT_ROOT = Path(
    os.environ.get(
        "ECERC_OSRAM_OUTPUT",
        "/data2/yb/paper/05_reproduction/runs/OSRAM_ECERC_features_20261012",
    )
)
FEATURE_ROOT = Path(
    os.environ.get(
        "ECERC_GCNET_FEATURE_ROOT",
        "/data2/yb/paper/05_reproduction/data/ecerc_iemocap/features",
    )
)
DATASET_ROOT = Path(
    os.environ.get("GCNET_DATASET_ROOT", "/data2/yb/paper/GCNet_TPAMI/dataset")
)


def tasks(seed: int = 66) -> list[dict[str, int]]:
    return [{"seed": seed, "fold": fold} for fold in range(1, 6)]


def build_config(seed: int, fold: int, config_class=None):
    if config_class is None:
        from gcnet_missing_m3.train_gcnet import TrainConfig
        config_class = TrainConfig

    return config_class(
        dataset="IEMOCAPSix",
        fold=fold,
        seed=seed,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
        evaluation_protocol="official",
        checkpoint_selection="test-oracle-per-rate",
        evaluate_test=True,
        train_rate_mode="cyclic",
        training_objective="emotion-only",
        emotion_loss_mode="sample-mean",
        fusion_type="mean",
        representation_type="slot",
        backbone_type="osram",
        osram_output_dim=1600,
        osram_num_heads=8,
        osram_key_dim=64,
        osram_value_dim=64,
        osram_bidirectional=False,
        osram_forward_slot_reuse=False,
        osram_write_step=0.6,
        osram_readout_fusion="flat",
        osram_meaningful_block="nested_gnn_rooted_evidence",
        disable_unused_aux_modules=True,
        completion_path="none",
        classification_completion=False,
    )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_fold(seed: int, fold: int) -> None:
    manifest_path = FEATURE_ROOT / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepare ECERC features first: {manifest_path}")
    roots = [FEATURE_ROOT / name for name in ("audio", "text", "visual")]
    if not all(root.is_dir() for root in roots):
        raise FileNotFoundError("one or more converted ECERC feature roots are absent")
    label_path = DATASET_ROOT / "IEMOCAP/IEMOCAP_features_raw_6way.pkl"
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    output = OUTPUT_ROOT / f"seed_{seed}" / f"fold_{fold}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    # config.py resolves these paths at import time, so set them before
    # importing TrainConfig/run_experiment through build_config.
    os.environ["GCNET_DATASET_ROOT"] = str(DATASET_ROOT)
    os.environ["GCNET_CACHE_ROOT"] = str(OUTPUT_ROOT / "cache")
    cfg = build_config(seed, fold)
    provenance = {
        "status": "running",
        "started_utc": _now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "feature_manifest": json.loads(manifest_path.read_text()),
        "label_path": str(label_path),
        "protocol_note": "ECERC-distributed features under existing five-session OSRAM protocol",
        "config": asdict(cfg),
    }
    _write_json(output / "PROVENANCE.json", provenance)
    try:
        from gcnet_missing_m3.train_gcnet import run_experiment

        metrics = run_experiment(cfg, *(str(root) for root in roots), output_dir=str(output))
        provenance.update(
            status="complete",
            finished_utc=_now(),
            selection_metric=metrics.get("selection_metric"),
        )
        _write_json(output / "PROVENANCE.json", provenance)
    except BaseException as error:
        provenance.update(status="failed", finished_utc=_now(), error=repr(error))
        _write_json(output / "PROVENANCE.json", provenance)
        raise


def dispatch(seed: int, gpu: int, concurrency: int) -> None:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pending = tasks(seed)
    children: list[tuple[subprocess.Popen, object, dict]] = []
    status = {"status": "running", "seed": seed, "gpu": gpu, "concurrency": concurrency,
              "started_utc": _now(), "tasks": []}
    status_path = OUTPUT_ROOT / f"STATUS_seed{seed}.json"
    _write_json(status_path, status)
    while pending or children:
        while pending and len(children) < concurrency:
            task = pending.pop(0)
            output = OUTPUT_ROOT / f"seed_{seed}" / f"fold_{task['fold']}"
            log = OUTPUT_ROOT / f"seed{seed}_fold{task['fold']}.log"
            if output.exists() or log.exists():
                raise FileExistsError(f"refusing duplicate {task}")
            command = [sys.executable, "-u", "-m", __package__ + ".run", "--run-fold",
                       "--seed", str(seed), "--fold", str(task["fold"])]
            stream = log.open("x")
            child = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                         MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            )
            row = dict(task=task, status="running", pid=child.pid, command=command,
                       log=str(log), output=str(output))
            status["tasks"].append(row)
            children.append((child, stream, row))
            _write_json(status_path, status)
        time.sleep(5)
        for item in list(children):
            child, stream, row = item
            code = child.poll()
            if code is None:
                continue
            stream.close()
            row.update(status="complete" if code == 0 else "failed", exit_code=code)
            children.remove(item)
            _write_json(status_path, status)
            if code != 0:
                status.update(status="failed", finished_utc=_now())
                _write_json(status_path, status)
                raise RuntimeError(f"failed {row['task']}; inspect {row['log']}")
    status.update(status="complete", finished_utc=_now())
    _write_json(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-fold", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.run_fold:
        if args.fold is None:
            parser.error("--fold is required with --run-fold")
        run_fold(args.seed, args.fold)
    elif args.dispatch:
        dispatch(args.seed, args.gpu, args.concurrency)
    else:
        parser.error("choose --run-fold or --dispatch")


if __name__ == "__main__":
    main()
