"""One persistent job: historical MELD baseline retrain, frozen audit, analysis.

No repository modules are imported until the train-only child selects the
historical code tree. Reproduction is a NEW run, never restoration of deleted
weights and never a claim of bit-identical historical performance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HISTORICAL_COMMIT = "b4a313841f098487a6482d2712bdac94aa8e319e"
EXPERIMENT = Path("experiments/osram_meld_write_retention_20261014")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def normalized(value):
    return json.loads(json.dumps(value))


def validate_config(config, source):
    expected = {
        "dataset": "MELD", "seed": 66, "epochs": 50,
        "missing_mask_protocol": "gcnet-random-mask-v1",
        "checkpoint_selection": "test-oracle-per-rate",
        "backbone_type": "osram", "osram_readout_fusion": "flat",
        "training_objective": "emotion-only", "osram_bidirectional": False,
        "osram_disable_local_skip": False,
    }
    for key, value in expected.items():
        if getattr(config, key, False if key == "osram_disable_local_skip" else None) != value:
            raise ValueError(f"Historical source violates {key}={value!r}")
    effective = normalized(asdict(config))
    for key, value in source.items():
        if key not in effective or effective[key] != normalized(value):
            raise ValueError(f"Source/effective config mismatch: {key}")
    return effective


def train_only(args):
    baseline = args.output_root / "baseline"
    if baseline.exists():
        raise FileExistsError(baseline)
    os.environ.update(
        GCNET_DATASET_ROOT=str(args.dataset_root), GCNET_MELD_PICKLE=str(args.meld_pickle),
        GCNET_CACHE_ROOT=str(args.output_root / "cache"), CUDA_VISIBLE_DEVICES=args.gpu_uuid,
    )
    sys.path.insert(0, str(args.training_code))
    import gcnet_missing_m3.train_gcnet as training
    if not Path(training.__file__).resolve().is_relative_to(args.training_code.resolve()):
        raise RuntimeError("Training import escaped the historical snapshot")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = json.loads(args.source_config.read_text())
    config = training.TrainConfig(**source)
    effective = validate_config(config, source)
    baseline.mkdir(parents=True)
    provenance = {
        "status": "running", "started_utc": now(), "pid": os.getpid(),
        "host": socket.gethostname(), "historical_code_commit": HISTORICAL_COMMIT,
        "training_module": training.__file__, "config": effective,
        "source_config_sha256": sha(args.source_config),
        "meld_pickle_sha256": sha(args.meld_pickle),
        "gpu_uuid": args.gpu_uuid, "torch_threads": 1, "torch_interop_threads": 1,
        "protocol": "INTERNAL DIAGNOSTIC ONLY; new baseline reproduction; per-rate Test-oracle",
        "checkpoint_policy": "retain all eight per-rate best checkpoints; no cleanup",
    }
    write(baseline / "PROVENANCE.json", provenance)
    try:
        training.run_experiment(config, "MELD-pickle", "MELD-pickle", "MELD-pickle", baseline)
        actual = json.loads((baseline / "config.json").read_text())
        for key, value in source.items():
            if actual.get(key) != normalized(value):
                raise ValueError(f"Saved config differs from source: {key}")
        for index in range(8):
            for name in (f"best_miss_0p{index}.pt", f"predictions_miss_0p{index}.npz"):
                if not (baseline / name).is_file():
                    raise FileNotFoundError(baseline / name)
        provenance.update(status="complete", finished_utc=now())
    except BaseException as error:
        provenance.update(status="failed", finished_utc=now(), error=repr(error))
        raise
    finally:
        write(baseline / "PROVENANCE.json", provenance)


def compare_results(args):
    """Keep historical selection scores separate from final saved-mask scores."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    current = args.output_root / "baseline"
    report = {"historical_source": str(args.historical_source), "new_source": str(current),
              "note": "New reproduction, not original checkpoint restoration. No exactness assumed.",
              "selected_metrics": {}, "saved_mask_metrics": {}}
    old_metrics_path = args.historical_source / "metrics.json"
    old_metrics = json.loads(old_metrics_path.read_text()) if old_metrics_path.exists() else {}
    new_metrics = json.loads((current / "metrics.json").read_text())
    for field in ("selected_score_by_rate", "selected_accuracy_by_rate", "selected_weighted_f1_by_rate", "selected_epoch_by_rate"):
        report["selected_metrics"][field] = {}
        for rate, new in new_metrics.get(field, {}).items():
            old = old_metrics.get(field, {}).get(rate)
            report["selected_metrics"][field][rate] = {
                "new": new, "historical": old, "delta": new - old if old is not None else None,
            }
    for index in range(8):
        name = f"predictions_miss_0p{index}.npz"
        old_path, new_path = args.historical_source / name, current / name
        values = {}
        for tag, path in (("new", new_path), ("historical", old_path)):
            if not path.exists():
                values[tag] = None
                continue
            with np.load(path) as data:
                y, pred = data["labels"], data["predictions"]
                values[tag] = {
                    "ACC": float(accuracy_score(y, pred)),
                    "WF1": float(f1_score(y, pred, average="weighted", zero_division=0)),
                    "UA": float(recall_score(y, pred, labels=list(range(7)), average="macro", zero_division=0)),
                    "N": int(len(y)), "sha256": sha(path),
                }
        values["delta"] = {k: values["new"][k] - values["historical"][k]
                           for k in ("ACC", "WF1", "UA")} if values["historical"] else None
        if old_path.exists():
            with np.load(old_path) as old, np.load(new_path) as new:
                values["same_labels"] = bool(np.array_equal(old["labels"], new["labels"]))
                values["same_masks"] = bool(np.array_equal(old["availability"], new["availability"]))
                values["same_predictions"] = bool(np.array_equal(old["predictions"], new["predictions"]))
        report["saved_mask_metrics"][str(index / 10)] = values
    write(args.output_root / "REPRODUCTION_COMPARISON.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-code", "audit-code", "source-config", "historical-source",
                 "dataset-root", "meld-pickle", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--python", default="/data2/yb/conda_envs/cmerc-repro/bin/python")
    parser.add_argument("--train-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("training_code", "audit_code", "source_config", "historical_source", "dataset_root", "meld_pickle", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.train_only:
        train_only(args)
        return
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    for path in (args.training_code / "gcnet_missing_m3/train_gcnet.py", args.source_config,
                 args.meld_pickle, args.audit_code / EXPERIMENT / "run.py",
                 args.audit_code / EXPERIMENT / "analyze.py", Path(args.python)):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.historical_source.is_dir() or not args.dataset_root.is_dir():
        raise FileNotFoundError("Historical source and dataset root must exist")
    # The audit records git provenance, so its snapshot must have working git metadata.
    audit_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.audit_code, text=True).strip()
    # Check core historical source bytes against the retained git object, not a path label.
    for relative in ("gcnet_missing_m3/train_gcnet.py", "gcnet_missing_m3/model.py", "gcnet_missing_m3/osram.py"):
        original = subprocess.check_output(["git", "show", f"{HISTORICAL_COMMIT}:{relative}"], cwd=args.audit_code)
        if hashlib.sha256(original).hexdigest() != sha(args.training_code / relative):
            raise ValueError(f"Historical snapshot mismatch: {relative}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid or not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("CUDA_VISIBLE_DEVICES must equal --gpu-uuid")
    listing = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    matched = [line.split(",") for line in listing.splitlines() if line.split(",")[-1].strip() == args.gpu_uuid]
    if len(matched) != 1 or matched[0][0].strip() == "4":
        raise ValueError("Unknown GPU UUID or forbidden host GPU4")
    args.output_root.mkdir(parents=True)
    state = {"status": "running", "pid": os.getpid(), "host": socket.gethostname(),
             "started_utc": now(), "gpu_uuid": args.gpu_uuid, "host_gpu_index": int(matched[0][0]),
             "historical_commit": HISTORICAL_COMMIT, "audit_commit": audit_commit,
             "source_config_sha256": sha(args.source_config), "stages": {}}
    target = args.output_root / "PIPELINE.json"
    write(target, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = time.monotonic()

    def stage(name, command, code):
        entry = {"status": "running", "command": command, "started_utc": now(), "cwd": str(code)}
        state["stages"][name] = entry
        write(target, state)
        tick = time.monotonic()
        with (args.output_root / f"{name}.log").open("x") as log:
            child_env = dict(env, PYTHONPATH=str(code))
            child = subprocess.Popen(command, cwd=code, env=child_env, stdout=log, stderr=subprocess.STDOUT)
            entry["pid"] = child.pid
            write(target, state)
            code_result = child.wait()
        entry.update(returncode=code_result, seconds=time.monotonic() - tick,
                     finished_utc=now(), status="complete" if code_result == 0 else "failed")
        write(target, state)
        if code_result:
            raise RuntimeError(f"{name} failed with returncode {code_result}; see {name}.log")

    try:
        command = [args.python, str(Path(__file__).resolve()), "--train-only"]
        for name in ("training_code", "audit_code", "source_config", "historical_source", "dataset_root", "meld_pickle", "output_root", "gpu_uuid", "python"):
            command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
        stage("training", command, args.training_code)
        compare_results(args)
        stage("audit", [args.python, str(args.audit_code / EXPERIMENT / "run.py"),
                        "--source", str(args.output_root / "baseline"), "--dataset-root", str(args.dataset_root),
                        "--meld-pickle", str(args.meld_pickle), "--output", str(args.output_root / "audit"),
                        "--gpu-uuid", args.gpu_uuid], args.audit_code)
        stage("analysis", [args.python, str(args.audit_code / EXPERIMENT / "analyze.py"),
                           "--input", str(args.output_root / "audit")], args.audit_code)
        summary = json.loads((args.output_root / "audit/SUMMARY.json").read_text())
        if summary["status"] != "complete" or not (args.output_root / "audit/AGGREGATE.csv").is_file():
            raise RuntimeError("Stage processes returned success without complete artifacts")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state.update(finished_utc=now(), seconds=time.monotonic() - started)
        write(target, state)


if __name__ == "__main__":
    main()
