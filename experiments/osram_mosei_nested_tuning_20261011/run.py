"""Dispatch exactly six CMU-MOSEI Nested capacity runs on biggpu GPU2/3."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time

from experiments.osram_nested_cross_dataset_20261006.run import (
    configuration as reference_configuration,
    now,
    read,
    reference_dir,
    sha,
    write,
)


LABEL = "INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle; NOT A FORMAL PAPER RESULT"
METHODS = (
    "mosei_nested_dim32",
    "mosei_nested_depth2",
    "mosei_nested_dim32_depth2",
    "mosei_nested_mlp128",
    "mosei_nested_groups4",
    "mosei_nested_dim32_depth2_mlp64",
)
GPU_ASSIGNMENT = dict(zip(METHODS, (2, 2, 2, 3, 3, 3)))
TASK = {"dataset": "CMUMOSEI", "seed": 66, "fold": 1}
RESERVED_MIB_PER_RUN = 8000
MIN_DISK_GIB = 30


def configuration_delta(method: str) -> dict[str, str]:
    if method not in METHODS:
        raise ValueError("method is outside the locked MOSEI screen")
    return {"osram_meaningful_block": method}


def validate_gpu_index(index: int) -> None:
    if index not in (2, 3):
        raise ValueError("only biggpu host GPU2/GPU3 are authorized")


def gpu_inventory() -> dict[int, tuple[str, int]]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result: dict[int, tuple[str, int]] = {}
    for line in rows.splitlines():
        index, uuid, free = (part.strip() for part in line.split(","))
        result[int(index)] = (uuid, int(free))
    for index in (2, 3):
        validate_gpu_index(index)
        if index not in result:
            raise ValueError(f"GPU{index} is missing")
    return result


def candidate_config(reference: dict[str, object], method: str) -> dict[str, object]:
    checked = reference_configuration(reference, TASK)
    checked.update(configuration_delta(method))
    changed = {key for key in checked if checked.get(key) != reference.get(key)}
    if changed != {"osram_meaningful_block"}:
        raise ValueError(f"unexpected configuration changes: {sorted(changed)}")
    return checked


def _verify_inputs(inputs_path: Path) -> tuple[dict[str, object], Path]:
    inputs = read(inputs_path)
    if TASK not in inputs.get("tasks", []):
        raise ValueError("seed66 CMU-MOSEI task is absent from the pinned manifest")
    reference = reference_dir(TASK)
    hashes = inputs.get("references", {}).get(str(reference))
    if not hashes:
        raise ValueError("same-seed Flat reference is absent from the pinned manifest")
    for name, digest in hashes.items():
        if sha(reference / name) != digest:
            raise ValueError(f"Flat reference changed: {name}")
    data = inputs["data"]["CMUMOSEI"]
    for name, digest in data["files"].items():
        if sha(name) != digest:
            raise ValueError(f"dataset changed: {name}")
    return data, reference


def _artifact_names() -> list[str]:
    names = ["config.json", "metrics.json", "history.json", "last_training.pt"]
    names += [f"{kind}_miss_0p{i}.{extension}" for i in range(8)
              for kind, extension in (("best", "pt"), ("predictions", "npz"))]
    return names


def train(root: Path, inputs_path: Path, method: str, gpu: int, gpu_uuid: str) -> None:
    validate_gpu_index(gpu)
    if GPU_ASSIGNMENT.get(method) != gpu:
        raise ValueError("method/GPU assignment differs from the locked matrix")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain the exact host GPU UUID")

    source = Path(__file__).resolve().parents[2]
    from experiments.osram_core20_20261005.run import verify_snapshot

    snapshot = verify_snapshot(source)
    data, reference = _verify_inputs(inputs_path)
    reference_config = read(reference / "config.json")
    config_dict = candidate_config(reference_config, method)

    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    from gcnet_missing_m3.training_resume import TrainingState
    import torch

    torch.set_num_threads(1)
    config = TrainConfig(**config_dict)
    output = root / "runs" / method / "seed_66"
    output.mkdir(parents=True, exist_ok=False)
    identity = {
        "snapshot": sha(source / "SNAPSHOT.json"),
        "inputs": sha(inputs_path),
        "config": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
    }
    record = {
        "label": LABEL,
        "status": "running",
        "method": method,
        "task": TASK,
        "server": socket.gethostname(),
        "gpu_index": gpu,
        "gpu_uuid": gpu_uuid,
        "pid": os.getpid(),
        "started_utc": now(),
        "source": str(source),
        "code_commit": snapshot["code_commit"],
        "snapshot_sha256": identity["snapshot"],
        "inputs_sha256": identity["inputs"],
        "reference": str(reference),
        "configuration_delta": configuration_delta(method),
        "config": asdict(config),
    }
    write(output / "PROVENANCE.json", record)
    try:
        os.environ["GCNET_DATASET_ROOT"] = data["dataset_root"]
        os.environ["GCNET_CACHE_ROOT"] = str(root / "cache_CMUMOSEI")
        torch.cuda.reset_peak_memory_stats()
        metrics = run_experiment(
            config,
            *data["feature_roots"],
            str(output),
            training_state=TrainingState(output, identity=identity),
        )
        if metrics.get("selection_metric") != "weighted_f1":
            raise ValueError("MOSEI checkpoint selection metric changed")
        if metrics.get("mask_sha256") != read(reference / "metrics.json").get("mask_sha256"):
            raise ValueError("ordered evaluation masks differ from same-seed Flat")
        if len(read(output / "history.json")) != 100:
            raise ValueError("training did not complete exactly 100 epochs")
        artifacts = _artifact_names()
        missing = [name for name in artifacts if not (output / name).is_file()]
        if missing:
            raise ValueError(f"missing artifacts: {missing}")
        record.update(
            status="complete",
            exit_code=0,
            outputs_verified=True,
            finished_utc=now(),
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            artifact_sha256={name: sha(output / name) for name in artifacts},
        )
        verify_snapshot(source)
        write(output / "PROVENANCE.json", record)
    except BaseException as error:
        record.update(status="failed", exit_code=1, finished_utc=now(), error=repr(error))
        write(output / "PROVENANCE.json", record)
        raise


def _scores(output: Path) -> dict[str, object]:
    metrics = read(output / "metrics.json")
    wf1 = metrics["selected_weighted_f1_by_rate"]
    acc = metrics["selected_accuracy_by_rate"]
    rates = [str(index / 10) for index in range(8)]
    high = [str(index / 10) for index in (5, 6, 7)]
    return {
        "per_rate_weighted_f1": wf1,
        "per_rate_accuracy": acc,
        "mean8_weighted_f1": 100 * statistics.mean(wf1[rate] for rate in rates),
        "mean8_accuracy": 100 * statistics.mean(acc[rate] for rate in rates),
        "high_weighted_f1": 100 * statistics.mean(wf1[rate] for rate in high),
        "high_accuracy": 100 * statistics.mean(acc[rate] for rate in high),
        "selected_epoch_by_rate": metrics["selected_epoch_by_rate"],
        "parameter_count": metrics.get("parameter_count"),
    }


def _publish_summary(root: Path, rows: list[dict[str, object]]) -> None:
    results = []
    for row in rows:
        item = {key: row.get(key) for key in (
            "method", "status", "gpu", "gpu_uuid", "pid", "output", "error"
        ) if row.get(key) is not None}
        if row.get("status") == "complete":
            item.update(_scores(Path(str(row["output"]))))
        results.append(item)
    write(root / "SUMMARY.json", {"label": LABEL, "runs": results})


def dispatch(root: Path, inputs_path: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "DISPATCH.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "DISPATCH.json"
    if state_path.exists():
        raise FileExistsError("reconcile the existing dispatcher before restarting")
    if shutil.disk_usage(root).free < MIN_DISK_GIB * 2**30:
        raise ValueError("insufficient disk for six isolated runs")
    inventory = gpu_inventory()
    for gpu in (2, 3):
        required = 3 * RESERVED_MIB_PER_RUN + 2048
        if inventory[gpu][1] < required:
            raise ValueError(f"GPU{gpu} has {inventory[gpu][1]} MiB free; {required} required")

    source = Path(__file__).resolve().parents[2]
    from experiments.osram_core20_20261005.run import verify_snapshot

    snapshot = verify_snapshot(source)
    _verify_inputs(inputs_path)
    rows = [
        {
            "method": method,
            "status": "pending",
            "gpu": GPU_ASSIGNMENT[method],
            "gpu_uuid": inventory[GPU_ASSIGNMENT[method]][0],
            "output": str(root / "runs" / method / "seed_66"),
        }
        for method in METHODS
    ]
    state = {
        "label": LABEL,
        "status": "running",
        "pid": os.getpid(),
        "server": socket.gethostname(),
        "started_utc": now(),
        "source": str(source),
        "code_commit": snapshot["code_commit"],
        "snapshot_sha256": sha(source / "SNAPSHOT.json"),
        "inputs": str(inputs_path),
        "inputs_sha256": sha(inputs_path),
        "gpu_inventory": {str(key): {"uuid": value[0], "free_mib": value[1]}
                          for key, value in inventory.items()},
        "runs": rows,
    }
    children: dict[str, tuple[subprocess.Popen[bytes], object]] = {}
    (root / "logs").mkdir(exist_ok=True)
    write(state_path, state)
    try:
        for row in rows:
            method = str(row["method"])
            output = Path(str(row["output"]))
            log = root / "logs" / f"{method}.log"
            if output.exists() or log.exists():
                raise FileExistsError(f"duplicate output/log for {method}")
            command = [
                sys.executable,
                "-u",
                "-m",
                "experiments.osram_mosei_nested_tuning_20261011.run",
                "--train",
                "--root",
                str(root),
                "--inputs",
                str(inputs_path),
                "--method",
                method,
                "--gpu",
                str(row["gpu"]),
                "--gpu-uuid",
                str(row["gpu_uuid"]),
            ]
            stream = log.open("xb")
            child = subprocess.Popen(
                command,
                cwd=source,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(row["gpu_uuid"]),
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                ),
            )
            row.update(status="running", pid=child.pid, command=command,
                       log=str(log), started_utc=now())
            children[method] = child, stream
            write(state_path, state)
        while children:
            for method, (child, stream) in list(children.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                row = next(item for item in rows if item["method"] == method)
                provenance = Path(str(row["output"])) / "PROVENANCE.json"
                verified = read(provenance).get("outputs_verified", False) if provenance.exists() else False
                row.update(
                    status="complete" if code == 0 and verified else "failed",
                    exit_code=code,
                    finished_utc=now(),
                )
                if row["status"] == "failed":
                    row["error"] = read(provenance).get("error", "inspect log") if provenance.exists() else "missing provenance"
                del children[method]
            _publish_summary(root, rows)
            write(state_path, state)
            if children:
                time.sleep(15)
        failed = [row["method"] for row in rows if row["status"] != "complete"]
        state.update(status="failed" if failed else "complete", finished_utc=now(), failed=failed)
        _publish_summary(root, rows)
        write(state_path, state)
    except BaseException as error:
        state.update(status="failed_needs_reconciliation", error=repr(error), finished_utc=now())
        write(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.train:
        if args.method is None or args.gpu is None or args.gpu_uuid is None:
            parser.error("--train requires --method, --gpu and --gpu-uuid")
        train(args.root.resolve(), args.inputs.resolve(), args.method, args.gpu, args.gpu_uuid)
    else:
        dispatch(args.root.resolve(), args.inputs.resolve())


if __name__ == "__main__":
    main()
