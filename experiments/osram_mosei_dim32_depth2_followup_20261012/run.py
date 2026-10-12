"""Dispatch the locked MOSEI dim32+depth2 follow-up in parallel."""
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


LABEL = "INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle"
SPECS = {
    "seed67_lr1e3_e60": dict(seed=67, learning_rate=1e-3, epochs=60, gpu=5,
                              method="mosei_nested_dim32_depth2"),
    "seed68_lr1e3_e60": dict(seed=68, learning_rate=1e-3, epochs=60, gpu=5,
                              method="mosei_nested_dim32_depth2"),
    "seed66_lr1e4_e60": dict(seed=66, learning_rate=1e-4, epochs=60, gpu=6,
                              method="mosei_nested_dim32_depth2"),
    "seed68_mlp128_lr1e3_e60": dict(seed=68, learning_rate=1e-3, epochs=60, gpu=5,
                                     method="mosei_nested_mlp128"),
    "seed68_depth2_lr1e3_e60": dict(seed=68, learning_rate=1e-3, epochs=60, gpu=6,
                                     method="mosei_nested_depth2"),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(seed):
    return Path(f"/data2/yb/remote_experiments/osram_mosei_cfg84_nojepa_20260919/seed_{seed}")


def data_info():
    row = read("/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006/INPUTS.json")["data"]["CMUMOSEI"]
    return row["dataset_root"], row["feature_roots"]


def candidate_config(base, name):
    spec = SPECS[name]
    required = dict(dataset="CMUMOSEI", seed=spec["seed"], fold=1, epochs=100,
                    batch_size=32, learning_rate=1e-3, training_objective="emotion-only",
                    backbone_type="osram", train_rate_mode="cyclic",
                    checkpoint_selection="test-oracle-per-rate", osram_readout_fusion="flat",
                    osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64,
                    osram_value_dim=64, osram_write_step=.6)
    for key, expected in required.items():
        if base.get(key) != expected:
            raise ValueError(f"reference mismatch {key}: {base.get(key)!r}")
    if base.get("osram_meaningful_block") not in (None, "none"):
        raise ValueError("reference is not Flat")
    result = dict(base)
    result.update(epochs=spec["epochs"], learning_rate=spec["learning_rate"],
                  osram_meaningful_block=spec["method"])
    return result


def gpu_inventory():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    return {int(a.strip()): b.strip() for a, b in (line.split(",") for line in text.splitlines())}


def train(root, name, gpu_uuid):
    spec = SPECS[name]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("GPU UUID binding mismatch")
    source = Path(__file__).resolve().parents[2]
    from experiments.osram_core20_20261005.run import verify_snapshot
    snapshot = verify_snapshot(source)
    ref = reference(spec["seed"])
    config_dict = candidate_config(read(ref / "config.json"), name)
    dataset_root, feature_roots = data_info()
    os.environ["GCNET_DATASET_ROOT"] = dataset_root
    os.environ["GCNET_CACHE_ROOT"] = str(root / "cache_CMUMOSEI")
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    from gcnet_missing_m3.training_resume import TrainingState
    from experiments.osram_mosi_hparam_sweep_20260918.run import canonical_mask_hashes
    import torch
    torch.set_num_threads(1)
    cfg = TrainConfig(**config_dict)
    output = root / "runs" / name
    output.mkdir(parents=True, exist_ok=False)
    identity = {"snapshot": sha(source / "SNAPSHOT.json"),
                "config": hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()}
    record = dict(label=LABEL, status="running", name=name, spec=spec, pid=os.getpid(),
                  server=socket.gethostname(), gpu_uuid=gpu_uuid, started_utc=now(),
                  code_commit=snapshot["code_commit"], config=asdict(cfg), reference=str(ref))
    write(output / "PROVENANCE.json", record)
    try:
        metrics = run_experiment(cfg, *feature_roots, str(output),
                                 training_state=TrainingState(output, identity=identity))
        if len(read(output / "history.json")) != spec["epochs"]:
            raise ValueError("epoch count mismatch")
        if canonical_mask_hashes(output) != canonical_mask_hashes(ref):
            raise ValueError("canonical evaluation masks changed")
        record.update(status="complete", finished_utc=now(), outputs_verified=True,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      selection_metric=metrics.get("selection_metric"))
        write(output / "PROVENANCE.json", record)
    except BaseException as error:
        record.update(status="failed", finished_utc=now(), error=repr(error))
        write(output / "PROVENANCE.json", record)
        raise


def dispatch(root):
    root.mkdir(parents=True, exist_ok=False)
    inventory = gpu_inventory()
    rows = []
    (root / "logs").mkdir()
    for name, spec in SPECS.items():
        gpu = spec["gpu"]
        if gpu == 4:
            raise ValueError("biggpu GPU4 is forbidden")
        log = root / "logs" / f"{name}.log"
        command = [sys.executable, "-u", "-m", "experiments.osram_mosei_dim32_depth2_followup_20261012.run",
                   "--train", "--root", str(root), "--name", name, "--gpu-uuid", inventory[gpu]]
        stream = log.open("x")
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=inventory[gpu],
                                          OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        rows.append(dict(name=name, gpu=gpu, gpu_uuid=inventory[gpu], pid=child.pid,
                         command=command, log=str(log), output=str(root / "runs" / name)))
    write(root / "LAUNCH.json", dict(label=LABEL, status="running", started_utc=now(), runs=rows))
    while any(subprocess.call(["kill", "-0", str(row["pid"])], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL) == 0 for row in rows):
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--name", choices=SPECS)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.train:
        train(args.root.resolve(), args.name, args.gpu_uuid)
    else:
        dispatch(args.root.resolve())


if __name__ == "__main__":
    main()
