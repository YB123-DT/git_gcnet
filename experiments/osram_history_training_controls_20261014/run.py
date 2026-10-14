"""P2 controls on IEMOCAP ECERC features or MELD's retained fixed masks.

No launch occurs without --run-fold or --dispatch; --dry-run only emits tasks.
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

from experiments.osram_ecerc_features_20261012.run import DATASET_ROOT, FEATURE_ROOT
from experiments.osram_ecerc_features_20261012.run_meld import MELD_PICKLE


CONTROLS = ("local-only", "causal-gru")
DATASETS = ("IEMOCAPFour", "IEMOCAPSix", "MELD")
IEMOCAP_DATASETS = ("IEMOCAPFour", "IEMOCAPSix")
BAD_GPU_UUID = "GPU-46fb379f-cc90-dc82-9b5e-5d011f552264"
PROVENANCE_SOURCE_FILES = (
    "gcnet_missing_m3/causal_gru_history.py",
    "gcnet_missing_m3/legacy_fixed_mask.py",
    "gcnet_missing_m3/osram.py",
    "gcnet_missing_m3/model.py",
    "gcnet_missing_m3/train_gcnet.py",
    "experiments/osram_history_training_controls_20261014/run.py",
    "experiments/osram_meld_flat_source_combinations_20261013/legacy_schedule.py",
)
OUTPUT_ROOT = Path(os.environ.get(
    "OSRAM_P2_OUTPUT", "/data2/yb/paper/05_reproduction/runs/OSRAM_P2_history_controls_20261014"))


def build_config(dataset: str, control: str, seed: int, fold: int = 1, config_class=None):
    if dataset not in DATASETS or control not in CONTROLS:
        raise ValueError("P2 requires IEMOCAPFour/IEMOCAPSix/MELD and local-only/causal-gru")
    if fold not in (range(1, 6) if dataset in IEMOCAP_DATASETS else (1,)):
        raise ValueError("invalid dataset fold")
    if config_class is None:
        from gcnet_missing_m3.train_gcnet import TrainConfig
        config_class = TrainConfig
    return config_class(
        dataset=dataset, fold=fold, seed=seed,
        epochs=100 if dataset in IEMOCAP_DATASETS else 50,
        batch_size=32, learning_rate=1e-3, weight_decay=1e-5,
        evaluation_protocol="official", checkpoint_selection="test-oracle-per-rate",
        checkpoint_selection_metric="weighted_f1", evaluate_test=True,
        train_rate_mode="cyclic",
        missing_mask_protocol="conversation-mask-v2" if dataset in IEMOCAP_DATASETS else "gcnet-legacy-fixed-v1",
        training_objective="emotion-only", emotion_loss_mode="sample-mean",
        fusion_type="mean", representation_type="slot", backbone_type="osram",
        osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
        osram_bidirectional=False, osram_forward_slot_reuse=False, osram_write_step=.6,
        osram_ablation="local-only" if control == "local-only" else "full",
        osram_emotion_ablation="full", osram_readout_fusion="flat",
        osram_meaningful_block="none", osram_causal_gru_history=control == "causal-gru",
        disable_unused_aux_modules=True, completion_path="none", classification_completion=False,
    )


def tasks(dataset: str, seeds: list[int], controls=CONTROLS) -> list[dict]:
    if dataset not in DATASETS or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("dataset and distinct seeds are required")
    if not controls or len(set(controls)) != len(controls) or any(c not in CONTROLS for c in controls):
        raise ValueError("controls must be distinct supported P2 controls")
    return [dict(dataset=dataset, control=control, seed=seed, fold=fold)
            for control in controls for seed in seeds
            for fold in (range(1, 6) if dataset in IEMOCAP_DATASETS else (1,))]


def dataset_inputs(dataset: str):
    """Use the trainer's real label map and the existing ECERC feature bundle.

    Four-way and six-way labels select their existing pickle; utterance-aligned
    ECERC feature roots are shared. Class counts remain the trainer's existing
    _dataset_shape contract, rather than being inferred from pickle names.
    """
    if dataset not in DATASETS:
        raise ValueError("unsupported P2 dataset")
    import config
    source = Path(config.PATH_TO_LABEL[dataset])
    if dataset in IEMOCAP_DATASETS:
        return (source, [FEATURE_ROOT / modality for modality in ("audio", "text", "visual")],
                FEATURE_ROOT / "MANIFEST.json")
    return source, ["MELD-pickle"] * 3, None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_code_provenance(source_root: Path | None = None) -> tuple[dict, str]:
    """Identify either a real Git worktree or an immutable archive deployment.

    An archive has no .git marker, so it never runs Git. Actual source-file I/O
    failures and Git failures inside a declared worktree remain hard failures.
    """
    source_root = (Path(__file__).resolve().parents[2] if source_root is None
                   else Path(source_root).resolve())
    source_hashes = {name: _sha256(source_root / name) for name in PROVENANCE_SOURCE_FILES}
    source_identity = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    if (source_root / ".git").exists():
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
        diff = subprocess.check_output([
            "git", "diff", "HEAD", "--", "gcnet_missing_m3",
            "experiments/osram_history_training_controls_20261014/run.py"],
            cwd=source_root, text=True)
        metadata = dict(code_source_kind="git-worktree", commit=commit,
                        code_diff_available=True, code_diff_sha256=hashlib.sha256(diff.encode()).hexdigest())
    else:
        diff = ""
        metadata = dict(
            code_source_kind="snapshot", commit=os.environ.get("OSRAM_P2_SOURCE_COMMIT") or "unknown",
            code_diff_available=False, code_diff_sha256=None,
            snapshot_identity=dict(name=source_root.name, path=str(source_root), source_sha256=source_identity),
        )
    metadata["source_sha256"] = source_hashes
    metadata["source_identity_sha256"] = source_identity
    return metadata, diff


def output_path(root: Path, task: dict) -> Path:
    return root / task["dataset"] / task["control"] / f"seed_{task['seed']}" / f"fold_{task['fold']}"


def child_command(task: dict, root: Path) -> list[str]:
    return [sys.executable, "-u", "-m", "experiments.osram_history_training_controls_20261014.run",
            "--run-fold", "--dataset", task["dataset"], "--control", task["control"],
            "--seed", str(task["seed"]), "--fold", str(task["fold"]), "--output-root", str(root)]


def validate_visible_devices(visibility: str | None = None) -> list[dict]:
    """Resolve host selectors against nvidia-smi before CUDA remaps indices.

    biggpu's SSH alias is user22 on the host. Its inventory defines the explicit
    whitelist: every listed host card except physical index4 and its known UUID.
    Full UUIDs and unique prefixes resolve identically to physical selectors.
    """
    host = socket.gethostname().lower().split(".")[0]
    biggpu = host in {"user22", "biggpu"} or os.environ.get("OSRAM_P2_SERVER_PROFILE") == "biggpu"
    if visibility is None:
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visibility is None or not visibility.strip():
        if biggpu:
            raise ValueError("biggpu/user22 requires explicit CUDA_VISIBLE_DEVICES restricted to healthy host GPUs")
        return []
    selectors = [device.strip() for device in visibility.split(",")]
    for selector in selectors:
        if selector == "4" or (selector.lower().startswith("gpu-")
                               and BAD_GPU_UUID.lower().startswith(selector.lower())):
            raise ValueError("physical GPU4 and its UUID/prefix are prohibited in the P2 project runner")
        if not selector:
            raise ValueError("CUDA_VISIBLE_DEVICES contains an empty selector")
    inventory_text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    inventory = []
    for line in inventory_text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit() or not fields[1].startswith("GPU-"):
            raise ValueError("cannot resolve the nvidia-smi host GPU inventory")
        inventory.append(dict(physical_index=int(fields[0]), uuid=fields[1]))
    whitelist = [gpu for gpu in inventory if gpu["physical_index"] != 4
                 and gpu["uuid"].lower() != BAD_GPU_UUID.lower()]
    resolved = []
    for selector in selectors:
        if selector.isdigit():
            matches = [gpu for gpu in inventory if gpu["physical_index"] == int(selector)]
        elif selector.lower().startswith("gpu-"):
            matches = [gpu for gpu in inventory if gpu["uuid"].lower().startswith(selector.lower())]
        else:
            raise ValueError(f"unsupported physical GPU selector: {selector}")
        if len(matches) != 1:
            raise ValueError(f"physical GPU selector must identify exactly one listed host card: {selector}")
        if matches[0] not in whitelist:
            raise ValueError("physical GPU4 and its UUID are outside the P2 healthy-card whitelist")
        if matches[0] in resolved:
            raise ValueError("CUDA_VISIBLE_DEVICES selects the same host GPU more than once")
        resolved.append(matches[0])
    return resolved


def run_fold(task: dict, root: Path):
    visible_devices = validate_visible_devices()
    if visible_devices:
        # UUID normalization avoids CUDA_DEVICE_ORDER-dependent numeric maps.
        # A dispatcher child receives physical6 but internally sees cuda:0;
        # resolve physical6 here before importing torch, never from cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu["uuid"] for gpu in visible_devices)
    # Set import-time data paths before loading TrainConfig or the trainer.
    os.environ["GCNET_DATASET_ROOT"] = str(DATASET_ROOT)
    os.environ["GCNET_MELD_PICKLE"] = str(MELD_PICKLE)
    os.environ["GCNET_CACHE_ROOT"] = str(root / "cache")
    config = build_config(**task)
    source, roots, manifest_path = dataset_inputs(task["dataset"])
    if manifest_path is not None:
        if not manifest_path.is_file() or not all(path.is_dir() for path in roots):
            raise FileNotFoundError(f"prepare ECERC IEMOCAP features first: {manifest_path}")
        feature_manifest = json.loads(manifest_path.read_text())
    else:
        feature_manifest = None
    if not source.is_file():
        raise FileNotFoundError(source)
    output = output_path(root, task)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    threads = int(os.environ.get("GCNET_CPU_THREADS", "1"))
    if threads < 1:
        raise ValueError("GCNET_CPU_THREADS must be positive")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(min(threads, 4))
    code_provenance, diff = collect_code_provenance()
    (output / "CODE_DIFF.patch").write_text(diff)
    provenance = dict(
        status="running", task=task, started_utc=_now(), host=socket.gethostname(),
        pid=os.getpid(), argv=sys.argv, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        visible_host_gpus=visible_devices,
        **code_provenance,
        config=asdict(config), dataset_source=str(source), dataset_sha256=_sha256(source),
        feature_manifest=feature_manifest, environment=dict(python=sys.version, torch=torch.__version__),
        protocol="per-rate test-oracle weighted-F1 exploratory control; same cyclic rates and cfg84 Flat",
    )
    _write(output / "PROVENANCE.json", provenance)
    try:
        from gcnet_missing_m3.train_gcnet import run_experiment
        metrics = run_experiment(config, *(str(path) for path in roots), output_dir=str(output))
        if metrics.get("selection_metric") != "weighted_f1":
            raise RuntimeError("P2 requires explicit per-rate test-oracle weighted-F1 selection")
        # The trainer verifies and evaluates all retained per-rate states itself.
        provenance.update(status="complete", finished_utc=_now(), selection_metric=metrics["selection_metric"])
    except BaseException as error:
        provenance.update(status="failed", finished_utc=_now(), error=repr(error))
        raise
    finally:
        _write(output / "PROVENANCE.json", provenance)


def dispatch(dataset: str, seeds: list[int], controls, root: Path, gpu: int, concurrency: int):
    if concurrency < 1 or gpu < 0:
        raise ValueError("concurrency must be positive and GPU index non-negative")
    # This project dispatcher refuses physical GPU4 regardless of hostname:
    # ssh biggpu's real hostname is user22, so alias matching is insufficient.
    # The child's remapped cuda:0 must not be mistaken for this physical index.
    if gpu == 4:
        raise ValueError("physical GPU4 is prohibited in the P2 project dispatcher")
    resolved_devices = validate_visible_devices(str(gpu))
    identity = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=index,uuid,name,memory.total,memory.free",
        "--format=csv,noheader"], text=True).strip()
    pending = tasks(dataset, seeds, controls)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / f"STATUS_{dataset}_{'-'.join(controls)}_{'-'.join(map(str, seeds))}.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    for task in pending:
        output = output_path(root, task)
        if output.exists() or output.with_suffix(".log").exists():
            raise FileExistsError(f"refusing duplicate P2 task: {output}")
    status = dict(status="running", dataset=dataset, started_utc=_now(), gpu=gpu,
                  visible_host_gpus=resolved_devices,
                  gpu_identity=identity, concurrency=concurrency, tasks=[])
    children = []
    _write(status_path, status)
    failed = False
    try:
        while pending or children:
            while pending and len(children) < concurrency and not failed:
                task = pending.pop(0)
                output = output_path(root, task)
                output.parent.mkdir(parents=True, exist_ok=True)
                log = output.with_suffix(".log")
                stream = log.open("x")
                command = child_command(task, root)
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                             MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
                row = dict(task=task, pid=child.pid, status="running", command=command,
                           log=str(log), output=str(output))
                status["tasks"].append(row)
                children.append((child, stream, row))
                _write(status_path, status)
            if failed:
                pending.clear()
            for item in list(children):
                child, stream, row = item
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                provenance = Path(row["output"]) / "PROVENANCE.json"
                complete = (code == 0 and provenance.is_file()
                            and json.loads(provenance.read_text()).get("status") == "complete")
                row.update(status="complete" if complete else "failed", exit_code=code)
                failed |= not complete
                children.remove(item)
                _write(status_path, status)
            if children:
                time.sleep(5)
        status.update(status="failed" if failed else "complete", finished_utc=_now())
        _write(status_path, status)
        if failed:
            raise RuntimeError(f"P2 task failed; inspect {status_path}")
    except BaseException as error:
        status.update(status="failed", error=repr(error), finished_utc=_now())
        _write(status_path, status)
        # Existing training processes are preserved and their PIDs are durable.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-fold", action="store_true")
    mode.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--control", choices=CONTROLS)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    seeds = args.seeds or [args.seed]
    controls = (args.control,) if args.control else CONTROLS
    if args.dry_run:
        print(json.dumps([dict(task=task, config=asdict(build_config(**task)),
                               output=str(output_path(args.output_root, task)))
                          for task in tasks(args.dataset, seeds, controls)], indent=2))
    elif args.run_fold:
        if args.control is None or args.seeds:
            parser.error("--run-fold requires one --control and --seed")
        run_fold(dict(dataset=args.dataset, control=args.control, seed=args.seed, fold=args.fold), args.output_root)
    else:
        if args.gpu is None:
            parser.error("--dispatch requires explicit physical --gpu")
        dispatch(args.dataset, seeds, controls, args.output_root, args.gpu, args.concurrency)


if __name__ == "__main__":
    main()
