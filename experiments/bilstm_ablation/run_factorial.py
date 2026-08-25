"""Run the immutable four-dataset GCNet BiLSTM factorial experiment."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Tuple


DATASETS = {
    "IEMOCAPFour": {
        "directory": "IEMOCAP",
        "fold": 5,
        "split_tag": "fold5_screening",
        "metric": "multiclass_weighted_f1",
    },
    "IEMOCAPSix": {
        "directory": "IEMOCAP",
        "fold": 5,
        "split_tag": "fold5_screening",
        "metric": "multiclass_weighted_f1",
    },
    "CMUMOSI": {
        "directory": "CMUMOSI",
        "fold": None,
        "split_tag": "official_split",
        "metric": "nonzero_binary_weighted_f1",
    },
    "CMUMOSEI": {
        "directory": "CMUMOSEI",
        "fold": None,
        "split_tag": "official_split",
        "metric": "nonzero_binary_weighted_f1",
    },
}
ARMS = {
    "original": ("bilstm", "bilstm"),
    "no_pre_bilstm": ("linear", "bilstm"),
    "no_post_bilstm": ("bilstm", "linear"),
    "no_all_bilstm": ("linear", "linear"),
}
FEATURE_NAMES = {
    "audio": "wav2vec-large-c-UTT",
    "text": "deberta-large-4-UTT",
    "video": "manet_UTT",
}
LABEL_FILENAMES = {
    "IEMOCAPFour": "IEMOCAP_features_raw_4way.pkl",
    "IEMOCAPSix": "IEMOCAP_features_raw_6way.pkl",
    "CMUMOSI": "CMUMOSI_features_raw_2way.pkl",
    "CMUMOSEI": "CMUMOSEI_features_raw_2way.pkl",
}
SMOKE_RATES = (0.7,)
SMOKE_SEEDS = (66,)
PILOT_RATES = (0.0, 0.7)
PILOT_SEEDS = (66, 67, 68)
FORMAL_RATES = tuple(index / 10.0 for index in range(8))
FORMAL_SEEDS = (66, 67, 68, 69, 70)
MAX_WORKERS_PER_GPU = 3
LOCKED_TRAINING = {
    "audio_feature": FEATURE_NAMES["audio"],
    "text_feature": FEATURE_NAMES["text"],
    "video_feature": FEATURE_NAMES["video"],
    "base_model": "LSTM",
    "graph_conv_variant": "original",
    "windowp": 2,
    "windowf": 2,
    "hidden": 200,
    "lr": 0.001,
    "l2": 0.00001,
    "dropout": 0.5,
    "batch_size": 32,
    "num_threads": 6,
    "epochs": 100,
    "loss_recon": True,
    "reccls_flag": False,
    "lower_bound": False,
    "time_attn": False,
}


@dataclass(frozen=True)
class Job:
    stage: str
    dataset: str
    arm: str
    missing_rate: float
    seed: int
    split_tag: str
    output_directory: Path


def _rate_tag(rate):
    return "{:.1f}".format(rate).replace(".", "p")


def _stage_defaults(stage):
    if stage == "smoke":
        return SMOKE_RATES, SMOKE_SEEDS
    if stage == "pilot":
        return PILOT_RATES, PILOT_SEEDS
    if stage == "formal":
        return FORMAL_RATES, FORMAL_SEEDS
    raise ValueError("unknown stage: {!r}".format(stage))


def _validated_subset(values, registry, label):
    values = tuple(values)
    unknown = set(values) - set(registry)
    if unknown:
        raise ValueError("unknown {}: {!r}".format(label, sorted(unknown)))
    if len(values) != len(set(values)):
        raise ValueError("duplicate {} are not allowed".format(label))
    return values


def build_jobs(
    stage,
    output_root,
    datasets=tuple(DATASETS),
    arms=tuple(ARMS),
    rates=None,
    seeds=None,
):
    default_rates, default_seeds = _stage_defaults(stage)
    datasets = _validated_subset(datasets, DATASETS, "datasets")
    arms = _validated_subset(arms, ARMS, "arms")
    rates = tuple(default_rates if rates is None else rates)
    seeds = tuple(default_seeds if seeds is None else seeds)
    if len(rates) != len(set(rates)) or len(seeds) != len(set(seeds)):
        raise ValueError("duplicate rates or seeds are not allowed")
    if any(rate < 0.0 or rate > 0.7 for rate in rates):
        raise ValueError("rates must be between 0.0 and 0.7")
    jobs = []
    for dataset in datasets:
        split_tag = DATASETS[dataset]["split_tag"]
        for arm in arms:
            for rate in rates:
                for seed in seeds:
                    directory = (
                        Path(output_root)
                        / stage
                        / dataset
                        / arm
                        / "miss_{}".format(_rate_tag(rate))
                        / "seed_{}".format(seed)
                        / split_tag
                    )
                    jobs.append(
                        Job(stage, dataset, arm, rate, seed, split_tag, directory)
                    )
    return jobs


def _validate_epoch_override(job, epochs, allow_short_run):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if epochs < LOCKED_TRAINING["epochs"]:
        if job.stage != "smoke" or not allow_short_run:
            raise ValueError(
                "short runs require smoke stage and explicit --allow-short-run"
            )
    elif epochs != LOCKED_TRAINING["epochs"]:
        raise ValueError("locked runs use exactly 100 epochs")
    if allow_short_run and job.stage != "smoke":
        raise ValueError("--allow-short-run is restricted to smoke stage")
    if allow_short_run and epochs == LOCKED_TRAINING["epochs"]:
        raise ValueError("--allow-short-run requires an explicit short epoch count")


def build_command(
    job,
    python,
    repository,
    data_root_root,
    mask_root,
    epochs=100,
    allow_short_run=False,
):
    _validate_epoch_override(job, epochs, allow_short_run)
    dataset = DATASETS[job.dataset]
    pre_context, post_context = ARMS[job.arm]
    command = [
        str(python),
        "-u",
        str(Path(repository) / "gcnet" / "train_gcnet.py"),
        "--audio-feature",
        FEATURE_NAMES["audio"],
        "--text-feature",
        FEATURE_NAMES["text"],
        "--video-feature",
        FEATURE_NAMES["video"],
        "--dataset",
        job.dataset,
        "--data-root",
        str(Path(data_root_root) / dataset["directory"]),
        "--base-model",
        "LSTM",
        "--windowp",
        "2",
        "--windowf",
        "2",
        "--hidden",
        "200",
        "--lr",
        "0.001",
        "--l2",
        "0.00001",
        "--dropout",
        "0.5",
        "--batch-size",
        "32",
        "--num-threads",
        "6",
        "--epochs",
        str(epochs),
        "--seed",
        str(job.seed),
        "--mask-seed",
        str(job.seed),
        "--mask-type",
        "constant-{:.1f}".format(job.missing_rate),
        "--graph-conv-variant",
        "original",
        "--pre-graph-context",
        pre_context,
        "--post-graph-context",
        post_context,
        "--mask-bank-root",
        str(Path(mask_root) / job.dataset / job.split_tag),
        "--output-dir",
        str(job.output_directory / "saved"),
        "--loss-recon",
    ]
    if dataset["fold"] is not None:
        command.extend(("--fold-index", str(dataset["fold"])))
    if allow_short_run:
        command.append("--allow-short-run")
    return command


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _job_payload(
    job,
    gpu,
    python,
    repository,
    data_root_root,
    mask_root,
    epochs=100,
    allow_short_run=False,
):
    dataset = DATASETS[job.dataset]
    return {
        "stage": job.stage,
        "dataset": job.dataset,
        "dataset_directory": dataset["directory"],
        "arm": job.arm,
        "pre_graph_context": ARMS[job.arm][0],
        "post_graph_context": ARMS[job.arm][1],
        "missing_rate": job.missing_rate,
        "seed": job.seed,
        "fold": dataset["fold"],
        "split_tag": dataset["split_tag"],
        "metric": dataset["metric"],
        "epochs": epochs,
        "allow_short_run": bool(allow_short_run),
        "gpu": str(gpu),
        "command": build_command(
            job,
            python,
            repository,
            data_root_root,
            mask_root,
            epochs,
            allow_short_run,
        ),
    }


def _completed_log(path, epochs, allow_short_run):
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    expected_smoke = "SMOKE_ONLY={}".format(bool(allow_short_run))
    return (
        sum(line.startswith("epoch:") for line in text.splitlines()) == epochs
        and "Finish" in text
        and "save results in " in text
        and expected_smoke in text
    )


def _completed(
    job,
    gpus,
    python,
    repository,
    data_root_root,
    mask_root,
    epochs=100,
    allow_short_run=False,
):
    directory = job.output_directory
    if not directory.exists():
        return False
    lock = directory / ".active.lock"
    if lock.exists():
        raise RuntimeError("active or stale job lock exists: {}".format(lock))
    if not directory.is_dir() or not any(directory.iterdir()):
        raise RuntimeError("partial job directory exists: {}".format(directory))
    missing = [
        name
        for name in ("command.json", "status.json", "train.log", "saved")
        if not (directory / name).exists()
    ]
    if missing:
        raise RuntimeError(
            "partial job artifacts in {}: missing {}".format(directory, missing)
        )
    try:
        payload = json.loads(
            (directory / "command.json").read_text(encoding="utf-8")
        )
        status = json.loads(
            (directory / "status.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "partial or invalid job metadata: {}".format(directory)
        ) from error
    gpu = str(payload.get("gpu"))
    if gpu not in tuple(map(str, gpus)):
        raise RuntimeError(
            "command.json GPU is outside requested GPUs: {}".format(gpu)
        )
    expected = _job_payload(
        job,
        gpu,
        python,
        repository,
        data_root_root,
        mask_root,
        epochs,
        allow_short_run,
    )
    if payload != expected:
        raise RuntimeError("command.json mismatch for {}".format(directory))
    if status.get("status") != "success" or status.get("return_code") != 0:
        raise RuntimeError(
            "status return_code is not successful for {}".format(directory)
        )
    archives = list((directory / "saved").glob("*.npz"))
    if len(archives) != 1:
        raise RuntimeError(
            "expected exactly one NPZ archive in {}, found {}".format(
                directory / "saved", len(archives)
            )
        )
    if not _completed_log(directory / "train.log", epochs, allow_short_run):
        raise RuntimeError("train.log completion markers or epoch count are invalid")
    return True


def _claim_job(job):
    job.output_directory.mkdir(parents=True, exist_ok=True)
    lock = job.output_directory / ".active.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise RuntimeError("active or stale job lock exists: {}".format(lock)) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {"pid": os.getpid(), "runner": str(Path(__file__).resolve())}, handle
        )
        handle.write("\n")
    return lock


def _release_job(job):
    lock = job.output_directory / ".active.lock"
    if lock.exists():
        lock.unlink()


def _write_json_exclusive_or_equal(path, payload):
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("immutable JSON mismatch: {}".format(path))


def _launch(
    job,
    gpu,
    python,
    repository,
    data_root_root,
    mask_root,
    epochs=100,
    allow_short_run=False,
):
    _claim_job(job)
    payload = _job_payload(
        job,
        gpu,
        python,
        repository,
        data_root_root,
        mask_root,
        epochs,
        allow_short_run,
    )
    log_handle = None
    try:
        _write_json_exclusive_or_equal(job.output_directory / "command.json", payload)
        log_handle = (job.output_directory / "train.log").open("x", encoding="utf-8")
    except Exception:
        if log_handle is not None:
            log_handle.close()
        _release_job(job)
        raise
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository), str(Path(repository) / "gcnet"))
    )
    environment["GCNET_CACHE_ROOT"] = str(Path(mask_root) / "dataset_cache")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONHASHSEED"] = "0"
    try:
        process = subprocess.Popen(
            payload["command"],
            cwd=repository,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_handle.close()
        _release_job(job)
        raise
    return process, log_handle


def _validate_gpu_capacity(gpus, workers_per_gpu):
    if not gpus:
        raise ValueError("at least one GPU is required")
    if workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    if workers_per_gpu > MAX_WORKERS_PER_GPU:
        raise ValueError("each GPU may host at most 3 workers")


def run_jobs(
    jobs,
    gpus,
    workers_per_gpu,
    python,
    repository,
    data_root_root,
    mask_root,
    epochs=100,
    allow_short_run=False,
):
    gpus = tuple(map(str, gpus))
    _validate_gpu_capacity(gpus, workers_per_gpu)
    pending = [
        job
        for job in jobs
        if not _completed(
            job,
            gpus,
            python,
            repository,
            data_root_root,
            mask_root,
            epochs,
            allow_short_run,
        )
    ]
    running = []  # type: List[Tuple[Job, str, subprocess.Popen, object, float]]
    failures = []
    try:
        while pending or running:
            counts = {gpu: 0 for gpu in gpus}  # type: Dict[str, int]
            for _, gpu, _, _, _ in running:
                counts[gpu] += 1
            while pending:
                available = [
                    gpu for gpu in gpus if counts[gpu] < workers_per_gpu
                ]
                if not available:
                    break
                gpu = min(available, key=lambda item: (counts[item], gpus.index(item)))
                job = pending.pop(0)
                process, handle = _launch(
                    job,
                    gpu,
                    python,
                    repository,
                    data_root_root,
                    mask_root,
                    epochs,
                    allow_short_run,
                )
                running.append((job, gpu, process, handle, time.time()))
                counts[gpu] += 1
            time.sleep(1.0)
            survivors = []
            for job, gpu, process, handle, started_at in running:
                return_code = process.poll()
                if return_code is None:
                    survivors.append((job, gpu, process, handle, started_at))
                    continue
                handle.close()
                _write_json(
                    job.output_directory / "status.json",
                    {
                        "status": "success" if return_code == 0 else "failed",
                        "return_code": return_code,
                        "elapsed_seconds": time.time() - started_at,
                        "gpu": gpu,
                    },
                )
                _release_job(job)
                if return_code != 0:
                    failures.append(str(job.output_directory))
            running = survivors
    finally:
        for job, _, process, handle, _ in running:
            if process.poll() is None:
                process.terminate()
                process.wait()
            handle.close()
            _release_job(job)
    if failures:
        raise RuntimeError("failed jobs: {}".format(failures))


def _python_provenance(python, command_output=subprocess.check_output):
    script = """
import json
import platform
import sys

def version(name):
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, '__version__', 'unknown'))

try:
    import torch
    cuda = torch.version.cuda
    cudnn = torch.backends.cudnn.version()
except ImportError:
    cuda = None
    cudnn = None

print(json.dumps({
    'python': {'executable': sys.executable, 'version': platform.python_version()},
    'versions': {
        'torch': version('torch'),
        'torch_geometric': version('torch_geometric'),
        'cuda': cuda,
        'cudnn': cudnn,
    },
}, sort_keys=True))
"""
    try:
        raw = command_output([str(python), "-c", script], text=True)
        provenance = json.loads(raw)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "failed to query requested Python interpreter: {}".format(python)
        ) from error
    provenance["python"]["requested"] = str(python)
    return provenance


def _fingerprint(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError("dataset artifact does not exist: {}".format(path))
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    files = [path] if path.is_file() else sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
    if not files:
        raise RuntimeError("dataset artifact contains no files: {}".format(path))
    for candidate in files:
        relative = (
            candidate.name
            if path.is_file()
            else candidate.relative_to(path).as_posix()
        )
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        with candidate.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                total_size += len(block)
        file_count += 1
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": total_size,
        "file_count": file_count,
    }


def _dataset_manifest(data_root_root, dataset):
    entry = DATASETS[dataset]
    root = Path(data_root_root) / entry["directory"]
    features = root / "features"
    return {
        "root": str(root.resolve()),
        "directory": entry["directory"],
        "fold": entry["fold"],
        "split_tag": entry["split_tag"],
        "metric": entry["metric"],
        "fingerprints": {
            "label": _fingerprint(root / LABEL_FILENAMES[dataset]),
            "audio": _fingerprint(features / FEATURE_NAMES["audio"]),
            "text": _fingerprint(features / FEATURE_NAMES["text"]),
            "video": _fingerprint(features / FEATURE_NAMES["video"]),
        },
    }


def _ensure_run_manifest(
    output_root,
    stage,
    repository,
    data_root_root,
    mask_root,
    datasets,
    arms,
    rates,
    seeds,
    gpus,
    workers_per_gpu,
    python,
    command_output=subprocess.check_output,
    epochs=100,
    allow_short_run=False,
):
    _validate_gpu_capacity(tuple(gpus), workers_per_gpu)

    def output(command):
        return command_output(command, cwd=repository, text=True).strip()

    if output(["git", "status", "--porcelain"]):
        raise RuntimeError("locked experiments require a clean git worktree")
    try:
        gpu_text = output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        )
        gpu_names = [line for line in gpu_text.splitlines() if line]
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu_names = []
    interpreter = _python_provenance(python, command_output)
    manifest = {
        "git": {"head": output(["git", "rev-parse", "HEAD"]), "clean": True},
        "stage": stage,
        "python": interpreter["python"],
        "versions": interpreter["versions"],
        "gpu_names": gpu_names,
        "roots": {
            "repository": str(Path(repository).resolve()),
            "data_root_root": str(Path(data_root_root).resolve()),
            "mask_root": str(Path(mask_root).resolve()),
        },
        "datasets": {
            dataset: _dataset_manifest(data_root_root, dataset)
            for dataset in datasets
        },
        "locked_training": LOCKED_TRAINING,
        "arms": {name: list(modes) for name, modes in ARMS.items()},
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
        },
    }
    path = Path(output_root) / stage / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("run manifest mismatch: {}".format(path)) from error
        if existing != manifest:
            raise RuntimeError("run manifest mismatch: {}".format(path))
    invocation = {
        "stage": stage,
        "datasets": list(datasets),
        "arms": list(arms),
        "rates": list(rates),
        "seeds": list(seeds),
        "gpus": list(gpus),
        "workers_per_gpu": workers_per_gpu,
        "epochs": epochs,
        "allow_short_run": bool(allow_short_run),
        "job_count": len(datasets) * len(arms) * len(rates) * len(seeds),
    }
    canonical = json.dumps(
        invocation, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    invocation_path = (
        path.parent
        / "invocations"
        / "{}.json".format(hashlib.sha256(canonical).hexdigest())
    )
    invocation_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive_or_equal(invocation_path, invocation)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/yangbin/miniconda3/envs/gcnet-official/bin/python"),
    )
    parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3"))
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=tuple(DATASETS)
    )
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=tuple(ARMS))
    parser.add_argument("--rates", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--allow-short-run", action="store_true", default=False)
    args = parser.parse_args()
    _validate_gpu_capacity(tuple(args.gpus), args.workers_per_gpu)
    repository = Path(__file__).resolve().parents[2]
    jobs = build_jobs(
        args.stage,
        args.output_root,
        datasets=args.datasets,
        arms=args.arms,
        rates=args.rates,
        seeds=args.seeds,
    )
    for job in jobs:
        _validate_epoch_override(job, args.epochs, args.allow_short_run)
    rates = tuple(dict.fromkeys(job.missing_rate for job in jobs))
    seeds = tuple(dict.fromkeys(job.seed for job in jobs))
    _ensure_run_manifest(
        args.output_root,
        args.stage,
        repository,
        args.data_root_root,
        args.mask_root,
        tuple(args.datasets),
        tuple(args.arms),
        rates,
        seeds,
        tuple(args.gpus),
        args.workers_per_gpu,
        args.python,
        epochs=args.epochs,
        allow_short_run=args.allow_short_run,
    )
    # Arms are sequenced so paired jobs cannot race while creating mask bundles.
    for arm in args.arms:
        run_jobs(
            (job for job in jobs if job.arm == arm),
            tuple(args.gpus),
            args.workers_per_gpu,
            args.python,
            repository,
            args.data_root_root,
            args.mask_root,
            args.epochs,
            args.allow_short_run,
        )


if __name__ == "__main__":
    main()
