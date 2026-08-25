"""Run the immutable four-dataset GCNet BiLSTM factorial experiment."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import tempfile
import time
from typing import Dict, List, Tuple

import numpy as np


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
PARAMETER_COUNTS = {
    "IEMOCAPFour": {
        "total": 36_165_564,
        "selected": {
            "original": 34_139_164,
            "no_pre_bilstm": 29_781_164,
            "no_post_bilstm": 15_109_164,
            "no_all_bilstm": 10_751_164,
        },
    },
    "IEMOCAPSix": {
        "total": 36_166_566,
        "selected": {
            "original": 34_140_166,
            "no_pre_bilstm": 29_782_166,
            "no_post_bilstm": 15_110_166,
            "no_all_bilstm": 10_752_166,
        },
    },
    "CMUMOSI": {
        "total": 36_044_061,
        "selected": {
            "original": 34_017_661,
            "no_pre_bilstm": 29_659_661,
            "no_post_bilstm": 14_987_661,
            "no_all_bilstm": 10_629_661,
        },
    },
    "CMUMOSEI": {
        "total": 36_044_061,
        "selected": {
            "original": 34_017_661,
            "no_pre_bilstm": 29_659_661,
            "no_post_bilstm": 14_987_661,
            "no_all_bilstm": 10_629_661,
        },
    },
}
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

    @property
    def artifact_scope(self):
        return "smoke" if self.stage == "smoke" else "formal"


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
    raw_rates = tuple(default_rates if rates is None else rates)
    normalized_rates = []
    for raw_rate in raw_rates:
        rate = float(raw_rate)
        scaled = rate * 10.0
        if (
            not math.isfinite(rate)
            or not 0.0 <= rate <= 0.7
            or abs(scaled - round(scaled)) > 1e-9
        ):
            raise ValueError("rates must be finite exact tenths between 0.0 and 0.7")
        normalized_rates.append(round(scaled) / 10.0)
    rates = tuple(normalized_rates)
    seeds = tuple(default_seeds if seeds is None else seeds)
    if len(rates) != len(set(rates)) or len(seeds) != len(set(seeds)):
        raise ValueError("duplicate rates or seeds are not allowed")
    jobs = []
    for dataset in datasets:
        split_tag = DATASETS[dataset]["split_tag"]
        for arm in arms:
            for rate in rates:
                for seed in seeds:
                    directory = (
                        Path(output_root)
                        / ("smoke" if stage == "smoke" else "formal")
                        / dataset
                        / arm
                        / "miss_{}".format(_rate_tag(rate))
                        / "seed_{}".format(seed)
                        / split_tag
                    )
                    jobs.append(
                        Job(stage, dataset, arm, rate, seed, split_tag, directory)
                    )
    if len({job.output_directory for job in jobs}) != len(jobs):
        raise RuntimeError("job output directories are not unique")
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
        "stage": job.artifact_scope,
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


def _sha256_file(path):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError("refusing symlink artifact: {}".format(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise RuntimeError("cannot open stable artifact: {}".format(path)) from error
    with os.fdopen(descriptor, "rb") as handle:
        return _hash_open_file(handle)


def _hash_open_file(handle):
    digest = hashlib.sha256()
    handle.seek(0)
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
    handle.seek(0)
    return digest.hexdigest()


def _stable_stat(file_stat):
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _open_stable_archive(path):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError("trusted NPZ archive must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise RuntimeError("trusted NPZ archive cannot be opened safely") from error
    opened_stat = os.fstat(descriptor)
    if not stat.S_ISREG(opened_stat.st_mode):
        os.close(descriptor)
        raise RuntimeError("trusted NPZ archive is not a regular file")
    return descriptor, opened_stat


def _verify_stable_archive(path, descriptor, opened_stat):
    after_stat = os.fstat(descriptor)
    try:
        path_stat = os.stat(str(path), follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("trusted NPZ archive path changed during validation") from error
    expected = _stable_stat(opened_stat)
    if _stable_stat(after_stat) != expected or _stable_stat(path_stat) != expected:
        raise RuntimeError("trusted NPZ archive changed during validation")


def _mask_artifact_paths(job, mask_root, epochs):
    stem = "mask_stage_v2_rate_{}_seed_{}_epochs_{}".format(
        _rate_tag(job.missing_rate), job.seed, epochs
    )
    root = Path(mask_root) / job.dataset / job.split_tag
    return root / (stem + ".npz"), root / (stem + ".json")


def _archive_scalar(archive, name):
    if name not in archive.files:
        raise RuntimeError("trusted NPZ is missing {!r}".format(name))
    value = archive[name]
    if value.size != 1:
        raise RuntimeError("trusted NPZ field {!r} is not scalar".format(name))
    return value.item()


def _require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(
            "trusted NPZ {} mismatch: {!r} != {!r}".format(
                label, actual, expected
            )
        )


def _validate_mask_manifest(job, manifest, mask_root, epochs):
    if not isinstance(manifest, dict):
        raise RuntimeError("trusted NPZ mask manifest is not a dictionary")
    expected = {
        "generator": "gcnet-random-mask-v2-stage-aware",
        "requested_missing_rate": job.missing_rate,
        "seed": job.seed,
        "epochs": epochs,
    }
    for key, value in expected.items():
        _require_equal(manifest.get(key), value, "mask manifest {}".format(key))
    hashes = [
        manifest.get("sha256"),
        manifest.get("validation_sha256"),
        manifest.get("test_sha256"),
    ]
    train_hashes = manifest.get("train_sha256")
    if not isinstance(train_hashes, (list, tuple)) or len(train_hashes) != epochs:
        raise RuntimeError("trusted NPZ mask manifest train hashes mismatch")
    hashes.extend(train_hashes)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        raise RuntimeError("trusted NPZ mask manifest contains an invalid hash")
    bank_path, manifest_path = _mask_artifact_paths(job, mask_root, epochs)
    if not bank_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("trusted mask bundle artifacts are incomplete")
    try:
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("trusted mask manifest cannot be read") from error
    if persisted != manifest:
        raise RuntimeError("trusted NPZ mask manifest differs from mask-bank manifest")
    return bank_path, manifest_path


def _validate_archive(
    job,
    archive_path,
    data_root_root,
    mask_root,
    epochs,
    allow_short_run,
    expected_sha256=None,
):
    descriptor = None
    try:
        descriptor, opened_stat = _open_stable_archive(archive_path)
        with os.fdopen(descriptor, "rb") as archive_handle:
            descriptor = None
            archive_sha256 = _hash_open_file(archive_handle)
            if (
                expected_sha256 is not None
                and archive_sha256 != expected_sha256
            ):
                raise RuntimeError("trusted NPZ archive digest mismatch")
            with np.load(archive_handle, allow_pickle=True) as archive:
                required = {
                    "args",
                    "fold_numbers",
                    "mask_bank_manifest",
                    "parameter_count",
                    "selected_path_parameter_count",
                    "smoke_only",
                }
                missing = required - set(archive.files)
                if missing:
                    raise RuntimeError(
                        "trusted NPZ is missing fields: {}".format(sorted(missing))
                    )
                arguments = _archive_scalar(archive, "args")
                if hasattr(arguments, "__dict__"):
                    arguments = vars(arguments)
                if not isinstance(arguments, dict):
                    raise RuntimeError(
                        "trusted NPZ args are not a namespace or dictionary"
                    )
                pre_context, post_context = ARMS[job.arm]
                expected_arguments = {
                    "dataset": job.dataset,
                    "data_root": str(
                        Path(data_root_root)
                        / DATASETS[job.dataset]["directory"]
                    ),
                    "audio_feature": FEATURE_NAMES["audio"],
                    "text_feature": FEATURE_NAMES["text"],
                    "video_feature": FEATURE_NAMES["video"],
                    "base_model": "LSTM",
                    "graph_conv_variant": "original",
                    "pre_graph_context": pre_context,
                    "post_graph_context": post_context,
                    "windowp": 2,
                    "windowf": 2,
                    "hidden": 200,
                    "lr": 0.001,
                    "l2": 0.00001,
                    "dropout": 0.5,
                    "batch_size": 32,
                    "num_threads": 6,
                    "epochs": epochs,
                    "seed": job.seed,
                    "mask_seed": job.seed,
                    "mask_type": "constant-{:.1f}".format(job.missing_rate),
                    "mask_bank_root": str(
                        Path(mask_root) / job.dataset / job.split_tag
                    ),
                    "fold_index": DATASETS[job.dataset]["fold"],
                    "output_dir": str(job.output_directory / "saved"),
                    "loss_recon": True,
                    "reccls_flag": False,
                    "lower_bound": False,
                    "time_attn": False,
                    "allow_short_run": bool(allow_short_run),
                }
                for name, expected in expected_arguments.items():
                    _require_equal(
                        arguments.get(name), expected, "args.{}".format(name)
                    )
                fold_numbers = archive["fold_numbers"].tolist()
                expected_folds = [DATASETS[job.dataset]["fold"] or 1]
                _require_equal(fold_numbers, expected_folds, "fold/split")
                _require_equal(
                    bool(_archive_scalar(archive, "smoke_only")),
                    bool(allow_short_run),
                    "smoke provenance",
                )
                counts = PARAMETER_COUNTS[job.dataset]
                _require_equal(
                    int(_archive_scalar(archive, "parameter_count")),
                    counts["total"],
                    "stored parameter count",
                )
                _require_equal(
                    int(
                        _archive_scalar(
                            archive, "selected_path_parameter_count"
                        )
                    ),
                    counts["selected"][job.arm],
                    "selected parameter count",
                )
                mask_manifest = _archive_scalar(archive, "mask_bank_manifest")
            _verify_stable_archive(archive_path, archive_handle.fileno(), opened_stat)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            "trusted NPZ cannot be opened: {}".format(archive_path)
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    mask_paths = _validate_mask_manifest(job, mask_manifest, mask_root, epochs)
    return mask_paths[0], mask_paths[1], archive_sha256


def _artifact_hashes(
    job, archive_path, mask_root, epochs, archive_sha256=None
):
    bank_path, mask_manifest_path = _mask_artifact_paths(job, mask_root, epochs)
    paths = {
        "command.json": job.output_directory / "command.json",
        "train.log": job.output_directory / "train.log",
        "archive": archive_path,
        "mask_bundle": bank_path,
        "mask_manifest": mask_manifest_path,
    }
    hashes = {name: _sha256_file(path) for name, path in paths.items() if name != "archive"}
    hashes["archive"] = (
        _sha256_file(archive_path)
        if archive_sha256 is None
        else archive_sha256
    )
    return hashes


def _status_checksum(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_completion_status(
    job,
    gpu,
    archive_path,
    python,
    repository,
    data_root_root,
    mask_root,
    epochs=100,
    allow_short_run=False,
    elapsed_seconds=0.0,
):
    _, _, archive_sha256 = _validate_archive(
        job,
        archive_path,
        data_root_root,
        mask_root,
        epochs,
        allow_short_run,
    )
    payload = {
        "status": "success",
        "return_code": 0,
        "elapsed_seconds": elapsed_seconds,
        "gpu": str(gpu),
        "artifacts": _artifact_hashes(
            job, archive_path, mask_root, epochs, archive_sha256
        ),
    }
    payload["status_sha256"] = _status_checksum(payload)
    return payload


def _validate_completion_status(job, status, archive_path, mask_root, epochs):
    if not isinstance(status, dict):
        raise RuntimeError("status metadata is invalid")
    checksum = status.get("status_sha256")
    unsigned = dict(status)
    unsigned.pop("status_sha256", None)
    if checksum != _status_checksum(unsigned):
        raise RuntimeError("status integrity mismatch")
    if status.get("status") != "success" or status.get("return_code") != 0:
        raise RuntimeError(
            "status return_code is not successful for {}".format(
                job.output_directory
            )
        )
    expected_hashes = _artifact_hashes(job, archive_path, mask_root, epochs)
    recorded_hashes = status.get("artifacts")
    if not isinstance(recorded_hashes, dict):
        raise RuntimeError("status artifact hashes are missing")
    for name, expected in expected_hashes.items():
        if recorded_hashes.get(name) != expected:
            raise RuntimeError("{} artifact hash mismatch".format(name))
    return recorded_hashes["archive"]


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
    saved_entries = list((directory / "saved").iterdir())
    archives = [
        path
        for path in saved_entries
        if path.is_file() and path.suffix == ".npz"
    ]
    if len(archives) != 1 or len(saved_entries) != 1:
        raise RuntimeError(
            "expected exactly one NPZ archive and no other saved artifacts "
            "in {}, found {} entries".format(
                directory / "saved",
                len(saved_entries),
            )
        )
    if not _completed_log(directory / "train.log", epochs, allow_short_run):
        raise RuntimeError("train.log completion markers or epoch count are invalid")
    expected_archive_sha256 = _validate_completion_status(
        job, status, archives[0], mask_root, epochs
    )
    # Hash verification precedes allow_pickle=True so only the exact archive
    # accepted at process completion is deserialized on resume.
    _validate_archive(
        job,
        archives[0],
        data_root_root,
        mask_root,
        epochs,
        allow_short_run,
        expected_archive_sha256,
    )
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


def _claim_invocation(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / ".runner.lock"
    try:
        descriptor = os.open(
            str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
        )
    except FileExistsError as error:
        raise RuntimeError("output-root runner lock already exists: {}".format(lock)) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "runner": str(Path(__file__).resolve())}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return lock


def _release_invocation(lock):
    lock = Path(lock)
    if lock.exists():
        lock.unlink()


def _atomic_replace_text(path, encoded):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_exclusive_or_equal(path, payload):
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("immutable JSON mismatch: {}".format(path))
        return
    _atomic_replace_text(path, encoded)


def _write_json_exclusive(path, payload):
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError("immutable status already exists: {}".format(path)) from error


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
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        _release_job(job)
        raise
    return process, log_handle


def _terminate_process_group(process, timeout=5.0):
    if process.poll() is not None:
        process.wait()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=timeout)
    else:
        # The group leader may exit on TERM while a child ignores it.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


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
                if return_code == 0:
                    archives = list((job.output_directory / "saved").glob("*.npz"))
                    if len(archives) != 1:
                        raise RuntimeError(
                            "expected exactly one NPZ archive after training, found {}".format(
                                len(archives)
                            )
                        )
                    status = _build_completion_status(
                        job,
                        gpu,
                        archives[0],
                        python,
                        repository,
                        data_root_root,
                        mask_root,
                        epochs,
                        allow_short_run,
                        time.time() - started_at,
                    )
                else:
                    status = {
                        "status": "success" if return_code == 0 else "failed",
                        "return_code": return_code,
                        "elapsed_seconds": time.time() - started_at,
                        "gpu": gpu,
                    }
                    status["status_sha256"] = _status_checksum(status)
                _write_json_exclusive(
                    job.output_directory / "status.json", status
                )
                _release_job(job)
                if return_code != 0:
                    failures.append(str(job.output_directory))
            running = survivors
    finally:
        for job, _, process, handle, _ in running:
            _terminate_process_group(process)
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


def _fingerprint_uncached(path):
    path = Path(path).resolve()
    if not path.exists():
        raise RuntimeError("dataset artifact does not exist: {}".format(path))
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    if path.is_file():
        files = iter((path,))
    else:
        def walk_files():
            for root, directories, filenames in os.walk(str(path)):
                directories.sort()
                for filename in sorted(filenames):
                    candidate = Path(root) / filename
                    if candidate.is_file():
                        yield candidate

        files = walk_files()
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
    if file_count == 0:
        raise RuntimeError("dataset artifact contains no files: {}".format(path))
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": total_size,
        "file_count": file_count,
    }


def _fingerprint(path, cache):
    resolved = str(Path(path).resolve())
    if resolved not in cache:
        cache[resolved] = _fingerprint_uncached(resolved)
    return cache[resolved]


def _dataset_manifest(data_root_root, dataset, fingerprint_cache):
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
            "label": _fingerprint(
                root / LABEL_FILENAMES[dataset], fingerprint_cache
            ),
            "audio": _fingerprint(
                features / FEATURE_NAMES["audio"], fingerprint_cache
            ),
            "text": _fingerprint(
                features / FEATURE_NAMES["text"], fingerprint_cache
            ),
            "video": _fingerprint(
                features / FEATURE_NAMES["video"], fingerprint_cache
            ),
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
    artifact_scope = "smoke" if stage == "smoke" else "formal"
    manifest = {
        "git": {"head": output(["git", "rev-parse", "HEAD"]), "clean": True},
        "artifact_scope": artifact_scope,
        "python": interpreter["python"],
        "versions": interpreter["versions"],
        "gpu_names": gpu_names,
        "roots": {
            "repository": str(Path(repository).resolve()),
            "data_root_root": str(Path(data_root_root).resolve()),
            "mask_root": str(Path(mask_root).resolve()),
        },
        "locked_training": LOCKED_TRAINING,
        "arms": {name: list(modes) for name, modes in ARMS.items()},
        "parameter_counts": PARAMETER_COUNTS,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
        },
    }
    path = Path(output_root) / artifact_scope / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_json_exclusive_or_equal(path, manifest)
    except (OSError, RuntimeError) as error:
        raise RuntimeError("run manifest mismatch: {}".format(path)) from error
    dataset_records = {}
    fingerprint_cache = {}
    dataset_root = path.parent / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        record = _dataset_manifest(data_root_root, dataset, fingerprint_cache)
        dataset_path = dataset_root / "{}.json".format(dataset)
        try:
            _write_json_exclusive_or_equal(dataset_path, record)
        except RuntimeError as error:
            raise RuntimeError(
                "run manifest mismatch for dataset {}".format(dataset)
            ) from error
        dataset_records[dataset] = record
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
    result = dict(manifest)
    result["datasets"] = dataset_records
    return result


def _create_runner_parser():
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
    return parser


def _execute_invocation(args):
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


def main():
    args = _create_runner_parser().parse_args()
    lock = _claim_invocation(args.output_root)
    try:
        _execute_invocation(args)
    finally:
        _release_invocation(lock)


if __name__ == "__main__":
    main()
