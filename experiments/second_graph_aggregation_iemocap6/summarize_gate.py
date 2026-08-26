"""Validate, pair, summarize, and gate second-graph aggregation experiments.

NPZ archives contain pickled ``argparse.Namespace`` values.  They are opened
with pickle enabled only after the surrounding immutable job artifacts, file
type, symlink policy, command, status, and completion log have been validated.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from experiments.mpfilm_iemocap6.run_locked_ab import (
    ARM_TO_BRANCH_FUSION,
    ARM_TO_GRAPH_VARIANT,
    ARM_TO_RELATION_TRACK_ROUTING,
    ARM_TO_SECOND_GRAPH_AGGREGATION,
    Job,
    LOCKED_TRAINING,
    _completed_log,
    _job_payload,
)


CANDIDATE_ARMS = ("genagg", "soft_medoid", "ssma", "rtdr")
RATES = (0.0, 0.7)
SEEDS = (66, 67, 68)

# These totals are locked by GraphModel's full-size parameter-count regression
# test.  The inherited Original predates the later wrapper parameters and stores
# its selected path count in ``parameter_count``.
ORIGINAL_ARCHIVE_PARAMETER_COUNT = 34_140_166
CURRENT_BASE_PARAMETER_COUNT = 36_419_816
SELECTED_BASE_PARAMETER_COUNT = 34_140_166
PARAMETER_DELTAS = {
    "genagg": 118,
    "soft_medoid": 0,
    "ssma": 595_400,
    "rtdr": 0,
}
STORED_PARAMETER_COUNTS = {
    "original": ORIGINAL_ARCHIVE_PARAMETER_COUNT,
    **{
        arm: CURRENT_BASE_PARAMETER_COUNT + delta
        for arm, delta in PARAMETER_DELTAS.items()
    },
}
SELECTED_PARAMETER_COUNTS = {
    "original": SELECTED_BASE_PARAMETER_COUNT,
    **{
        arm: SELECTED_BASE_PARAMETER_COUNT + delta
        for arm, delta in PARAMETER_DELTAS.items()
    },
}
LEGACY_DEFAULTS = {
    "pre_graph_context": "bilstm",
    "post_graph_context": "bilstm",
    "branch_fusion": "addition",
    "second_graph_aggregation": "add",
    "relation_track_routing": "early",
}
LEGACY_COMMAND_DEFAULTS = {
    "--pre-graph-context": "bilstm",
    "--post-graph-context": "bilstm",
    "--branch-fusion": "addition",
    "--second-graph-aggregation": "add",
    "--relation-track-routing": "early",
}
OFFICIAL_PYTHON = "/data2/yb/reproduction_envs/gcnet-official/bin/python"
OFFICIAL_DATA_ROOT = "/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset/IEMOCAP"
OFFICIAL_MASK_ROOT = "/data2/yb/paper/experiments/mpfilm_iemocap6_20260824/mask_banks"
HISTORICAL_REPOSITORY = "/data2/yb/paper/GCNet_cp_lecc_20260824"
HISTORICAL_INITIAL_DATA_ROOT = HISTORICAL_REPOSITORY + "/dataset/IEMOCAP"
HISTORICAL_HEAD = "d64fa9b6003d9a37fef5f135ce194fd206baac2a"
PHASE_A_HEAD = "24ea3e7bfb65621d48d935291cb233db69f54dcc"
OFFICIAL_PYTHON_RECORD = {
    "executable": OFFICIAL_PYTHON,
    "requested": OFFICIAL_PYTHON,
    "version": "3.8.20",
}
OFFICIAL_VERSIONS = {
    "cuda": "10.2",
    "cudnn": 7605,
    "torch": "1.8.0",
    "torch_geometric": "2.0.1",
}
LOCKED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
}


def _historical_invocations() -> Tuple[dict, ...]:
    common = {
        "fold": 5,
        "gpus": ["0", "1", "2", "3"],
        "seeds": [66, 67, 68, 69, 70],
        "stage": "formal",
        "workers_per_gpu": 3,
    }
    return (
        dict(common, arms=["original"], rates=[0.5, 0.7], job_count=10),
        dict(common, arms=["original", "full"], rates=[0.0, 0.1, 0.3], job_count=30),
        dict(common, arms=["original", "full"], rates=[0.2, 0.4, 0.6], job_count=30),
        dict(common, arms=["full", "cp_lecc"], rates=[0.5, 0.7], job_count=20),
    )


def _candidate_invocation(arm: str) -> dict:
    phase_arms = (
        ["genagg", "soft_medoid"]
        if arm in ("genagg", "soft_medoid")
        else ["ssma", "rtdr"]
    )
    return {
        "arms": phase_arms,
        "fold": 5,
        "job_count": 12,
        "parallel_arms": True,
        "rates": [0.0, 0.7],
        "seeds": [66, 67, 68],
        "stage": "formal",
        "workers_per_gpu": 3,
    }


def _rtdr_extension_invocation() -> dict:
    return {
        "arms": ["rtdr"],
        "fold": 5,
        "gpus": ["5", "6", "7"],
        "job_count": 15,
        "parallel_arms": True,
        "rates": [0.0, 0.5, 0.7],
        "seeds": [66, 67, 68, 69, 70],
        "stage": "formal",
        "workers_per_gpu": 3,
    }


def _rtdr_full_invocation() -> dict:
    return {
        "arms": ["rtdr"],
        "fold": 5,
        "gpus": ["5", "6", "7"],
        "job_count": 40,
        "parallel_arms": True,
        "rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "seeds": [66, 67, 68, 69, 70],
        "stage": "formal",
        "workers_per_gpu": 3,
    }


def _phase_a_uniform_invocation() -> dict:
    return {
        "arms": ["genagg", "soft_medoid"],
        "fold": 5,
        "gpus": ["1", "2", "3", "4", "5", "6", "7"],
        "job_count": 30,
        "parallel_arms": True,
        "rates": [0.0, 0.5, 0.7],
        "seeds": [66, 67, 68, 69, 70],
        "stage": "formal",
        "workers_per_gpu": 3,
    }


def _ssma_uniform_invocation() -> dict:
    return {
        "arms": ["ssma"],
        "fold": 5,
        "gpus": ["5", "6", "7"],
        "job_count": 15,
        "parallel_arms": True,
        "rates": [0.0, 0.5, 0.7],
        "seeds": [66, 67, 68, 69, 70],
        "stage": "formal",
        "workers_per_gpu": 3,
    }


def _valid_gpu_list(gpus: Any) -> bool:
    return (
        isinstance(gpus, list)
        and bool(gpus)
        and len(set(gpus)) == len(gpus)
        and all(isinstance(gpu, str) and gpu.isdigit() for gpu in gpus)
    )


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.size == 1 else value


def _concatenate(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value
    pieces = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
    return np.concatenate([np.asarray(piece) for piece in pieces], axis=0)


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> Tuple[float, float]:
    accuracy = float(np.mean(labels == predictions))
    scores = []
    supports = []
    for label in np.unique(labels):
        true_positive = float(np.sum((labels == label) & (predictions == label)))
        false_positive = float(np.sum((labels != label) & (predictions == label)))
        false_negative = float(np.sum((labels == label) & (predictions != label)))
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0.0 else 2.0 * true_positive / denominator)
        supports.append(float(np.sum(labels == label)))
    return float(np.average(np.asarray(scores), weights=np.asarray(supports))), accuracy


def _selected_snapshot(data: Mapping[str, Any]) -> Tuple[int, Mapping[str, Any]]:
    saves = data["folder_savewhole"]
    if len(saves) != 1 or len(saves[0]) < 2:
        raise ValueError("folder_savewhole must contain one fold and a selected snapshot")
    epoch = int(_scalar(saves[0][0]))
    snapshot = saves[0][-1]
    if isinstance(snapshot, np.ndarray) and snapshot.shape == ():
        snapshot = snapshot.item()
    if not isinstance(snapshot, Mapping):
        raise ValueError("selected folder_savewhole snapshot is not a mapping")
    return epoch, snapshot


def _locked_paths(command: Sequence[str]) -> Tuple[Path, Path, Path, Path]:
    if not isinstance(command, list) or len(command) < 3:
        raise ValueError("command payload is not a complete command list")
    try:
        python = Path(command[0])
        train_script = Path(command[2])
        repository = train_script.parents[1]
        data_root = Path(command[command.index("--data-root") + 1])
        mask_root = Path(command[command.index("--mask-bank-root") + 1])
    except (ValueError, IndexError):
        raise ValueError("command list is missing locked paths") from None
    if train_script != repository / "gcnet" / "train_gcnet.py":
        raise ValueError("command training script mismatch: {}".format(train_script))
    return python, repository, data_root, mask_root


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_invocations(
    formal: Path, arm: str, historical: bool, rate: float, seed: int
) -> None:
    directory = formal / "invocations"
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("invocation directory is missing or unsafe: {}".format(directory))
    paths = sorted(directory.glob("*.json"))
    expected = _historical_invocations() if historical else (_candidate_invocation(arm),)
    phase_a = not historical and arm in ("genagg", "soft_medoid")
    phase_b = not historical and arm in ("ssma", "rtdr")
    valid_counts = (1, 2) if phase_a else ((1, 2, 3, 4) if phase_b else (len(expected),))
    if len(paths) not in valid_counts:
        raise ValueError(
            "invocation set mismatch: expected {}, found {}".format(
                " or ".join(str(value) for value in valid_counts), len(paths)
            )
        )
    found = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("invocation file is missing or unsafe: {}".format(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name != _canonical_digest(payload) + ".json":
            raise ValueError("invocation filename hash mismatch: {}".format(path))
        found.append(payload)
    if historical:
        expected_by_digest = {_canonical_digest(payload): payload for payload in expected}
        found_by_digest = {_canonical_digest(payload): payload for payload in found}
        if found_by_digest != expected_by_digest:
            raise ValueError("invocation payload mismatch for {}".format(formal))
    else:
        base_matches = []
        for payload in found:
            normalized = dict(payload)
            gpus = normalized.pop("gpus", None)
            if normalized == expected[0] and _valid_gpu_list(gpus):
                base_matches.append(payload)
        if len(base_matches) != 1:
            raise ValueError("invocation payload mismatch for {}".format(formal))
        remaining = [payload for payload in found if payload is not base_matches[0]]
        if phase_a:
            expected_remaining = (
                [] if not remaining else [_phase_a_uniform_invocation()]
            )
        else:
            expected_chain = [
                _rtdr_extension_invocation(),
                _rtdr_full_invocation(),
                _ssma_uniform_invocation(),
            ]
            expected_remaining = expected_chain[: len(remaining)]
        remaining_by_digest = {
            _canonical_digest(payload): payload for payload in remaining
        }
        expected_by_digest = {
            _canonical_digest(payload): payload for payload in expected_remaining
        }
        if remaining_by_digest != expected_by_digest:
            raise ValueError("invocation payload mismatch for {}".format(formal))
    if not any(
        arm in payload["arms"]
        and float(rate) in payload["rates"]
        and int(seed) in payload["seeds"]
        for payload in found
    ):
        raise ValueError(
            "invocation set does not cover arm={} rate={} seed={}".format(
                arm, rate, seed
            )
        )


def _validate_run_manifest(
    fold_directory: Path,
    arm: str,
    historical: bool,
    rate: float,
    seed: int,
) -> Mapping[str, Any]:
    # Locked layout: formal/<arm>/miss_*/seed_*/fold_5.
    try:
        formal = fold_directory.parents[3]
    except IndexError:
        raise ValueError("run manifest path cannot be derived from job directory") from None
    if formal.name != "formal" or fold_directory.parents[2].name != arm:
        raise ValueError("run manifest layout mismatch for {}".format(fold_directory))
    path = formal / "run_manifest.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("run manifest is missing or unsafe: {}".format(path))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {
        "environment",
        "fold",
        "git",
        "gpu_names",
        "locked_training",
        "python",
        "roots",
        "stage",
        "versions",
    }
    if set(manifest) != required_keys:
        raise ValueError("run manifest keys mismatch")
    git = manifest.get("git", {})
    head = git.get("head")
    if historical:
        if head != HISTORICAL_HEAD:
            raise ValueError("run manifest git head mismatch")
    elif not isinstance(head, str) or len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise ValueError("run manifest git head is not a full clean commit")
    if git.get("clean") is not True:
        raise ValueError("run manifest git clean mismatch")
    if manifest.get("python") != OFFICIAL_PYTHON_RECORD:
        raise ValueError("run manifest python executable mismatch")
    versions = manifest.get("versions", {})
    for field, expected in OFFICIAL_VERSIONS.items():
        if versions.get(field) != expected:
            raise ValueError("run manifest {} mismatch".format(field))
    if manifest.get("locked_training") != LOCKED_TRAINING:
        raise ValueError("run manifest locked_training mismatch")
    if manifest.get("environment") != LOCKED_ENVIRONMENT:
        raise ValueError("run manifest environment mismatch")
    if manifest.get("stage") != "formal" or manifest.get("fold") != 5:
        raise ValueError("run manifest stage/fold mismatch")
    roots = manifest.get("roots", {})
    expected_fixed_roots = {
        "data": OFFICIAL_DATA_ROOT,
        "mask_bank": OFFICIAL_MASK_ROOT,
    }
    for field, expected in expected_fixed_roots.items():
        if roots.get(field) != expected:
            raise ValueError("run manifest roots.{} mismatch".format(field))
    if historical and roots.get("repository") != HISTORICAL_REPOSITORY:
        raise ValueError("run manifest roots.repository mismatch")
    repository = roots.get("repository")
    if not isinstance(repository, str) or not Path(repository).is_absolute():
        raise ValueError("run manifest roots.repository mismatch")
    gpu_names = manifest.get("gpu_names")
    if not isinstance(gpu_names, list) or not gpu_names or any(
        name != "Tesla V100-SXM2-32GB" for name in gpu_names
    ):
        raise ValueError("run manifest gpu_names mismatch")
    _validate_invocations(formal, arm, historical, rate, seed)
    return manifest


def _without_legacy_defaults(command: Sequence[str]) -> list:
    normalized = list(command)
    for flag, default in LEGACY_COMMAND_DEFAULTS.items():
        while flag in normalized:
            index = normalized.index(flag)
            if index + 1 >= len(normalized) or normalized[index + 1] != default:
                raise ValueError("historical command {} is not locked default".format(flag))
            del normalized[index : index + 2]
    return normalized


def _command_original_fold(
    command: Sequence[str], relocated_fold: Path, arm: str, rate: float, seed: int
) -> Path:
    if command.count("--output-dir") != 1:
        raise ValueError("command must contain exactly one --output-dir")
    index = command.index("--output-dir")
    if index + 1 >= len(command):
        raise ValueError("command --output-dir is missing its value")
    raw_output = str(command[index + 1])
    output = Path(raw_output)
    if not output.is_absolute() or ".." in output.parts or output.name != "saved":
        raise ValueError("command original output-dir is not a safe absolute saved path")
    original_fold = output.parent
    expected_suffix = (
        "formal",
        arm,
        "miss_{}".format(_rate_tag(rate)),
        "seed_{}".format(seed),
        "fold_5",
    )
    original_suffix = tuple(original_fold.parts[-5:])
    relocated_suffix = tuple(Path(relocated_fold).parts[-5:])
    if original_suffix != expected_suffix or relocated_suffix != expected_suffix:
        raise ValueError(
            "relocated artifact suffix mismatch: original={!r}, relocated={!r}, expected={!r}".format(
                original_suffix, relocated_suffix, expected_suffix
            )
        )
    return original_fold


def _validated_archive(
    fold_directory: Path,
    arm: str,
    rate: float,
    seed: int,
    historical_original: bool,
) -> Tuple[Path, float, Mapping[str, Any]]:
    fold_directory = Path(fold_directory)
    if arm not in ("original",) + CANDIDATE_ARMS:
        raise ValueError("unknown arm: {!r}".format(arm))
    if historical_original != (arm == "original"):
        raise ValueError("historical_original must be used exactly for inherited Original")
    if not fold_directory.is_dir() or fold_directory.is_symlink():
        raise ValueError("job directory is missing or unsafe: {}".format(fold_directory))
    manifest = _validate_run_manifest(
        fold_directory,
        arm,
        historical=historical_original,
        rate=rate,
        seed=seed,
    )
    lock = fold_directory / ".active.lock"
    if lock.exists():
        raise ValueError("active or stale lock present: {}".format(lock))
    required = ("command.json", "status.json", "train.log", "saved")
    missing = [name for name in required if not (fold_directory / name).exists()]
    if missing:
        raise ValueError("missing job artifacts in {}: {}".format(fold_directory, missing))
    saved = fold_directory / "saved"
    if not saved.is_dir() or saved.is_symlink():
        raise ValueError("saved directory is missing or unsafe: {}".format(saved))
    archives = sorted(saved.glob("*.npz"))
    if len(archives) != 1:
        raise ValueError("expected exactly one NPZ archive in {}, found {}".format(saved, len(archives)))
    archive = archives[0]
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("archive is missing or unsafe: {}".format(archive))

    payload = json.loads((fold_directory / "command.json").read_text(encoding="utf-8"))
    command = payload.get("command")
    python, repository, data_root, mask_root = _locked_paths(command)
    original_fold = _command_original_fold(command, fold_directory, arm, rate, seed)
    manifest_roots = manifest["roots"]
    expected_command_data_root = manifest_roots["data"]
    if historical_original and float(rate) in (0.5, 0.7):
        # The first historical Original invocation used the repository-local
        # data path.  Later recovery invocations used the manifest root.  This
        # exception is rate-locked and the full command remains exact.
        expected_command_data_root = HISTORICAL_INITIAL_DATA_ROOT
    path_bindings = {
        "python executable": (str(python), manifest["python"]["executable"]),
        "roots.repository": (str(repository), manifest_roots["repository"]),
        "roots.data": (str(data_root), expected_command_data_root),
        "roots.mask_bank": (str(mask_root), manifest_roots["mask_bank"]),
    }
    for label, (actual, expected_path) in path_bindings.items():
        if actual != expected_path:
            raise ValueError(
                "command {} mismatch: {!r} != {!r}".format(label, actual, expected_path)
            )
    job = Job("formal", arm, float(rate), int(seed), original_fold)
    expected = _job_payload(
        job, str(payload.get("gpu")), python, repository, data_root, mask_root
    )
    if historical_original:
        actual_command = _without_legacy_defaults(payload.get("command"))
        expected_command = _without_legacy_defaults(expected["command"])
        actual_payload = dict(payload, command=actual_command)
        expected_payload = dict(expected, command=expected_command)
    else:
        actual_payload, expected_payload = payload, expected
    if actual_payload != expected_payload:
        raise ValueError("command.json mismatch for {}".format(fold_directory))

    status = json.loads((fold_directory / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "success" or status.get("return_code") != 0:
        raise ValueError("status return_code is not successful for {}".format(fold_directory))
    runtime = status.get("elapsed_seconds")
    if not isinstance(runtime, (int, float)) or not math.isfinite(float(runtime)) or runtime < 0:
        raise ValueError("status elapsed_seconds is invalid for {}".format(fold_directory))
    if not _completed_log(fold_directory / "train.log"):
        raise ValueError("train.log does not contain exactly 100 epoch records and completion markers")
    return archive, float(runtime), manifest


def _archive_metrics(archive: Path) -> Dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(str(archive), flags)
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            with np.load(handle, allow_pickle=True) as data:
                if "folder_losswhole" not in data.files:
                    raise ValueError("folder_losswhole is missing")
                losses = data["folder_losswhole"]
                if len(losses) != 1 or len(losses[0]) != 100:
                    raise ValueError(
                        "folder_losswhole must contain one fold and exactly 100 epochs"
                    )
                if any(not isinstance(epoch_loss, Mapping) for epoch_loss in losses[0]):
                    raise ValueError("folder_losswhole epoch entries must be mappings")
                epoch, snapshot = _selected_snapshot(data)
                labels = _concatenate(snapshot["test_labels"]).reshape(-1)
                scores = _concatenate(snapshot["test_preds"])
                if scores.ndim != 2 or scores.shape[1] != 6:
                    raise ValueError(
                        "test prediction scores must have shape [N, six columns]"
                    )
                if not np.all(np.isfinite(scores)):
                    raise ValueError("test prediction scores must be finite")
                try:
                    numeric_labels = labels.astype(np.float64)
                except (TypeError, ValueError):
                    raise ValueError("test label range is invalid") from None
                if (
                    not np.all(np.isfinite(numeric_labels))
                    or not np.all(numeric_labels == np.floor(numeric_labels))
                    or np.any(numeric_labels < 0)
                    or np.any(numeric_labels > 5)
                ):
                    raise ValueError("test label range must be integer labels 0..5")
                predictions = np.argmax(scores, axis=-1).reshape(-1)
                if labels.size == 0 or labels.shape != predictions.shape:
                    raise ValueError("label/prediction shape mismatch: {} != {}".format(labels.shape, predictions.shape))
                weighted_f1, accuracy = _classification_metrics(labels, predictions)
                counts = np.unique(predictions, return_counts=True)[1]
                manifest = _scalar(data["mask_bank_manifest"])
                if not isinstance(manifest, Mapping):
                    raise ValueError("mask_bank_manifest is not a mapping")
                result = {
                    "weighted_f1": weighted_f1,
                    "accuracy": accuracy,
                    "class_coverage": int(len(np.unique(predictions))),
                    "dominant_ratio": float(counts.max() / predictions.size),
                    "epoch": epoch,
                    "mask_sha256": str(manifest["sha256"]),
                    "requested_missing_rate": float(manifest["requested_missing_rate"]),
                    "manifest_seed": int(manifest["seed"]),
                    "parameter_count": int(_scalar(data["parameter_count"])),
                    "fold_numbers": [int(value) for value in np.asarray(data["fold_numbers"]).flat],
                    "smoke_only": bool(_scalar(data["smoke_only"])),
                    "args": _scalar(data["args"]),
                }
                if "selected_path_parameter_count" in data.files:
                    result["selected_path_parameter_count"] = int(
                        _scalar(data["selected_path_parameter_count"])
                    )
                return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def collect_job(
    fold_directory: Path,
    arm: str,
    rate: float,
    seed: int,
    historical_original: bool = False,
) -> Dict[str, Any]:
    """Validate one complete job before opening its sole trusted archive."""
    archive, runtime, manifest = _validated_archive(
        fold_directory, arm, rate, seed, historical_original
    )
    row = _archive_metrics(archive)
    arguments = row.pop("args")
    expected_args = dict(LOCKED_TRAINING)
    expected_args.pop("fold")
    expected_args.update(
        graph_conv_variant=ARM_TO_GRAPH_VARIANT[arm],
        seed=int(seed),
        mask_seed=int(seed),
        mask_type="constant-{:.1f}".format(rate),
        fold_index=5,
    )
    mechanism_args = dict(
        pre_graph_context="bilstm",
        post_graph_context="bilstm",
        branch_fusion=ARM_TO_BRANCH_FUSION[arm],
        second_graph_aggregation=ARM_TO_SECOND_GRAPH_AGGREGATION.get(arm, "add"),
        relation_track_routing=ARM_TO_RELATION_TRACK_ROUTING.get(arm, "early"),
    )
    expected_args.update(mechanism_args)
    for field, expected in expected_args.items():
        phase_a_legacy_relation = (
            arm in ("genagg", "soft_medoid")
            and manifest["git"]["head"] == PHASE_A_HEAD
            and field == "relation_track_routing"
        )
        if (
            (historical_original and field in LEGACY_DEFAULTS)
            or phase_a_legacy_relation
        ) and not hasattr(arguments, field):
            actual = LEGACY_DEFAULTS[field]
        else:
            if not hasattr(arguments, field):
                raise ValueError("archive provenance {} is missing".format(field))
            actual = getattr(arguments, field)
        if not _matches(actual, expected):
            raise ValueError("archive provenance {} mismatch: {!r} != {!r}".format(field, actual, expected))

    if historical_original and "selected_path_parameter_count" not in row:
        row["selected_path_parameter_count"] = SELECTED_PARAMETER_COUNTS["original"]
    expected_fields = {
        "fold_numbers": [5],
        "smoke_only": False,
        "requested_missing_rate": float(rate),
        "manifest_seed": int(seed),
        "parameter_count": STORED_PARAMETER_COUNTS[arm],
        "selected_path_parameter_count": SELECTED_PARAMETER_COUNTS[arm],
    }
    for field, expected in expected_fields.items():
        if field not in row or not _matches(row[field], expected):
            raise ValueError("archive provenance {} mismatch: {!r} != {!r}".format(field, row.get(field), expected))
    row.update(
        arm=arm,
        rate=float(rate),
        seed=int(seed),
        runtime_seconds=runtime,
        graph_conv_variant=expected_args["graph_conv_variant"],
        **mechanism_args
    )
    return row


def _rate_tag(rate: float) -> str:
    return "{:.1f}".format(rate).replace(".", "p")


def _formal_root(root: Path) -> Path:
    root = Path(root)
    return root if root.name == "formal" else root / "formal"


def _candidate_job_path(root: Path, arm: str, rate: float, seed: int) -> Path:
    suffix = Path(arm) / "miss_{}".format(_rate_tag(rate)) / "seed_{}".format(seed) / "fold_5"
    path = _formal_root(root) / suffix
    if not path.exists():
        raise ValueError("candidate job path for {} rate={} seed={} is missing: {}".format(arm, rate, seed, path))
    return path


def _original_job_path(root: Path, rate: float, seed: int) -> Path:
    root = Path(root)
    base = root if root.name == "original" else root / "original"
    return base / "miss_{}".format(_rate_tag(rate)) / "seed_{}".format(seed) / "fold_5"


def collect_grid(
    original_root: Path,
    phase_a_root: Path,
    phase_b_root: Path,
    rates: Sequence[float] = RATES,
    seeds: Sequence[int] = SEEDS,
) -> Dict[str, list]:
    rows = {"original": []}
    rows.update({arm: [] for arm in CANDIDATE_ARMS})
    arm_roots = {
        "genagg": Path(phase_a_root),
        "soft_medoid": Path(phase_a_root),
        "ssma": Path(phase_b_root),
        "rtdr": Path(phase_b_root),
    }
    for arm in ("genagg", "soft_medoid"):
        if (_formal_root(phase_b_root) / arm).exists():
            raise ValueError("Phase A arm {} was found under Phase B root".format(arm))
    for arm in ("ssma", "rtdr"):
        if (_formal_root(phase_a_root) / arm).exists():
            raise ValueError("Phase B arm {} was found under Phase A root".format(arm))
    for rate in rates:
        for seed in seeds:
            original = collect_job(
                _original_job_path(original_root, rate, seed),
                "original",
                float(rate),
                int(seed),
                historical_original=True,
            )
            rows["original"].append(original)
            for arm in CANDIDATE_ARMS:
                candidate = collect_job(
                    _candidate_job_path(arm_roots[arm], arm, rate, seed),
                    arm,
                    float(rate),
                    int(seed),
                )
                if candidate["mask_sha256"] != original["mask_sha256"]:
                    raise ValueError("paired mask SHA256 mismatch for {} at rate={}, seed={}".format(arm, rate, seed))
                rows[arm].append(candidate)
    return rows


def index_rows(
    rows: Iterable[Mapping[str, Any]],
    rates: Sequence[float] = RATES,
    seeds: Sequence[int] = SEEDS,
) -> Dict[Tuple[float, int], Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        key = (float(row["rate"]), int(row["seed"]))
        if key in indexed:
            raise ValueError("duplicate result key: rate={}, seed={}".format(*key))
        indexed[key] = row
    expected = {(float(rate), int(seed)) for rate in rates for seed in seeds}
    if set(indexed) != expected:
        raise ValueError("result grid mismatch: expected {}, found {}".format(sorted(expected), sorted(indexed)))
    return indexed


def _finite_run(row: Mapping[str, Any]) -> bool:
    fields = ("weighted_f1", "accuracy", "dominant_ratio", "runtime_seconds")
    return all(math.isfinite(float(row[field])) for field in fields)


def _noncollapsed_run(row: Mapping[str, Any]) -> bool:
    return int(row["class_coverage"]) == 6 and float(row["dominant_ratio"]) < 1.0


def summarize_candidate(
    arm: str,
    original_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    rates: Sequence[float] = RATES,
    seeds: Sequence[int] = SEEDS,
) -> Dict[str, Any]:
    original = index_rows(original_rows, rates, seeds)
    candidate = index_rows(candidate_rows, rates, seeds)
    tasks = []
    rate_means = {}
    seed_deltas = {int(seed): [] for seed in seeds}
    all_finite = True
    all_noncollapsed = True
    for rate in rates:
        deltas = []
        for seed in seeds:
            key = (float(rate), int(seed))
            left, right = original[key], candidate[key]
            if left["mask_sha256"] != right["mask_sha256"]:
                raise ValueError("paired mask SHA256 mismatch for {} at rate={}, seed={}".format(arm, rate, seed))
            delta = float(right["weighted_f1"]) - float(left["weighted_f1"])
            deltas.append(delta)
            seed_deltas[int(seed)].append(delta)
            all_finite = all_finite and _finite_run(right)
            all_noncollapsed = all_noncollapsed and _noncollapsed_run(right)
            tasks.append(
                {
                    "rate": float(rate),
                    "seed": int(seed),
                    "original": dict(left),
                    "candidate": dict(right),
                    "delta_weighted_f1": delta,
                }
            )
        rate_means[str(float(rate))] = {
            "original_mean": float(np.mean([original[(float(rate), int(seed))]["weighted_f1"] for seed in seeds])),
            "candidate_mean": float(np.mean([candidate[(float(rate), int(seed))]["weighted_f1"] for seed in seeds])),
            "mean_delta": float(np.mean(np.asarray(deltas, dtype=np.float64))),
        }
    seed_macro = {
        str(seed): float(np.mean(np.asarray(seed_deltas[int(seed)], dtype=np.float64)))
        for seed in seeds
    }
    macro_delta = float(np.mean(np.asarray(list(seed_macro.values()), dtype=np.float64)))
    positive_seeds = sum(value > 0.0 for value in seed_macro.values())
    minimum_positive_seeds = (len(seeds) + 1) // 2
    positive_rates = all(row["mean_delta"] > 0.0 for row in rate_means.values())
    passed = (
        all_finite
        and all_noncollapsed
        and positive_rates
        and macro_delta > 0.0
        and positive_seeds >= minimum_positive_seeds
    )
    return {
        "arm": arm,
        "tasks": tasks,
        "rate_means": rate_means,
        "seed_macro_deltas": seed_macro,
        "macro_delta": macro_delta,
        "gate": {
            "passed": bool(passed),
            "all_finite": bool(all_finite),
            "all_noncollapsed": bool(all_noncollapsed),
            "both_rate_means_positive": bool(positive_rates),
            "all_rate_means_positive": bool(positive_rates),
            "seed_macro_positive": bool(macro_delta > 0.0),
            "positive_seed_macros": int(positive_seeds),
            "minimum_positive_seed_macros": int(minimum_positive_seeds),
        },
    }


def summarize_grid(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    return {
        "experiment": "Second graph mechanisms vs inherited Original, locked IEMOCAPSix fold 5",
        "rates": list(RATES),
        "seeds": list(SEEDS),
        "candidates": {
            arm: summarize_candidate(arm, rows["original"], rows[arm])
            for arm in CANDIDATE_ARMS
        },
    }


def _render(summary: Mapping[str, Any], language: str) -> str:
    title = {
        "main": "# Second Graph Aggregation Results",
        "zh": "# 第二层图机制配对实验结果",
        "en": "# Second-Graph Mechanism Paired Results",
    }[language]
    if language == "zh":
        lines = [
            title,
            "",
            "PASS 表示该候选模块满足全部预注册晋级条件；FAIL 表示至少一项条件未满足。",
            "",
            "| 候选模块 | 是否晋级 | seed 宏平均 F1 差值 |",
            "|---|---:|---:|",
        ]
    else:
        lines = [
            title,
            "",
            "PASS means every preregistered advancement condition was satisfied; FAIL means at least one condition was not.",
            "",
            "| candidate | gate | seed-macro F1 delta |",
            "|---|---:|---:|",
        ]
    for arm, result in summary.get("candidates", {}).items():
        gate = result.get("gate", {})
        verdict = "PASS" if gate.get("passed") else "FAIL"
        macro = result.get("macro_delta")
        rendered = "NA" if macro is None else "{:+.9f}".format(float(macro))
        lines.append("| {} | {} | {} |".format(arm, verdict, rendered))
    lines.append("")
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(path.parent), delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(output_directory: Path, summary: Mapping[str, Any]) -> None:
    output_directory = Path(output_directory)
    _write_atomic(output_directory / "summary.json", json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_atomic(output_directory / "RESULTS.md", _render(summary, "main"))
    _write_atomic(output_directory / "RESULTS.zh.md", _render(summary, "zh"))
    _write_atomic(output_directory / "RESULTS.en.md", _render(summary, "en"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and gate trusted second-graph experiments")
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    rows = collect_grid(arguments.original_root, arguments.phase_a_root, arguments.phase_b_root)
    write_outputs(arguments.output_dir, summarize_grid(rows))


if __name__ == "__main__":
    main()
