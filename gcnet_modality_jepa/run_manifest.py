"""Versioned, auditable run manifests for paired GCNet experiments."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


MANIFEST_NAME = "gcnet-unified-run-manifest"
MANIFEST_VERSION = 1


class ManifestValidationError(ValueError):
    """Raised when a run manifest is missing required audit evidence."""


_REQUIRED_PATHS = (
    "schema.name",
    "schema.version",
    "run.dataset",
    "run.fold",
    "run.master_seed",
    "environment.python",
    "environment.torch",
    "environment.cuda",
    "environment.cudnn",
    "environment.pyg",
    "environment.numpy",
    "environment.sklearn",
    "environment.gpu.index",
    "environment.gpu.model",
    "environment.gpu.driver",
    "provenance.command",
    "provenance.cwd",
    "provenance.git_revision",
    "provenance.git_status",
    "features.audio.path",
    "features.audio.metadata_sha256",
    "features.text.path",
    "features.text.metadata_sha256",
    "features.visual.path",
    "features.visual.metadata_sha256",
    "split.indices.train",
    "split.indices.validation",
    "split.indices.test",
    "split.hash",
    "samplers.train.seed",
    "samplers.train.signature",
    "samplers.validation.seed",
    "samplers.validation.signature",
    "samplers.test.seed",
    "samplers.test.signature",
    "masks.requested_missing_rate",
    "masks.config_hashes.train",
    "masks.config_hashes.validation",
    "masks.config_hashes.test",
    "masks.realized_missing_rates.train",
    "masks.realized_missing_rates.validation",
    "masks.realized_missing_rates.test",
    "seeds.model_init",
    "seeds.training_stochasticity",
    "seeds.split",
    "seeds.data_order.train",
    "seeds.data_order.validation",
    "seeds.data_order.test",
    "seeds.missing_mask",
    "seeds.stability_mask",
    "initialization.shared_hash",
    "stability.enabled",
    "stability.mask_rate",
    "stability.weight",
    "method.model_variant",
    "method.jepa_weight",
    "method.loss_reconstruction",
    "lifecycle.checkpoint_metric",
    "lifecycle.best_epoch",
    "lifecycle.best_validation_f1",
    "lifecycle.test_call_count",
    "metrics.weighted_f1",
    "metrics.accuracy",
    "outputs.result_archive",
)


_PAIRED_INVARIANT_PATHS = (
    "run.dataset",
    "run.fold",
    "run.master_seed",
    "environment.python",
    "environment.torch",
    "environment.cuda",
    "environment.cudnn",
    "environment.pyg",
    "environment.numpy",
    "environment.sklearn",
    "environment.gpu.model",
    "environment.gpu.driver",
    "provenance.git_revision",
    "provenance.git_status",
    "features",
    "split",
    "samplers",
    "masks",
    "seeds",
    "initialization.shared_hash",
    "stability",
    "lifecycle.checkpoint_metric",
)


PathLike = Union[str, os.PathLike]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def feature_metadata_hash(path: PathLike) -> str:
    """Hash sorted feature-tree metadata without loading feature contents."""
    root = Path(path)
    if not root.exists() and not root.is_symlink():
        raise FileNotFoundError(str(root))

    if root.is_symlink():
        paths = [root]
    elif root.is_dir():
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    else:
        paths = [root]

    digest = hashlib.sha256()
    for item in paths:
        stat_result = os.lstat(str(item))
        if item == root:
            relative = "."
        else:
            relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            kind = "symlink"
        elif item.is_dir():
            kind = "directory"
        elif item.is_file():
            kind = "file"
        else:
            kind = "other"
        record = {
            "kind": kind,
            "mtime_ns": int(stat_result.st_mtime_ns),
            "path": relative,
            "size": int(stat_result.st_size),
        }
        if kind == "symlink":
            record["target"] = os.readlink(str(item))
        encoded = _canonical_json(record)
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def sampler_signature(indices: Iterable[int], seed: int) -> str:
    """Return an order-sensitive signature of a protocol sampler definition."""
    payload = {"indices": [int(index) for index in indices], "seed": int(seed)}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    raise TypeError("value of type {} is not JSON serializable".format(type(value).__name__))


def _get_path(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ManifestValidationError("missing required manifest field {}".format(path))
        current = current[component]
    return current


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the required versioned audit contract."""
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("manifest must be a mapping")
    for path in _REQUIRED_PATHS:
        _get_path(manifest, path)
    if _get_path(manifest, "schema.name") != MANIFEST_NAME:
        raise ManifestValidationError("schema.name must be {!r}".format(MANIFEST_NAME))
    schema_version = _get_path(manifest, "schema.version")
    if type(schema_version) is not int or schema_version != MANIFEST_VERSION:
        raise ManifestValidationError(
            "schema.version must be {}".format(MANIFEST_VERSION)
        )
    def require_string(path: str, nullable: bool = False) -> None:
        value = _get_path(manifest, path)
        if nullable and value is None:
            return
        if not isinstance(value, str) or not value:
            raise ManifestValidationError("{} must be a non-empty string".format(path))

    def require_integer(path: str, minimum: int = 0, nullable: bool = False) -> None:
        value = _get_path(manifest, path)
        if nullable and value is None:
            return
        if (not isinstance(value, numbers.Integral) or isinstance(value, bool)
                or int(value) < minimum):
            raise ManifestValidationError(
                "{} must be an integer >= {}".format(path, minimum)
            )

    def require_number(
        path: str, minimum: float, maximum: Optional[float] = None
    ) -> None:
        value = _get_path(manifest, path)
        if (not isinstance(value, numbers.Real) or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < minimum
                or (maximum is not None and float(value) > maximum)):
            bound = "{}..{}".format(minimum, maximum) if maximum is not None else ">={}".format(minimum)
            raise ManifestValidationError("{} must be finite and in {}".format(path, bound))

    def require_hash(path: str) -> None:
        value = _get_path(manifest, path)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ManifestValidationError("{} must be a lowercase SHA-256 hash".format(path))

    require_string("run.dataset")
    require_integer("run.fold", minimum=1)
    require_integer("run.master_seed")
    for path in (
        "environment.python", "environment.torch", "environment.numpy",
        "environment.sklearn", "provenance.cwd", "method.model_variant",
        "lifecycle.checkpoint_metric", "outputs.result_archive",
    ):
        require_string(path)
    for path in ("environment.cuda", "environment.pyg"):
        require_string(path, nullable=True)
    require_integer("environment.cudnn", nullable=True)
    require_integer("environment.gpu.index", nullable=True)
    require_string("environment.gpu.model", nullable=True)
    require_string("environment.gpu.driver", nullable=True)
    require_string("provenance.git_revision", nullable=True)
    if _get_path(manifest, "provenance.git_status") is not None and not isinstance(
        _get_path(manifest, "provenance.git_status"), str
    ):
        raise ManifestValidationError("provenance.git_status must be a string or null")
    command = _get_path(manifest, "provenance.command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) for item in command
    ):
        raise ManifestValidationError("provenance.command must be a nonempty string list")

    for modality in ("audio", "text", "visual"):
        require_string("features.{}.path".format(modality))
        require_hash("features.{}.metadata_sha256".format(modality))

    split_groups = {}
    for split in ("train", "validation", "test"):
        path = "split.indices.{}".format(split)
        values = _get_path(manifest, path)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, numbers.Integral) or isinstance(value, bool)
            or int(value) < 0 for value in values
        ):
            raise ManifestValidationError("{} must be a nonempty integer list".format(path))
        if len(set(values)) != len(values):
            raise ManifestValidationError("{} contains duplicates".format(path))
        split_groups[split] = set(int(value) for value in values)
    if (split_groups["train"] & split_groups["validation"]
            or split_groups["train"] & split_groups["test"]
            or split_groups["validation"] & split_groups["test"]):
        raise ManifestValidationError("split.indices must be disjoint")
    combined = set().union(*split_groups.values())
    if combined != set(range(len(combined))):
        raise ManifestValidationError("split.indices must cover every dataset index")
    require_hash("split.hash")

    for split in ("train", "validation", "test"):
        require_integer("samplers.{}.seed".format(split))
        require_hash("samplers.{}.signature".format(split))
        require_hash("masks.config_hashes.{}".format(split))
    require_number("masks.requested_missing_rate", 0.0, 0.7)
    for split in ("train", "validation", "test"):
        path = "masks.realized_missing_rates.{}".format(split)
        value = _get_path(manifest, path)
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ManifestValidationError("{} must not be empty".format(path))
        for item in values:
            if (not isinstance(item, numbers.Real) or isinstance(item, bool)
                    or not math.isfinite(float(item)) or not 0.0 <= float(item) <= 1.0):
                raise ManifestValidationError("{} must contain rates in 0..1".format(path))

    for path in (
        "seeds.model_init", "seeds.training_stochasticity", "seeds.split",
        "seeds.data_order.train", "seeds.data_order.validation",
        "seeds.data_order.test", "seeds.missing_mask", "seeds.stability_mask",
    ):
        require_integer(path)
    require_hash("initialization.shared_hash")
    if not isinstance(_get_path(manifest, "stability.enabled"), bool):
        raise ManifestValidationError("stability.enabled must be Boolean")
    require_number("stability.mask_rate", 0.0, 0.7)
    require_number("stability.weight", 0.0)
    require_number("method.jepa_weight", 0.0)
    if not isinstance(_get_path(manifest, "method.loss_reconstruction"), bool):
        raise ManifestValidationError("method.loss_reconstruction must be Boolean")
    require_integer("lifecycle.best_epoch", minimum=1)
    require_number("lifecycle.best_validation_f1", 0.0, 1.0)
    require_integer("lifecycle.test_call_count", minimum=0)
    require_number("metrics.weighted_f1", 0.0, 1.0)
    require_number("metrics.accuracy", 0.0, 1.0)


def write_manifest_atomic(path: PathLike, manifest: Mapping[str, Any]) -> Path:
    """Validate and atomically persist one JSON manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = _json_safe(manifest)
    validate_manifest(normalized)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=".{}.".format(destination.name),
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(normalized, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(str(temporary_path), str(destination))
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(str(destination.parent), directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise
    return destination


def load_manifest(path: PathLike) -> Dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestValidationError("manifest is unreadable: {}".format(path)) from error
    validate_manifest(manifest)
    return manifest


def audit_paired_manifests(
    baseline: Mapping[str, Any], jepa: Mapping[str, Any]
) -> List[str]:
    """Return human-readable fairness violations for a Baseline/JEPA pair."""
    validate_manifest(baseline)
    validate_manifest(jepa)
    mismatches: List[str] = []
    def compare(path: str, baseline_value: Any, jepa_value: Any) -> None:
        if isinstance(baseline_value, Mapping) and isinstance(jepa_value, Mapping):
            keys = sorted(set(baseline_value) | set(jepa_value))
            for key in keys:
                child_path = "{}.{}".format(path, key)
                if key not in baseline_value:
                    mismatches.append("{} is missing from baseline".format(child_path))
                elif key not in jepa_value:
                    mismatches.append("{} is missing from jepa".format(child_path))
                else:
                    compare(child_path, baseline_value[key], jepa_value[key])
            return
        if baseline_value != jepa_value:
            mismatches.append(
                "{} differs: baseline={!r}, jepa={!r}".format(
                    path, baseline_value, jepa_value
                )
            )

    for path in _PAIRED_INVARIANT_PATHS:
        compare(path, _get_path(baseline, path), _get_path(jepa, path))
    for name, manifest in (("baseline", baseline), ("jepa", jepa)):
        count = _get_path(manifest, "lifecycle.test_call_count")
        if count != 1:
            mismatches.append(
                "lifecycle.test_call_count must be 1 for {} (got {!r})".format(
                    name, count
                )
            )
    return mismatches


def _optional_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _gpu_driver_version() -> Optional[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[0] if lines else None


def collect_environment() -> Dict[str, Any]:
    """Collect version and hardware evidence without requiring a GPU."""
    import numpy
    import sklearn
    import torch

    gpu_index: Optional[int] = None
    gpu_model: Optional[str] = None
    if torch.cuda.is_available():
        gpu_index = int(torch.cuda.current_device())
        gpu_model = str(torch.cuda.get_device_name(gpu_index))
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "pyg": _optional_version("torch_geometric"),
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "gpu": {
            "index": gpu_index,
            "model": gpu_model,
            "driver": _gpu_driver_version(),
        },
    }


def collect_provenance(
    command: Optional[Sequence[str]] = None, cwd: Optional[PathLike] = None
) -> Dict[str, Any]:
    """Collect the exact invocation and current repository state."""
    working_directory = Path(cwd or os.getcwd()).resolve()

    def git_output(arguments: Sequence[str]) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git"] + list(arguments),
                cwd=str(working_directory),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    revision = git_output(("rev-parse", "HEAD"))
    status = git_output(("status", "--porcelain"))
    return {
        "command": list(command if command is not None else sys.argv),
        "cwd": str(working_directory),
        "git_revision": revision,
        "git_status": "clean" if status == "" else status,
    }
