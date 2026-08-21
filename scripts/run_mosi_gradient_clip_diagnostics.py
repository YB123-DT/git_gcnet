#!/usr/bin/env python3
"""Run the bounded MOSI gradient-clipping collapse diagnostic."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gcnet_modality_jepa.run_manifest import ManifestValidationError, load_manifest


CONDITIONS = ((0.3, 73), (0.4, 72), (0.4, 73))
METHODS = ("baseline", "jepa")
GRADIENT_CLIP_NORM = 1.0
EPOCHS = 100
EXPECTED_JOB_COUNT = 6
EXPECTED_PAIR_COUNT = 3
CLAIM_FILE = ".gradient-clip-diagnostics.claim"
GPU_LEASE_PREFIX = "gcnet-mosi-gradient-clip-gpu"


@dataclass(frozen=True)
class AttemptSnapshot:
    manifest_paths: frozenset[Path]
    artifact_signatures: Dict[Path, Tuple[int, int, int]]


@dataclass(frozen=True)
class GPULease:
    path: Path
    token: str
    host: str
    pid: int
    descriptor: int


@dataclass(frozen=True)
class GradientClipJob:
    method: str
    missing_rate: float
    seed: int
    gpu: int
    slot: int
    epochs: int
    source_contract_sha256: str
    output_dir: Path
    command: Tuple[str, ...]

    @property
    def identity(self) -> str:
        return "CMUMOSI:{}:{:.1f}:{}:clip{:.1f}:epochs{}".format(
            self.method,
            self.missing_rate,
            self.seed,
            GRADIENT_CLIP_NORM,
            self.epochs,
        )


def _rate_directory(rate: float) -> str:
    return "miss_{:.1f}".format(rate).replace(".", "p")


def _training_command(
    python: str,
    rate: float,
    seed: int,
    epochs: int,
    output_dir: Path,
) -> List[str]:
    return [
        python,
        "-u",
        "-m",
        "gcnet_modality_jepa.train_gcnet",
        "--audio-feature",
        "wav2vec-large-c-UTT",
        "--text-feature",
        "deberta-large-4-UTT",
        "--video-feature",
        "manet_UTT",
        "--dataset",
        "CMUMOSI",
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
        "--dropout",
        "0.5",
        "--batch-size",
        "32",
        "--num-threads",
        "4",
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--mask-type",
        "constant-{:.1f}".format(rate),
        "--evaluation-protocol",
        "official",
        "--stability-aux-mask-rate",
        "0.1",
        "--stability-recon-weight",
        "0.01",
        "--gradient-clip-norm",
        str(GRADIENT_CLIP_NORM),
        "--epoch-collapse-diagnostics",
        "--output-dir",
        str(output_dir),
    ]


def build_jobs(
    output_root: Path,
    python: str,
    gpu: int,
    max_concurrent: int = 3,
    epochs: int = EPOCHS,
    source_contract_sha256: str | None = None,
) -> List[GradientClipJob]:
    gpu = int(gpu)
    if gpu == 4:
        raise ValueError("broken GPU 4 must be excluded")
    if not 1 <= max_concurrent <= 3:
        raise ValueError("max_concurrent must be between 1 and 3")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if source_contract_sha256 is None:
        source_contract_sha256 = _source_contract_sha256()
    if len(source_contract_sha256) != 64:
        raise ValueError("source contract must be a SHA-256 digest")

    jobs: List[GradientClipJob] = []
    for job_index, ((rate, seed), method) in enumerate(
        (condition, method)
        for condition in CONDITIONS
        for method in METHODS
    ):
        output_dir = (
            output_root
            / _rate_directory(rate)
            / "seed_{}".format(seed)
            / method
        )
        command = _training_command(python, rate, seed, epochs, output_dir)
        if epochs < 60:
            command.append("--allow-short-run")
        if method == "baseline":
            command.extend(
                (
                    "--loss-recon",
                    "--jepa-weight",
                    "0",
                    "--model-variant",
                    "addon",
                )
            )
        else:
            command.extend(
                ("--jepa-weight", "0.1", "--model-variant", "replacement")
            )
        jobs.append(
            GradientClipJob(
                method=method,
                missing_rate=rate,
                seed=seed,
                gpu=gpu,
                slot=job_index % max_concurrent,
                epochs=epochs,
                source_contract_sha256=source_contract_sha256,
                output_dir=output_dir,
                command=tuple(command),
            )
        )
    return jobs


def _require_repository_root(root: Path | None = None) -> Path:
    if root is not None and Path(root).resolve() != REPOSITORY_ROOT:
        raise ValueError(
            "repository root mismatch: {} != {}".format(
                Path(root).resolve(), REPOSITORY_ROOT
            )
        )
    return REPOSITORY_ROOT


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_contract_sha256(job: GradientClipJob) -> str:
    payload = {
        "command": list(job.command),
        "identity": job.identity,
        "repository_root": str(REPOSITORY_ROOT),
        "source_contract_sha256": job.source_contract_sha256,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_contract_sha256(root: Path = REPOSITORY_ROOT) -> str:
    root = Path(root).resolve()
    paths = [root / "config.py"]
    for directory in (
        root / "gcnet_modality_jepa",
        root / "gcnet_jepa_replacement",
    ):
        paths.extend(directory.rglob("*.py"))
    paths.append(root / "scripts" / "run_mosi_gradient_clip_diagnostics.py")
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        path_bytes = relative.as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _code_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPOSITORY_ROOT),
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    if not revision:
        raise RuntimeError("repository code revision is unavailable")
    return revision


def _git_status() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=str(REPOSITORY_ROOT),
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    return "clean" if status == "" else status


def _python_executable(job: GradientClipJob) -> str:
    return str(Path(job.command[0]).expanduser().resolve())


def _expected_manifest_command(job: GradientClipJob) -> List[str]:
    return [
        str(REPOSITORY_ROOT / "gcnet_modality_jepa" / "train_gcnet.py"),
        *job.command[4:],
    ]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _file_signature(path: Path) -> Tuple[int, int, int]:
    stat_result = path.stat()
    return (
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


def _attempt_artifact_paths(job: GradientClipJob) -> List[Path]:
    paths = []
    run_records = job.output_dir / "run_records"
    if run_records.exists():
        paths.extend(path for path in run_records.rglob("*") if path.is_file())
    epoch_path = job.output_dir / "epoch_collapse_diagnostics.json"
    if epoch_path.is_file():
        paths.append(epoch_path)
    paths.extend(path for path in job.output_dir.glob("*.npz") if path.is_file())
    return paths


def _snapshot_attempt_outputs(job: GradientClipJob) -> AttemptSnapshot:
    artifacts = {
        path.resolve(): _file_signature(path)
        for path in _attempt_artifact_paths(job)
    }
    manifests = frozenset(
        path
        for path in artifacts
        if path.name == "run_manifest_fold_1.json"
    )
    return AttemptSnapshot(manifests, artifacts)


def _artifact_changed(path: Path, snapshot: AttemptSnapshot) -> bool:
    resolved = path.resolve()
    if not resolved.is_file():
        return False
    return snapshot.artifact_signatures.get(resolved) != _file_signature(resolved)


def _manifest_matches_job(
    manifest: dict,
    job: GradientClipJob,
    code_revision: str,
    git_status: str,
) -> bool:
    expected_method = (
        {
            "model_variant": "addon",
            "jepa_weight": 0.0,
            "loss_reconstruction": True,
        }
        if job.method == "baseline"
        else {
            "model_variant": "replacement",
            "jepa_weight": 0.1,
            "loss_reconstruction": False,
        }
    )
    try:
        return (
            manifest["run"]["dataset"] == "CMUMOSI"
            and manifest["run"]["fold"] == 1
            and manifest["run"]["master_seed"] == job.seed
            and manifest["masks"]["requested_missing_rate"] == job.missing_rate
            and manifest["lifecycle"]["evaluation_protocol"] == "official"
            and manifest["lifecycle"]["epochs_completed"] == job.epochs
            and manifest["provenance"]["cwd"] == str(REPOSITORY_ROOT)
            and manifest["provenance"]["git_revision"] == code_revision
            and manifest["provenance"]["git_status"] == git_status
            and manifest["provenance"]["command"]
            == _expected_manifest_command(job)
            and all(
                manifest["method"][key] == value
                for key, value in expected_method.items()
            )
        )
    except (KeyError, TypeError):
        return False


def _fold_metrics_match_job(
    manifest: dict,
    manifest_path: Path | None = None,
) -> bool:
    try:
        metrics_path = Path(manifest["outputs"]["fold_metrics"]).resolve()
        if manifest_path is not None and metrics_path != (
            manifest_path.parent / "fold_metrics.json"
        ).resolve():
            return False
        records = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(records, list)
        and len(records) == 1
        and records[0].get("fold") == 1
        and records[0].get("gradient_clip_norm") == GRADIENT_CLIP_NORM
    )


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _numeric_diagnostics_valid(value: object, key: str = "") -> bool:
    if isinstance(value, dict):
        return all(
            _numeric_diagnostics_valid(child, str(child_key))
            for child_key, child in value.items()
        )
    if isinstance(value, list):
        return all(_numeric_diagnostics_valid(child, key) for child in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value)) and (
            "norm" not in key.lower() or float(value) >= 0.0
        )
    return True


def _epoch_diagnostics_match_job(
    job: GradientClipJob,
    diagnostics_path: Path | None = None,
) -> bool:
    diagnostics_path = (
        diagnostics_path
        if diagnostics_path is not None
        else job.output_dir / "epoch_collapse_diagnostics.json"
    )
    try:
        records = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(records, list) or len(records) != job.epochs:
        return False
    epochs = []
    for record in records:
        if not isinstance(record, dict):
            return False
        if not _numeric_diagnostics_valid(record):
            return False
        epoch = record.get("epoch")
        if (
            record.get("fold") != 1
            or not isinstance(epoch, int)
            or isinstance(epoch, bool)
        ):
            return False
        epochs.append(epoch)
        gradient_clip = record.get("gradient_clip")
        if not isinstance(gradient_clip, dict):
            return False
        configured_norm = gradient_clip.get("configured_norm")
        optimizer_steps = gradient_clip.get("optimizer_steps")
        clipped_steps = gradient_clip.get("clipped_steps")
        clipped_fraction = gradient_clip.get("clipped_fraction")
        pre_clip_norm_mean = gradient_clip.get("pre_clip_norm_mean")
        pre_clip_norm_max = gradient_clip.get("pre_clip_norm_max")
        if (
            not _is_finite_number(configured_norm)
            or configured_norm != GRADIENT_CLIP_NORM
        ):
            return False
        if (
            not isinstance(optimizer_steps, int)
            or isinstance(optimizer_steps, bool)
            or optimizer_steps <= 0
        ):
            return False
        if (
            not isinstance(clipped_steps, int)
            or isinstance(clipped_steps, bool)
            or not 0 <= clipped_steps <= optimizer_steps
        ):
            return False
        if (
            not _is_finite_number(clipped_fraction)
            or not 0.0 <= float(clipped_fraction) <= 1.0
        ):
            return False
        if not _is_finite_number(pre_clip_norm_mean) or not _is_finite_number(
            pre_clip_norm_max
        ):
            return False
        if (
            float(pre_clip_norm_mean) < 0.0
            or float(pre_clip_norm_max) < float(pre_clip_norm_mean)
        ):
            return False
        expected_fraction = float(clipped_steps) / float(optimizer_steps)
        if not math.isclose(
            float(clipped_fraction),
            expected_fraction,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return False
        if clipped_steps == 0 and float(pre_clip_norm_max) > float(configured_norm):
            return False
        if clipped_steps > 0 and float(pre_clip_norm_max) <= float(configured_norm):
            return False
    return set(epochs) == set(range(1, job.epochs + 1))


def _load_status(job: GradientClipJob) -> dict | None:
    try:
        status = json.loads(
            (job.output_dir / "status.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    return status if isinstance(status, dict) else None


def _bound_manifest_path(job: GradientClipJob) -> Path | None:
    status = _load_status(job)
    if status is None or not isinstance(status.get("manifest_path"), str):
        return None
    path = Path(status["manifest_path"]).resolve()
    expected_root = (job.output_dir / "run_records").resolve()
    if (
        str(path) != status["manifest_path"]
        or not _is_within(path, expected_root)
        or path.name != "run_manifest_fold_1.json"
    ):
        return None
    return path


def _bound_epoch_path(job: GradientClipJob, status: dict) -> Path | None:
    value = status.get("epoch_diagnostics_path")
    if not isinstance(value, str):
        return None
    path = Path(value).resolve()
    expected = (job.output_dir / "epoch_collapse_diagnostics.json").resolve()
    return path if str(path) == value and path == expected else None


def _hash_matches(path: Path, expected: object) -> bool:
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
        return False
    try:
        return _sha256_file(path) == expected
    except OSError:
        return False


def is_complete(job: GradientClipJob) -> bool:
    status = _load_status(job)
    if status is None:
        return False
    if status.get("identity") != job.identity:
        return False
    returncode = status.get("returncode")
    if type(returncode) is not int or returncode != 0:
        return False
    attempt_started = status.get("attempt_started_at_unix")
    if not _is_finite_number(attempt_started) or float(attempt_started) <= 0.0:
        return False
    if status.get("command_contract_sha256") != _command_contract_sha256(job):
        return False
    if status.get("source_contract_sha256") != job.source_contract_sha256:
        return False
    if _source_contract_sha256() != job.source_contract_sha256:
        return False
    code_revision = status.get("code_revision")
    try:
        current_revision = _code_revision()
        current_git_status = _git_status()
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        return False
    if not isinstance(code_revision, str) or code_revision != current_revision:
        return False
    git_status = status.get("git_status")
    if not isinstance(git_status, str) or git_status != current_git_status:
        return False
    if status.get("python_executable") != _python_executable(job):
        return False
    manifest_path = _bound_manifest_path(job)
    epoch_path = _bound_epoch_path(job, status)
    if manifest_path is None or epoch_path is None:
        return False
    if not _hash_matches(manifest_path, status.get("manifest_sha256")):
        return False
    if not _hash_matches(epoch_path, status.get("epoch_diagnostics_sha256")):
        return False
    try:
        manifest = load_manifest(manifest_path)
    except ManifestValidationError:
        return False
    try:
        metrics_path = Path(manifest["outputs"]["fold_metrics"]).resolve()
        result_path = Path(manifest["outputs"]["result_archive"]).resolve()
    except (KeyError, TypeError):
        return False
    if status.get("fold_metrics_path") != str(metrics_path):
        return False
    if status.get("result_archive_path") != str(result_path):
        return False
    if not _hash_matches(metrics_path, status.get("fold_metrics_sha256")):
        return False
    if not _hash_matches(result_path, status.get("result_archive_sha256")):
        return False
    return (
        _manifest_matches_job(manifest, job, code_revision, git_status)
        and _fold_metrics_match_job(manifest, manifest_path)
        and _epoch_diagnostics_match_job(job, epoch_path)
        and _is_within(result_path, job.output_dir.resolve())
        and result_path.is_file()
    )


def _new_attempt_evidence(
    job: GradientClipJob,
    snapshot: AttemptSnapshot,
    code_revision: str,
    git_status: str,
) -> Dict[str, object]:
    current_manifests = {
        path.resolve()
        for path in job.output_dir.glob(
            "run_records/*/run_manifest_fold_1.json"
        )
        if path.is_file()
    }
    new_manifests = current_manifests - set(snapshot.manifest_paths)
    if len(new_manifests) != 1:
        raise RuntimeError(
            "expected exactly one new run-record manifest, found {}".format(
                len(new_manifests)
            )
        )
    manifest_path = next(iter(new_manifests))
    try:
        manifest = load_manifest(manifest_path)
    except ManifestValidationError as error:
        raise RuntimeError("new run-record manifest is invalid") from error
    if not _manifest_matches_job(manifest, job, code_revision, git_status):
        raise RuntimeError("new run-record manifest does not match job contract")

    try:
        metrics_path = Path(manifest["outputs"]["fold_metrics"]).resolve()
        result_path = Path(manifest["outputs"]["result_archive"]).resolve()
    except (KeyError, TypeError) as error:
        raise RuntimeError("new run-record manifest lacks output artifacts") from error
    expected_metrics = (manifest_path.parent / "fold_metrics.json").resolve()
    output_root = job.output_dir.resolve()
    if metrics_path != expected_metrics or not _is_within(result_path, output_root):
        raise RuntimeError("new run-record artifact paths escape their attempt")
    if (
        metrics_path in snapshot.artifact_signatures
        or result_path in snapshot.artifact_signatures
    ):
        raise RuntimeError("new run-record reused an artifact from before this attempt")
    epoch_path = (job.output_dir / "epoch_collapse_diagnostics.json").resolve()
    for label, path in (
        ("fold metrics", metrics_path),
        ("result archive", result_path),
        ("epoch diagnostics", epoch_path),
    ):
        if not _artifact_changed(path, snapshot):
            raise RuntimeError("{} were not created by this attempt".format(label))
    if not _fold_metrics_match_job(manifest, manifest_path):
        raise RuntimeError("new fold metrics do not match job contract")
    if not _epoch_diagnostics_match_job(job, epoch_path):
        raise RuntimeError("new epoch diagnostics do not match job contract")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "epoch_diagnostics_path": str(epoch_path),
        "epoch_diagnostics_sha256": _sha256_file(epoch_path),
        "fold_metrics_path": str(metrics_path),
        "fold_metrics_sha256": _sha256_file(metrics_path),
        "result_archive_path": str(result_path),
        "result_archive_sha256": _sha256_file(result_path),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".{}.".format(path.name),
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def _acquire_claim(job: GradientClipJob) -> Path | None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    claim_path = job.output_dir / CLAIM_FILE
    try:
        descriptor = os.open(
            str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
        )
    except FileExistsError:
        return None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "identity": job.identity,
                    "pid": os.getpid(),
                    "claimed_at_unix": time.time(),
                },
                handle,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            claim_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return claim_path


def _release_claim(claim_path: Path | None) -> None:
    if claim_path is None:
        return
    try:
        claim_path.unlink()
    except FileNotFoundError:
        pass


def acquire_gpu_lease(
    gpu: int,
    lease_root: Path | None = None,
) -> GPULease:
    root = Path(tempfile.gettempdir()) if lease_root is None else Path(lease_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "{}-{}.lease".format(GPU_LEASE_PREFIX, int(gpu))
    token = uuid.uuid4().hex
    host = socket.gethostname()
    pid = os.getpid()
    descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(
            "GPU {} lease is held at {}".format(gpu, path)
        ) from error
    try:
        payload = json.dumps(
            {"token": token, "host": host, "pid": pid}, sort_keys=True
        ).encode("utf-8") + b"\n"
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, payload)
        os.fsync(descriptor)
    except Exception:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise
    return GPULease(
        path=path,
        token=token,
        host=host,
        pid=pid,
        descriptor=descriptor,
    )


def release_gpu_lease(lease: GPULease) -> bool:
    try:
        fcntl.flock(lease.descriptor, fcntl.LOCK_UN)
        os.close(lease.descriptor)
    except OSError:
        return False
    return True


def _gpu_memory_mb(gpu: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id={}".format(gpu),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip())


def assert_gpu_available(gpu: int, maximum_idle_memory_mb: int = 768) -> None:
    if gpu == 4:
        raise ValueError("broken GPU 4 must be excluded")
    memory = _gpu_memory_mb(gpu)
    if memory > maximum_idle_memory_mb:
        raise RuntimeError(
            "refusing to use occupied GPU {}: {} MiB used".format(gpu, memory)
        )


def _environment(root: Path, gpu: int) -> Dict[str, str]:
    root = _require_repository_root(root)
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "GCNET_DATASET_ROOT": str(root / "dataset"),
            "GCNET_CACHE_ROOT": "/data2/yb/gcnet_unified_cache",
            "PYTHONPATH": str(root),
        }
    )
    return environment


def run_job(
    job: GradientClipJob,
    root: Path,
    stop_event: threading.Event,
) -> bool:
    root = _require_repository_root(root)
    if is_complete(job):
        return True
    if stop_event.is_set():
        return False
    started_at = None
    returncode = None
    claim_path: Path | None = None
    code_revision = None
    git_status = None
    python_executable = _python_executable(job)
    try:
        claim_path = _acquire_claim(job)
        if claim_path is None:
            stop_event.set()
            return False
        if is_complete(job):
            return True
        if stop_event.is_set():
            return False
        started_at = time.time()
        code_revision = _code_revision()
        git_status = _git_status()
        _write_json(
            job.output_dir / "command.json",
            {
                **asdict(job),
                "output_dir": str(job.output_dir),
                "command": list(job.command),
            },
        )
        snapshot = _snapshot_attempt_outputs(job)
        with (job.output_dir / "train.log").open("w", encoding="utf-8") as log:
            if _source_contract_sha256() != job.source_contract_sha256:
                raise RuntimeError("source contract changed before subprocess")
            result = subprocess.run(
                job.command,
                cwd=str(root),
                env=_environment(root, job.gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if _source_contract_sha256() != job.source_contract_sha256:
                raise RuntimeError("source contract changed during subprocess")
        returncode = result.returncode
        status = {
            "identity": job.identity,
            "gpu": job.gpu,
            "slot": job.slot,
            "returncode": returncode,
            "attempt_started_at_unix": started_at,
            "finished_at_unix": time.time(),
            "command_contract_sha256": _command_contract_sha256(job),
            "source_contract_sha256": job.source_contract_sha256,
            "code_revision": code_revision,
            "git_status": git_status,
            "python_executable": python_executable,
        }
        if returncode != 0:
            status["error"] = (
                "training process exited with return code {}".format(returncode)
            )
        _write_json(job.output_dir / "status.json", status)
        if returncode != 0:
            stop_event.set()
            return False
        status.update(
            _new_attempt_evidence(
                job,
                snapshot,
                code_revision,
                git_status,
            )
        )
        _write_json(job.output_dir / "status.json", status)
        if not is_complete(job):
            status["error"] = "return code 0 without matching diagnostic evidence"
            status["finished_at_unix"] = time.time()
            _write_json(job.output_dir / "status.json", status)
            stop_event.set()
            return False
        return True
    except Exception as error:
        stop_event.set()
        try:
            _write_json(
                job.output_dir / "status.json",
                {
                    "identity": job.identity,
                    "gpu": job.gpu,
                    "slot": job.slot,
                    "returncode": returncode,
                    "attempt_started_at_unix": started_at,
                    "finished_at_unix": time.time(),
                    "command_contract_sha256": _command_contract_sha256(job),
                    "source_contract_sha256": job.source_contract_sha256,
                    "code_revision": code_revision,
                    "git_status": git_status,
                    "python_executable": python_executable,
                    "error": "{}: {}".format(type(error).__name__, error),
                },
            )
        except Exception:
            pass
        return False
    finally:
        _release_claim(claim_path)


def _run_lane(
    jobs: Sequence[GradientClipJob],
    root: Path,
    stop_event: threading.Event,
) -> bool:
    for job in jobs:
        if not run_job(job, root, stop_event):
            return False
    return True


def audit_completed_pairs(
    jobs: Sequence[GradientClipJob],
    root: Path,
    python: str,
) -> Tuple[int, int]:
    root = _require_repository_root(root)
    pairs: Dict[Tuple[float, int], Dict[str, GradientClipJob]] = {}
    for job in jobs:
        pairs.setdefault((job.missing_rate, job.seed), {})[job.method] = job
    failures = 0
    audited_pairs = 0
    audit_script = root / "scripts" / "audit_paired_runs.py"
    for pair_jobs in pairs.values():
        if set(pair_jobs) != set(METHODS):
            failures += 1
            continue
        baseline = pair_jobs["baseline"]
        jepa = pair_jobs["jepa"]
        if not is_complete(baseline) or not is_complete(jepa):
            failures += 1
            continue
        baseline_manifest = _bound_manifest_path(baseline)
        jepa_manifest = _bound_manifest_path(jepa)
        if baseline_manifest is None or jepa_manifest is None:
            failures += 1
            continue
        audit_path = baseline.output_dir.parent / "paired_audit.log"
        try:
            result = subprocess.run(
                [
                    python,
                    str(audit_script),
                    str(baseline_manifest),
                    str(jepa_manifest),
                ],
                cwd=str(root),
                env=_environment(root, baseline.gpu),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            audit_path.write_text(result.stdout, encoding="utf-8")
        except Exception as error:
            failures += 1
            try:
                audit_path.write_text(
                    "audit exception: {}: {}\n".format(
                        type(error).__name__, error
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass
            continue
        audited_pairs += 1
        failures += int(result.returncode != 0)
    return audited_pairs, failures


def _collect_worker_results(futures, stop_event):
    failed_workers = 0
    errors = []
    for future in futures:
        try:
            if future.result() is not True:
                failed_workers += 1
        except Exception as error:
            stop_event.set()
            failed_workers += 1
            errors.append("{}: {}".format(type(error).__name__, error))
    return failed_workers == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--python",
        default="/data2/yb/reproduction_envs/gcnet-official/bin/python",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = REPOSITORY_ROOT
    output_root = args.output_root.resolve()
    source_contract_sha256 = _source_contract_sha256()
    jobs = build_jobs(
        output_root,
        args.python,
        args.gpu,
        max_concurrent=args.max_concurrent,
        epochs=args.epochs,
        source_contract_sha256=source_contract_sha256,
    )
    _write_json(
        output_root / "task_manifest.json",
        [
            {
                **asdict(job),
                "output_dir": str(job.output_dir),
                "command": list(job.command),
            }
            for job in jobs
        ],
    )
    if args.dry_run:
        print("generated {} jobs".format(len(jobs)))
        return 0

    lease = acquire_gpu_lease(args.gpu)
    try:
        assert_gpu_available(args.gpu)
        lanes: Dict[int, List[GradientClipJob]] = {
            slot: [] for slot in range(args.max_concurrent)
        }
        for job in jobs:
            lanes[job.slot].append(job)
        stop_event = threading.Event()
        assert_gpu_available(args.gpu)
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
            futures = [
                executor.submit(_run_lane, lane_jobs, root, stop_event)
                for lane_jobs in lanes.values()
            ]
            worker_success, worker_errors = _collect_worker_results(
                futures, stop_event
            )

        audited_pairs, audit_failures = audit_completed_pairs(
            jobs, root, args.python
        )
        complete_jobs = sum(is_complete(job) for job in jobs)
        scheduler_success = (
            worker_success
            and len(jobs) == EXPECTED_JOB_COUNT
            and complete_jobs == EXPECTED_JOB_COUNT
            and audited_pairs == EXPECTED_PAIR_COUNT
            and audit_failures == 0
        )
        _write_json(
            output_root / "scheduler_status.json",
            {
                "complete_jobs": complete_jobs,
                "total_jobs": len(jobs),
                "worker_success": worker_success,
                "worker_errors": worker_errors,
                "paired_audits": audited_pairs,
                "expected_pair_audits": EXPECTED_PAIR_COUNT,
                "paired_audit_failures": audit_failures,
            },
        )
        return 0 if scheduler_success else 1
    finally:
        if not release_gpu_lease(lease):
            raise RuntimeError("failed to release owned GPU lease")


if __name__ == "__main__":
    raise SystemExit(main())
