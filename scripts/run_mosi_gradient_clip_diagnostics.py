#!/usr/bin/env python3
"""Run the bounded MOSI gradient-clipping collapse diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
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


@dataclass(frozen=True)
class GradientClipJob:
    method: str
    missing_rate: float
    seed: int
    gpu: int
    slot: int
    epochs: int
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
) -> List[GradientClipJob]:
    gpu = int(gpu)
    if gpu == 4:
        raise ValueError("broken GPU 4 must be excluded")
    if not 1 <= max_concurrent <= 3:
        raise ValueError("max_concurrent must be between 1 and 3")
    if epochs < 1:
        raise ValueError("epochs must be positive")

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
                output_dir=output_dir,
                command=tuple(command),
            )
        )
    return jobs


def _latest_manifest(output_dir: Path) -> Path | None:
    manifests = sorted(
        output_dir.glob("run_records/*/run_manifest_fold_*.json")
    )
    return manifests[-1] if manifests else None


def _manifest_matches_job(manifest: dict, job: GradientClipJob) -> bool:
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
    return (
        manifest["run"]["dataset"] == "CMUMOSI"
        and manifest["run"]["fold"] == 1
        and manifest["run"]["master_seed"] == job.seed
        and manifest["masks"]["requested_missing_rate"] == job.missing_rate
        and manifest["lifecycle"]["evaluation_protocol"] == "official"
        and all(
            manifest["method"][key] == value
            for key, value in expected_method.items()
        )
    )


def _fold_metrics_match_job(manifest: dict) -> bool:
    try:
        metrics_path = Path(manifest["outputs"]["fold_metrics"])
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


def _epoch_diagnostics_match_job(job: GradientClipJob) -> bool:
    diagnostics_path = job.output_dir / "epoch_collapse_diagnostics.json"
    try:
        records = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(records, list) or len(records) != job.epochs:
        return False
    for expected_epoch, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            return False
        if record.get("fold") != 1 or record.get("epoch") != expected_epoch:
            return False
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
    return True


def is_complete(job: GradientClipJob) -> bool:
    status_path = job.output_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if status.get("identity") != job.identity:
        return False
    returncode = status.get("returncode")
    if type(returncode) is not int or returncode != 0:
        return False
    manifest_path = _latest_manifest(job.output_dir)
    if manifest_path is None:
        return False
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ManifestValidationError):
        return False
    return (
        _manifest_matches_job(manifest, job)
        and _fold_metrics_match_job(manifest)
        and _epoch_diagnostics_match_job(job)
    )


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
    if is_complete(job):
        return True
    if stop_event.is_set():
        return False
    started_at = time.time()
    returncode = None
    claim_path: Path | None = None
    try:
        claim_path = _acquire_claim(job)
        if claim_path is None:
            stop_event.set()
            return False
        if is_complete(job):
            return True
        if stop_event.is_set():
            return False
        _write_json(
            job.output_dir / "command.json",
            {
                **asdict(job),
                "output_dir": str(job.output_dir),
                "command": list(job.command),
            },
        )
        with (job.output_dir / "train.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                job.command,
                cwd=str(root),
                env=_environment(root, job.gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        returncode = result.returncode
        status = {
            "identity": job.identity,
            "gpu": job.gpu,
            "slot": job.slot,
            "returncode": returncode,
            "started_at_unix": started_at,
            "finished_at_unix": time.time(),
        }
        if returncode != 0:
            status["error"] = (
                "training process exited with return code {}".format(returncode)
            )
        _write_json(job.output_dir / "status.json", status)
        if returncode != 0:
            stop_event.set()
            return False
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
                    "started_at_unix": started_at,
                    "finished_at_unix": time.time(),
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
        baseline_manifest = _latest_manifest(baseline.output_dir)
        jepa_manifest = _latest_manifest(jepa.output_dir)
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

    root = Path.cwd().resolve()
    output_root = args.output_root.resolve()
    jobs = build_jobs(
        output_root,
        args.python,
        args.gpu,
        max_concurrent=args.max_concurrent,
        epochs=args.epochs,
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

    assert_gpu_available(args.gpu)
    lanes: Dict[int, List[GradientClipJob]] = {
        slot: [] for slot in range(args.max_concurrent)
    }
    for job in jobs:
        lanes[job.slot].append(job)
    stop_event = threading.Event()
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


if __name__ == "__main__":
    raise SystemExit(main())
