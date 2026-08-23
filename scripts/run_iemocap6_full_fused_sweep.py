#!/usr/bin/env python3
"""Run the isolated IEMOCAPSix missing-vs-full-fused paired sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gcnet_modality_jepa.run_manifest import (
    ManifestValidationError,
    audit_paired_manifests,
    load_manifest,
)


DATASET = "IEMOCAPSix"
CONDITION = "full_fused"
RATES = tuple(round(index / 10.0, 1) for index in range(8))
SEEDS = tuple(range(66, 76))
DEFAULT_GPUS = (0, 1, 2)
DEFAULT_BASELINE_ROOT = Path(
    "/data2/yb/experiments/gcnet_official_4dataset_10seed_20260820/IEMOCAPSix"
)
CLAIM_FILE = ".iemocap6-full-fused-sweep.claim"


@dataclass(frozen=True)
class FullFusedJob:
    condition: str
    rate: float
    seed: int
    gpu: int
    slot: int
    epochs: int
    baseline_dir: Path
    output_dir: Path
    command: Tuple[str, ...]

    @property
    def identity(self) -> str:
        return "{}:{}:{:.1f}:{}:epochs{}".format(
            DATASET, self.condition, self.rate, self.seed, self.epochs
        )


def _rate_directory(rate: float) -> str:
    return "miss_{:.1f}".format(rate).replace(".", "p")


def _training_command(
    python: str,
    rate: float,
    seed: int,
    epochs: int,
    reconstruction_target: str,
    output_dir: Path,
) -> List[str]:
    command = [
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
        DATASET,
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
        "--fold",
        "5",
        "--stability-aux-mask-rate",
        "0.1",
        "--stability-recon-weight",
        "0.01",
        "--loss-recon",
        "--jepa-weight",
        "0",
        "--model-variant",
        "addon",
        "--reconstruction-target",
        reconstruction_target,
        "--output-dir",
        str(output_dir),
    ]
    if epochs < 60:
        command.append("--allow-short-run")
    return command


def _normalize_selection(
    values: Sequence[object], allowed: Sequence[object], label: str
) -> Tuple[object, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError("at least one {} is required".format(label))
    if len(set(normalized)) != len(normalized):
        raise ValueError("{} list contains duplicates".format(label))
    unsupported = [value for value in normalized if value not in allowed]
    if unsupported:
        raise ValueError("unsupported {}: {}".format(label, unsupported))
    return normalized


def build_jobs(
    output_root: Path,
    python: str,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    gpus: Sequence[int] = DEFAULT_GPUS,
    jobs_per_gpu: int = 3,
    rates: Sequence[float] = RATES,
    seeds: Sequence[int] = SEEDS,
    epochs: int = 100,
) -> List[FullFusedJob]:
    normalized_gpus = tuple(int(gpu) for gpu in gpus)
    if not normalized_gpus:
        raise ValueError("at least one GPU is required")
    if 4 in normalized_gpus:
        raise ValueError("broken GPU 4 must be excluded")
    if len(normalized_gpus) > 3:
        raise ValueError("at most 3 GPUs may be selected")
    if len(set(normalized_gpus)) != len(normalized_gpus):
        raise ValueError("GPU list contains duplicates")
    if not 1 <= jobs_per_gpu <= 3:
        raise ValueError("jobs_per_gpu must be between 1 and 3")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    normalized_rates = tuple(
        float(value)
        for value in _normalize_selection(
            tuple(round(float(rate), 1) for rate in rates), RATES, "rate"
        )
    )
    normalized_seeds = tuple(
        int(value)
        for value in _normalize_selection(
            tuple(int(seed) for seed in seeds), SEEDS, "seed"
        )
    )
    lanes = [
        (gpu, slot)
        for gpu in normalized_gpus
        for slot in range(jobs_per_gpu)
    ]
    jobs: List[FullFusedJob] = []
    job_index = 0
    for rate in normalized_rates:
        for seed in normalized_seeds:
            gpu, slot = lanes[job_index % len(lanes)]
            relative_pair = Path(_rate_directory(rate)) / "seed_{}".format(seed)
            output_dir = Path(output_root) / relative_pair / CONDITION
            baseline_dir = Path(baseline_root) / relative_pair / "baseline"
            jobs.append(
                FullFusedJob(
                    condition=CONDITION,
                    rate=rate,
                    seed=seed,
                    gpu=gpu,
                    slot=slot,
                    epochs=epochs,
                    baseline_dir=baseline_dir,
                    output_dir=output_dir,
                    command=tuple(
                        _training_command(
                            python,
                            rate,
                            seed,
                            epochs,
                            "full_fused",
                            output_dir,
                        )
                    ),
                )
            )
            job_index += 1
    return jobs


def _latest_manifest(output_dir: Path) -> Path | None:
    manifests = sorted(
        output_dir.glob("run_records/*/run_manifest_fold_5.json")
    )
    return manifests[-1] if manifests else None


def _manifest_matches_job(manifest: Mapping[str, object], job: FullFusedJob) -> bool:
    expected_method = {
        "model_variant": "addon",
        "jepa_weight": 0.0,
        "loss_reconstruction": True,
        "reconstruction_target": "full_fused",
    }
    try:
        return (
            manifest["run"]["dataset"] == DATASET
            and manifest["run"]["fold"] == 5
            and manifest["run"]["master_seed"] == job.seed
            and manifest["masks"]["requested_missing_rate"] == job.rate
            and manifest["lifecycle"]["evaluation_protocol"] == "official"
            and manifest["lifecycle"]["epochs_completed"] == job.epochs
            and all(
                manifest["method"][key] == value
                for key, value in expected_method.items()
            )
        )
    except (KeyError, TypeError):
        return False


def _fold_metrics_match_job(
    manifest: Mapping[str, object], manifest_path: Path, job: FullFusedJob
) -> bool:
    try:
        metrics_path = Path(manifest["outputs"]["fold_metrics"]).resolve()
        expected_path = (manifest_path.parent / "fold_metrics.json").resolve()
        records = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False
    return (
        metrics_path == expected_path
        and isinstance(records, list)
        and len(records) == 1
        and isinstance(records[0], dict)
        and records[0].get("fold") == 5
        and records[0].get("reconstruction_target") == "full_fused"
    )


def is_complete(job: FullFusedJob) -> bool:
    status_path = job.output_dir / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(status, dict) or status.get("identity") != job.identity:
        return False
    returncode = status.get("returncode")
    if type(returncode) is not int or returncode != 0:
        return False
    manifest_path = _latest_manifest(job.output_dir)
    if manifest_path is None:
        return False
    try:
        manifest = load_manifest(manifest_path)
    except ManifestValidationError:
        return False
    return _manifest_matches_job(manifest, job) and _fold_metrics_match_job(
        manifest, manifest_path, job
    )


def _baseline_identity(job: FullFusedJob) -> str:
    return "{}:baseline:{:.1f}:{}".format(DATASET, job.rate, job.seed)


def _baseline_manifest_matches_job(
    manifest: Mapping[str, object], job: FullFusedJob
) -> bool:
    try:
        method = manifest["method"]
        return (
            manifest["run"]["dataset"] == DATASET
            and manifest["run"]["fold"] == 5
            and manifest["run"]["master_seed"] == job.seed
            and manifest["masks"]["requested_missing_rate"] == job.rate
            and manifest["lifecycle"]["evaluation_protocol"] == "official"
            and method["model_variant"] == "addon"
            and method["jepa_weight"] == 0.0
            and method["loss_reconstruction"] is True
            and method.get("reconstruction_target", "missing") == "missing"
        )
    except (KeyError, TypeError):
        return False


def _baseline_fold_metrics_match(
    manifest: Mapping[str, object], manifest_path: Path
) -> bool:
    try:
        metrics_path = Path(manifest["outputs"]["fold_metrics"]).resolve()
        expected_path = (manifest_path.parent / "fold_metrics.json").resolve()
        records = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False
    return (
        metrics_path == expected_path
        and isinstance(records, list)
        and len(records) == 1
        and isinstance(records[0], dict)
        and records[0].get("fold") == 5
        and records[0].get("reconstruction_target", "missing") == "missing"
    )


def baseline_is_complete(job: FullFusedJob) -> bool:
    try:
        status = json.loads(
            (job.baseline_dir / "status.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not isinstance(status, dict)
        or status.get("identity") != _baseline_identity(job)
    ):
        return False
    returncode = status.get("returncode")
    if type(returncode) is not int or returncode != 0:
        return False
    manifest_path = _latest_manifest(job.baseline_dir)
    if manifest_path is None:
        return False
    try:
        manifest = load_manifest(manifest_path)
    except ManifestValidationError:
        return False
    return _baseline_manifest_matches_job(
        manifest, job
    ) and _baseline_fold_metrics_match(manifest, manifest_path)


def validate_baselines(jobs: Sequence[FullFusedJob]) -> List[str]:
    return [job.identity for job in jobs if not baseline_is_complete(job)]


def audit_pair_manifests(
    baseline: Mapping[str, object],
    full_fused: Mapping[str, object],
    allow_epoch_mismatch: bool = False,
) -> List[str]:
    try:
        mismatches = list(audit_paired_manifests(baseline, full_fused))
    except ManifestValidationError as error:
        return ["invalid paired manifest: {}".format(error)]
    if allow_epoch_mismatch:
        mismatches = [
            mismatch
            for mismatch in mismatches
            if not mismatch.startswith("lifecycle.epochs_completed differs:")
        ]
    expected = (
        ("baseline", baseline, "missing"),
        ("full_fused", full_fused, "full_fused"),
    )
    for condition, manifest, target in expected:
        try:
            method = manifest["method"]
            values = {
                "model_variant": method["model_variant"],
                "jepa_weight": method["jepa_weight"],
                "loss_reconstruction": method["loss_reconstruction"],
                "reconstruction_target": method.get(
                    "reconstruction_target", "missing"
                ),
            }
        except (KeyError, TypeError):
            mismatches.append("{} method evidence is incomplete".format(condition))
            continue
        required = {
            "model_variant": "addon",
            "jepa_weight": 0.0,
            "loss_reconstruction": True,
            "reconstruction_target": target,
        }
        for key, required_value in required.items():
            if values[key] != required_value:
                mismatches.append(
                    "{} method.{} must be {!r}, got {!r}".format(
                        condition, key, required_value, values[key]
                    )
                )
    return mismatches


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


def _job_payload(job: FullFusedJob) -> Dict[str, object]:
    return {
        **asdict(job),
        "baseline_dir": str(job.baseline_dir),
        "output_dir": str(job.output_dir),
        "command": list(job.command),
        "identity": job.identity,
    }


def _acquire_claim(job: FullFusedJob) -> Path | None:
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


def assert_gpus_available(
    gpus: Iterable[int], maximum_idle_memory_mb: int = 768
) -> None:
    occupied = {}
    for gpu in sorted(set(gpus)):
        memory = _gpu_memory_mb(gpu)
        if memory > maximum_idle_memory_mb:
            occupied[gpu] = memory
    if occupied:
        raise RuntimeError("refusing to use occupied GPUs: {}".format(occupied))


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


def run_job(job: FullFusedJob, root: Path, stop_event: threading.Event) -> bool:
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
        _write_json(job.output_dir / "command.json", _job_payload(job))
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
            status["error"] = "return code 0 without matching fold5 evidence"
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
    jobs: Sequence[FullFusedJob], root: Path, stop_event: threading.Event
) -> bool:
    for job in jobs:
        if not run_job(job, root, stop_event):
            return False
    return True


def audit_completed_pairs(jobs: Sequence[FullFusedJob]) -> Tuple[int, int]:
    audited_pairs = 0
    failures = 0
    for job in jobs:
        issues: List[str] = []
        if not baseline_is_complete(job):
            issues.append("existing baseline evidence is incomplete")
        elif not is_complete(job):
            issues.append("full_fused job evidence is incomplete")
        else:
            baseline_path = _latest_manifest(job.baseline_dir)
            full_fused_path = _latest_manifest(job.output_dir)
            try:
                baseline = load_manifest(baseline_path)
                full_fused = load_manifest(full_fused_path)
            except (ManifestValidationError, TypeError) as error:
                issues.append("paired manifest load failed: {}".format(error))
            else:
                issues.extend(
                    audit_pair_manifests(
                        baseline,
                        full_fused,
                        allow_epoch_mismatch=job.epochs < 60,
                    )
                )
                audited_pairs += 1
        passed = not issues
        failures += int(not passed)
        _write_json(
            job.output_dir.parent / "paired_audit.json",
            {
                "passed": passed,
                "issues": issues,
                "baseline_manifest": str(_latest_manifest(job.baseline_dir)),
                "full_fused_manifest": str(_latest_manifest(job.output_dir)),
            },
        )
    return audited_pairs, failures


def _collect_worker_results(
    futures: Sequence[object], stop_event: threading.Event
) -> Tuple[bool, List[str]]:
    failed_workers = 0
    errors: List[str] = []
    for future in futures:
        try:
            if future.result() is not True:
                failed_workers += 1
        except Exception as error:
            stop_event.set()
            failed_workers += 1
            errors.append("{}: {}".format(type(error).__name__, error))
    return failed_workers == 0, errors


def _scheduler_succeeded(
    worker_success: bool,
    complete_jobs: int,
    total_jobs: int,
    audited_pairs: int,
    expected_pairs: int,
    audit_failures: int,
) -> bool:
    return (
        worker_success
        and complete_jobs == total_jobs
        and audited_pairs == expected_pairs
        and audit_failures == 0
    )


def _parse_gpus(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_rates(value: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_seeds(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT
    )
    parser.add_argument(
        "--python",
        default="/data2/yb/reproduction_envs/gcnet-official/bin/python",
    )
    parser.add_argument("--gpus", type=_parse_gpus, default=DEFAULT_GPUS)
    parser.add_argument("--jobs-per-gpu", type=int, default=3)
    parser.add_argument("--rates", type=_parse_rates, default=RATES)
    parser.add_argument("--seeds", type=_parse_seeds, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = REPOSITORY_ROOT
    output_root = args.output_root.resolve()
    jobs = build_jobs(
        output_root=output_root,
        python=args.python,
        baseline_root=args.baseline_root.resolve(),
        gpus=args.gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        rates=args.rates,
        seeds=args.seeds,
        epochs=args.epochs,
    )
    _write_json(
        output_root / "task_manifest.json", [_job_payload(job) for job in jobs]
    )
    if args.dry_run:
        print(
            "generated {} full_fused jobs for {} pairs".format(
                len(jobs), len(jobs)
            )
        )
        return 0

    baseline_failures = validate_baselines(jobs)
    if baseline_failures:
        _write_json(
            output_root / "scheduler_status.json",
            {
                "complete_jobs": sum(is_complete(job) for job in jobs),
                "total_jobs": len(jobs),
                "worker_success": False,
                "worker_errors": [],
                "paired_audits": 0,
                "expected_pair_audits": len(jobs),
                "paired_audit_failures": len(baseline_failures),
                "baseline_preflight_failures": baseline_failures,
            },
        )
        return 1

    assert_gpus_available(args.gpus)
    lanes: Dict[Tuple[int, int], List[FullFusedJob]] = {
        (gpu, slot): []
        for gpu in args.gpus
        for slot in range(args.jobs_per_gpu)
    }
    for job in jobs:
        lanes[(job.gpu, job.slot)].append(job)
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = [
            executor.submit(_run_lane, lane_jobs, root, stop_event)
            for lane_jobs in lanes.values()
        ]
        worker_success, worker_errors = _collect_worker_results(
            futures, stop_event
        )
    audited_pairs, audit_failures = audit_completed_pairs(jobs)
    complete_jobs = sum(is_complete(job) for job in jobs)
    expected_pairs = len(jobs)
    _write_json(
        output_root / "scheduler_status.json",
        {
            "complete_jobs": complete_jobs,
            "total_jobs": len(jobs),
            "worker_success": worker_success,
            "worker_errors": worker_errors,
            "paired_audits": audited_pairs,
            "expected_pair_audits": expected_pairs,
            "paired_audit_failures": audit_failures,
            "baseline_preflight_failures": [],
        },
    )
    return 0 if _scheduler_succeeded(
        worker_success,
        complete_jobs,
        len(jobs),
        audited_pairs,
        expected_pairs,
        audit_failures,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
