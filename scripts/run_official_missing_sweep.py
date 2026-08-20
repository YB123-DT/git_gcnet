#!/usr/bin/env python3
"""Run the auditable four-dataset GCNet official-protocol sweep."""

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
from typing import Dict, Iterable, List, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gcnet_modality_jepa.run_manifest import ManifestValidationError, load_manifest


DATASETS = ("IEMOCAPFour", "IEMOCAPSix", "CMUMOSI", "CMUMOSEI")
METHODS = ("baseline", "jepa")
RATES = tuple(round(index / 10.0, 1) for index in range(8))
SEEDS = tuple(range(66, 76))
DEFAULT_GPUS = (0, 1, 2, 3, 5, 6, 7)
CLAIM_FILE = ".official-sweep.claim"
EXPECTED_JOB_COUNT = 640
EXPECTED_PAIR_COUNT = 320


@dataclass(frozen=True)
class OfficialJob:
    dataset: str
    method: str
    missing_rate: float
    seed: int
    gpu: int
    slot: int
    output_dir: Path
    command: Tuple[str, ...]

    @property
    def identity(self) -> str:
        return "{}:{}:{:.1f}:{}".format(
            self.dataset, self.method, self.missing_rate, self.seed
        )


def _rate_directory(rate: float) -> str:
    return "miss_{:.1f}".format(rate).replace(".", "p")


def _common_command(
    python: str,
    dataset: str,
    rate: float,
    seed: int,
    epochs: int,
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
        dataset,
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
        "--output-dir",
        str(output_dir),
    ]
    if dataset.startswith("IEMOCAP"):
        command.extend(("--fold", "5"))
    if epochs < 60:
        command.append("--allow-short-run")
    return command


def build_jobs(
    output_root: Path,
    python: str,
    gpus: Sequence[int] = DEFAULT_GPUS,
    jobs_per_gpu: int = 3,
    epochs: int = 100,
) -> List[OfficialJob]:
    normalized_gpus = tuple(int(gpu) for gpu in gpus)
    if not normalized_gpus:
        raise ValueError("at least one GPU is required")
    if 4 in normalized_gpus:
        raise ValueError("broken GPU 4 must be excluded")
    if len(set(normalized_gpus)) != len(normalized_gpus):
        raise ValueError("GPU list contains duplicates")
    if not 1 <= jobs_per_gpu <= 3:
        raise ValueError("jobs_per_gpu must be between 1 and 3")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    lanes = [
        (gpu, slot)
        for gpu in normalized_gpus
        for slot in range(jobs_per_gpu)
    ]
    jobs: List[OfficialJob] = []
    job_index = 0
    for dataset in DATASETS:
        for rate in RATES:
            for seed in SEEDS:
                for method in METHODS:
                    gpu, slot = lanes[job_index % len(lanes)]
                    output_dir = (
                        output_root
                        / dataset
                        / _rate_directory(rate)
                        / "seed_{}".format(seed)
                        / method
                    )
                    command = _common_command(
                        python, dataset, rate, seed, epochs, output_dir
                    )
                    if method == "baseline":
                        command.extend(
                            ("--loss-recon", "--jepa-weight", "0", "--model-variant", "addon")
                        )
                    else:
                        command.extend(
                            ("--jepa-weight", "0.1", "--model-variant", "replacement")
                        )
                    jobs.append(
                        OfficialJob(
                            dataset=dataset,
                            method=method,
                            missing_rate=rate,
                            seed=seed,
                            gpu=gpu,
                            slot=slot,
                            output_dir=output_dir,
                            command=tuple(command),
                        )
                    )
                    job_index += 1
    return jobs


def _latest_manifest(output_dir: Path) -> Path | None:
    manifests = sorted(
        output_dir.glob("run_records/*/run_manifest_fold_*.json")
    )
    return manifests[-1] if manifests else None


def _manifest_matches_job(manifest: dict, job: OfficialJob) -> bool:
    expected_fold = 5 if job.dataset.startswith("IEMOCAP") else 1
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
        manifest["run"]["dataset"] == job.dataset
        and manifest["run"]["fold"] == expected_fold
        and manifest["run"]["master_seed"] == job.seed
        and manifest["masks"]["requested_missing_rate"] == job.missing_rate
        and manifest["lifecycle"]["evaluation_protocol"] == "official"
        and all(
            manifest["method"][key] == value
            for key, value in expected_method.items()
        )
    )


def is_complete(job: OfficialJob) -> bool:
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
    except ManifestValidationError:
        return False
    return _manifest_matches_job(manifest, job)


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


def _acquire_claim(job: OfficialJob) -> Path | None:
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


def run_job(job: OfficialJob, root: Path, stop_event: threading.Event) -> bool:
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
            status["error"] = "training process exited with return code {}".format(
                returncode
            )
        _write_json(job.output_dir / "status.json", status)
        if returncode != 0:
            stop_event.set()
            return False
        if not is_complete(job):
            status["error"] = "return code 0 without a matching valid manifest"
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
    jobs: Sequence[OfficialJob], root: Path, stop_event: threading.Event
) -> bool:
    for job in jobs:
        if not run_job(job, root, stop_event):
            return False
    return True


def audit_completed_pairs(
    jobs: Sequence[OfficialJob], root: Path, python: str
) -> Tuple[int, int]:
    pairs: Dict[Tuple[str, float, int], Dict[str, OfficialJob]] = {}
    for job in jobs:
        pairs.setdefault(
            (job.dataset, job.missing_rate, job.seed), {}
        )[job.method] = job
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
                    "audit exception: {}: {}\n".format(type(error).__name__, error),
                    encoding="utf-8",
                )
            except OSError:
                pass
            continue
        audited_pairs += 1
        failures += int(result.returncode != 0)
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
    audit_failures: int,
) -> bool:
    return (
        worker_success
        and total_jobs == EXPECTED_JOB_COUNT
        and complete_jobs == EXPECTED_JOB_COUNT
        and audited_pairs == EXPECTED_PAIR_COUNT
        and audit_failures == 0
    )


def _parse_gpus(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        default="/data2/yb/reproduction_envs/gcnet-official/bin/python",
    )
    parser.add_argument("--gpus", type=_parse_gpus, default=DEFAULT_GPUS)
    parser.add_argument("--jobs-per-gpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output_root = args.output_root.resolve()
    jobs = build_jobs(
        output_root,
        args.python,
        args.gpus,
        args.jobs_per_gpu,
        args.epochs,
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

    assert_gpus_available(args.gpus)
    lanes: Dict[Tuple[int, int], List[OfficialJob]] = {
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
        success, worker_errors = _collect_worker_results(futures, stop_event)
    audited_pairs, audit_failures = audit_completed_pairs(jobs, root, args.python)
    complete_jobs = sum(is_complete(job) for job in jobs)
    _write_json(
        output_root / "scheduler_status.json",
        {
            "complete_jobs": complete_jobs,
            "total_jobs": len(jobs),
            "worker_success": success,
            "worker_errors": worker_errors,
            "paired_audits": audited_pairs,
            "expected_pair_audits": EXPECTED_PAIR_COUNT,
            "paired_audit_failures": audit_failures,
        },
    )
    return 0 if _scheduler_succeeded(
        success,
        complete_jobs,
        len(jobs),
        audited_pairs,
        audit_failures,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
