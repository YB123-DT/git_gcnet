"""Run the locked Original/MPFiLM IEMOCAP-6 fold-5 comparison."""

import argparse
from dataclasses import dataclass
import json
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Dict, Iterable, List, Sequence, Tuple


ARMS = ("original", "full")
ARM_TO_GRAPH_VARIANT = {
    "original": "original",
    "pattern_only": "pattern_only",
    "full": "full",
    "linearized_film": "full",
    "faithful_edgewise_film": "faithful_edgewise",
    "parameter_matched": "content_film_control",
    "cp_lecc": "cp_lecc",
}
GATE_RATES = (0.0, 0.7)
FORMAL_RATES = tuple(index / 10 for index in range(8))
GATE_SEEDS = (66, 67)
FORMAL_SEEDS = (66, 67, 68, 69, 70)
LOCKED_TRAINING = {
    "dataset": "IEMOCAPSix",
    "audio_feature": "wav2vec-large-c-UTT",
    "text_feature": "deberta-large-4-UTT",
    "video_feature": "manet_UTT",
    "base_model": "LSTM",
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
    "fold": 5,
}


@dataclass(frozen=True)
class Job:
    stage: str
    arm: str
    missing_rate: float
    seed: int
    output_directory: Path


def _rate_tag(rate: float) -> str:
    return f"{rate:.1f}".replace(".", "p")


def build_jobs(
    stage: str,
    output_root: Path,
    arms: Sequence[str] = ARMS,
    rates: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
) -> List[Job]:
    if stage == "gate":
        default_rates, default_seeds = GATE_RATES, GATE_SEEDS
    elif stage == "formal":
        default_rates, default_seeds = FORMAL_RATES, FORMAL_SEEDS
    else:
        raise ValueError(f"unknown stage: {stage!r}")
    rates = tuple(default_rates if rates is None else rates)
    seeds = tuple(default_seeds if seeds is None else seeds)
    arms = tuple(arms)
    unknown_arms = set(arms) - set(ARM_TO_GRAPH_VARIANT)
    if unknown_arms:
        raise ValueError(f"unknown arms: {sorted(unknown_arms)!r}")
    jobs = []
    for arm in arms:
        for rate in rates:
            for seed in seeds:
                output = (
                    Path(output_root)
                    / stage
                    / arm
                    / f"miss_{_rate_tag(rate)}"
                    / f"seed_{seed}"
                    / "fold_5"
                )
                jobs.append(Job(stage, arm, rate, seed, output))
    return jobs


def build_command(
    job: Job,
    python: Path,
    repository: Path,
    data_root: Path,
    mask_bank_root: Path,
) -> List[str]:
    return [
        str(python),
        "-u",
        str(Path(repository) / "gcnet" / "train_gcnet.py"),
        "--audio-feature",
        "wav2vec-large-c-UTT",
        "--text-feature",
        "deberta-large-4-UTT",
        "--video-feature",
        "manet_UTT",
        "--dataset",
        "IEMOCAPSix",
        "--data-root",
        str(data_root),
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
        "6",
        "--epochs",
        "100",
        "--seed",
        str(job.seed),
        "--mask-seed",
        str(job.seed),
        "--mask-type",
        f"constant-{job.missing_rate:.1f}",
        "--fold-index",
        "5",
        "--graph-conv-variant",
        ARM_TO_GRAPH_VARIANT[job.arm],
        "--mask-bank-root",
        str(mask_bank_root),
        "--output-dir",
        str(job.output_directory / "saved"),
        "--loss-recon",
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _job_payload(
    job: Job,
    gpu: str,
    python: Path,
    repository: Path,
    data_root: Path,
    mask_bank_root: Path,
) -> dict:
    return {
        "stage": job.stage,
        "arm": job.arm,
        "missing_rate": job.missing_rate,
        "seed": job.seed,
        "fold": 5,
        "gpu": str(gpu),
        "command": build_command(job, python, repository, data_root, mask_bank_root),
    }


def _completed_log(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return (
        sum(line.startswith("epoch:") for line in text.splitlines()) == 100
        and "Finish" in text
        and "save results in " in text
        and "SMOKE_ONLY=False" in text
    )


def _completed(
    job: Job,
    gpus: Sequence[str],
    python: Path,
    repository: Path,
    data_root: Path,
    mask_bank_root: Path,
) -> bool:
    directory = job.output_directory
    if not directory.exists():
        return False
    lock = directory / ".active.lock"
    if lock.exists():
        raise RuntimeError(f"active or stale job lock exists: {lock}")
    if not directory.is_dir() or not any(directory.iterdir()):
        raise RuntimeError(f"partial job directory exists: {directory}")
    required = ("command.json", "status.json", "train.log", "saved")
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"partial job artifacts in {directory}: missing {missing}")
    payload = json.loads((directory / "command.json").read_text(encoding="utf-8"))
    gpu = str(payload.get("gpu"))
    if gpu not in tuple(map(str, gpus)):
        raise RuntimeError(f"command.json GPU is outside requested GPUs: {gpu}")
    expected = _job_payload(
        job, gpu, python, repository, data_root, mask_bank_root
    )
    if payload != expected:
        raise RuntimeError(f"command.json mismatch for {directory}")
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "success" or status.get("return_code") != 0:
        raise RuntimeError(f"status return_code is not successful for {directory}")
    archives = list((directory / "saved").glob("*.npz"))
    if len(archives) != 1:
        raise RuntimeError(
            f"expected exactly one NPZ archive in {directory / 'saved'}, found {len(archives)}"
        )
    if not _completed_log(directory / "train.log"):
        raise RuntimeError(f"train.log does not contain exactly 100 epoch records and completion markers")
    return True


def _claim_job(job: Job) -> Path:
    job.output_directory.mkdir(parents=True, exist_ok=True)
    lock = job.output_directory / ".active.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"active or stale job lock exists: {lock}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "runner": str(Path(__file__).resolve())}, handle)
        handle.write("\n")
    return lock


def _release_job(job: Job) -> None:
    (job.output_directory / ".active.lock").unlink(missing_ok=True)


def _write_json_exclusive_or_equal(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable JSON mismatch: {path}")


def _launch(
    job: Job,
    gpu: str,
    python: Path,
    repository: Path,
    data_root: Path,
    mask_bank_root: Path,
) -> Tuple[subprocess.Popen, object]:
    _claim_job(job)
    payload = _job_payload(job, gpu, python, repository, data_root, mask_bank_root)
    command = payload["command"]
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
    environment["GCNET_CACHE_ROOT"] = str(mask_bank_root.parent / "dataset_cache")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONHASHSEED"] = "0"
    try:
        process = subprocess.Popen(
            command,
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


def run_jobs(
    jobs: Iterable[Job],
    gpus: Sequence[str],
    workers_per_gpu: int,
    python: Path,
    repository: Path,
    data_root: Path,
    mask_bank_root: Path,
) -> None:
    pending = [
        job
        for job in jobs
        if not _completed(
            job, gpus, python, repository, data_root, mask_bank_root
        )
    ]
    running: List[Tuple[Job, str, subprocess.Popen, object, float]] = []
    failures = []
    try:
        while pending or running:
            counts: Dict[str, int] = {gpu: 0 for gpu in gpus}
            for _, gpu, _, _, _ in running:
                counts[gpu] += 1
            while pending:
                available = [gpu for gpu in gpus if counts[gpu] < workers_per_gpu]
                if not available:
                    break
                gpu = min(available, key=lambda item: (counts[item], gpus.index(item)))
                job = pending.pop(0)
                process, handle = _launch(
                    job, gpu, python, repository, data_root, mask_bank_root
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
                status = {
                    "status": "success" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "elapsed_seconds": time.time() - started_at,
                    "gpu": gpu,
                }
                _write_json(job.output_directory / "status.json", status)
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
        raise RuntimeError(f"failed jobs: {failures}")


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _ensure_run_manifest(
    output_root: Path,
    stage: str,
    repository: Path,
    data_root: Path,
    mask_bank_root: Path,
    arms: Sequence[str],
    rates: Sequence[float],
    seeds: Sequence[int],
    gpus: Sequence[str],
    workers_per_gpu: int,
    python: Path,
    command_output=subprocess.check_output,
) -> dict:
    def output(command):
        return command_output(command, cwd=repository, text=True).strip()

    status = output(["git", "status", "--porcelain"])
    if status:
        raise RuntimeError("locked experiments require a clean git worktree")
    try:
        gpu_text = output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        )
        gpu_names = [line for line in gpu_text.splitlines() if line]
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu_names = []
    torch_version = _package_version("torch")
    try:
        import torch
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    except ImportError:
        cuda_version = cudnn_version = None
    manifest = {
        "git": {"head": output(["git", "rev-parse", "HEAD"]), "clean": True},
        "python": {"executable": str(python), "version": platform.python_version()},
        "versions": {
            "torch": torch_version,
            "torch_geometric": _package_version("torch_geometric"),
            "cuda": cuda_version,
            "cudnn": cudnn_version,
        },
        "gpu_names": gpu_names,
        "roots": {
            "repository": str(Path(repository).resolve()),
            "data": str(Path(data_root).resolve()),
            "mask_bank": str(Path(mask_bank_root).resolve()),
        },
        "stage": stage,
        "arms": list(arms),
        "rates": list(rates),
        "seeds": list(seeds),
        "fold": 5,
        "locked_training": LOCKED_TRAINING,
        "gpus": list(gpus),
        "workers_per_gpu": workers_per_gpu,
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
            raise RuntimeError(f"run manifest mismatch: {path}") from error
        if existing != manifest:
            raise RuntimeError(f"run manifest mismatch: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mask-bank-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/yangbin/miniconda3/envs/multimodalerc310/bin/python"),
    )
    parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3"))
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(ARM_TO_GRAPH_VARIANT),
        default=ARMS,
    )
    parser.add_argument("--rates", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    repository = Path(__file__).resolve().parents[2]
    jobs = build_jobs(
        args.stage,
        args.output_root,
        arms=args.arms,
        rates=args.rates,
        seeds=args.seeds,
    )
    _ensure_run_manifest(
        args.output_root,
        args.stage,
        repository,
        args.data_root,
        args.mask_bank_root,
        tuple(args.arms),
        tuple(dict.fromkeys(job.missing_rate for job in jobs)),
        tuple(dict.fromkeys(job.seed for job in jobs)),
        tuple(args.gpus),
        args.workers_per_gpu,
        args.python,
    )
    # Run arms sequentially so every comparison uses the same immutable banks
    # without oversubscribing the requested per-GPU worker limit.
    for arm in args.arms:
        run_jobs(
            (job for job in jobs if job.arm == arm),
            tuple(args.gpus),
            args.workers_per_gpu,
            args.python,
            repository,
            args.data_root,
            args.mask_bank_root,
        )


if __name__ == "__main__":
    main()
