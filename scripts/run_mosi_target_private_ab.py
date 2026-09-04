"""Run the paired MOSI shared-vs-target-private Missing-M3 experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SEEDS = (66, 67, 68, 69, 70)
RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
ARMS = {"shared": 0, "target-private": 32}


@dataclass(frozen=True)
class Job:
    arm: str
    seed: int
    target_private_rank: int
    gpu: int
    output_dir: Path
    command: tuple[str, ...]


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _build_jobs(repo_root, output_root, python, feature_root, gpus):
    if not gpus or 4 in gpus:
        raise ValueError("provide at least one healthy GPU and exclude GPU 4")
    jobs = []
    index = 0
    for seed in SEEDS:
        for arm, rank in ARMS.items():
            output_dir = Path(output_root) / arm / f"seed_{seed}"
            command = (
                str(python), "-u", "-m", "gcnet_missing_m3.train_gcnet",
                "--dataset", "CMUMOSI", "--fold", "1",
                "--audio-feature", "wav2vec-large-c-UTT",
                "--text-feature", "deberta-large-4-UTT",
                "--video-feature", "manet_UTT",
                "--feature-root", str(feature_root),
                "--output-dir", str(output_dir),
                "--seed", str(seed), "--epochs", "100", "--batch-size", "32",
                "--train-rate-mode", "stratified", "--hidden", "200",
                "--windowp", "2", "--windowf", "2", "--lr", "0.001",
                "--l2", "0.00001", "--dropout", "0.5",
                "--gradient-clip-norm", "1.0", "--evaluation-protocol", "official",
                "--num-threads", "2", "--fusion-type", "slot",
                "--representation-type", "slot", "--graph-branch-mode", "both",
                "--mmoe-variant", "dual-gate", "--latent-dim", "256",
                "--num-experts", "4", "--top-k", "2", "--jepa-weight", "0.1",
                "--temperature", "0.03", "--ema-tau", "0.996",
                "--target-private-rank", str(rank),
            )
            jobs.append(Job(arm, seed, rank, gpus[index % len(gpus)], output_dir, command))
            index += 1
    return jobs


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_complete(job):
    config = _read_json(job.output_dir / "config.json")
    history = _read_json(job.output_dir / "history.json")
    metrics = _read_json(job.output_dir / "metrics.json")
    if config is None or history is None or metrics is None or not (job.output_dir / "best.pt").exists():
        return False
    return (
        config.get("dataset") == "CMUMOSI"
        and config.get("seed") == job.seed
        and config.get("train_rate_mode") == "stratified"
        and config.get("target_private_rank") == job.target_private_rank
        and len(history) == 100
        and metrics.get("selection_split") == "validation"
        and int(metrics.get("best_epoch", 0)) in range(1, 101)
        and set(metrics.get("test", {})) == {str(rate) for rate in RATES}
    )


def summarize_jobs(jobs):
    if not all(_is_complete(job) for job in jobs):
        return {"complete": False}
    by_key = {(job.arm, job.seed): _read_json(job.output_dir / "metrics.json") for job in jobs}
    paired = {}
    rate_summary = {}
    seed_macros = {}
    mask_match = True
    for seed in SEEDS:
        control = by_key[("shared", seed)]
        treatment = by_key[("target-private", seed)]
        paired[str(seed)] = {}
        nonzero = []
        for rate in RATES:
            key = str(rate)
            left, right = control["test"][key], treatment["test"][key]
            mask_match &= left.get("mask_sha256") == right.get("mask_sha256")
            delta = float(right["weighted_f1"]) - float(left["weighted_f1"])
            paired[str(seed)][key] = {
                "shared": float(left["weighted_f1"]),
                "target_private": float(right["weighted_f1"]),
                "delta": delta,
            }
            if rate > 0:
                nonzero.append(delta)
        seed_macros[str(seed)] = sum(nonzero) / len(nonzero)
    for rate in RATES:
        key = str(rate)
        deltas = [paired[str(seed)][key]["delta"] for seed in SEEDS]
        rate_summary[key] = {
            "mean_delta": sum(deltas) / len(deltas),
            "positive_seed_count": sum(value > 0 for value in deltas),
            "deltas": deltas,
        }
    positive_rates = sum(rate_summary[str(rate)]["mean_delta"] > 0 for rate in RATES[1:])
    positive_seeds = sum(value > 0 for value in seed_macros.values())
    miss0_delta = rate_summary["0.0"]["mean_delta"]
    treatment_values = [paired[str(seed)][str(rate)]["target_private"] for seed in SEEDS for rate in RATES]
    parameter_delta = by_key[("target-private", 66)].get("parameter_count", 0) - by_key[("shared", 66)].get("parameter_count", 0)
    gate = mask_match and positive_rates >= 4 and positive_seeds >= 3 and miss0_delta >= -0.005 and min(treatment_values) > 0.5
    return {
        "complete": True,
        "mask_hashes_match": mask_match,
        "paired": paired,
        "rate_summary": rate_summary,
        "nonzero_macro": {
            "by_seed": seed_macros,
            "mean_delta": sum(seed_macros.values()) / len(seed_macros),
            "positive_seed_count": positive_seeds,
        },
        "parameter_delta": parameter_delta,
        "verdict": {
            "positive_nonzero_rate_count": positive_rates,
            "miss0_mean_delta": miss0_delta,
            "no_collapse": min(treatment_values) > 0.5,
            "passes_predefined_gate": gate,
        },
    }


def _start(job, repo_root):
    job.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    environment["PYTHONPATH"] = f"{repo_root}:{repo_root / 'gcnet'}"
    log = (job.output_dir / "train.log").open("w", encoding="utf-8")
    process = subprocess.Popen(job.command, cwd=repo_root, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
    return process, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--max-concurrent-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo_root, output_root = args.repo_root.resolve(), args.output_root.resolve()
    jobs = _build_jobs(repo_root, output_root, args.python, args.feature_root, tuple(args.gpus))
    if args.dry_run:
        for job in jobs:
            print(job.arm, job.seed, job.gpu, job.output_dir)
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "manifest.json", {"jobs": [{**asdict(job), "output_dir": str(job.output_dir), "command": list(job.command)} for job in jobs]})
    pending = [job for job in jobs if not _is_complete(job)]
    running = {}
    failures = []
    while pending or running:
        counts = {gpu: 0 for gpu in args.gpus}
        for job, _, _ in running.values():
            counts[job.gpu] += 1
        for job in list(pending):
            if counts[job.gpu] >= args.max_concurrent_per_gpu:
                continue
            process, log = _start(job, repo_root)
            running[process.pid] = (job, process, log)
            counts[job.gpu] += 1
            pending.remove(job)
        _atomic_json(output_root / "runner_status.json", {"pending": len(pending), "running": [{"pid": pid, "arm": value[0].arm, "seed": value[0].seed, "gpu": value[0].gpu} for pid, value in running.items()], "failures": failures})
        time.sleep(args.poll_seconds)
        for pid, (job, process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code != 0 or not _is_complete(job):
                failures.append({"arm": job.arm, "seed": job.seed, "gpu": job.gpu, "returncode": code})
            del running[pid]
    summary = summarize_jobs(jobs)
    summary["failures"] = failures
    _atomic_json(output_root / "paired_summary.json", summary)
    return 0 if summary.get("complete") and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
