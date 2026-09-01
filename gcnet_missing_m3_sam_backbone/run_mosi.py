"""Run and audit the five-seed CMU-MOSI Stage-1 SAM experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .train_mosi import write_json


SEEDS: Tuple[int, ...] = (66, 67, 68, 69, 70)
DEFAULT_GPUS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_PYTHON = "/data2/yb/reproduction_envs/gcnet-official/bin/python3"
DEFAULT_FEATURE_ROOT = (
    "/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/"
    "dataset/CMUMOSI/features"
)
DEFAULT_CONTROL_SUMMARY = (
    "/data2/yb/remote_experiments/"
    "m3_mosi_complete_20260828/formal/summary.json"
)
REQUIRED_ARTIFACTS = (
    "config.json",
    "history.json",
    "best_checkpoint.pt",
    "metrics.json",
    "predictions.npz",
)


@dataclass(frozen=True)
class SAMJob:
    seed: int
    gpu: int
    output_dir: Path
    command: Tuple[str, ...]


@dataclass(frozen=True)
class ResultInspection:
    complete: bool
    reason: str
    collapsed: bool = False
    weighted_f1: Optional[float] = None


def build_jobs(
    output_root: Path,
    gpus: Sequence[int] = DEFAULT_GPUS,
    python: str = DEFAULT_PYTHON,
    feature_root: str = DEFAULT_FEATURE_ROOT,
    epochs: int = 100,
) -> List[SAMJob]:
    """Build exactly five candidate jobs; controls are never trained here."""
    if not gpus:
        raise ValueError("at least one GPU is required")
    root = Path(output_root)
    jobs: List[SAMJob] = []
    for index, seed in enumerate(SEEDS):
        gpu = int(gpus[index % len(gpus)])
        output = root / "seed_{}".format(seed)
        command = (
            str(python),
            "-m",
            "gcnet_missing_m3_sam_backbone.train_mosi",
            "--feature-root",
            str(feature_root),
            "--output-dir",
            str(output),
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--device",
            "cuda",
        )
        jobs.append(SAMJob(seed, gpu, output, command))
    return jobs


def inspect_result(
    output_dir: Path,
    expected_seed: int,
    expected_epochs: int,
) -> ResultInspection:
    """Accept only a fully written, validation-selected formal result."""
    output = Path(output_dir)
    metrics_path = output / "metrics.json"
    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        try:
            preliminary = json.loads(metrics_path.read_text())
        except (OSError, ValueError, TypeError) as error:
            return ResultInspection(False, "invalid metrics JSON: {}".format(error))
        if preliminary.get("selection_split") != "validation":
            return ResultInspection(False, "selection split is not validation")
    for name in REQUIRED_ARTIFACTS:
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            return ResultInspection(False, "missing or empty {}".format(name))
    try:
        metrics = json.loads(metrics_path.read_text())
        history = json.loads((output / "history.json").read_text())
        config = json.loads((output / "config.json").read_text())
    except (OSError, ValueError, TypeError) as error:
        return ResultInspection(False, "invalid JSON: {}".format(error))
    if metrics.get("variant") != "mask-aware-sam-backbone":
        return ResultInspection(False, "variant provenance mismatch")
    if int(metrics.get("seed", -1)) != int(expected_seed):
        return ResultInspection(False, "seed provenance mismatch")
    if int(metrics.get("history_epochs", -1)) != int(expected_epochs):
        return ResultInspection(False, "history epoch count mismatch")
    if len(history) != int(expected_epochs):
        return ResultInspection(False, "history.json epoch count mismatch")
    if int(config.get("seed", -1)) != int(expected_seed):
        return ResultInspection(False, "config seed mismatch")
    if int(config.get("epochs", -1)) != int(expected_epochs):
        return ResultInspection(False, "config epoch count mismatch")
    test = metrics.get("test")
    if not isinstance(test, dict):
        return ResultInspection(False, "test metrics are absent")
    try:
        weighted_f1 = float(test["weighted_f1"])
    except (KeyError, TypeError, ValueError):
        return ResultInspection(False, "test weighted_f1 is invalid")
    if not math.isfinite(weighted_f1):
        return ResultInspection(False, "test weighted_f1 is not finite")
    collapsed = bool(metrics.get("collapsed", True))
    if collapsed:
        return ResultInspection(
            False,
            "formal prediction collapsed",
            collapsed=True,
            weighted_f1=weighted_f1,
        )
    return ResultInspection(True, "complete", False, weighted_f1)


def load_control(summary_path: Path) -> Dict[int, float]:
    """Load the inherited strongest complete-MOSI five-seed control."""
    payload = json.loads(Path(summary_path).read_text())
    records = payload.get("records", [])
    control: Dict[int, float] = {}
    for record in records:
        if record.get("variant") != "m3_mosi":
            continue
        seed = int(record["seed"])
        if seed in SEEDS:
            control[seed] = float(record["test_f1"])
    if set(control) != set(SEEDS):
        raise ValueError("control summary lacks exact m3_mosi seeds 66--70")
    if not all(math.isfinite(value) for value in control.values()):
        raise ValueError("control summary contains non-finite F1")
    return control


def summarize(
    candidate: Mapping[int, float],
    control: Mapping[int, float],
    collapsed_seeds: Iterable[int] = (),
) -> Dict[str, object]:
    """Apply the pre-registered paired five-seed Stage-1 gate."""
    if set(candidate) != set(SEEDS) or set(control) != set(SEEDS):
        raise ValueError("candidate and control must contain exact seeds 66--70")
    collapsed = sorted(set(int(seed) for seed in collapsed_seeds))
    records = []
    deltas = []
    for seed in SEEDS:
        candidate_f1 = float(candidate[seed])
        control_f1 = float(control[seed])
        delta = candidate_f1 - control_f1
        deltas.append(delta)
        records.append(
            {
                "seed": seed,
                "candidate_f1": candidate_f1,
                "control_f1": control_f1,
                "delta": delta,
            }
        )
    candidate_mean = sum(float(candidate[seed]) for seed in SEEDS) / len(SEEDS)
    control_mean = sum(float(control[seed]) for seed in SEEDS) / len(SEEDS)
    mean_delta = candidate_mean - control_mean
    positive_count = sum(delta > 0 for delta in deltas)
    passed = mean_delta > 0 and positive_count >= 3 and not collapsed
    return {
        "variant": "mask-aware-sam-backbone",
        "dataset": "CMUMOSI",
        "missing_rate": 0.0,
        "selection_split": "validation",
        "control_variant": "m3_mosi",
        "control_provenance": "inherited; not retrained by this runner",
        "seeds": list(SEEDS),
        "records": records,
        "candidate_mean_f1": candidate_mean,
        "control_mean_f1": control_mean,
        "mean_delta": mean_delta,
        "positive_seed_count": positive_count,
        "collapsed_seeds": collapsed,
        "gate": "mean_delta>0 and positive_seed_count>=3 and no collapse",
        "passed": passed,
    }


def _run_gpu_queue(jobs: Sequence[SAMJob], cwd: Path, epochs: int) -> None:
    for job in jobs:
        inspection = inspect_result(job.output_dir, job.seed, epochs)
        if inspection.complete:
            print("inherit complete seed={} gpu={}".format(job.seed, job.gpu), flush=True)
            continue
        job.output_dir.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        log_path = job.output_dir / "train.log"
        print("start seed={} gpu={}".format(job.seed, job.gpu), flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                job.command,
                cwd=str(cwd),
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=7200,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "seed {} exited {} (see {})".format(
                    job.seed,
                    completed.returncode,
                    log_path,
                )
            )
        inspection = inspect_result(job.output_dir, job.seed, epochs)
        if not inspection.complete:
            raise RuntimeError(
                "seed {} failed audit: {}".format(job.seed, inspection.reason)
            )
        print(
            "done seed={} gpu={} test_f1={:.2f}".format(
                job.seed,
                job.gpu,
                100.0 * float(inspection.weighted_f1),
            ),
            flush=True,
        )


def _write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# Mask-Aware SAM Backbone — CMU-MOSI miss0",
        "",
        "Selection: validation loss. Test was evaluated once after training.",
        "Control: inherited `m3_mosi`; it was not retrained.",
        "",
        "| Seed | SAM W-F1 | Control W-F1 | Delta |",
        "|---:|---:|---:|---:|",
    ]
    for record in summary["records"]:
        lines.append(
            "| {seed} | {candidate_f1:.4%} | {control_f1:.4%} | {delta:+.4%} |".format(
                **record
            )
        )
    lines.extend(
        [
            "",
            "Candidate mean: {:.4%}".format(summary["candidate_mean_f1"]),
            "Control mean: {:.4%}".format(summary["control_mean_f1"]),
            "Mean delta: {:+.4%}".format(summary["mean_delta"]),
            "Positive seeds: {}/5".format(summary["positive_seed_count"]),
            "Collapsed seeds: {}".format(summary["collapsed_seeds"]),
            "Gate: **{}**".format("PASS" if summary["passed"] else "FAIL"),
            "",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(str(temporary), str(path))


def run_formal(
    output_root: Path,
    gpus: Sequence[int],
    python: str,
    feature_root: str,
    control_summary: Path,
    epochs: int,
) -> Dict[str, object]:
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".runner.lock"
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another formal runner owns this output root") from error
        jobs = build_jobs(output, gpus, python, feature_root, epochs)
        queues: Dict[int, List[SAMJob]] = {int(gpu): [] for gpu in gpus}
        for job in jobs:
            queues[job.gpu].append(job)
        cwd = Path(__file__).resolve().parent.parent
        with ThreadPoolExecutor(max_workers=len(queues)) as executor:
            futures = [
                executor.submit(_run_gpu_queue, queue, cwd, epochs)
                for queue in queues.values()
                if queue
            ]
            for future in futures:
                future.result()
        candidate: Dict[int, float] = {}
        collapsed: List[int] = []
        for job in jobs:
            inspection = inspect_result(job.output_dir, job.seed, epochs)
            if inspection.weighted_f1 is not None:
                candidate[job.seed] = inspection.weighted_f1
            if inspection.collapsed:
                collapsed.append(job.seed)
            if not inspection.complete:
                raise RuntimeError(
                    "formal result audit failed for seed {}: {}".format(
                        job.seed,
                        inspection.reason,
                    )
                )
        summary = summarize(candidate, load_control(control_summary), collapsed)
        write_json(output / "summary.json", summary)
        _write_markdown(output / "summary.md", summary)
        return summary


def _parse_gpus(value: str) -> Tuple[int, ...]:
    gpus = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not gpus:
        raise argparse.ArgumentTypeError("GPU list must not be empty")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("GPU list contains duplicates")
    return gpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=_parse_gpus, default=DEFAULT_GPUS)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--feature-root", default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--control-summary", default=DEFAULT_CONTROL_SUMMARY)
    parser.add_argument("--epochs", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_formal(
        Path(args.output_root),
        args.gpus,
        args.python,
        args.feature_root,
        Path(args.control_summary),
        args.epochs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "SEEDS",
    "ResultInspection",
    "SAMJob",
    "build_jobs",
    "inspect_result",
    "load_control",
    "run_formal",
    "summarize",
]
