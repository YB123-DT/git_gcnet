"""Three-GPU five-seed runner for the Text-Anchored backbone."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Sequence, Tuple

from gcnet_missing_m3_sam_backbone.run_mosi import (
    DEFAULT_CONTROL_SUMMARY,
    DEFAULT_FEATURE_ROOT,
    DEFAULT_GPUS,
    DEFAULT_PYTHON,
    SEEDS,
    load_control,
    summarize,
)
from gcnet_missing_m3_sam_backbone.train_mosi import write_json


def _job(seed: int, gpu: int, output: Path, epochs: int) -> Tuple[int, int, Path, Tuple[str, ...]]:
    return (
        seed,
        gpu,
        output,
        (
            DEFAULT_PYTHON,
            "-m",
            "gcnet_missing_m3_text_anchor.train_mosi",
            "--feature-root",
            DEFAULT_FEATURE_ROOT,
            "--output-dir",
            str(output),
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--device",
            "cuda",
        ),
    )


def _inspect(output: Path, seed: int, epochs: int):
    required = ("config.json", "history.json", "best_checkpoint.pt", "metrics.json", "predictions.npz")
    for name in required:
        if not (output / name).is_file():
            return None
    metrics = json.loads((output / "metrics.json").read_text())
    if metrics.get("variant") != "text-anchored-residual-backbone":
        return None
    if metrics.get("selection_split") != "validation_weighted_f1":
        raise RuntimeError("test-oracle result rejected")
    if int(metrics.get("seed", -1)) != seed or int(metrics.get("history_epochs", -1)) != epochs:
        return None
    if bool(metrics.get("collapsed", True)):
        raise RuntimeError("seed {} collapsed".format(seed))
    return float(metrics["test"]["weighted_f1"])


def _queue(jobs, cwd: Path, epochs: int) -> None:
    for seed, gpu, output, command in jobs:
        inherited = _inspect(output, seed, epochs)
        if inherited is not None:
            print("inherit seed={} f1={:.2f}".format(seed, inherited * 100), flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print("start seed={} gpu={}".format(seed, gpu), flush=True)
        with (output / "train.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=7200,
                check=False,
            )
        if result.returncode:
            raise RuntimeError("seed {} failed; inspect {}".format(seed, output / "train.log"))
        score = _inspect(output, seed, epochs)
        if score is None:
            raise RuntimeError("seed {} produced incomplete artifacts".format(seed))
        print("done seed={} gpu={} f1={:.2f}".format(seed, gpu, score * 100), flush=True)


def run(output_root: Path, gpus: Sequence[int], epochs: int):
    if not gpus:
        raise ValueError("at least one GPU is required")
    root = output_root.resolve()
    jobs = [
        _job(seed, int(gpus[index % len(gpus)]), root / "seed_{}".format(seed), epochs)
        for index, seed in enumerate(SEEDS)
    ]
    queues = {int(gpu): [] for gpu in gpus}
    for job in jobs:
        queues[job[1]].append(job)
    cwd = Path(__file__).resolve().parent.parent
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [executor.submit(_queue, queue, cwd, epochs) for queue in queues.values() if queue]
        for future in futures:
            future.result()
    candidate: Dict[int, float] = {
        seed: _inspect(output, seed, epochs)
        for seed, _, output, _ in jobs
    }
    summary = summarize(candidate, load_control(Path(DEFAULT_CONTROL_SUMMARY)))
    summary["variant"] = "text-anchored-residual-backbone"
    summary["selection_split"] = "validation_weighted_f1"
    write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    print(json.dumps(run(Path(args.output_root), gpus, args.epochs), indent=2), flush=True)


if __name__ == "__main__":
    main()
