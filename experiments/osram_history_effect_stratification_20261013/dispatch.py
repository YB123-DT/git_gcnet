"""Dispatch the 30 IEMOCAP and three MELD frozen P0 source replays."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.osram_history_effect_stratification_20261013.evaluate import _write, validate_source

DATASETS = {"iemocap4": "IEMOCAPFour", "iemocap6": "IEMOCAPSix", "meld": "MELD"}


def build_tasks(iemocap_source_root: Path, meld_source_root: Path, output_root: Path,
                datasets=("iemocap4", "iemocap6", "meld"), seeds=(66, 67, 68)) -> list[dict]:
    tasks = []
    for dataset in datasets:
        for seed in seeds:
            for fold in ((1,) if dataset == "meld" else range(1, 6)):
                source = (meld_source_root / f"flat_seed{seed}" if dataset == "meld" else
                          iemocap_source_root / dataset / f"seed_{seed}" / f"fold_{fold}")
                tasks.append({"dataset": dataset, "seed": seed, "fold": fold, "source": str(source),
                              "output": str(output_root / dataset / f"seed_{seed}" / f"fold_{fold}")})
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iemocap-source-root", type=Path)
    parser.add_argument("--meld-source-root", type=Path, help="directory containing flat_seed66/67/68")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", choices=tuple(DATASETS), nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[66, 67, 68])
    parser.add_argument("--rates", type=float, nargs="+", default=[index / 10 for index in range(8)])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.workers < 1 or len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("workers must be positive; datasets and seeds must be unique")
    if len(set(args.rates)) != len(args.rates) or any(rate not in {index / 10 for index in range(8)} for rate in args.rates):
        raise ValueError("rates must be unique members of 0.0--0.7")
    if any(dataset != "meld" for dataset in args.datasets) and args.iemocap_source_root is None:
        raise ValueError("IEMOCAP requires --iemocap-source-root")
    if "meld" in args.datasets and (args.meld_source_root is None or args.meld_pickle is None or not args.meld_pickle.is_file()):
        raise ValueError("MELD requires --meld-source-root and the source --meld-pickle")
    tasks = build_tasks(args.iemocap_source_root, args.meld_source_root, args.output_root, args.datasets, args.seeds)
    # Preflight all sources before starting or creating any output.
    for task in tasks:
        source = Path(task["source"])
        config = json.loads((source / "config.json").read_text())
        validate_source(config)
        if (config["dataset"], config["seed"], config["fold"]) != (DATASETS[task["dataset"]], task["seed"], task["fold"]):
            raise ValueError(f"source identity mismatch: {source}")
        for rate in args.rates:
            suffix = f"{rate:.1f}".replace(".", "p")
            for filename in ([f"best_miss_{suffix}.pt", f"predictions_miss_{suffix}.npz"] if task["dataset"] != "meld" else [f"best_miss_{suffix}.pt"]):
                if not (source / filename).is_file():
                    raise FileNotFoundError(source / filename)
    args.output_root.mkdir(parents=True)
    logs = args.output_root / "logs"
    logs.mkdir()
    manifest = {"status": "running", "training": False, "workers": args.workers,
                "device": args.device, "rates": args.rates, "tasks": tasks, "completed": [], "failed": []}
    _write(args.output_root / "DISPATCH.json", manifest)

    def run(task):
        command = [sys.executable, "-u", "-m", "experiments.osram_history_effect_stratification_20261013.evaluate",
                   "--source", task["source"], "--dataset-root", str(args.dataset_root), "--output", task["output"],
                   "--device", args.device, "--rates", *[str(rate) for rate in args.rates]]
        if task["dataset"] == "meld":
            command.extend(["--meld-pickle", str(args.meld_pickle)])
        log = logs / f"{task['dataset']}_seed{task['seed']}_fold{task['fold']}.log"
        environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        with log.open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=False)
        completed_summary = Path(task["output"]) / "SUMMARY.json"
        status = json.loads(completed_summary.read_text()).get("status") if completed_summary.is_file() else None
        return {**task, "command": command, "log": str(log), "returncode": result.returncode,
                "summary_status": status, "success": result.returncode == 0 and status == "complete"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {**task, "success": False, "error": repr(error)}
            manifest["completed" if result["success"] else "failed"].append(result)
            _write(args.output_root / "DISPATCH.json", manifest)
            print(f"{'COMPLETE' if result['success'] else 'FAILED'} {task['dataset']} seed={task['seed']} fold={task['fold']}", flush=True)
    manifest["status"] = "failed" if manifest["failed"] else "complete"
    _write(args.output_root / "DISPATCH.json", manifest)
    if manifest["failed"]:
        raise RuntimeError(f"{len(manifest['failed'])} frozen runs failed; inspect DISPATCH.json and logs")


if __name__ == "__main__":
    main()
