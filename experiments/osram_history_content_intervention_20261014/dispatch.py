"""Dispatch 30 IEMOCAP4/6 and three MELD P1 sources without training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.osram_history_effect_stratification_20261013.dispatch import DATASETS, build_tasks
from experiments.osram_history_effect_stratification_20261013.evaluate import _write, validate_source


def preflight(tasks, rates):
    for task in tasks:
        source = Path(task["source"])
        config = json.loads((source/"config.json").read_text())
        validate_source(config)
        if (config["dataset"], config["seed"], config["fold"]) != (DATASETS[task["dataset"]], task["seed"], task["fold"]):
            raise ValueError(f"source identity mismatch: {source}")
        for rate in rates:
            suffix = f"{rate:.1f}".replace(".", "p")
            names = [f"best_miss_{suffix}.pt"]
            if task["dataset"] != "meld":
                names.append(f"predictions_miss_{suffix}.npz")
            for name in names:
                if not (source/name).is_file():
                    raise FileNotFoundError(source/name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iemocap-source-root", type=Path)
    parser.add_argument("--meld-source-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--meld-pickle", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", choices=tuple(DATASETS), nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[66, 67, 68])
    parser.add_argument("--rates", type=float, nargs="+", default=[i/10 for i in range(8)])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if min(args.workers, args.chunk_size) < 1 or len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("positive workers/chunk size; unique datasets/seeds required")
    if not args.rates or len(set(args.rates)) != len(args.rates) or any(rate not in {i/10 for i in range(8)} for rate in args.rates):
        raise ValueError("unique nonempty 0.0--0.7 rate grid required")
    if any(dataset != "meld" for dataset in args.datasets) and args.iemocap_source_root is None:
        raise ValueError("IEMOCAP source root required")
    if "meld" in args.datasets and (args.meld_source_root is None or args.meld_pickle is None or not args.meld_pickle.is_file()):
        raise ValueError("MELD source root and source pickle required")
    tasks = build_tasks(args.iemocap_source_root, args.meld_source_root, args.output_root, args.datasets, args.seeds)
    preflight(tasks, args.rates)
    args.output_root.mkdir(parents=True)
    (args.output_root/"logs").mkdir()
    manifest = dict(status="running", training=False, tasks=tasks, completed=[], failed=[], rates=args.rates)
    _write(args.output_root/"DISPATCH.json", manifest)

    def run(task):
        command = [sys.executable, "-u", "-m", "experiments.osram_history_content_intervention_20261014.evaluate",
                   "--source", task["source"], "--output", task["output"], "--dataset-root", str(args.dataset_root),
                   "--device", args.device, "--chunk-size", str(args.chunk_size), "--rates", *map(str, args.rates)]
        if task["dataset"] == "meld":
            command.extend(["--meld-pickle", str(args.meld_pickle)])
        log = args.output_root/"logs"/f"{task['dataset']}_seed{task['seed']}_fold{task['fold']}.log"
        with log.open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"), check=False)
        path = Path(task["output"])/"SUMMARY.json"
        status = json.loads(path.read_text()).get("status") if path.is_file() else None
        return dict(task, command=command, log=str(log), returncode=result.returncode,
                    success=result.returncode == 0 and status == "complete")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = dict(futures[future], success=False, error=repr(error))
            manifest["completed" if result["success"] else "failed"].append(result)
            _write(args.output_root/"DISPATCH.json", manifest)
    manifest["status"] = "failed" if manifest["failed"] else "complete"
    _write(args.output_root/"DISPATCH.json", manifest)
    if manifest["failed"]:
        raise RuntimeError(f"{len(manifest['failed'])} P1 sources failed")


if __name__ == "__main__":
    main()
