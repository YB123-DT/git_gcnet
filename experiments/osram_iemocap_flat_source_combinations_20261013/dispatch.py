"""Run all IEMOCAP source-combination replays with bounded concurrency."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


TASKS = ("iemocap4", "iemocap6")
SEEDS = (66, 67, 68)
FOLDS = (1, 2, 3, 4, 5)


def complete(path: Path) -> bool:
    summary = path / "SUMMARY.json"
    if not summary.is_file():
        return False
    try:
        return json.loads(summary.read_text()).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(exist_ok=True)
    tasks = []
    for dataset in TASKS:
        for seed in SEEDS:
            for fold in FOLDS:
                output = args.output_root / dataset / f"seed_{seed}" / f"fold_{fold}"
                if complete(output):
                    continue
                if output.exists():
                    raise FileExistsError(f"Incomplete output requires inspection: {output}")
                tasks.append((dataset, seed, fold, output))

    def run(task):
        dataset, seed, fold, output = task
        source = args.source_root / dataset / f"seed_{seed}" / f"fold_{fold}"
        log_path = log_root / f"{dataset}_seed{seed}_fold{fold}.log"
        command = [
            sys.executable,
            "-u",
            "-m",
            "experiments.osram_iemocap_flat_source_combinations_20261013.evaluate",
            "--source",
            str(source),
            "--dataset-root",
            str(args.dataset_root),
            "--output",
            str(output),
            "--device",
            args.device,
        ]
        environment = dict(
            os.environ,
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
        )
        with log_path.open("w") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{dataset} seed={seed} fold={fold} failed; see {log_path}")
        return dataset, seed, fold

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            dataset, seed, fold = future.result()
            print(f"COMPLETE {dataset} seed={seed} fold={fold}", flush=True)


if __name__ == "__main__":
    main()
