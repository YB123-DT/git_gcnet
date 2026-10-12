"""Run the authorized 70-run MELD Flat/Nested screen in five 14-run waves."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gcnet_missing_m3.meaningful_new40_registry import MELD_NESTED_GRID


GPU_UUIDS = {
    0: "GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45",
    1: "GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049",
}


def _cost(method: str) -> int:
    config = MELD_NESTED_GRID[method]
    return config["dim"] * config["depth"] * config["groups"] * config["mlp_hidden"]


def _interleave_extremes(values: list[str]) -> list[str]:
    ordered = sorted(values, key=lambda method: (_cost(method), method))
    result = []
    left, right = 0, len(ordered) - 1
    while left <= right:
        result.append(ordered[right])
        right -= 1
        if left <= right:
            result.append(ordered[left])
            left += 1
    return result


def build_jobs() -> list[dict]:
    anchors = [
        {
            "name": f"{variant}_seed{seed}",
            "kind": "anchor",
            "variant": variant,
            "seed": seed,
            "meaningful_block": None,
        }
        for seed in (66, 67, 68)
        for variant in ("flat", "nested")
    ]
    grid = [
        {
            "name": method,
            "kind": "grid",
            "variant": "nested",
            "seed": 66,
            "meaningful_block": method,
        }
        for method in _interleave_extremes(list(MELD_NESTED_GRID))
    ]
    jobs = anchors + grid
    if len(jobs) != 70:
        raise AssertionError("the MELD screen must contain exactly 70 runs")
    for index, job in enumerate(jobs):
        job["wave"] = index // 14
        job["gpu"] = (index % 14) % 2
    return jobs


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_gpus() -> None:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    found = {}
    for line in raw.splitlines():
        index, identifier, name, total = [value.strip() for value in line.split(",")]
        found[int(index)] = (identifier, name, int(total))
    for index, identifier in GPU_UUIDS.items():
        if index == 4 or index not in found or found[index][0] != identifier:
            raise RuntimeError(f"GPU {index} failed the healthy UUID whitelist")
        if found[index][2] < 32000:
            raise RuntimeError(f"GPU {index} does not expose the expected 32GB capacity")


def _command(job: dict, output: Path, epochs: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "experiments.osram_ecerc_features_20261012.run_meld",
        "--seed",
        str(job["seed"]),
        "--epochs",
        str(epochs),
        "--model-variant",
        job["variant"],
        "--output",
        str(output),
    ]
    if job["meaningful_block"] is not None:
        command += ["--meaningful-block", job["meaningful_block"]]
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.epochs != 50:
        raise ValueError("this authorized screen is locked to 50 epochs")
    _check_gpus()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "runs").mkdir()
    source = Path(__file__).resolve().parents[2]
    rows = build_jobs()
    state = {
        "status": "running",
        "started_utc": _now(),
        "source": str(source),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "epochs": args.epochs,
        "selection_metric": "accuracy",
        "protocol": "MELD cfg84 cyclic missing; per-rate Test-oracle",
        "gpu_uuid_whitelist": GPU_UUIDS,
        "runs": rows,
    }
    _write(root / "DISPATCH.json", state)

    for wave in range(5):
        active = []
        for row in (item for item in rows if item["wave"] == wave):
            output = root / "runs" / row["name"]
            log = root / "logs" / f"{row['name']}.log"
            command = _command(row, output, args.epochs)
            stream = log.open("x")
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=GPU_UUIDS[row["gpu"]],
                GCNET_CPU_THREADS="2",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                NUMEXPR_NUM_THREADS="2",
            )
            child = subprocess.Popen(
                command,
                cwd=source,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            row.update(
                status="running",
                pid=child.pid,
                output=str(output),
                log=str(log),
                command=command,
                gpu_uuid=GPU_UUIDS[row["gpu"]],
                started_utc=_now(),
            )
            active.append((child, stream, row))
            time.sleep(1)
        _write(root / "DISPATCH.json", state)

        while active:
            for child, stream, row in list(active):
                code = child.poll()
                if code is None:
                    continue
                stream.close()
                provenance_path = Path(row["output"]) / "PROVENANCE.json"
                provenance = (
                    json.loads(provenance_path.read_text())
                    if provenance_path.exists()
                    else {}
                )
                complete = code == 0 and provenance.get("status") == "complete"
                row.update(
                    status="complete" if complete else "failed",
                    exit_code=code,
                    finished_utc=_now(),
                    error=None if complete else provenance.get("error", "see log"),
                )
                active.remove((child, stream, row))
            _write(root / "DISPATCH.json", state)
            if active:
                time.sleep(20)

    state.update(
        status=("complete" if all(row["status"] == "complete" for row in rows) else "failed"),
        finished_utc=_now(),
    )
    _write(root / "DISPATCH.json", state)


if __name__ == "__main__":
    main()
