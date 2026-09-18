"""Repair completed screen artifacts, extend the queue, and launch top-5 verification."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.osram_mosi_hparam_sweep_20260918.run import (
    ROOT,
    SOURCE_ROOT,
    SPECS,
    canonical_mask_hashes,
)


PYTHON = "/data2/yb/reproduction_envs/s0/bin/python3.10"
RUNNER = Path(__file__).resolve()
EXPECTED = tuple(item["id"] for item in SPECS)
FIRST_WAVES = tuple(item["id"] for item in SPECS[:60])
FOLLOWUP_IDS = tuple(item["id"] for item in SPECS[60:])


def _repair_completed() -> tuple[int, list[str]]:
    root = ROOT / "screen_seed66"
    source = SOURCE_ROOT / "seed_66"
    complete = 0
    pending: list[str] = []
    for spec_id in EXPECTED:
        provenance_path = root / spec_id / "PROVENANCE.json"
        if not provenance_path.exists():
            pending.append(spec_id)
            continue
        output = provenance_path.parent
        try:
            provenance = json.loads(provenance_path.read_text())
            config = json.loads((output / "config.json").read_text())
            metrics = json.loads((output / "metrics.json").read_text())
            history = json.loads((output / "history.json").read_text())
            expected_epochs = int(config["epochs"])
            valid = (
                len(history) == expected_epochs
                and metrics.get("selection_protocol") == "per-rate-test-oracle"
                and all((output / (f"predictions_miss_{i / 10:.1f}".replace(".", "p") + ".npz")).exists()
                        for i in range(8))
                and canonical_mask_hashes(output) == canonical_mask_hashes(source)
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
        if valid:
            if provenance.get("status") != "complete" or provenance.get("mask_validation") != "canonical_row_multiset":
                provenance["status"] = "complete"
                provenance.pop("error", None)
                provenance["mask_validation"] = "canonical_row_multiset"
                provenance.setdefault("completed_utc", datetime.now(timezone.utc).isoformat())
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
            complete += 1
        else:
            pending.append(spec_id)
    return complete, pending


def _launch_followup() -> None:
    queue_path = ROOT / "FOLLOWUP_QUEUE.json"
    if queue_path.exists():
        return
    buckets = {1: [], 2: [], 3: []}
    for index, spec_id in enumerate(FOLLOWUP_IDS):
        buckets[(index % 3) + 1].append(spec_id)
    tasks = []
    for gpu, spec_ids in buckets.items():
        for spec_id in spec_ids:
            log_path = ROOT / f"gpu{gpu}_{spec_id}.log"
            log = log_path.open("a")
            env = dict(
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                PYTHONPATH=str(REPO),
            )
            child = subprocess.Popen(
                [PYTHON, "-u", str(RUNNER.with_name("run.py")), "--train", spec_id],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.close()
            tasks.append({"gpu": gpu, "spec_id": spec_id, "pid": child.pid,
                          "log": str(log_path), "status": "running"})
    queue_path.write_text(json.dumps({
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configs": list(FOLLOWUP_IDS),
        "tasks": tasks,
    }, indent=2) + "\n")


def main() -> None:
    log = ROOT / "WATCHER.log"
    ROOT.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        while True:
            complete, pending = _repair_completed()
            handle.write(f"{datetime.now(timezone.utc).isoformat()} complete={complete}/90 pending={pending}\n")
            handle.flush()
            if complete >= len(FIRST_WAVES) and not (ROOT / "FOLLOWUP_QUEUE.json").exists():
                _launch_followup()
                handle.write(f"{datetime.now(timezone.utc).isoformat()} launched follow-up cfg61-cfg90\n")
                handle.flush()
            if complete == 90:
                summarize = RUNNER.with_name("summarize.py")
                verify = RUNNER.with_name("verify_top.py")
                subprocess.run([PYTHON, str(summarize), "--root", str(ROOT)], check=True)
                subprocess.run([PYTHON, str(verify), "--launch", "--top", "5"], check=True)
                handle.write(f"{datetime.now(timezone.utc).isoformat()} top-five verification finished\n")
                handle.flush()
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
