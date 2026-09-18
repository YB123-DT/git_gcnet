"""Repair completed screen artifacts, summarize all 60 configs, and launch top-5 verification."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.osram_mosi_hparam_sweep_20260918.run import (
    ROOT,
    SOURCE_ROOT,
    canonical_mask_hashes,
)


PYTHON = "/data2/yb/reproduction_envs/s0/bin/python3.10"
RUNNER = Path(__file__).resolve()
EXPECTED = tuple(f"cfg{i:02d}" for i in range(1, 61))


def _repair_completed() -> tuple[int, list[str]]:
    root = ROOT / "screen_seed66"
    source = SOURCE_ROOT / "seed_66"
    complete = 0
    pending: list[str] = []
    for prefix in EXPECTED:
        paths = sorted(root.glob(prefix + "_*/PROVENANCE.json"))
        if len(paths) != 1:
            pending.append(prefix)
            continue
        provenance_path = paths[0]
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
            pending.append(prefix)
    return complete, pending


def main() -> None:
    log = ROOT / "WATCHER.log"
    ROOT.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        while True:
            complete, pending = _repair_completed()
            handle.write(f"{datetime.now(timezone.utc).isoformat()} complete={complete}/60 pending={pending}\n")
            handle.flush()
            if complete == 60:
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
