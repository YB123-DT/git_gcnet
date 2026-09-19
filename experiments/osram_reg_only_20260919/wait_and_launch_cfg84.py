"""Wait for the preceding capacity run, then launch reg-only five-seed check."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON = "/data2/yb/reproduction_envs/s0/bin/python3.10"
PREVIOUS_QUEUE = Path("/data2/yb/remote_experiments/osram_mosi_cfg84_cfg23_20260919/QUEUE.json")
TARGET = Path(__file__).with_name("run_cfg84_capacity.py")


def main() -> None:
    while True:
        if PREVIOUS_QUEUE.exists():
            status = json.loads(PREVIOUS_QUEUE.read_text()).get("status")
            if status in {"complete", "failed"}:
                break
        time.sleep(60)
    subprocess.run([PYTHON, "-u", str(TARGET), "--launch"], cwd=REPO, check=False)


if __name__ == "__main__":
    main()
