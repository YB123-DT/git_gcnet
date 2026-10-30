"""Finite, restart-safe local scheduler for the four approved MOSEI ablations."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VARIANTS = ("A0", "A1", "A2", "A3")
BAD_UUID = "GPU-46fb379f-cc90-dc82-9b5e-5d011f552264"


def admissible_gpus(cards, *, reserve_mib=4096, job_mib=6000, per_gpu=4):
    if job_mib <= 0 or reserve_mib < 0 or not 1 <= per_gpu <= 4:
        raise ValueError("Require measured positive memory and 1–4 jobs/device")
    result = []
    for card in cards:
        if card["index"] == 4 or card["uuid"] == BAD_UUID:
            continue
        slots = min(per_gpu, max(0, (card["free_mib"] - reserve_mib) // job_mib))
        if slots:
            result.append({"index": card["index"], "uuid": card["uuid"], "slots": slots})
    return result


def pending_variants(statuses):
    if set(statuses) - set(VARIANTS):
        raise ValueError("Only four preregistered arms are allowed")
    # Failures need an explicit inspected recovery, never an automatic loop.
    return [v for v in VARIANTS if v not in statuses]


def inventory(allowed):
    response = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    cards = []
    for line in response.strip().splitlines():
        index, uuid, free = (part.strip() for part in line.split(","))
        if int(index) in allowed:
            cards.append(dict(index=int(index), uuid=uuid, free_mib=int(free)))
    if {c["index"] for c in cards} != set(allowed):
        raise RuntimeError("Could not verify every requested physical GPU")
    return cards


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="7")
    parser.add_argument("--job-mib", type=int, required=True)
    parser.add_argument("--reserve-mib", type=int, default=4096)
    parser.add_argument("--per-gpu", type=int, default=4)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    allowed = [int(i) for i in args.gpus.split(",")]
    if not allowed or 4 in allowed or len(allowed) != len(set(allowed)):
        raise ValueError("Explicit unique healthy physical GPUs required; GPU4 forbidden")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "SCHEDULER.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # All source bytes must have been pinned and checked before training.
    preflight = args.output / "PREFLIGHT.json"
    proof = json.loads(preflight.read_text())
    if proof.get("status") != "passed" or proof.get("source_commit") != args.source_commit:
        raise ValueError("Passed preflight for this immutable source is required")
    from .bundle import verify
    source = verify(Path(__file__).resolve().parents[2])
    if source["source_commit"] != args.source_commit:
        raise ValueError("Bundled source identity differs from preflight")
    if args.job_mib < proof["job_memory_budget_mib"]:
        raise ValueError("Requested reservation is below measured preflight budget")
    statuses = {}
    for variant in VARIANTS:
        path = args.output / "runs" / variant / "SUMMARY.json"
        if path.exists():
            statuses[variant] = json.loads(path.read_text())["status"]
    if any(s not in ("complete", "failed") for s in statuses.values()):
        raise RuntimeError("Existing unfinished runs: inspect/explicitly resume; do not duplicate")
    pending = pending_variants(statuses)
    running = {}
    history_path = args.output / "QUEUE.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    while pending or running:
        for variant, item in list(running.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close()
            path = args.output / "runs" / variant / "SUMMARY.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            success = code == 0 and state.get("status") == "complete"
            history[variant].update(status="complete" if success else "failed",
                                    returncode=code, finished_at=time.time())
            del running[variant]
        if pending:
            cards = inventory(allowed)
            capacity = admissible_gpus(cards, job_mib=args.job_mib,
                                       reserve_mib=args.reserve_mib, per_gpu=args.per_gpu)
            for card in capacity:
                active = sum(x["uuid"] == card["uuid"] for x in running.values())
                # memory.free already excludes running jobs. Process limit is separate.
                slots = min(card["slots"], args.per_gpu - active)
                for _ in range(slots):
                    if not pending:
                        break
                    variant = pending.pop(0)
                    run = args.output / "runs" / variant
                    if run.exists():
                        raise FileExistsError(run)
                    log_path = args.output / f"{variant}.log"
                    log = log_path.open("x")
                    command = [sys.executable, "-m", "experiments.cam_unified_20261030.train",
                               "--variant", variant, "--output", str(run),
                               "--gpu-uuid", card["uuid"], "--source-commit", args.source_commit]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=card["uuid"],
                               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    running[variant] = dict(process=process, log=log, uuid=card["uuid"])
                    history[variant] = dict(status="running", pid=process.pid, command=command,
                                            uuid=card["uuid"], host_gpu=card["index"],
                                            started_at=time.time(), log=str(log_path))
                    print(f"Started {variant}: pid={process.pid} GPU{card['index']}", flush=True)
        write_json(history_path, history)
        if pending or running:
            time.sleep(10)
    subprocess.run([sys.executable, "-m", "experiments.cam_unified_20261030.summarize",
                    "--output", str(args.output)], check=True)
    if any(row.get("status") != "complete" for row in history.values()):
        raise SystemExit("One or more runs failed; see QUEUE.json and separate logs")


if __name__ == "__main__":
    main()
