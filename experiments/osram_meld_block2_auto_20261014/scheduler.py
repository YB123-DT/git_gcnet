"""Finite, recoverable UUID-pinned queue; never manages external processes."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
WORKER = "experiments.osram_meld_block2_auto_20261014.worker"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    """Return Linux process birth identity, or None for dead/zombie processes."""
    try:
        tail = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        if tail[0] == "Z":
            return None
        return tail[19]
    except (OSError, ValueError, IndexError):
        return None


class Scheduler:
    def __init__(self, root, gpus=(0, 2, 3), per_gpu=4, python=sys.executable,
                 code_root=ROOT, smoke_costs=None, per_gpu_limit=4):
        self.root = Path(root).resolve()
        self.gpus = tuple(gpus)
        if (not self.gpus or len(set(self.gpus)) != len(self.gpus)
                or len(self.gpus) > 3 or any(g < 0 or g == 4 for g in self.gpus)
                or type(per_gpu_limit) is not int or not 1 <= per_gpu_limit <= 6
                or not 1 <= per_gpu <= per_gpu_limit):
            raise ValueError("use at most three healthy GPUs, never GPU4; >4 jobs/GPU requires an explicit <=6 limit")
        self.root.mkdir(parents=True, exist_ok=True)
        self.python = str(python)
        self.code_root = Path(code_root).resolve()
        self.smoke_costs = smoke_costs or {}
        self.poll_seconds = 3.0
        self.disk_margin_bytes = 2 * 1024 ** 3
        self.capacity = {g: per_gpu for g in self.gpus}
        self.children = {}
        self.state_path = self.root / "QUEUE.json"
        self.state = self.load()
        for gpu, capacity in self.state.get("capacity", {}).items():
            if int(gpu) in self.capacity:
                self.capacity[int(gpu)] = min(self.capacity[int(gpu)], int(capacity))

    def load(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"status": "idle", "jobs": {}, "gpu_uuids": {}, "created_at": now()}

    def save(self):
        self.state["capacity"] = self.capacity
        self.state["updated_at"] = now()
        write_json(self.state_path, self.state)

    @contextmanager
    def lock(self):
        with (self.root / ".scheduler.lock").open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another scheduler owns this queue") from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def gpu_snapshot(self):
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits"], text=True)
        result = {}
        for line in raw.splitlines():
            index, uuid, free = (part.strip() for part in line.split(","))
            result[int(index)] = {"uuid": uuid, "free_mib": float(free)}
        return result

    def costs(self, job):
        cost = self.smoke_costs.get(job["method"])
        if cost is None:
            raise ValueError(f"missing measured smoke peak for {job['method']}")
        if isinstance(cost, dict):
            peak = cost.get("peak_memory_mib", cost.get("peak_mib"))
            disk = cost.get("disk_bytes", 2 * 1024 ** 3)
        else:
            peak, disk = cost, 2 * 1024 ** 3
        if peak is None or not math.isfinite(float(peak)) or float(peak) <= 0:
            raise ValueError("smoke peak must be a positive finite MiB value")
        if int(disk) < 0:
            raise ValueError("disk budget must be nonnegative")
        return float(peak), int(disk)

    def admit(self, job):
        peak, disk = self.costs(job)
        active = [row for row in self.state["jobs"].values()
                  if row["status"] in ("running", "launching")]
        outstanding_disk = sum(self.costs(row)[1] for row in active)
        if shutil.disk_usage(self.root).free < disk + outstanding_disk + self.disk_margin_bytes:
            return None
        snapshot = self.gpu_snapshot()  # fresh immediately before each Popen
        uuids = self.state["gpu_uuids"]
        for gpu in self.gpus:
            if gpu not in snapshot:
                raise RuntimeError(f"healthy GPU {gpu} is unavailable")
            uuid = snapshot[gpu]["uuid"]
            if not uuid.startswith("GPU-") or uuid == snapshot.get(4, {}).get("uuid"):
                raise RuntimeError("GPU UUID failed healthy-device admission")
            if str(gpu) in uuids and uuids[str(gpu)] != uuid:
                raise RuntimeError(f"GPU {gpu} UUID changed since queue creation")
            uuids[str(gpu)] = uuid
        counts = {g: sum(row.get("gpu") == g for row in active) for g in self.gpus}
        # Reserve complete peaks for active children. This deliberately covers
        # startup allocations before nvidia-smi reports the new process.
        for gpu in sorted(self.gpus, key=lambda g: (counts[g], -snapshot[g]["free_mib"])):
            reserved = sum(self.costs(row)[0] for row in active if row.get("gpu") == gpu)
            if (counts[gpu] < self.capacity[gpu]
                    and snapshot[gpu]["free_mib"] - reserved >= peak + 1024):
                return gpu, snapshot[gpu]["uuid"]
        return None

    def command(self, job, config_path, gpu_uuid):
        return [self.python, "-u", "-m", WORKER, "--config", str(config_path),
                "--output", job["output"], "--gpu-uuid", gpu_uuid]

    def owns_process(self, row):
        pid = row.get("pid")
        if not pid or process_identity(pid) != row.get("process_start"):
            return False
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            return [part.decode() for part in command] == row.get("command")
        except OSError:
            return False

    def summary(self, row):
        path = Path(row["output"]) / "SUMMARY.json"
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return summary if summary.get("status") == "complete" else None

    def failure(self, row, reason, returncode=None):
        row.setdefault("failures", []).append({"at": now(), "reason": reason,
            "returncode": returncode, "gpu": row.get("gpu"), "pid": row.get("pid"),
            "attempt": row.get("attempts", 0), "log": row.get("log")})
        is_oom = "out of memory" in reason.lower()
        previous_ooms = sum("out of memory" in f["reason"].lower() for f in row["failures"])
        if is_oom and previous_ooms == 1:
            gpu = row["gpu"]
            self.capacity[gpu] = max(1, self.capacity[gpu] - 1)
            row["status"] = "pending"
            row["resume_reason"] = "single OOM retry, same configuration and complete epoch state"
        else:
            row["status"] = "failed"
            row["finished_at"] = now()

    def finish(self, row, returncode):
        summary = self.summary(row)
        if returncode in (0, None) and summary is not None:
            row.update(status="complete", summary=summary, finished_at=now(), returncode=returncode)
            return
        try:
            tail = Path(row["log"]).read_text(errors="replace")[-16000:]
        except OSError:
            tail = ""
        reason = "CUDA out of memory" if "out of memory" in tail.lower() else (
            f"worker exited {returncode}; complete SUMMARY.json missing or unsuccessful")
        self.failure(row, reason, returncode)

    def recover(self):
        for row in self.state["jobs"].values():
            if row["status"] == "complete":
                if self.summary(row) is None:
                    self.failure(row, "completed artifact missing or invalid")
                continue
            if row["status"] == "launching" and not row.get("pid"):
                # Recover the Popen -> state-save crash window using the exact
                # persisted launch command; never signal the matching process.
                for proc in Path("/proc").iterdir():
                    if not proc.name.isdigit():
                        continue
                    candidate = dict(row, pid=int(proc.name), process_start=process_identity(proc.name))
                    if candidate["process_start"] and self.owns_process(candidate):
                        row.update(pid=candidate["pid"], process_start=candidate["process_start"])
                        break
                if not row.get("pid"):
                    row["status"] = "pending"
                    continue
            if row["status"] in ("running", "launching"):
                if self.owns_process(row):
                    row["status"] = "running"
                elif self.summary(row) is not None:
                    self.finish(row, None)
                else:
                    alive = process_identity(row.get("pid", 0)) is not None
                    reason = "process identity mismatch; external PID left untouched" if alive else (
                        "owned worker interrupted; retry from its complete epoch checkpoint")
                    row.setdefault("failures", []).append({"at": now(), "reason": reason})
                    row["status"] = "pending"

    def launch(self, row, device):
        gpu, uuid = device
        config_path = self.root / "configs" / f"{row['id']}.json"
        write_json(config_path, row["config"])
        log = self.root / "logs" / f"{row['id']}.attempt{row['attempts'] + 1}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = self.command(row, config_path, uuid)
        row.update(status="launching", gpu=gpu, gpu_uuid=uuid, command=command,
                   config_path=str(config_path), log=str(log), pid=None,
                   attempts=row["attempts"] + 1, started_at=now())
        self.save()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           GCNET_CPU_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
        try:
            with log.open("a") as stream:
                child = subprocess.Popen(command, cwd=self.code_root, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.children[row["id"]] = child
            row.update(status="running", pid=child.pid, process_start=process_identity(child.pid))
        except OSError as error:
            self.failure(row, f"launch failed: {error}")
        self.save()

    def run_jobs(self, jobs, *, return_on_first_completion=False):
        jobs = list(jobs)
        ids = [job["id"] for job in jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("job IDs must be unique")
        with self.lock():
            self.state = self.load()
            for gpu, capacity in self.state.get("capacity", {}).items():
                if int(gpu) in self.capacity:
                    self.capacity[int(gpu)] = min(self.capacity[int(gpu)], int(capacity))
            outputs = {row["output"]: identifier for identifier, row in self.state["jobs"].items()}
            for job in jobs:
                if not job["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in job["id"]):
                    raise ValueError("job ID must be a safe filename")
                job = dict(job, output=str(Path(job["output"]).resolve()))
                if job["output"] in outputs and outputs[job["output"]] != job["id"]:
                    raise ValueError("distinct jobs must use independent output directories")
                outputs[job["output"]] = job["id"]
                self.costs(job)
                existing = self.state["jobs"].get(job["id"])
                if existing:
                    if any(existing[key] != job[key] for key in ("method", "config", "output")):
                        raise ValueError(f"immutable job configuration changed: {job['id']}")
                else:
                    self.state["jobs"][job["id"]] = dict(job, status="pending", attempts=0, failures=[])
            self.recover()
            unfinished_at_entry = {identifier for identifier in ids
                if self.state['jobs'][identifier]['status'] not in ('complete', 'failed')}
            self.state.update(status="running", scheduler_pid=os.getpid())
            self.save()
            try:
                while any(self.state["jobs"][identifier]["status"] not in ("complete", "failed") for identifier in ids):
                    changed = False
                    for row in self.state["jobs"].values():
                        if row["status"] != "running":
                            continue
                        child = self.children.get(row["id"])
                        code = child.poll() if child is not None else None
                        if (child is not None and code is not None) or (child is None and not self.owns_process(row)):
                            self.finish(row, code)
                            self.children.pop(row["id"], None)
                            changed = True
                    for identifier in ids:
                        row = self.state["jobs"][identifier]
                        if row["status"] == "pending":
                            device = self.admit(row)
                            if device is not None:
                                self.launch(row, device)
                                changed = True
                    if changed:
                        self.save()
                    if return_on_first_completion and any(
                            self.state['jobs'][identifier]['status'] in ('complete', 'failed')
                            for identifier in unfinished_at_entry):
                        break
                    if any(self.state["jobs"][identifier]["status"] not in ("complete", "failed") for identifier in ids):
                        time.sleep(min(30.0, max(0.01, self.poll_seconds)))
            except BaseException:
                self.state["status"] = "interrupted"
                self.save()
                raise
            if any(row['status'] not in ('complete', 'failed') for row in self.state['jobs'].values()):
                self.state['status'] = 'running'
            else:
                self.state["status"] = "failed" if any(row["status"] == "failed" for row in self.state["jobs"].values()) else "complete"
            self.save()
            return [dict(self.state["jobs"][identifier]) for identifier in ids]
