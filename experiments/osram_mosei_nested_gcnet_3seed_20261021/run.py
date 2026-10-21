"""Finite five-job MOSEI queue, preceded by one real Nested GPU smoke."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .report import summarize

BAD_UUID = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'
MODULE = 'experiments.osram_mosei_nested_gcnet_3seed_20261021.worker'


def plan_jobs():
    return [dict(name=f'{mode}_seed{seed}', mode=mode, seed=seed)
            for mode, seeds in [('nested', (66, 67, 68)), ('flat', (67, 68))] for seed in seeds]


def parse_gpus(raw, requested):
    if not requested or len(requested) != len(set(requested)):
        raise ValueError('Provide distinct GPU UUIDs')
    found = {}
    for line in raw.splitlines():
        index, uuid, free = [v.strip() for v in line.split(',')]
        if uuid not in requested:
            continue
        if int(index) == 4 or uuid == BAD_UUID:
            raise ValueError('Forbidden failed biggpu GPU4')
        found[uuid] = dict(index=int(index), free_mib=int(free))
    if set(found) != set(requested) or any(not u.startswith('GPU-') for u in requested):
        raise ValueError('Unknown GPU UUID; physical indices are not accepted')
    return found


def gpu_snapshot(requested):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                  '--format=csv,noheader,nounits'], text=True)
    return parse_gpus(raw, requested)


def ram_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024**2
    raise RuntimeError('Cannot verify host RAM headroom')


def job_reserve_mib(mode, nested_peak_mib):
    if mode == 'flat':
        return 4096  # existing original Flat runs, unchanged model/batch
    if mode != 'nested' or not math.isfinite(nested_peak_mib) or nested_peak_mib <= 0:
        raise ValueError('Need a successful measured Nested peak')
    # First three cyclic smoke batches are not a worst-case .7-mask guarantee.
    # Retain the historical Nested eight-GiB floor as well as the measured peak.
    return max(8192, math.ceil((1.5*nested_peak_mib+512)/1024)*1024)


def can_admit(*, free_mib, existing_reserved_mib, proposed_mib, ram_gib, active_count):
    # Deliberately conservative: reserve *all* own active jobs again even if some
    # allocations already appear in nvidia-smi. CPU-loading jobs cannot hide usage.
    return (ram_gib >= 64 and active_count < 4
            and free_mib >= existing_reserved_mib+proposed_mib+2048)


def worker_command(args, job, phase, uuid, resume=False):
    output = args.output/('smoke' if phase == 'smoke' else 'runs')/'CMUMOSEI'/f"seed_{job['seed']}"/job['mode']
    command = [sys.executable, '-u', '-m', MODULE, '--phase', phase, '--mode', job['mode'],
               '--seed', str(job['seed']), '--gpu-uuid', uuid,
               '--source-commit', args.source_commit, '--output', str(output)]
    if resume:
        command.append('--resume')
    return command


def atomic_write(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', required=True, help='Comma-separated real GPU UUIDs, never physical indices')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--resume', action='store_true', help='Explicit stopped-controller/full-state recovery only')
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()
    requested = args.gpus.split(',')
    identity = dict(source_commit=args.source_commit, gpus=requested, jobs=plan_jobs())
    cards = gpu_snapshot(requested)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 15*2**30:
        raise RuntimeError('Need at least15GiB output disk headroom')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Existing run: inspect queue; no automatic duplicate submission')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'QUEUE.json'
    state = dict(status='preflight', identity=identity, pid=os.getpid(), jobs=plan_jobs(),
                 smoke=None, source_commit=args.source_commit, started_at=time.time())
    if args.resume:
        old = json.loads(path.read_text())
        if old['identity'] != identity:
            raise ValueError('Queue recovery identity differs')
        if any(j.get('status') == 'running' and process_alive(j['pid']) for j in old['jobs']):
            raise RuntimeError('A prior worker is still alive; refuse duplicate recovery')
        state = old
        state.update(pid=os.getpid(), recovery_started_at=time.time(), status='resuming')
    active = []

    def save():
        state['updated_at'] = time.time()
        atomic_write(path, state)

    def launch(job, phase, uuid, resume=False):
        command = worker_command(args, job, phase, uuid, resume)
        log = args.output/f"{phase}_{job['name']}_{time.time_ns()}.log"
        with log.open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid,
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        job.update(command=command, pid=process.pid, log=str(log), status='running', gpu_uuid=uuid,
                   physical_gpu=cards[uuid]['index'], output=command[command.index('--output')+1])
        save()
        return process

    save()
    try:
        if not state.get('smoke') or state['smoke'].get('status') != 'complete':
            smoke = dict(name='nested_seed66', mode='nested', seed=66)
            while True:
                cards = gpu_snapshot(requested)
                eligible = [u for u in requested if ram_available_gib() >= 64 and cards[u]['free_mib'] >= 7000]
                if eligible:
                    break
                state.update(status='waiting_for_smoke_resources', resources=cards)
                save(); time.sleep(max(10, args.interval))
            uuid = sorted(eligible, key=lambda u: (cards[u]['index'] != 7, -cards[u]['free_mib']))[0]
            state['smoke'] = smoke
            proc = launch(smoke, 'smoke', uuid)
            code = proc.wait()
            spath = Path(smoke['output'])/'SUMMARY.json'
            summary = json.loads(spath.read_text()) if spath.exists() else {}
            smoke.update(returncode=code, status='complete' if code == 0 and summary.get('status') == 'complete' else 'failed',
                         peak_memory_mib=summary.get('peak_memory_mib'))
            save()
            if smoke['status'] != 'complete':
                raise RuntimeError('Nested real-batch GPU smoke failed; no training admitted')
        peak = state['smoke']['peak_memory_mib']
        job_reserve_mib('nested', peak)  # validate measured peak before scheduling
        pending = []
        for job in state['jobs']:
            target = args.output/'runs/CMUMOSEI'/f"seed_{job['seed']}"/job['mode']
            summary_path = target/'SUMMARY.json'
            if args.resume and summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary.get('status') == 'complete':
                    job.update(status='complete', output=str(target), returncode=0)
                    continue
            pending.append(job)
        failed = False
        while pending or active:
            for process, job in list(active):
                code = process.poll()
                if code is None:
                    continue
                summary_path = Path(job['output'])/'SUMMARY.json'
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                job.update(returncode=code, status='complete' if code == 0 and summary.get('status') == 'complete' else 'failed')
                failed |= job['status'] == 'failed'
                active.remove((process, job))
            if failed:
                state['status'] = 'failure_draining_active'
            else:
                cards = gpu_snapshot(requested)
                for job in list(pending):
                    reserve = job_reserve_mib(job['mode'], peak)
                    preferred = 7 if job['mode'] == 'nested' else 0
                    for uuid in sorted(requested, key=lambda u: (cards[u]['index'] != preferred, -cards[u]['free_mib'])):
                        # Fresh query before *each* admission, not stale initial free memory.
                        cards = gpu_snapshot(requested)
                        own = [j for _, j in active if j['gpu_uuid'] == uuid]
                        if can_admit(free_mib=cards[uuid]['free_mib'],
                            existing_reserved_mib=sum(j['reserved_mib'] for j in own),
                            proposed_mib=reserve, ram_gib=ram_available_gib(), active_count=len(own)):
                            job['reserved_mib'] = reserve
                            target = args.output/'runs/CMUMOSEI'/f"seed_{job['seed']}"/job['mode']
                            resume = args.resume and (target/'last.pt').exists()
                            active.append((launch(job, 'train', uuid, resume), job))
                            pending.remove(job)
                            break
                state['status'] = 'running' if active else 'waiting_for_resources'
                state['resources'] = cards
            state['pending_count'] = len(pending)
            save()
            if failed and not active:
                break
            if pending or active:
                time.sleep(max(10, args.interval))
        summarize(args.output)
        state['status'] = 'complete' if not failed and not pending else 'failed'
        state['completed_at'] = time.time()
    except BaseException as error:
        state.update(status='failure_draining_active', error=repr(error))
        save()
        # No new admissions after an infrastructure exception, but admitted
        # children are allowed to finish; never kill peer or unrelated jobs.
        for process, job in active:
            code = process.wait()
            summary_path = Path(job['output'])/'SUMMARY.json'
            result = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            job.update(returncode=code, status='complete' if code == 0 and result.get('status') == 'complete' else 'failed')
            save()
        state.update(status='failed', completed_at=time.time())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
