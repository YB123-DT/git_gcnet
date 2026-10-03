"""Durable single-host queue. Standard library only: never imports torch."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_readout20_20261003.run import (
    CANDIDATES, DATASET, GPU_UUIDS, REFERENCE, completion_status, gpu_check,
    now, read, sha, validate_gpu, verify_source, write, require_active_screen,
)


def process_identity(pid):
    """Linux process start time + host boot ID prevent accepting recycled PIDs."""
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), start_ticks=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, IndexError, ValueError):
        return None


def process_matches(record):
    current = process_identity(record.get('pid', -1))
    return current is not None and all(current[k] == record.get(k) for k in current)


def validate_args(args):
    validate_gpu(args.gpu)
    if not 1 <= args.max_concurrent <= 5:
        raise ValueError('Concurrency must be between one and five')
    if args.min_free_mib < 5000 or args.disk_reserve_gib < 16:
        raise ValueError('Do not lower 5000 MiB GPU / 16 GiB disk safety floors')
    if min(args.warmup_seconds, args.launch_interval, args.poll_seconds) < 0:
        raise ValueError('Intervals cannot be negative')
    if args.poll_seconds == 0:
        raise ValueError('Polling interval must be positive')


def first_epoch_ready(record):
    path = Path(record['output']) / 'history.json'
    try:
        return len(read(path)) >= 1
    except (OSError, ValueError, TypeError):
        return False


def coordinate(args):
    require_active_screen()
    validate_args(args)
    manifest = read(args.manifest)
    sources = verify_source(manifest, args.commit)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'runs').mkdir(exist_ok=True)
    (args.root / 'logs').mkdir(exist_ok=True)
    with (args.root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = dict(code_commit=args.commit, manifest_sha256=sha(args.manifest),
                    source_sha256=sources, gpu=args.gpu, gpu_uuid=GPU_UUIDS[args.gpu],
                    max_concurrent=args.max_concurrent, min_free_mib=args.min_free_mib,
                    disk_reserve_gib=args.disk_reserve_gib, warmup_seconds=args.warmup_seconds,
                    launch_interval=args.launch_interval, reference=str(args.reference.resolve()),
                    dataset=str(args.dataset.resolve()))
        state_path = args.root / 'QUEUE.json'
        if state_path.exists():
            state = read(state_path)
            if state['spec'] != spec:
                raise ValueError('Existing queue belongs to a different locked protocol')
        else:
            state = dict(spec=spec, created_at=now(), status='running', children={})
            write(state_path, state)
        children = state['children']
        handles = {}
        last_launch = max((v.get('launched_unix', 0.) for v in children.values()), default=0.)
        while True:
            # Same-host restart attaches only when the complete process identity matches.
            active = []
            for candidate, child in children.items():
                if candidate in handles:
                    exit_code = handles[candidate].poll()
                    if exit_code is not None:
                        child['exit_code'] = exit_code
                if process_matches(child):
                    child['status'] = 'running'
                    active.append(candidate)
                    continue
                status, detail = completion_status(Path(child['output']))
                child['status'] = status if status in ('complete', 'failed') else 'interrupted'
                child['detail'] = detail
                # launch_requested without a matching identity is uncertain, never auto-retried.
            pending = [c for c in CANDIDATES if c not in children]
            for candidate in list(pending):
                output = args.root / 'runs' / candidate
                if output.exists():
                    status, detail = completion_status(output)
                    children[candidate] = dict(output=str(output), status='orphaned',
                        detail=f'Unregistered existing output ({status}): {detail}; not overwritten')
                    pending.remove(candidate)
            state.update(status='running', updated_at=now(), coordinator=process_identity(os.getpid()),
                         running=len(active), pending=len(pending))
            if not active and not pending:
                state['status'] = ('complete' if all(c['status'] == 'complete' for c in children.values())
                                   else 'finished_with_failures')
                write(state_path, state)
                return
            first = children.get(CANDIDATES[0])
            warmed = (first is None or (first_epoch_ready(first) and
                      time.time() - first.get('launched_unix', time.time()) >= args.warmup_seconds))
            if first and first.get('status') in ('failed', 'interrupted', 'orphaned') and not first_epoch_ready(first):
                state.update(status='blocked', reason='First candidate failed before warmup; no automatic retries')
                write(state_path, state)
                return
            can_launch = pending and len(active) < args.max_concurrent and warmed
            state['waiting_reason'] = ('initial warmup / first completed epoch' if not warmed else
                                       'concurrency limit' if len(active) >= args.max_concurrent else
                                       'launch interval or polling')
            if can_launch and time.time() - last_launch >= args.launch_interval:
                free = gpu_check(args.gpu)
                disk = shutil.disk_usage(args.root).free / 1024 ** 3
                state['resources'] = dict(free_gpu_mib=free, free_disk_gib=disk, checked_at=now())
                if free < args.min_free_mib:
                    state['waiting_reason'] = 'insufficient free GPU memory; preserving batch protocol'
                elif disk < args.disk_reserve_gib:
                    state['waiting_reason'] = 'disk reserve; no automatic artifact deletion'
                if free >= args.min_free_mib and disk >= args.disk_reserve_gib:
                    # Revalidate immediately before every launch; never synchronize mutable source.
                    verify_source(manifest, args.commit)
                    candidate = pending[0]
                    output = args.root / 'runs' / candidate
                    command = [sys.executable, '-u', str(Path(__file__).with_name('run.py')),
                        '--candidate', candidate, '--output', str(output), '--manifest', str(args.manifest),
                        '--commit', args.commit, '--gpu', args.gpu, '--reference', str(args.reference),
                        '--dataset', str(args.dataset)]
                    entry = dict(status='launch_requested', output=str(output), command=command,
                                 gpu=args.gpu, gpu_uuid=GPU_UUIDS[args.gpu], launched_at=now(), launched_unix=time.time())
                    children[candidate] = entry
                    write(state_path, state)  # Durable intent before side effect.
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='2',
                               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                               GCNET_DATASET_ROOT=str(args.dataset), PYTHONPATH=str(REPO))
                    try:
                        with (args.root / 'logs' / f'{candidate}.log').open('x') as log:
                            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
                        handles[candidate] = child
                        identity = process_identity(child.pid)
                        entry.update(identity or dict(pid=child.pid))
                        entry['status'] = 'running' if identity else 'starting'
                    except BaseException as error:
                        entry.update(status='interrupted', detail=repr(error))
                        write(state_path, state)
                        raise
                    last_launch = time.time()
                    print(f'{now()} launched {candidate}: pid={child.pid}, host GPU{args.gpu}', flush=True)
            write(state_path, state)
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--gpu', default='1')
    p.add_argument('--reference', type=Path, default=REFERENCE)
    p.add_argument('--dataset', type=Path, default=DATASET)
    p.add_argument('--max-concurrent', type=int, default=5)
    p.add_argument('--warmup-seconds', type=float, default=120)
    p.add_argument('--launch-interval', type=float, default=30)
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--min-free-mib', type=int, default=5000)
    p.add_argument('--disk-reserve-gib', type=float, default=16)
    coordinate(p.parse_args())
