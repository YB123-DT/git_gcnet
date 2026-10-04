"""Persistent M01--M40 admission on physical GPUs 2/6; reuse the existing runner.

No training, GPU smoke, cancellation or model mutation is implemented here.
Only pre-creation resource rejections may be retried. Other failures stay failed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


GPUS = {'2': 'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4',
        '6': 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def alive(job):
    tick = identity(job.get('pid', -1))
    return tick is not None and (not job.get('start_ticks') or tick == str(job['start_ticks']))


def load_candidates(paths):
    if len(paths) != 2:
        raise ValueError('Exactly two immutable twenty-card manifests required')
    result = []
    for group, path in enumerate(paths):
        cards = read(path)['cards']
        if len(cards) != 20:
            raise ValueError('Each manifest must contain twenty accepted candidates')
        for card in cards:
            result.append(dict(candidate=card['id'], manifest=str(Path(path).resolve()), group=group))
    numbers = [re.fullmatch(r'm(\d{2})_[a-z0-9_]+', row['candidate']) for row in result]
    if any(match is None for match in numbers) or [int(match[1]) for match in numbers] != list(range(1, 41)):
        raise ValueError('Only M01--M40, in order, may enter this dispatcher')
    return result


def admission(gpu, jobs, peak_mib, disk_free_gib, disk_reserved_gib):
    same = [job for job in jobs if job['gpu_uuid'] == gpu['uuid']]
    if len(same) >= 11:
        return False, f'global concurrency {len(same)}/11, including initializing processes'
    values = [gpu['free_mib'], gpu['temperature'], peak_mib, disk_free_gib, disk_reserved_gib]
    values += [value for job in same for value in (job['used_mib'], job['estimate_mib'])]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values) or peak_mib <= 0:
        return False, 'invalid resource evidence'
    if gpu['temperature'] >= 90:
        return False, 'temperature >=90C'
    growth = sum(max(0., 1.2 * max(job['used_mib'], job['estimate_mib']) + 512 - job['used_mib']) for job in same)
    required = growth + 1.2 * peak_mib + 512 + 2048
    if gpu['free_mib'] < required:
        return False, f'memory need {required:.1f}MiB; free {gpu["free_mib"]:.1f}MiB; existing growth reserve {growth:.1f}MiB'
    if disk_free_gib - disk_reserved_gib < 10 + 16:
        return False, f'disk free {disk_free_gib:.1f}GiB minus active growth {disk_reserved_gib:.1f}GiB <26GiB'
    return True, f'admit {len(same)+1}/11; reserve 2GiB global plus measured growth'


def retryable_admission(log, created):
    return not created and 'Occupied-lane admission rejected before job creation:' in log


def gpu_resources():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,temperature.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    result = {}
    for line in raw.splitlines():
        index, gpu_uuid, free, temperature = [item.strip() for item in line.split(',')]
        if index in GPUS:
            if gpu_uuid != GPUS[index]:
                raise ValueError('Physical GPU index/UUID changed; refuse migration')
            result[index] = dict(uuid=gpu_uuid, free_mib=float(free), temperature=float(temperature))
    if set(result) != set(GPUS):
        raise ValueError('Cannot verify both authorized GPUs')
    return result


def queue_jobs(roots):
    result = {}
    for root in roots:
        path = Path(root) / 'QUEUE.json'
        if path.exists():
            for job in read(path)['jobs'].values():
                result[str(job['output'])] = job
    return list(result.values())


def census(roots, reservations, high_water, default_peak):
    """Union real CUDA PIDs, all-root initializing train children, own intents."""
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    actors = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        gpu_uuid, pid, used = [item.strip() for item in line.split(',')]
        if gpu_uuid in GPUS.values():
            actors[(gpu_uuid, int(pid))] = dict(gpu_uuid=gpu_uuid, pid=int(pid), used_mib=float(used), estimate_mib=default_peak)
    known = queue_jobs(roots)
    for job in known:
        if alive(job) and job['gpu_uuid'] in GPUS.values():
            actor = actors.setdefault((job['gpu_uuid'], job['pid']), dict(gpu_uuid=job['gpu_uuid'], pid=job['pid'], used_mib=0.))
            actor.update(estimate_mib=max(default_peak, job.get('profile', {}).get('peak_mib', 0)), candidate=job['candidate'], output=job['output'])
    # Read actual children even if their queue write has not yet become visible.
    external_intents = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / 'cmdline').read_bytes().decode().split('\0')
            if '--train-only' in command:
                record = read(command[command.index('--train-only') + 1])
                gpu_uuid = record['gpu_uuid']
                if gpu_uuid in GPUS.values() and identity(int(path.name)):
                    actor = actors.setdefault((gpu_uuid, int(path.name)), dict(gpu_uuid=gpu_uuid, pid=int(path.name), used_mib=0., estimate_mib=default_peak))
                    actor.update(candidate=record['candidate'], output=record['output'])
            elif any(Path(part).name.startswith('occupied_lane') and part.endswith('.py') for part in command) and '--candidate' in command:
                candidate = command[command.index('--candidate') + 1]
                root = command[command.index('--root') + 1]
                gpu_uuid = command[command.index('--gpu-uuid') + 1]
                if gpu_uuid in GPUS.values():
                    external_intents.append(dict(pid=int(path.name), gpu_uuid=gpu_uuid, candidate=candidate,
                                                 output=str(Path(root) / 'runs' / candidate / 'seed_66')))
        except (OSError, ValueError, KeyError, IndexError):
            continue
    for job in [*external_intents, *reservations]:
        if not alive(job):
            continue
        if any(actor.get('output') == job.get('output') and job.get('output') for actor in actors.values()):
            continue
        actors[(job['gpu_uuid'], job['pid'])] = dict(gpu_uuid=job['gpu_uuid'], pid=job['pid'], used_mib=0., estimate_mib=default_peak,
                                                   candidate=job['candidate'], output=job.get('output'), initializing=True)
    for actor in actors.values():
        key = f'{actor["gpu_uuid"]}:{actor["pid"]}:{identity(actor["pid"])}'
        high_water[key] = max(high_water.get(key, 0), actor['used_mib'])
        actor['estimate_mib'] = max(actor['estimate_mib'], high_water[key])
    reserved = 0.
    known_outputs = set()
    for job in known:
        if not alive(job):
            continue
        known_outputs.add(job['output'])
        used = sum(path.stat().st_size for path in Path(job['output']).rglob('*') if path.is_file()) / 1024**3
        reserved += max(0., max(10., job.get('profile', {}).get('artifact_gib', 10.)) - used)
    for actor in actors.values():
        output = actor.get('output')
        if output in known_outputs:
            continue
        if output:
            used = sum(path.stat().st_size for path in Path(output).rglob('*') if path.is_file()) / 1024**3
            reserved += max(0., 10. - used)
            known_outputs.add(output)
        else:
            reserved += 10.  # unknown external CUDA process: do not assume zero artifact growth
    return list(actors.values()), reserved


def reconcile(state, children):
    for candidate, job in state['jobs'].items():
        if job['status'] in ('complete', 'failed', 'pending'):
            continue
        queue = Path(job['lane_root']) / 'QUEUE.json'
        entry = read(queue).get('jobs', {}).get(candidate + ':66') if queue.exists() else None
        process = children.get(candidate)
        if process is not None:
            job['controller_exit_code'] = process.poll()
        if entry and entry.get('status') == 'complete' and entry.get('process_exit_code') == 0:
            provenance = Path(entry['output']) / 'PROVENANCE.json'
            result = read(provenance) if provenance.exists() else {}
            job['status'] = 'complete' if result.get('status') == 'complete' and result.get('outputs_verified') else 'failed'
        elif entry and alive(entry):
            job.update(status='running', training_pid=entry['pid'])
        elif alive(job):
            job['status'] = 'initializing' if entry is None else 'finishing'
        elif job.get('pid') is None and entry is None:
            job.update(status='inspection_pending', reason='Crash window before controller PID persisted; do not schedule around an unknown launch')
        else:
            text = Path(job['log']).read_text() if Path(job['log']).exists() else ''
            created = entry is not None or Path(job['output']).exists()
            if retryable_admission(text, created):
                job['status'] = 'pending'
            else:
                job.update(status='failed', reason='Controller exited without verified completion; no automatic rerun')


def launch(args, row, gpu_index, state):
    candidate = row['candidate']
    lane = args.root / f'group{row["group"] + 1}'
    previous = state['jobs'].get(candidate, {})
    attempt = previous.get('attempt', 0) + 1
    log = args.root / 'controller_logs' / f'{candidate}.attempt{attempt}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [args.python, '-u', str(args.controller), '--root', str(lane), '--snapshot', str(args.snapshot),
        '--manifest', row['manifest'], '--reference-root', str(args.reference_root), '--baseline-audit', str(args.baseline_audit),
        '--data-manifest', str(args.data_manifest), '--cpu-log', str(args.cpu_log), '--cpu-command', args.cpu_command,
        '--candidate', candidate, '--gpu', gpu_index, '--gpu-uuid', GPUS[gpu_index], '--max-per-gpu', '11',
        '--defer-cuda-smoke-by-user', '--estimated-peak-mib', str(args.estimated_peak_mib), '--live-allocation-floor']
    job = dict(candidate=candidate, status='launch_intent', gpu_uuid=GPUS[gpu_index], gpu_index=gpu_index,
        lane_root=str(lane), output=str(lane / 'runs' / candidate / 'seed_66'), command=command, log=str(log), attempt=attempt)
    state['jobs'][candidate] = job
    write(args.root / 'DISPATCH.json', state)  # crash here is inspection-required, not an implicit retry
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=args.snapshot, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu_index], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
    job.update(pid=process.pid, start_ticks=identity(process.pid), status='initializing')
    write(args.root / 'DISPATCH.json', state)
    return process


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'snapshot', 'controller', 'reference-root', 'baseline-audit', 'data-manifest', 'cpu-log'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--manifest', type=Path, action='append', required=True)
    result.add_argument('--existing-root', type=Path, action='append', default=[])
    result.add_argument('--cpu-command', required=True)
    result.add_argument('--python', default=sys.executable)
    result.add_argument('--estimated-peak-mib', type=float, default=2000.)
    result.add_argument('--poll-seconds', type=float, default=30.)
    result.add_argument('--once', action='store_true', help='One admission cycle; submitted jobs remain persistent')
    return result


def main():
    args = parser().parse_args()
    for key in ('root', 'snapshot', 'controller', 'reference_root', 'baseline_audit', 'data_manifest', 'cpu_log'):
        setattr(args, key, getattr(args, key).resolve())
    if args.poll_seconds < 1 or not math.isfinite(args.estimated_peak_mib) or args.estimated_peak_mib <= 0:
        raise ValueError('Positive polling interval and finite memory estimate required')
    candidates = load_candidates(args.manifest)
    identity_files = [*args.manifest, args.controller, args.snapshot / 'SNAPSHOT.json', args.baseline_audit, args.data_manifest, args.cpu_log]
    binding = {str(path.resolve()): sha(path) for path in identity_files}
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.root / 'DISPATCH.json'
        state = read(path) if path.exists() else dict(binding=binding, jobs={}, high_water={}, status='waiting', authorized_gpus=GPUS)
        if state['binding'] != binding:
            raise ValueError('Dispatcher source/config binding changed; refuse implicit migration')
        children = {}
        roots = [*args.existing_root, args.root / 'group1', args.root / 'group2']
        while True:
            reconcile(state, children)
            try:
                if any(job['status'] == 'inspection_pending' for job in state['jobs'].values()):
                    raise ValueError('Ambiguous launch identity requires inspection; new submissions paused')
                gpus = gpu_resources()
                reservations = [job for job in state['jobs'].values() if job['status'] in ('initializing', 'launch_intent', 'running', 'finishing')]
                actors, disk_reserved = census(roots, reservations, state['high_water'], args.estimated_peak_mib)
                disk_free = shutil.disk_usage(args.root).free / 1024**3
                state.update(at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), resources=gpus,
                             active_processes=actors, disk_free_gib=disk_free, disk_reserved_gib=disk_reserved, wait_reasons={})
                next_row = next((row for row in candidates if state['jobs'].get(row['candidate'], {}).get('status', 'pending') == 'pending'), None)
                if next_row:
                    for gpu_index in sorted(GPUS, key=lambda index: -gpus[index]['free_mib']):
                        allowed, reason = admission(gpus[gpu_index], actors, args.estimated_peak_mib, disk_free, disk_reserved)
                        state['wait_reasons'][gpu_index] = reason
                        if allowed:
                            children[next_row['candidate']] = launch(args, next_row, gpu_index, state)
                            print(state['at'], 'controller submitted', next_row['candidate'], 'GPU' + gpu_index, flush=True)
                            break
                state['status'] = 'active' if any(job['status'] in ('running', 'initializing', 'finishing') for job in state['jobs'].values()) else 'waiting'
                if len(state['jobs']) == 40 and all(job['status'] in ('complete', 'failed') for job in state['jobs'].values()):
                    state['status'] = 'finished_with_failures' if any(job['status'] == 'failed' for job in state['jobs'].values()) else 'complete'
                print(state['at'], state['status'], json.dumps(state['wait_reasons'], sort_keys=True), flush=True)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                state.update(status='resource_inspection_wait', last_error=repr(error))
                print('Inspection failed, no launch:', repr(error), flush=True)
            write(path, state)
            if args.once or state['status'] in ('complete', 'finished_with_failures'):
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
