"""Finite resource-admitted MOSI GCNet-missing Flat/Joint three-seed queue.

This stdlib-only controller also serves the following IEMOCAP queue. It never
kills peers, changes batch size, or retries a failed training job automatically.
"""
from __future__ import annotations

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

from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib, write

ROOT = Path(__file__).resolve().parents[2]
BAD_UUID = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'
MODULE = 'experiments.osram_mosi_gcnet_router_3seed_20261019.worker'
RATES = tuple(f'{i / 10:.1f}' for i in range(8))


def devices(uuids):
    """Resolve physical indices; a process-local cuda:0 is not a physical card."""
    if not uuids or len(set(uuids)) != len(uuids) or BAD_UUID in uuids:
        raise ValueError('Require distinct healthy GPU UUIDs; physical GPU4 forbidden')
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], text=True, timeout=20)
    found = {}
    for line in output.strip().splitlines():
        index, uuid, used, total = [v.strip() for v in line.split(',')]
        if uuid in uuids:
            if int(index) == 4:
                raise ValueError('Physical GPU4 forbidden')
            found[uuid] = dict(index=int(index), used_mib=int(used), total_mib=int(total))
    if set(found) != set(uuids):
        raise ValueError('Some requested GPU UUIDs are unavailable')
    return found


def gpu_capacities(found, uuids, peak_mib):
    if not math.isfinite(peak_mib) or peak_mib <= 0:
        raise ValueError('Smoke must report positive finite peak_memory_mib')
    return {uuid: max(0, min(2, int((found[uuid]['total_mib'] - found[uuid]['used_mib'] - 2048)
                                   // (1.5 * peak_mib)))) for uuid in uuids}


def command(module, output, source_commit, uuid, mode, seed, *, phase='train', fold=None):
    argv = [sys.executable, '-m', module, '--phase', phase, '--mode', mode, '--seed', str(seed),
            '--gpu-uuid', uuid, '--source-commit', source_commit, '--output', str(output)]
    if fold is not None:
        argv += ['--fold', str(fold)]
    return argv


def make_job(module, output, source_commit, uuid, mode, seed, dataset, *, fold=None, phase='train'):
    return dict(dataset=dataset, seed=seed, fold=fold, mode=mode, phase=phase, source_commit=source_commit,
                gpu_uuid=uuid, output=str(output), log=str(output) + '.log', status='pending',
                mask_protocol='gcnet-random-mask-v1',
                command=command(module, output, source_commit, uuid, mode, seed, phase=phase, fold=fold))


def plan_jobs(output, source_commit, gpus):
    if len(gpus) != 3:
        raise ValueError('This finite plan expects three GPU UUIDs')
    return [make_job(MODULE, Path(output)/'runs'/'MOSI'/f'seed_{seed}'/mode,
                     source_commit, gpus[index], mode, seed, 'CMUMOSI')
            for index, seed in enumerate((66, 67, 68)) for mode in ('flat', 'joint')]


def dependency_state(path):
    path = Path(path)
    if not path.exists():
        return 'waiting'
    state = json.loads(path.read_text())
    if state.get('status') in ('failed', 'blocked', 'interrupted'):
        raise RuntimeError('The upstream MOSI queue did not complete: ' + str(path))
    return 'ready' if state.get('status') == 'complete' else 'waiting'


def validate_summary(job):
    value = json.loads((Path(job['output'])/'SUMMARY.json').read_text())
    for key in ('status', 'phase', 'mode', 'seed', 'source_commit'):
        expected = 'complete' if key == 'status' else job[key]
        if value.get(key) != expected:
            raise ValueError(f'Child summary {key} mismatch: {job["output"]}')
    if value.get('dataset') != job['dataset']:
        raise ValueError('Child summary dataset mismatch')
    if job.get('fold') is not None:
        actual = value.get('fold', value.get('identity', {}).get('config', {}).get('fold'))
        if actual != job['fold']:
            raise ValueError('Child summary fold mismatch')
    if job['phase'] == 'train':
        if set(value.get('evaluation', {})) != set(RATES) or 'router' not in value.get('aggregate', {}):
            raise ValueError('Training completion requires eight evaluated rates and aggregate')
    elif not value.get('smoke_only') or not value.get('full_checkpoint_reloaded'):
        raise ValueError('Smoke must verify checkpoint reload and identify smoke_only')
    return value


def read_smoke_peak(summary):
    peak = float(summary['peak_memory_mib'])
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError('Invalid GPU smoke memory peak')
    return peak


def launch(job):
    output = Path(job['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(job['log']).open('a' if '--resume' in job['command'] else 'x') as stream:
        process = subprocess.Popen(job['command'], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=job['gpu_uuid'], OMP_NUM_THREADS='2',
                     MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
    job.update(status='running', pid=process.pid, started_at=time.time())
    return process


def poll(process, job):
    code = process.poll()
    if code is None:
        return False
    try:
        if code != 0:
            raise RuntimeError('Child exited with status ' + str(code))
        validate_summary(job)
        job.update(status='complete')
    except (OSError, ValueError, RuntimeError) as error:
        job.update(status='failed', error=repr(error))
    job.update(exit_code=code, finished_at=time.time())
    return True


def recover_jobs(state):
    """Explicit queue resume only; never repeat completed or previously failed jobs."""
    for job in [state.get('smoke', {})] + state['jobs']:
        if not job:
            continue
        if job['status'] == 'complete':
            validate_summary(job)
        elif job['status'] == 'running':
            try:
                os.kill(job['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Recorded child still alive; refuse duplicate submission')
            try:
                validate_summary(job)
                job['status'] = 'complete'
                continue
            except (OSError, ValueError):
                pass
            if job['phase'] != 'train' or not (Path(job['output'])/'last.pt').exists():
                raise RuntimeError('Interrupted child has no compatible complete recovery checkpoint')
            job.update(status='pending', previous_pid=job['pid'])
            if '--resume' not in job['command']:
                job['command'].append('--resume')
        elif job['status'] in ('failed', 'blocked'):
            raise RuntimeError('Failed jobs are not automatically retried')


def run_queue(args, jobs, report, *, references=None):
    """One smoke, bounded train waves, complete-only report. Called under tmux."""
    output = args.output
    if output.exists() and not args.resume:
        raise FileExistsError('Independent output required; pass --resume only for this exact queue')
    output.mkdir(parents=True, exist_ok=args.resume)
    lock = (output/'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = output/'QUEUE.json'
    if args.resume:
        state = json.loads(path.read_text())
        if state['source_commit'] != args.source_commit or state['gpus'] != args.gpus:
            raise ValueError('Resume source/GPU identity mismatch')
        if state['status'] == 'complete':
            raise ValueError('Queue already complete; do not repeat training')
        old = [{k: j[k] for k in ('seed', 'fold', 'mode', 'output', 'gpu_uuid')} for j in state['jobs']]
        if old != [{k: j[k] for k in ('seed', 'fold', 'mode', 'output', 'gpu_uuid')} for j in jobs]:
            raise ValueError('Resume plan differs')
        recover_jobs(state)
    else:
        first = jobs[0]
        smoke = make_job(first['command'][2], output/'smoke'/'joint', args.source_commit,
                         args.gpus[0], 'joint', 66, first['dataset'], fold=first['fold'], phase='smoke')
        state = dict(status='preflight', source_commit=args.source_commit, pid=os.getpid(),
                     gpus=args.gpus, jobs=jobs, smoke=smoke, references=references or [],
                     started_at=time.time(), minimum_available_gib=64, maximum_per_gpu=2,
                     protocol='gcnet-random-mask-v1', python=sys.executable,
                     after_queue=str(args.after_queue) if args.after_queue else None)
    children = {}

    def record():
        state['updated_at'] = time.time()
        write(path, state)

    try:
        record()
        while args.after_queue and dependency_state(args.after_queue) != 'ready':
            state.update(status='waiting_for_upstream')
            record()
            time.sleep(args.poll_seconds)
        state['gpu_preflight'] = devices(args.gpus)
        state['cpu_count'] = os.cpu_count()
        state['disk_free_gib'] = shutil.disk_usage(output).free / 2**30
        # Includes two full model/Adam checkpoints per job plus evaluation/provenance.
        if state['disk_free_gib'] < 4 + .75 * len(jobs):
            raise RuntimeError('Insufficient disk for this finite queue; no checkpoints deleted')
        state['minimum_disk_gib'] = 4 + .75 * len(jobs)
        smoke = state['smoke']
        while smoke['status'] != 'complete':
            free = available_gib()
            state.update(available_ram_gib=free, status='waiting_for_ram' if free < 64 else 'smoke')
            record()
            if free < 64:
                time.sleep(args.poll_seconds)
                continue
            if smoke['status'] == 'failed':
                raise RuntimeError('GPU smoke failed; formal training blocked')
            current = devices(args.gpus)
            first_free = current[smoke['gpu_uuid']]['total_mib'] - current[smoke['gpu_uuid']]['used_mib']
            if first_free < 2048:
                state.update(status='waiting_for_vram')
                record()
                time.sleep(args.poll_seconds)
                continue
            process = launch(smoke)
            record()
            while not poll(process, smoke):
                time.sleep(min(5, args.poll_seconds))
            record()
        peak = read_smoke_peak(validate_summary(smoke))
        state.update(smoke_peak_memory_mib=peak,
                     memory_policy='Per GPU <=2 jobs; reserve 1.5x observed joint smoke peak per job + 2GiB/card; MemAvailable>=64GiB per launch')
        while True:
            for key, process in list(children.items()):
                if poll(process, state['jobs'][key]):
                    del children[key]
            failed = any(j['status'] == 'failed' for j in state['jobs'])
            pending = [j for j in state['jobs'] if j['status'] == 'pending']
            if failed and not children:
                for job in pending:
                    job.update(status='blocked', error='An earlier job failed; no implicit retry/new submissions')
                raise RuntimeError('Training failed; other running peers were allowed to finish')
            if not pending and not children:
                break
            state.update(status='draining_after_failure' if failed else 'training')
            free = available_gib()
            state['available_ram_gib'] = free
            current = devices(args.gpus)
            capacities = gpu_capacities(current, args.gpus, peak)
            state['current_devices'] = current
            if not failed and free >= 64:
                for index, job in enumerate(state['jobs']):
                    if job['status'] != 'pending':
                        continue
                    uuid = job['gpu_uuid']
                    running = sum(j['status'] == 'running' and j['gpu_uuid'] == uuid for j in state['jobs'])
                    # First admission reserves all planned slots from fresh free VRAM.
                    # Subsequent admissions also require room for the next complete peak.
                    allowed = state.setdefault('admitted_capacity', {}).get(uuid)
                    if allowed is None or running == 0:
                        allowed = capacities[uuid]
                        state['admitted_capacity'][uuid] = allowed
                    available = current[uuid]['total_mib'] - current[uuid]['used_mib']
                    if running >= allowed or available < 1.5 * peak + 2048:
                        continue
                    if available_gib() < 64:
                        state['status'] = 'waiting_for_ram'
                        break
                    children[index] = launch(job)
                    record()
                    # Do not admit two processes from the same stale resource snapshot.
                    time.sleep(min(2, args.poll_seconds))
                    current = devices(args.gpus)
            elif not failed:
                state['status'] = 'waiting_for_ram'
            if not children and pending and state['status'] == 'training':
                state['status'] = 'waiting_for_vram'
            record()
            time.sleep(args.poll_seconds)
        report(output, state)
        state.update(status='complete', completed_at=time.time())
    except BaseException as error:
        # start_new_session children deliberately survive a controller error.
        state.update(status='failed', error=repr(error), surviving_children=[p.pid for p in children.values()])
        raise
    finally:
        record()


def parser(description):
    value = argparse.ArgumentParser(description=description)
    value.add_argument('--output', type=Path, required=True)
    value.add_argument('--source-commit', required=True)
    value.add_argument('--gpus', nargs=3, required=True, help='Three healthy physical GPU UUIDs')
    value.add_argument('--after-queue', type=Path)
    value.add_argument('--resume', action='store_true')
    value.add_argument('--poll-seconds', type=float, default=30)
    return value


def main():
    args = parser(__doc__).parse_args()
    if args.poll_seconds <= 0:
        raise ValueError('poll-seconds must be positive')
    from .report import write_report
    run_queue(args, plan_jobs(args.output, args.source_commit, args.gpus), write_report)


if __name__ == '__main__':
    main()
