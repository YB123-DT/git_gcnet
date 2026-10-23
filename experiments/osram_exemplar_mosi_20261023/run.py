"""Persistent bounded queue: two real-data smokes, then exactly two MOSI66 runs."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .worker import VARIANTS, RATES


def approved_jobs():
    return [dict(id='MOSI66_' + variant, dataset='CMUMOSI', variant=variant, seed=66) for variant in VARIANTS]


def worker_command(args, job, phase, uuid, resume=False):
    expected = {row['id']: row for row in approved_jobs()}.get(job.get('id'))
    if expected is None or any(job.get(k) != v for k, v in expected.items()) or phase not in ('smoke', 'train'):
        raise ValueError('Only the two approved MOSI66 retrieval arms')
    if resume and phase != 'train':
        raise ValueError('Only full-state formal training may resume')
    command = [sys.executable, '-u', '-m', 'experiments.osram_exemplar_mosi_20261023.worker',
        '--variant', job['variant'], '--phase', phase, '--seed', '66', '--gpu-uuid', uuid,
        '--source-commit', args.source_commit,
        '--output', str(args.output / ('smoke' if phase == 'smoke' else 'runs') / job['id'])]
    return command + (['--resume'] if resume else [])


def process_record(pid):
    try:
        root = Path('/proc') / str(pid)
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(start_ticks=stat[19], state=stat[0],
            command=(root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_is_running(job):
    actual = process_record(job.get('pid'))
    return bool(actual and actual['state'] != 'Z' and actual['start_ticks'] == job.get('start_ticks')
                and actual['command'] == job.get('command'))


def completion_matches(job, summary, phase):
    if (summary.get('status') != 'complete' or summary.get('phase') != phase or summary.get('seed') != 66
            or summary.get('dataset') != 'CMUMOSI' or summary.get('variant') != job['variant']):
        return False
    if phase == 'smoke':
        return bool(summary.get('smoke_only') and summary.get('smoke_optimizer_steps', 0) >= 4
                    and summary.get('optimizer_rng_prediction_reload_exact')
                    and summary.get('peak_memory_mib', 0) > 0)
    output = Path(job['output'])
    return (summary.get('epoch') == 150 and set(summary.get('evaluation', {})) == set(RATES)
        and set(summary.get('best_by_rate', {})) == set(RATES)
        and (output / 'last.pt').is_file() and (output / 'fixed_masks.pt').is_file()
        and (output / 'support_manifest.json').is_file()
        and all((output / row[field]).is_file() for row in summary['best_by_rate'].values()
                for field in ('checkpoint', 'predictions')))


def select_gpu(peak, resources, per_gpu):
    """Admit measured peaks, leaving room for live peers and CUDA overhead."""
    candidates = []
    for uuid, item in resources.items():
        peers = item['peers']
        required = 2000 + 1.5 * peak + sum(max(0, 1.5 * p - resident) for p, resident in peers)
        if len(peers) < per_gpu and item['free_mib'] >= required:
            candidates.append((len(peers), -item['free_mib'], uuid, required))
    return min(candidates)[2:] if candidates else (None, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuids', nargs='+', required=True)
    parser.add_argument('--per-gpu', type=int, choices=(1, 2), default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    from experiments.osram_source_relation_attention_20261021.run import gpu_free
    from experiments.osram_mosi_v3_capacity_20261022.run import resident_memory
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    from .report import summarize
    if not args.gpu_uuids or len(set(args.gpu_uuids)) != len(args.gpu_uuids):
        raise ValueError('Distinct healthy physical GPU UUIDs required')
    gpus = {uuid: gpu_free(uuid)[0] for uuid in args.gpu_uuids}
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent queue or explicit recovery required')
    if args.resume and not (args.output / 'QUEUE.json').is_file():
        raise ValueError('Recovery requires existing queue ledger')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick, active = args.output / 'QUEUE.json', time.monotonic(), {}
    if args.resume:
        state = json.loads(path.read_text())
        if (state['source_commit'] != args.source_commit or state['gpus'] != gpus
                or state['per_gpu'] != args.per_gpu):
            raise ValueError('Recovery changed source/GPU/concurrency contract')
        for group in ('jobs', 'smoke'):
            if [{k: row[k] for k in ('id', 'dataset', 'variant', 'seed')} for row in state[group]] != approved_jobs():
                raise ValueError('Recovery changed bounded experiment scope')
    else:
        state = dict(status='smoke', source_commit=args.source_commit, gpus=gpus, per_gpu=args.per_gpu,
            max_jobs=2, jobs=[dict(job, status='pending') for job in approved_jobs()],
            smoke=[dict(job, status='pending') for job in approved_jobs()])
    old_seconds = state.get('seconds', 0)
    state.update(pid=os.getpid(), command=[sys.executable] + sys.argv)

    def save():
        state['seconds'] = old_seconds + time.monotonic() - tick
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(path)

    def summary_for(job):
        path = Path(job['output']) / 'SUMMARY.json'
        return json.loads(path.read_text()) if path.is_file() else {}

    def headroom():
        return available_gib() >= 64 and shutil.disk_usage(args.output).free >= 50 * 2**30

    def refresh(jobs, phase):
        for job in jobs:
            if job['status'] != 'running':
                continue
            summary = summary_for(job)
            job.update(epoch=summary.get('epoch'), worker_status=summary.get('status'))
            process = active.get((phase, job['id']))
            code = process.poll() if process is not None else None
            running = code is None if process is not None else process_is_running(job)
            if running:
                continue
            done = code in (None, 0) and completion_matches(job, summary, phase)
            job.update(status='complete' if done else 'failed', returncode=code,
                peak_memory_mib=summary.get('peak_memory_mib'), error=summary.get('error'))

    def spawn(job, phase, uuid, required):
        resume = job.pop('resume_requested', False)
        output = args.output / ('smoke' if phase == 'smoke' else 'runs') / job['id']
        if output.exists() and not resume:
            raise FileExistsError('Existing worker output requires recovery: ' + str(output))
        if not headroom() or gpu_free(uuid)[1] < required:
            return False
        command = worker_command(args, job, phase, uuid, resume)
        log = args.output / (phase + '_' + job['id'] + '.log')
        with log.open('a' if resume else 'w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        record = process_record(process.pid)
        job.update(status='running', gpu_uuid=uuid, physical_gpu=gpus[uuid], pid=process.pid,
            start_ticks=record['start_ticks'] if record else None, command=command, output=str(output),
            log=str(log), launch_time=time.time(), admission_required_mib=required)
        active[(phase, job['id'])] = process
        save()
        return True

    try:
        if args.resume:
            for phase, group in (('smoke', 'smoke'), ('train', 'jobs')):
                for job in state[group]:
                    if job['status'] == 'running' and process_is_running(job):
                        continue
                    if job['status'] in ('running', 'failed', 'complete'):
                        summary = summary_for(job)
                        if completion_matches(job, summary, phase):
                            job.update(status='complete', peak_memory_mib=summary.get('peak_memory_mib'))
                        elif phase == 'train' and (Path(job['output']) / 'last.pt').is_file():
                            job.update(status='pending', resume_requested=True)
                        else:
                            raise RuntimeError('No compatible completed smoke or recoverable state: ' + job['id'])
        save()
        while any(job['status'] in ('pending', 'running') for job in state['smoke']):
            refresh(state['smoke'], 'smoke')
            if any(job['status'] == 'failed' for job in state['smoke']):
                raise RuntimeError('Real-data smoke failed; formal training not launched')
            pending = next((job for job in state['smoke'] if job['status'] == 'pending'), None)
            # Serialize real-data smoke: measure each arm's peak before scheduling training.
            if pending and not any(job['status'] == 'running' for job in state['smoke']) and headroom():
                candidates = [(gpu_free(uuid)[1], uuid) for uuid in args.gpu_uuids]
                free, uuid = max(candidates)
                if free >= 12000:
                    spawn(pending, 'smoke', uuid, 12000)
            save()
            if any(job['status'] in ('pending', 'running') for job in state['smoke']):
                time.sleep(10)
        if not all(job['status'] == 'complete' for job in state['smoke']):
            raise RuntimeError('Both real-data smokes must pass')
        peaks = {job['id']: job['peak_memory_mib'] for job in state['smoke']}
        state['status'] = 'running'
        save()
        while any(job['status'] in ('pending', 'running') for job in state['jobs']):
            refresh(state['jobs'], 'train')
            if headroom():
                resident = resident_memory()
                resources = {uuid: dict(free_mib=gpu_free(uuid)[1], peers=[
                    (max(peaks[p['id']], float(summary_for(p).get('peak_memory_mib', 0))), resident.get(p['pid'], 0))
                    for p in state['jobs'] if p['status'] == 'running' and p['gpu_uuid'] == uuid]) for uuid in args.gpu_uuids}
                for job in state['jobs']:
                    if job['status'] == 'pending':
                        uuid, required = select_gpu(peaks[job['id']], resources, args.per_gpu)
                        if uuid is not None and spawn(job, 'train', uuid, required):
                            break
            save()
            summarize(args.output)
            if any(job['status'] in ('pending', 'running') for job in state['jobs']):
                time.sleep(15)
        state['status'] = 'complete' if all(job['status'] == 'complete' for job in state['jobs']) else 'failed'
        save()
        summarize(args.output)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
