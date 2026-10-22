"""Persistent six-job controller with measured admission and explicit recovery."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.osram_source_relation_attention_20261021.run import gpu_free
from experiments.osram_mosi_v3_capacity_20261022.run import admission_required, resident_memory
from .worker import REFERENCES, VARIANTS, RATES, settings


def approved_jobs():
    return [dict(id=dataset + '_' + variant, dataset=dataset, variant=variant, seed=66)
            for variant in VARIANTS for dataset in REFERENCES]


def worker_command(args, job, phase, uuid, resume=False):
    approved = {entry['id']: entry for entry in approved_jobs()}
    if (job.get('id') not in approved or phase not in ('smoke', 'train')
            or any(job.get(name) != value for name, value in approved.get(job.get('id'), {}).items())):
        raise ValueError('Only the six approved seed66 jobs may run')
    if resume and phase != 'train':
        raise ValueError('Only full-state formal training can resume')
    command = [sys.executable, '-u', '-m', 'experiments.osram_v3_connection_screen_20261022.worker',
        '--dataset', job['dataset'], '--variant', job['variant'], '--phase', phase, '--seed', '66',
        '--gpu-uuid', uuid, '--source-commit', args.source_commit,
        '--output', str(args.output / ('smoke' if phase == 'smoke' else 'runs') / job['id'])]
    return command + (['--resume'] if resume else [])


def select_gpu(new_peak, resources, per_gpu):
    if not 1 <= per_gpu <= 6:
        raise ValueError('Per-GPU concurrency must be between one and six')
    eligible = []
    for uuid, resource in resources.items():
        peers = resource['peers']
        required = admission_required(new_peak, peers)
        if len(peers) < per_gpu and resource['free_mib'] >= required:
            eligible.append((len(peers), -(resource['free_mib'] - required), uuid))
    return min(eligible)[-1] if eligible else None


def process_record(pid):
    """Identify a process without sending signals, including through PID reuse."""
    try:
        root = Path('/proc') / str(pid)
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(start_ticks=stat[19], state=stat[0], command=command)
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_is_running(job):
    actual = process_record(job.get('pid'))
    return bool(actual and actual['state'] != 'Z'
                and actual['start_ticks'] == job.get('start_ticks')
                and actual['command'] == job.get('command'))


def completion_matches(job, summary, phase):
    if (summary.get('status') != 'complete' or summary.get('phase') != phase
            or summary.get('dataset') != job['dataset']
            or summary.get('variant') != job['variant'] or summary.get('seed') != 66):
        return False
    if phase == 'smoke':
        return (summary.get('smoke_only') is True and summary.get('smoke_optimizer_steps', 0) >= 4
                and summary.get('optimizer_rng_prediction_reload_exact') is True
                and summary.get('peak_memory_mib', 0) > 0)
    output = Path(job['output'])
    return (summary.get('epoch') == settings(job['dataset'])['epochs']
        and set(summary.get('evaluation', {})) == set(RATES)
        and set(summary.get('best_by_rate', {})) == set(RATES)
        and (output / 'last.pt').is_file() and (output / 'fixed_masks.pt').is_file()
        and all((output / row[field]).is_file() for row in summary['best_by_rate'].values()
                for field in ('checkpoint', 'predictions')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuids', nargs='+', required=True)
    parser.add_argument('--per-gpu', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--resume', action='store_true', help='Reattach live jobs or resume incomplete train last.pt explicitly.')
    args = parser.parse_args(argv)
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    from .report import summarize
    if not args.gpu_uuids or len(args.gpu_uuids) != len(set(args.gpu_uuids)):
        raise ValueError('One or more distinct healthy GPU UUIDs required')
    # gpu_free rejects both physical host GPU4 and the recorded failed UUID.
    gpus = {uuid: gpu_free(uuid)[0] for uuid in args.gpu_uuids}
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh queue required, or explicit --resume')
    if args.resume and not (args.output / 'QUEUE.json').is_file():
        raise ValueError('Explicit recovery requires the existing QUEUE.json')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick = args.output / 'QUEUE.json', time.monotonic()
    if args.resume:
        state = json.loads(path.read_text())
        if (state['source_commit'] != args.source_commit or state['gpus'] != gpus
                or state['per_gpu'] != args.per_gpu):
            raise ValueError('Immutable queue source, GPUs or concurrency changed')
        expected = approved_jobs()
        for phase in ('jobs', 'smoke'):
            actual = [{name: row[name] for name in ('id', 'dataset', 'variant', 'seed')}
                      for row in state[phase]]
            if actual != expected:
                raise ValueError('Immutable six-job queue changed')
        state['recoveries'] = state.get('recoveries', 0) + 1
    else:
        state = dict(status='smoke', source_commit=args.source_commit, gpus=gpus,
            per_gpu=args.per_gpu, max_jobs=6, seconds=0,
            jobs=[dict(job, status='pending') for job in approved_jobs()],
            smoke=[dict(job, status='pending') for job in approved_jobs()])
    previous_seconds = state.get('seconds', 0)
    state.update(pid=os.getpid(), command=[sys.executable] + sys.argv)
    active = {}

    def save():
        state['seconds'] = previous_seconds + time.monotonic() - tick
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(path)

    def resource_headroom():
        return available_gib() >= 64 and shutil.disk_usage(args.output).free >= 50 * 2**30

    def summary_for(job):
        summary_path = Path(job['output']) / 'SUMMARY.json'
        return json.loads(summary_path.read_text()) if summary_path.is_file() else {}

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
            done = (code in (0, None)) and completion_matches(job, summary, phase)
            job.update(returncode=code, status='complete' if done else 'failed',
                process_exit_observed=True, peak_memory_mib=summary.get('peak_memory_mib'),
                seconds=summary.get('seconds'), error=summary.get('error'))

    def spawn(job, phase, uuid):
        resume = job.get('resume_requested', False)
        command = worker_command(args, job, phase, uuid, resume=resume)
        output = args.output / ('smoke' if phase == 'smoke' else 'runs') / job['id']
        if output.exists() and not resume:
            raise FileExistsError('Existing worker output requires explicit recovery: ' + str(output))
        log = args.output / (phase + '_' + job['id'] + '.log')
        # Repeat all resource checks immediately before this individual launch.
        if not resource_headroom():
            return False
        if gpu_free(uuid)[1] < job.get('admission_required_mib', 12000):
            return False
        with log.open('a' if resume else 'w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1',
                         MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        record = process_record(process.pid)
        job.pop('resume_requested', None)
        job.update(status='running', gpu_uuid=uuid, physical_gpu=gpus[uuid], pid=process.pid,
            start_ticks=record['start_ticks'] if record else None, command=command,
            output=str(output), log=str(log), launch_time=time.time(), resumed=resume)
        active[(phase, job['id'])] = process
        save()
        return True

    def recover(jobs, phase):
        for job in jobs:
            if job['status'] == 'running' and process_is_running(job):
                continue
            if job['status'] in ('running', 'failed'):
                summary = summary_for(job)
                if completion_matches(job, summary, phase):
                    job.update(status='complete', peak_memory_mib=summary.get('peak_memory_mib'))
                elif phase == 'train' and (Path(job['output']) / 'last.pt').is_file():
                    job.update(status='pending', resume_requested=True)
                else:
                    raise RuntimeError('No compatible completed smoke/full training state: ' + job['id'])
            elif job['status'] == 'complete' and not completion_matches(job, summary_for(job), phase):
                raise RuntimeError('Completed queue artifact is missing or incompatible: ' + job['id'])

    save()
    try:
        if args.resume:
            recover(state['smoke'], 'smoke')
            recover(state['jobs'], 'train')
        smokes, jobs = state['smoke'], state['jobs']
        # One real-data smoke per GPU; all six types must pass before training.
        while any(job['status'] in ('pending', 'running') for job in smokes):
            refresh(smokes, 'smoke')
            if any(job['status'] == 'failed' for job in smokes):
                raise RuntimeError('Real-data smoke failed; no implicit retry or formal launch')
            if resource_headroom():
                resources = {uuid: dict(free_mib=gpu_free(uuid)[1], peers=[])
                    for uuid in args.gpu_uuids
                    if not any(job['status'] == 'running' and job['gpu_uuid'] == uuid for job in smokes)}
                # Until a measured peak exists, require 12GiB free conservatively.
                candidates = [uuid for uuid, row in resources.items() if row['free_mib'] >= 12000]
                pending = next((job for job in smokes if job['status'] == 'pending'), None)
                if pending is not None and candidates:
                    uuid = max(candidates, key=lambda value: resources[value]['free_mib'])
                    spawn(pending, 'smoke', uuid)
            save()
            summarize(args.output)
            if any(job['status'] in ('pending', 'running') for job in smokes):
                time.sleep(10)
        if any(job['status'] != 'complete' for job in smokes):
            raise RuntimeError('All six real-data smokes must pass')
        peaks = {job['id']: job['peak_memory_mib'] for job in smokes}
        if any(not isinstance(value, (int, float)) or value <= 0 for value in peaks.values()):
            raise RuntimeError('Six measured positive smoke peaks required')
        state['status'] = 'running'
        save()
        while any(job['status'] in ('pending', 'running') for job in jobs):
            refresh(jobs, 'train')
            if resource_headroom():
                resident = resident_memory()
                resources = {uuid: dict(free_mib=gpu_free(uuid)[1], peers=[
                    (peaks[peer['id']], resident.get(peer['pid'], 0)) for peer in jobs
                    if peer['status'] == 'running' and peer['gpu_uuid'] == uuid]) for uuid in args.gpu_uuids}
                # One launch per observation interval lets resident memory settle.
                for job in jobs:
                    if job['status'] != 'pending':
                        continue
                    uuid = select_gpu(peaks[job['id']], resources, args.per_gpu)
                    if uuid is not None:
                        job['admission_required_mib'] = admission_required(peaks[job['id']], resources[uuid]['peers'])
                        if spawn(job, 'train', uuid):
                            break
            save()
            summarize(args.output)
            if any(job['status'] in ('pending', 'running') for job in jobs):
                time.sleep(15)
        state['status'] = 'complete' if all(job['status'] == 'complete' for job in jobs) else 'failed'
        save()
        summarize(args.output)
    except BaseException as error:
        # Running jobs keep ownership of their outputs. Never signal other jobs.
        state.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
