"""Finite eight-rate MOSI Flat queue; train=test rate per independent model."""
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
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib

RATES = tuple(f'{i/10:.1f}' for i in range(8))


def run_directory(root, rate):
    if rate not in RATES:
        raise ValueError('Only eight approved fixed rates')
    return Path(root)/'runs'/('rate_'+rate.replace('.', 'p'))/'seed_66'


def worker_command(args, rate):
    return [sys.executable, '-u', '-m', 'experiments.osram_mosi_fixed_rate_seed66_20261021.worker',
        '--phase', 'train', '--rate', rate, '--gpu-uuid', args.gpu_uuid,
        '--source-commit', args.source_commit, '--output', str(run_directory(args.output, rate))]


def capacity(free_mib, active, peak_mib, unmeasured=0):
    """Reserve2GiB globally, >=4GiB/job; account for loading children."""
    reservation = max(4096, int(1.4*peak_mib))
    budget = free_mib - 2048 - unmeasured*reservation
    return max(0, min(4-active, int(budget//reservation)))


def read_state(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh independent queue required')
    if not args.output.parent.is_dir() or shutil.disk_usage(args.output.parent).free < 15*2**30:
        raise RuntimeError('Require existing output parent with15GiB free')
    index, free = gpu_free(args.gpu_uuid)
    if available_gib() < 64 or capacity(free, 0, 4096) == 0:
        raise RuntimeError('Need64GiB host RAM and>=4GiB/job plus2GiB GPU reserve')
    args.output.mkdir()
    lock = (args.output/'QUEUE.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'QUEUE.json'; tick = time.monotonic(); children = []
    state = dict(status='running', pid=os.getpid(), source_commit=args.source_commit,
        gpu_uuid=args.gpu_uuid, physical_gpu=index, max_parallel=4, jobs=[],
        pending_rates=list(RATES), first_epoch_measured=False,
        reserve_mib=2048, minimum_job_reservation_mib=4096,
        minimum_available_ram_gib=64, seed=66, label='INTERNAL TEST-ORACLE SCREEN')
    def save():
        state['seconds'] = time.monotonic()-tick
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True)+'\n'); temporary.replace(path)
    def launch(rate):
        command = worker_command(args, rate); log = args.output/('rate_'+rate.replace('.', 'p')+'.log')
        with log.open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid,
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        job = dict(rate=rate, pid=process.pid, command=command, log=str(log), status='running',
            output=str(run_directory(args.output, rate)))
        children.append((process, job)); state['jobs'].append(job)
        state['pending_rates'].remove(rate); save()
    try:
        save(); launch(RATES[0])
        failed = False
        while True:
            records = {}
            for process, job in children:
                record = read_state(Path(job['output'])/'SUMMARY.json'); records[job['rate']] = record
                job.update(epoch=record.get('epoch'), peak_memory_mib=record.get('peak_memory_mib'))
                code = process.poll()
                if code is None:
                    # A caught worker error may precede process exit; stop new
                    # submissions immediately, without killing running peers.
                    failed = failed or record.get('status') == 'failed'
                    continue
                if job['status'] == 'running':
                    ok = code == 0 and record.get('status') == 'complete' and record.get('epoch') == 150
                    job.update(returncode=code, status='complete' if ok else 'failed')
                    failed = failed or not ok
            first = records.get(RATES[0], {})
            measured = (first.get('epoch', 0) or 0) >= 1 and bool(first.get('peak_memory_mib'))
            state['first_epoch_measured'] = measured
            active = [(p, j) for p, j in children if p.poll() is None]
            if failed:
                state.update(status='draining_after_failure',
                    stop_reason='A worker failed; no new submissions, running jobs are not killed')
            elif measured and state['pending_rates']:
                _, free = gpu_free(args.gpu_uuid)
                peaks = [r.get('peak_memory_mib') or 0 for r in records.values()]
                peak = max(peaks)
                unmeasured = sum(not records[j['rate']].get('peak_memory_mib') for _, j in active)
                slots = capacity(free, len(active), peak, unmeasured)
                state.update(current_free_memory_mib=free, measured_peak_mib=peak,
                    job_reservation_mib=max(4096, int(1.4*peak)), available_ram_gib=available_gib())
                if state['available_ram_gib'] < 64 or shutil.disk_usage(args.output).free < 15*2**30:
                    slots = 0; state['status'] = 'waiting_resources'
                elif slots > 0:
                    # One launch per poll allows imports/data allocations to
                    # become visible before admitting the next process.
                    launch(state['pending_rates'][0]); state['status'] = 'running'
            from .report import summarize
            summarize(args.output)
            save()
            if not any(p.poll() is None for p, _ in children) and (failed or not state['pending_rates']):
                break
            time.sleep(15)
        state['status'] = 'failed' if failed else 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
