"""Six finite MOSI arms, real smokes, then measured multi-GPU admission."""
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
from .worker import MODES
from .report import summarize


def worker_command(args, mode, phase, uuid):
    if mode not in MODES or phase not in ('smoke', 'train'):
        raise ValueError('Only six approved MOSI arms and smoke/train phases')
    return [sys.executable, '-u', '-m', 'experiments.osram_mosi_v3_capacity_20261022.worker',
            '--mode', mode, '--phase', phase, '--seed', '66', '--gpu-uuid', uuid,
            '--source-commit', args.source_commit,
            '--output', str(args.output / ('smoke' if phase == 'smoke' else 'runs') / mode)]


def admission_required(new_peak, peers):
    """Free VRAM needed, reserving only each live peer's unallocated margin."""
    return 2000 + 1.5 * new_peak + sum(max(0, 1.5 * peak - resident) for peak, resident in peers)


def resident_memory():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    values = {}
    for line in raw.splitlines():
        pid, memory = [v.strip() for v in line.split(',')]
        if pid.isdigit() and memory.isdigit():
            values[int(pid)] = values.get(int(pid), 0) + int(memory)
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--gpu-uuids', nargs=2, required=True)
    p.add_argument('--per-gpu', type=int, choices=(1, 2, 3), default=3)
    args = p.parse_args()
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    if args.output.exists():
        raise FileExistsError('Fresh independent queue required; no silent duplicate')
    if len(set(args.gpu_uuids)) != 2:
        raise ValueError('Two distinct healthy GPUs required')
    gpus = {uuid: gpu_free(uuid)[0] for uuid in args.gpu_uuids}
    if available_gib() < 64 or shutil.disk_usage(args.output.parent).free < 50 * 2**30:
        raise RuntimeError('Require64GiB host RAM and50GiB output headroom')
    args.output.mkdir()
    lock = (args.output / 'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Widths stay together for direct same-width comparisons, one on each GPU.
    jobs = [dict(mode=mode, gpu_uuid=args.gpu_uuids[0 if MODES[mode][0] == 512 else 1],
                 status='pending') for mode in MODES]
    state = dict(status='smoke', pid=os.getpid(), source_commit=args.source_commit,
                 gpus=gpus, per_gpu=args.per_gpu, jobs=jobs, smoke=[], max_jobs=6)
    start = time.monotonic()
    def save():
        state['seconds'] = time.monotonic() - start
        tmp = args.output / 'QUEUE.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(args.output / 'QUEUE.json')
    def spawn(job, phase):
        mode, uuid = job['mode'], job['gpu_uuid']
        log = args.output / (phase + '_' + mode + '.log')
        cmd = worker_command(args, mode, phase, uuid)
        with log.open('w') as stream:
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1',
                         MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        job.update(pid=proc.pid, command=cmd, log=str(log), status='running',
                   output=str(args.output / ('smoke' if phase == 'smoke' else 'runs') / mode))
        return proc
    save()
    try:
        smokes = [dict(mode=j['mode'], gpu_uuid=j['gpu_uuid'], status='pending') for j in jobs]
        state['smoke'] = smokes
        active = {}
        while any(j['status'] in ('pending', 'running') for j in smokes):
            for job in smokes:
                if job['status'] != 'running':
                    continue
                code = active[job['mode']].poll()
                if code is None:
                    continue
                f = Path(job['output']) / 'SUMMARY.json'
                s = json.loads(f.read_text()) if f.exists() else {}
                job.update(returncode=code, peak_memory_mib=s.get('peak_memory_mib'),
                           seconds=s.get('seconds'), status='complete' if code == 0 and s.get('status') == 'complete' else 'failed')
                if job['status'] == 'failed':
                    save(); raise RuntimeError('Real MOSI smoke failed: ' + job['mode'])
            for uuid in args.gpu_uuids:
                if any(j['status'] == 'running' and j['gpu_uuid'] == uuid for j in smokes):
                    continue
                pending = next((j for j in smokes if j['status'] == 'pending' and j['gpu_uuid'] == uuid), None)
                if pending is not None and available_gib() >= 64 and gpu_free(uuid)[1] >= 12000:
                    active[pending['mode']] = spawn(pending, 'smoke')
            save()
            if any(j['status'] in ('pending', 'running') for j in smokes):
                time.sleep(10)
        peaks = {j['mode']: j['peak_memory_mib'] for j in smokes}
        if any(not isinstance(v, (int, float)) or v <= 0 for v in peaks.values()):
            raise RuntimeError('Missing measured smoke peaks')
        state['status'] = 'running'; active = {}; save()
        while any(j['status'] in ('pending', 'running') for j in jobs):
            for job in jobs:
                if job['status'] != 'running':
                    continue
                proc = active[job['mode']]
                f = Path(job['output']) / 'SUMMARY.json'
                s = json.loads(f.read_text()) if f.exists() else {}
                job.update(epoch=s.get('epoch'), worker_status=s.get('status'))
                code = proc.poll()
                if code is not None:
                    job.update(returncode=code, status='complete' if code == 0 and s.get('status') == 'complete' else 'failed')
            for uuid in args.gpu_uuids:
                live = [j for j in jobs if j['status'] == 'running' and j['gpu_uuid'] == uuid]
                resident = resident_memory()
                for job in jobs:
                    if job['status'] != 'pending' or job['gpu_uuid'] != uuid or len(live) >= args.per_gpu:
                        continue
                    needed = admission_required(peaks[job['mode']],
                        [(peaks[j['mode']], resident.get(j['pid'], 0)) for j in live])
                    job['admission_required_mib'] = needed
                    if available_gib() < 64 or gpu_free(uuid)[1] < needed:
                        continue
                    active[job['mode']] = spawn(job, 'train'); live.append(job); save()
            save(); summarize(args.output)
            if any(j['status'] in ('pending', 'running') for j in jobs):
                time.sleep(15)
        state['status'] = 'complete' if all(j['status'] == 'complete' for j in jobs) else 'failed'
        summarize(args.output)
    except BaseException as error:
        # No process signals, no implicit retries or protocol changes.
        state.update(status='failed', error=repr(error)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
