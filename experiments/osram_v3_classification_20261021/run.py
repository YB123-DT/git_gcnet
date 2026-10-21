"""Finite overnight V3 queue; existing MOSEI is tracked, never resubmitted."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.osram_source_relation_attention_20261021.run import gpu_free
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib, write

PYTHON = '/data2/yb/conda_envs/cmerc-repro/bin/python'
GPU_LIMITS = {
    'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45': 4,
    'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4': 8,
}
MOSEI = Path('/data2/yb/remote_experiments/osram_source_role_v3_20261021/execution_v1/runs/CMUMOSEI/SUMMARY.json')


def validate_gpus(gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(g not in GPU_LIMITS for g in gpus):
        raise ValueError('Only reviewed healthy physical GPU0/2; GPU4 prohibited')


def plan_jobs(root):
    result = [dict(dataset='MELD', fold=1, seed=66)]
    result += [dict(dataset=d, fold=f, seed=66) for d in ('IEMOCAPFour', 'IEMOCAPSix') for f in range(1, 6)]
    for j in result:
        j.update(id=f"{j['dataset']}_f{j['fold']}_s66", status='pending',
            output=str(Path(root)/'runs'/j['dataset']/f"fold_{j['fold']}"/'seed_66'))
    return result


def worker_command(args, job, uuid, phase):
    output = job['output'] if phase == 'train' else str(Path(args.output)/'smoke'/job['dataset'])
    return [PYTHON, '-u', '-m', 'experiments.osram_v3_classification_20261021.worker',
        '--dataset', job['dataset'], '--fold', str(job['fold']), '--seed', '66',
        '--phase', phase, '--gpu-uuid', uuid, '--source-commit', args.source_commit,
        '--output', output]


def capacity(free_mib, active, peak_mib, unmeasured=0, *, limit=4):
    reservation = max(2048, int(1.4*peak_mib + 512))
    return max(0, min(limit-active, int((free_mib-2048-unmeasured*reservation)//reservation)))


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def summarize(root, state):
    rows = []
    for j in state['jobs']:
        s = read(Path(j['output'])/'SUMMARY.json')
        item = dict(dataset=j['dataset'], fold=j['fold'], seed=66, status=j['status'], epoch=s.get('epoch'))
        if j['status'] == 'complete':
            item['best_by_rate'] = s.get('best_by_rate')
            item['aggregate'] = s.get('aggregate')
        rows.append(item)
    mosei = read(MOSEI)
    result = dict(label='INTERNAL TEST-ORACLE SCREEN', source_commit=state['source_commit'],
        queue_status=state['status'], runs=rows, reused_mosei=dict(path=str(MOSEI),
            status=mosei.get('status'), epoch=mosei.get('epoch'), resubmitted=False))
    write(root/'REPORT.json', result)
    lines = ['# V3 finite classification queue', '', 'INTERNAL TEST-ORACLE SCREEN', '',
        '| Dataset | fold | seed | status | epoch |', '|---|---:|---:|---|---:|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['fold']} | 66 | {r['status']} | {r['epoch'] or '—'} |")
    lines += ['', f"Existing MOSEI V3: {mosei.get('status', 'unknown')}, epoch {mosei.get('epoch', '?')}/60; not duplicated."]
    (root/'RESULT.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    args = p.parse_args()
    validate_gpus(list(GPU_LIMITS))
    if args.output.exists():
        raise FileExistsError('Inspect existing queue; do not submit duplicate jobs')
    if shutil.disk_usage(args.output.parent).free < 35*2**30:
        raise RuntimeError('Need35GiB free for11 runs plus checkpoint headroom')
    args.output.mkdir()
    lock = (args.output/'QUEUE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tick = time.monotonic()
    state = dict(status='smoke', pid=os.getpid(), source_commit=args.source_commit,
        gpu_limits=GPU_LIMITS, jobs=plan_jobs(args.output), smoke=[], seed=66,
        original_mosei=str(MOSEI), no_extra_seeds=True)
    children = []
    def save():
        state['seconds'] = time.monotonic()-tick
        write(args.output/'QUEUE.json', state)
    def resources(uuid, peak):
        index, free = gpu_free(uuid)
        active = [(p,j) for p,j in children if p.poll() is None and j['gpu_uuid'] == uuid]
        loading = sum(not read(Path(j['output'])/'SUMMARY.json').get('peak_memory_mib') for _,j in active)
        return capacity(free, len(active), peak, loading, limit=GPU_LIMITS[uuid])
    try:
        save()
        peaks = {}
        for dataset in ('MELD', 'IEMOCAPFour', 'IEMOCAPSix'):
            j = next(j for j in state['jobs'] if j['dataset'] == dataset)
            while True:
                healthy = [g for g in GPU_LIMITS if resources(g, 7000)>0]
                if healthy and available_gib() >= 64:
                    break
                save(); time.sleep(15)
            uuid = healthy[0]; command = worker_command(args, j, uuid, 'smoke')
            log = args.output/('smoke_'+dataset+'.log')
            with log.open('x') as stream:
                code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid,
                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')).returncode
            record = read(args.output/'smoke'/dataset/'SUMMARY.json')
            state['smoke'].append(dict(dataset=dataset, returncode=code, status=record.get('status'),
                peak_memory_mib=record.get('peak_memory_mib'), log=str(log), gpu_uuid=uuid))
            save()
            if code or record.get('status') != 'complete' or not record.get('peak_memory_mib'):
                raise RuntimeError('Real-batch smoke failed: '+dataset)
            peaks[dataset] = record['peak_memory_mib']
        failed = False
        state['status'] = 'running'
        while True:
            for process, job in children:
                record = read(Path(job['output'])/'SUMMARY.json')
                job.update(epoch=record.get('epoch'), peak_memory_mib=record.get('peak_memory_mib'))
                code = process.poll()
                if code is not None and job['status'] == 'running':
                    ok = code == 0 and record.get('status') == 'complete'
                    job.update(status='complete' if ok else 'failed', returncode=code)
                    failed = failed or not ok
                if record.get('status') == 'failed':
                    failed = True
                if record.get('peak_memory_mib'):
                    peaks[job['dataset']] = max(peaks[job['dataset']], record['peak_memory_mib'])
            pending = [j for j in state['jobs'] if j['status'] == 'pending']
            if failed:
                state['status'] = 'draining_after_failure'
            elif pending and available_gib()>=64 and shutil.disk_usage(args.output).free>=20*2**30:
                job = pending[0]; peak = max(peaks.values())
                for uuid in reversed(GPU_LIMITS):
                    if resources(uuid, peak)>0:
                        command = worker_command(args, job, uuid, 'train')
                        log = args.output/(job['id']+'.log')
                        with log.open('x') as stream:
                            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid,
                                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                        job.update(status='running', pid=proc.pid, gpu_uuid=uuid, command=command, log=str(log))
                        children.append((proc, job))
                        break
            summarize(args.output, state); save()
            if not any(p.poll() is None for p,_ in children) and (failed or not pending):
                break
            time.sleep(15)
        state['status'] = 'failed' if failed else 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        save(); summarize(args.output, state)


if __name__ == '__main__':
    main()
