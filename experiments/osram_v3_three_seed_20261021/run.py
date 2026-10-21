"""Finish only the missing V3 seeds; preserve all seed66 workers and results."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.osram_v3_classification_20261021.run import GPU_LIMITS, capacity, read
from experiments.osram_source_relation_attention_20261021.run import gpu_free
from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib, write

CLASS_QUEUE = Path('/data2/yb/remote_experiments/osram_v3_classification_20261021/execution_v1/QUEUE.json')
ORIGINAL_QUEUE = Path('/data2/yb/remote_experiments/osram_source_role_v3_20261021/execution_v1/QUEUE.json')
PYTHONS = dict(classification='/data2/yb/conda_envs/cmerc-repro/bin/python',
    regression='/data2/yb/reproduction_workspace/envs/s0/bin/python')


def plan_jobs(root):
    jobs = []
    for seed in (67, 68):
        specs = [('CMUMOSEI', 1), ('MELD', 1)]
        specs += [(d, f) for f in range(1, 6) for d in ('IEMOCAPFour', 'IEMOCAPSix')]
        for dataset, fold in specs:
            jobs.append(dict(id=f'{dataset}_f{fold}_s{seed}', dataset=dataset, fold=fold,
                seed=seed, status='pending', output=str(Path(root)/'runs'/dataset/f'fold_{fold}'/f'seed_{seed}')))
    return jobs


def worker_command(args, job, uuid):
    mosei = job['dataset'] == 'CMUMOSEI'
    module = ('experiments.osram_v3_three_seed_20261021.mosei_worker' if mosei else
        'experiments.osram_v3_classification_20261021.worker')
    cmd = [PYTHONS['regression' if mosei else 'classification'], '-u', '-m', module,
        '--seed', str(job['seed']), '--phase', 'train', '--gpu-uuid', uuid,
        '--source-commit', args.source_commit, '--output', job['output']]
    if not mosei:
        cmd += ['--dataset', job['dataset'], '--fold', str(job['fold'])]
    return cmd


def dependency_complete(queue):
    if queue.get('status') in ('failed', 'draining_after_failure'):
        raise RuntimeError('Seed66 queue failed; inspect rather than hide failure or duplicate it')
    return queue.get('status') == 'complete' and bool(queue.get('jobs')) and all(
        j['status'] == 'complete' for j in queue['jobs'])


def measured_peaks(classification, original):
    result = {}
    for q in (classification, original):
        for item in q.get('smoke', []):
            if item['dataset'] not in ('MELD', 'IEMOCAPFour', 'IEMOCAPSix', 'CMUMOSEI'):
                continue
            if item.get('status') == 'complete' and item.get('returncode') == 0:
                result[item['dataset']] = float(item['peak_memory_mib'])
    if set(result) != {'MELD', 'IEMOCAPFour', 'IEMOCAPSix', 'CMUMOSEI'} or min(result.values()) <= 0:
        raise ValueError('Four successful original V3 real-batch smoke records required')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Inspect persistent queue; never duplicate submitted supplement')
    if shutil.disk_usage(args.output.parent).free < 48*2**30:
        raise RuntimeError('Need48GiB to retain24 new runs with safety reserve')
    classification, original = read(CLASS_QUEUE), read(ORIGINAL_QUEUE)
    peaks = measured_peaks(classification, original)
    args.output.mkdir()
    lock = (args.output/'QUEUE.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='waiting_seed66_queue', pid=os.getpid(), source_commit=args.source_commit,
        jobs=plan_jobs(args.output), gpu_limits=GPU_LIMITS, dependency=str(CLASS_QUEUE),
        reused_smoke_peaks_mib=peaks, requested_seeds=[66,67,68], added_seeds=[67,68],
        label='INTERNAL TEST-ORACLE SCREEN', no_seed66_resubmission=True)
    tick, children = time.monotonic(), []
    from .report import summarize
    def save():
        state['seconds'] = time.monotonic()-tick
        write(args.output/'QUEUE.json', state); summarize(args.output, state, read(CLASS_QUEUE), read(ORIGINAL_QUEUE))
    try:
        save()
        # The old immutable controller cannot count another controller's children.
        # Serialize controllers, not jobs: once it drains, use the4/8 caps here.
        while not dependency_complete(read(CLASS_QUEUE)):
            save(); time.sleep(15)
        state['status'] = 'running'; failed = False
        while True:
            for proc, job in children:
                summary = read(Path(job['output'])/'SUMMARY.json')
                job.update(epoch=summary.get('epoch'), peak_memory_mib=summary.get('peak_memory_mib'))
                if summary.get('peak_memory_mib'):
                    peaks[job['dataset']] = max(peaks[job['dataset']], summary['peak_memory_mib'])
                code = proc.poll()
                if code is not None and job['status'] == 'running':
                    ok = code == 0 and summary.get('status') == 'complete'
                    job.update(status='complete' if ok else 'failed', returncode=code)
                    failed = failed or not ok
                failed = failed or summary.get('status') == 'failed'
            pending = [j for j in state['jobs'] if j['status'] == 'pending']
            if failed:
                state['status'] = 'draining_after_failure'
            elif pending and available_gib()>=64 and shutil.disk_usage(args.output).free>=20*2**30:
                # Find a fitting pending job, rather than block cheap jobs behind
                # an IEMOCAP job that cannot currently fit. No configuration changes.
                admitted = False
                for job in pending:
                    for uuid in reversed(GPU_LIMITS):
                        _, free = gpu_free(uuid)
                        active = [(p,j) for p,j in children if p.poll() is None and j['gpu_uuid'] == uuid]
                        loading = [j for _,j in active if not read(Path(j['output'])/'SUMMARY.json').get('peak_memory_mib')]
                        reserve = sum(max(2048, int(1.4*peaks[j['dataset']]+512)) for j in loading)
                        if capacity(free-reserve, len(active), peaks[job['dataset']], limit=GPU_LIMITS[uuid])<=0:
                            continue
                        cmd = worker_command(args, job, uuid); log = args.output/(job['id']+'.log')
                        with log.open('x') as stream:
                            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid,
                                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                        job.update(status='running', pid=proc.pid, gpu_uuid=uuid, command=cmd, log=str(log))
                        children.append((proc, job)); admitted = True; break
                    if admitted:
                        break
            save()
            if not any(p.poll() is None for p,_ in children) and (failed or not pending):
                break
            time.sleep(15)
        state['status'] = 'failed' if failed else 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
