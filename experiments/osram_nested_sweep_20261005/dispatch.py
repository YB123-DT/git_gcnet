"""Ten locked seed66 comparisons; capacity checked on healthy GPUs0/1/6."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from experiments.osram_core20_20261005.run import LABEL, now, write
from experiments.osram_readout_top3_3seed_20261005.dispatch import read, REMOTE, ORIGINALS
from gcnet_missing_m3.nested_sweep import NESTED_SWEEP

GPUS = {0: 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
        1: 'GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049',
        6: 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'}
REFERENCE = REMOTE / 'osram_mosi_memory_gap_ablation_20260920/full/seed_66'


def scores(output):
    values = read(output / 'metrics.json')
    rates = values['selected_weighted_f1_by_rate']
    return dict(per_rate=rates, mean8=100 * statistics.mean(rates[str(i / 10)] for i in range(8)),
                high=100 * statistics.mean(rates[str(i / 10)] for i in (5, 6, 7)),
                parameter_count=values.get('parameter_count'))


def summarize(root, rows):
    table = []
    for row in rows:
        result = dict(method=row['method'], category=row['category'], status=row['status'],
                      settings=NESTED_SWEEP[row['method']])
        if row['status'] == 'complete':
            result.update(scores(Path(row['output'])))
            prov = read(Path(row['output']) / 'PROVENANCE.json')
            result['peak_allocated_mib'] = prov['peak_allocated_mib']
        elif row['status'] == 'failed':
            result['error'] = row.get('error')
        table.append(result)
    references = {}
    for name, output in (('flat', REFERENCE), ('old_nested', ORIGINALS['nested_gnn_rooted_evidence'])):
        try:
            references[name] = dict(status='available', output=str(output), **scores(output))
        except (OSError, ValueError, KeyError) as error:
            references[name] = dict(status='unavailable', error=repr(error))
    write(root / 'SUMMARY.json', dict(label=LABEL, references=references, methods=table))


def capacity(gpu, jobs):
    raw = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    index, identifier, free = [s.strip() for s in raw.split(',')]
    if int(index) != gpu or identifier != GPUS[gpu] or gpu == 4:
        raise ValueError('healthy GPU UUID whitelist mismatch')
    owned = [job for job in jobs.values() if job[2]['gpu'] == gpu]
    if len(owned) >= 2:
        return False
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    used = {int(line.split(',')[0]): float(line.split(',')[1]) for line in raw.splitlines()}
    # Reserve not-yet-materialized model/data allocations of newly started jobs.
    growth = sum(max(0, 6000 - used.get(child.pid, 0)) for child, _, _ in owned)
    return float(free) >= growth + 6000 + 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'DISPATCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'DISPATCH.json').exists():
        raise FileExistsError('Inspect live/completed runs; refusing duplicate dispatch')
    source = Path(__file__).resolve().parents[2]
    rows = [dict(method=method, seed=66, status='pending',
                 category='ablation' if method.startswith('nested_ab_') else 'sensitivity')
            for method in NESTED_SWEEP]
    data_root = read(args.data_manifest)['dataset_root']
    jobs = {}
    while True:
        for method, (child, stream, row) in list(jobs.items()):
            if child.poll() is None:
                continue
            stream.close()
            path = Path(row['output']) / 'PROVENANCE.json'
            prov = read(path) if path.exists() else {}
            row.update(status='complete' if child.returncode == 0 and prov.get('outputs_verified') else 'failed',
                       exit_code=child.returncode, error=prov.get('error', 'See train.log' if child.returncode else None),
                       finished_utc=now())
            del jobs[method]
        row = next((r for r in rows if r['status'] == 'pending'), None)
        if row and shutil.disk_usage(root).free / 2**30 >= 16 + 10 * (len(jobs) + 1):
            gpu = next((g for g in GPUS if capacity(g, jobs)), None)
            if gpu is not None:
                output = root / 'runs' / row['method'] / 'seed_66'
                output.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, '-u', '-m', 'experiments.osram_core20_20261005.run',
                    '--method', row['method'], '--seed', '66', '--reference', str(REFERENCE / 'config.json'),
                    '--data-manifest', str(args.data_manifest), '--output', str(output), '--gpu-uuid', GPUS[gpu]]
                log = output / 'train.log'
                stream = log.open('x')
                child = subprocess.Popen(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[gpu], GCNET_DATASET_ROOT=data_root,
                             OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                row.update(status='running', pid=child.pid, output=str(output), log=str(log), command=command,
                           gpu=gpu, gpu_uuid=GPUS[gpu], started_utc=now())
                jobs[row['method']] = child, stream, row
                print(row['method'], gpu, child.pid, flush=True)
        write(root / 'DISPATCH.json', dict(pid=os.getpid(), label=LABEL, source=str(source), runs=rows))
        summarize(root, rows)
        if not jobs and all(r['status'] in ('complete', 'failed') for r in rows):
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
