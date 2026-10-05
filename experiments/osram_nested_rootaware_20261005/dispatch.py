"""Three from-scratch seed confirmations on healthy GPU6; no automatic retry."""
import argparse
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from experiments.osram_core20_20261005.run import now, write
from experiments.osram_readout_top3_3seed_20261005.dispatch import read, GPU, REMOTE, ORIGINALS


def summarize(root, rows):
    results = []
    for row in rows:
        if row['status'] != 'complete':
            results.append(dict(seed=row['seed'], status=row['status'], error=row.get('error')))
            continue
        seed = row['seed']
        reference = REMOTE / 'osram_mosi_memory_gap_ablation_20260920/full' / f'seed_{seed}'
        old = ORIGINALS['nested_gnn_rooted_evidence'] if seed == 66 else (
            REMOTE / 'osram_readout_top3_3seed_20261005/runs/nested_gnn_rooted_evidence' / f'seed_{seed}')
        result = dict(seed=seed, status='complete')
        for name, output in (('flat', reference), ('old_nested', old), ('rootaware', Path(row['output']))):
            try:
                scores = read(output / 'metrics.json')['selected_weighted_f1_by_rate']
                result[name] = dict(status='available', per_rate=scores,
                    mean8=100 * statistics.mean(scores[str(i / 10)] for i in range(8)),
                    high=100 * statistics.mean(scores[str(i / 10)] for i in (5, 6, 7)))
            except (OSError, ValueError, KeyError, TypeError) as error:
                result[name] = dict(status='unavailable', error=repr(error), output=str(output))
        results.append(result)
    means = {}
    if all(r['status'] == 'complete' for r in results):
        for name in ('flat', 'old_nested', 'rootaware'):
            if all(r[name]['status'] == 'available' for r in results):
                means[name] = dict(mean8=statistics.mean(r[name]['mean8'] for r in results),
                    high=statistics.mean(r[name]['high'] for r in results),
                    sample_sd=statistics.stdev(r[name]['mean8'] for r in results))
    write(root / 'SUMMARY.json', dict(label='INTERNAL DIAGNOSTIC ONLY; Test-oracle', seeds=results, means=means))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-manifest', type=Path, required=True)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / 'DISPATCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.root / 'DISPATCH.json').exists():
        raise FileExistsError('Inspect existing run status before restart')
    source = Path(__file__).resolve().parents[2]
    rows = [dict(seed=seed, status='pending') for seed in (66, 67, 68)]
    jobs = {}
    while True:
        for seed, (child, stream, row) in list(jobs.items()):
            if child.poll() is None:
                continue
            stream.close()
            path = Path(row['output']) / 'PROVENANCE.json'
            prov = read(path) if path.exists() else {}
            row.update(status='complete' if child.returncode == 0 and prov.get('outputs_verified') else 'failed',
                       exit_code=child.returncode, error=prov.get('error'), finished_utc=now())
            del jobs[seed]
        raw = subprocess.check_output(['nvidia-smi', '--id=6', '--query-gpu=index,uuid,memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
        index, gpu_uuid, free = [v.strip() for v in raw.split(',')]
        if index != '6' or gpu_uuid != GPU:
            raise ValueError('GPU6 identity mismatch; GPU4 forbidden')
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                       '--format=csv,noheader,nounits'], text=True)
        used = {int(line.split(',')[0]): float(line.split(',')[1]) for line in raw.splitlines()}
        growth = sum(max(0, 3500 - used.get(child.pid, 0)) for child, _, _ in jobs.values())
        disk_free = __import__('shutil').disk_usage(args.root).free / 2**30
        if float(free) >= growth + 3500 + 2048 and disk_free >= 16 + 10 * (len(jobs) + 1):
            row = next((r for r in rows if r['status'] == 'pending'), None)
            if row:
                seed = row['seed']
                output = args.root / f'seed_{seed}'
                output.mkdir(exist_ok=False)
                reference = REMOTE / 'osram_mosi_memory_gap_ablation_20260920/full' / f'seed_{seed}/config.json'
                command = [sys.executable, '-u', '-m', 'experiments.osram_nested_rootaware_20261005.run',
                           '--seed', str(seed), '--reference', str(reference), '--data-manifest', str(args.data_manifest),
                           '--output', str(output), '--gpu-uuid', GPU]
                log = output / 'train.log'
                stream = log.open('x')
                data_root = read(args.data_manifest)['dataset_root']
                child = subprocess.Popen(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, GCNET_DATASET_ROOT=data_root,
                             OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                row.update(status='running', pid=child.pid, command=command, output=str(output), log=str(log),
                           started_utc=now(), gpu=6, gpu_uuid=GPU)
                jobs[seed] = child, stream, row
                print(seed, child.pid, flush=True)
        write(args.root / 'DISPATCH.json', dict(pid=os.getpid(), runs=rows,
              label='INTERNAL DIAGNOSTIC ONLY', source=str(source), seed66_reused=False))
        if not jobs and all(r['status'] in ('complete', 'failed') for r in rows):
            summarize(args.root, rows)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
