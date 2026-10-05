"""Six confirmation runs using the exact seed66 controllers and model snapshots."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

GPU = 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'
REMOTE = Path('/data2/yb/remote_experiments')
ORIGINALS = {
    'm28_xcit_xca': REMOTE / 'osram_priority40_20261004/group2/runs/m28_xcit_xca/seed_66',
    'nested_gnn_rooted_evidence': REMOTE / 'osram_new40_gpu0123_20261004/attempt2/runs/nested_gnn_rooted_evidence/seed_66',
    'conditional_new_07_neural_production': REMOTE / 'osram_new40_gpu0123_20261004/attempt2/runs/conditional_new_07_neural_production/seed_66',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def replace_arg(argv, name, value):
    argv[argv.index(name) + 1] = str(value)


def summarize(root, rows):
    table = []
    for method, original in ORIGINALS.items():
        results = []
        for seed in (66, 67, 68):
            output = original if seed == 66 else root / 'runs' / method / f'seed_{seed}'
            if seed != 66 and next(r for r in rows if r['method'] == method and r['seed'] == seed)['status'] != 'complete':
                continue
            m = read(output / 'metrics.json')
            values = m['selected_weighted_f1_by_rate']
            results.append(dict(seed=seed, per_rate=values,
                mean8=100 * statistics.mean(values[str(i / 10)] for i in range(8)),
                high=100 * statistics.mean(values[str(i / 10)] for i in (5, 6, 7))))
        table.append(dict(method=method, seeds=results,
            mean8=statistics.mean(r['mean8'] for r in results),
            high=statistics.mean(r['high'] for r in results),
            sample_sd=statistics.stdev(r['mean8'] for r in results) if len(results) > 1 else None,
            complete_three_seeds=len(results) == 3))
    write(root / 'SUMMARY.json', dict(label='INTERNAL TEST-ORACLE DIAGNOSTIC ONLY', methods=table, runs=rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'DISPATCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'DISPATCH.json').exists():
        raise FileExistsError('Inspect existing runs; do not resubmit blindly')
    templates = {}
    for method, original in ORIGINALS.items():
        prov = read(original / 'PROVENANCE.json')
        if prov['status'] != 'complete' or not prov['outputs_verified']:
            raise ValueError('Seed66 is not verified: ' + method)
        templates[method] = read(prov['external_controller']['record'])
        record = templates[method]
        if hashlib.sha256(Path(record['controller']).read_bytes()).hexdigest() != record['controller_sha256']:
            raise ValueError('Original controller changed')
    rows = [dict(method=m, seed=s, status='pending') for m in ORIGINALS for s in (67, 68)]
    jobs = {}
    while True:
        for key, (child, stream, row) in list(jobs.items()):
            if child.poll() is None:
                continue
            stream.close()
            path = Path(row['output']) / 'PROVENANCE.json'
            prov = read(path) if path.exists() else {}
            row.update(status='complete' if child.returncode == 0 and prov.get('outputs_verified') else 'failed',
                       exit_code=child.returncode, error=prov.get('error'))
            del jobs[key]
        raw = subprocess.check_output(['nvidia-smi', '--id=6', '--query-gpu=index,uuid,memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
        index, gpu_uuid, free = [s.strip() for s in raw.split(',')]
        if index != '6' or gpu_uuid != GPU:
            raise ValueError('GPU6 identity mismatch; GPU4 forbidden')
        allocs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                         '--format=csv,noheader,nounits'], text=True)
        allocations = {int(line.split(',')[0]): float(line.split(',')[1]) for line in allocs.splitlines()}
        growth = sum(max(0, 3500 - allocations.get(child.pid, 0)) for child, _, _ in jobs.values())
        disk_free = __import__('shutil').disk_usage(root).free / 2**30
        if float(free) >= growth + 3500 + 2048 and disk_free >= 16 + 10 * (len(jobs) + 1):
            row = next((r for r in rows if r['status'] == 'pending'), None)
            if row:
                record = copy.deepcopy(templates[row['method']])
                output = root / 'runs' / row['method'] / f"seed_{row['seed']}"
                record['output'] = str(output)
                record['seed'] = row['seed']
                record['gpu_uuid'] = GPU
                record['gpu_index'] = '6'
                for name, value in (('--seed', row['seed']), ('--output', output), ('--gpu', '6'),
                                    ('--gpu-uuid', GPU), ('--run-id', uuid.uuid4().hex)):
                    replace_arg(record['training_argv'], name, value)
                record['confirmation_seed66_run'] = str(ORIGINALS[row['method']])
                path = root / 'records' / f"{row['method']}_{row['seed']}.json"
                write(path, record)
                log = root / 'logs' / f"{row['method']}_{row['seed']}.log"
                log.parent.mkdir(exist_ok=True)
                stream = log.open('x')
                command = [sys.executable, '-u', record['controller'], '--train-only', str(path)]
                child = subprocess.Popen(command, cwd=record['snapshot'], stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                row.update(status='running', pid=child.pid, output=str(output), log=str(log), command=command,
                           gpu=6, gpu_uuid=GPU, snapshot=record['snapshot'])
                jobs[(row['method'], row['seed'])] = child, stream, row
                print(row['method'], row['seed'], child.pid, flush=True)
        write(root / 'DISPATCH.json', dict(pid=os.getpid(), seed66_reused=True, runs=rows,
            selection='locked top3 single-view one-stage seed66 mean8; six new runs only'))
        if not jobs and all(r['status'] in ('complete', 'failed') for r in rows):
            summarize(root, rows)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
