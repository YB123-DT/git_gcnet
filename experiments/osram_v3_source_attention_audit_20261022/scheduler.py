"""Four bounded frozen-replay jobs; no optimizer, no training, no process signals."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .run import DATASETS, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from experiments.osram_source_relation_attention_20261021.run import gpu_free
    from experiments.osram_text_path_router_3seed_20261019.resource_gate import available_gib
    if args.output.exists():
        raise FileExistsError('New independent audit queue required')
    if gpu_free(args.gpu_uuid)[1] < 10000 or available_gib() < 32:
        raise RuntimeError('Need10GB free GPU and32GiB RAM for at most2 frozen replays')
    args.output.mkdir(parents=True)
    jobs = [dict(dataset=d, status='pending') for d in DATASETS]
    state = dict(status='running', jobs=jobs, source_commit=args.source_commit,
                 gpu_uuid=args.gpu_uuid, max_parallel=2, train_steps=0)
    active = {}
    while any(j['status'] in ('pending', 'running') for j in jobs):
        for j in jobs:
            if j['status'] != 'running':
                continue
            code = active[j['dataset']].poll()
            if code is not None:
                path = args.output/j['dataset']/'STATUS.json'
                s = json.loads(path.read_text()) if path.exists() else {}
                j.update(returncode=code, status='complete' if code == 0 and s.get('status') == 'complete' else 'failed')
        live = sum(j['status'] == 'running' for j in jobs)
        for j in jobs:
            if live >= 2 or j['status'] != 'pending':
                continue
            if gpu_free(args.gpu_uuid)[1] < 6000 or available_gib() < 32:
                break
            executable = ('/data2/yb/reproduction_workspace/envs/s0/bin/python'
                          if j['dataset'].startswith('CMU') else '/data2/yb/conda_envs/cmerc-repro/bin/python')
            cmd = [executable, '-u', '-m', 'experiments.osram_v3_source_attention_audit_20261022.run',
                   '--dataset', j['dataset'], '--gpu-uuid', args.gpu_uuid,
                   '--source-commit', args.source_commit, '--output', str(args.output/j['dataset'])]
            with (args.output/(j['dataset']+'.log')).open('w') as f:
                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            active[j['dataset']] = process
            j.update(status='running', pid=process.pid, command=cmd)
            live += 1
        write(args.output/'QUEUE.json', state)
        if any(j['status'] in ('pending', 'running') for j in jobs):
            time.sleep(5)
    state['status'] = 'complete' if all(j['status'] == 'complete' for j in jobs) else 'failed'
    write(args.output/'QUEUE.json', state)
    from .report import summarize_all
    summarize_all(args.output)


if __name__ == '__main__':
    main()
