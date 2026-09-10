"""Approved 15-task queue: three readout variants per seed/GPU."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from run import ROOT, SEEDS, VARIANTS, write_json


def main():
    if sys.argv[1:] != ['--launch']:
        raise SystemExit('Explicit --launch required; GPUs 2,3,5,6,7, three jobs each.')
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'QUEUE.json'
    state = dict(status='starting', started_utc=datetime.now(timezone.utc).isoformat(),
                 reporting_protocol='per-seed-per-rate-test-oracle', tasks=[])
    with manifest.open('x') as f:
        json.dump(state, f)
    repo = Path(__file__).resolve().parents[2]
    children = []
    try:
        for seed, gpu in zip(SEEDS, (2,3,5,6,7)):
            for variant in VARIANTS:
                path = ROOT / f'{variant}_seed{seed}.log'
                with path.open('x') as log:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', PYTHONPATH=str(repo))
                    child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('run.py')),
                                              '--variant', variant, '--seed', str(seed)],
                                             cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                row = dict(seed=seed, variant=variant, gpu=gpu, pid=child.pid,
                           log=str(path), status='running')
                state['tasks'].append(row)
                children.append((child, row))
                write_json(manifest, state)
                print(f'START {variant} seed={seed} GPU={gpu} PID={child.pid}', flush=True)
        state['status']='running'
        write_json(manifest,state)
        for child,row in children:
            row['exit_code']=child.wait()
            row['status']='complete' if row['exit_code']==0 else 'failed'
            write_json(manifest,state)
        state['status']='complete' if all(r['exit_code']==0 for _,r in children) else 'failed'
        state['completed_utc']=datetime.now(timezone.utc).isoformat()
        write_json(manifest,state)
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py'))],cwd=repo)
        state['summary_exit_code']=result.returncode
        write_json(manifest,state)
        if state['status']!='complete' or result.returncode:
            raise SystemExit(1)
    except Exception as error:
        # Already-running children are left intact; inspect their logs, never duplicate them.
        state.update(status='launcher-error', error=f'{type(error).__name__}: {error}')
        write_json(manifest,state)
        raise


if __name__=='__main__':
    main()
