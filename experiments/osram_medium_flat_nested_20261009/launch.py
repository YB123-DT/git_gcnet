"""Ten fixed runs, five per healthy GPU; stagger admission through first epochs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.osram_core20_20261005.run import write, now

GPUS = {2:'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4',
        3:'GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a'}
DATA = '/data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/'BATCH.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'BATCH.json').exists():
        raise FileExistsError('Inspect existing batch, never duplicate launches')
    source = Path(__file__).resolve().parents[2]
    children, rows = [], []
    state = dict(status='launching',pid=os.getpid(),started_utc=now(),runs=rows)
    for width in (384,512,768,1024,1280):
        pair = []
        for gpu,suffix in ((2,''),(3,'_nested')):
            method = f'flat{width}{suffix}'
            output = root/'runs'/method
            command = [sys.executable,'-u','-m','experiments.osram_nps_local_20261009.dispatch',
                       '--method',method,'--root',str(output),'--data-manifest',DATA,
                       '--gpu-index',str(gpu),'--gpu-uuid',GPUS[gpu]]
            with (root/f'{method}_dispatch.log').open('x') as log:
                child = subprocess.Popen(command,cwd=source,stdout=log,stderr=subprocess.STDOUT)
            row = dict(method=method,gpu=gpu,dispatcher_pid=child.pid,root=str(output),status='submitted')
            rows.append(row)
            children.append((child,row))
            pair.append((child,row))
            write(root/'BATCH.json',state)
        # No next pair until both previous runs have materialized train/eval memory.
        while True:
            ready = []
            for child,row in pair:
                dispatch = Path(row['root'])/'DISPATCH.json'
                info = json.loads(dispatch.read_text()) if dispatch.exists() else {}
                if child.poll() is not None and info.get('status') != 'complete':
                    state.update(status='launch_failed',error=row['method'])
                    write(root/'BATCH.json',state)
                    raise RuntimeError('inspect dispatcher: '+row['method'])
                history = Path(row['root'])/'seed_66/history.json'
                ready.append(history.exists() and len(json.loads(history.read_text())) >= 1)
                row.update(status=info.get('status','submitted'),training_pid=info.get('pid'))
            write(root/'BATCH.json',state)
            if all(ready):
                break
            time.sleep(5)
    state['status'] = 'all_submitted'
    write(root/'BATCH.json',state)
    for child,row in children:
        child.wait()
        info = json.loads((Path(row['root'])/'DISPATCH.json').read_text())
        row.update(status=info['status'],exit_code=info.get('exit_code'))
        write(root/'BATCH.json',state)
    state.update(status='complete' if all(r['status']=='complete' for r in rows) else 'failed',finished_utc=now())
    write(root/'BATCH.json',state)


if __name__ == '__main__':
    main()
