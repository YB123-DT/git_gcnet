"""Finite IEMOCAP6 seed66/fivefold queue sharing GPU slots with MOSEI."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .scheduler import admissible_gpus, inventory, write_json


def live_pid(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False


def foreign_jobs(path, uuid):
    if not path.exists():
        raise FileNotFoundError('MOSEI queue identity must exist before sharing its card')
    rows = json.loads(path.read_text())
    return sum(r.get('status') == 'running' and r.get('uuid') == uuid
               and live_pid(r.get('pid')) for r in rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mosei-output', type=Path, required=True)
    parser.add_argument('--gpus', default='7')
    parser.add_argument('--per-gpu', type=int, default=4)
    parser.add_argument('--job-mib', type=int, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    allowed = [int(v) for v in args.gpus.split(',')]
    if 4 in allowed or not allowed or len(set(allowed)) != len(allowed):
        raise ValueError('Unique healthy GPUs only; physicalGPU4 prohibited')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'SCHEDULER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from .bundle import verify
    source = verify(Path(__file__).resolve().parents[2])
    proof = json.loads((args.output/'PREFLIGHT.json').read_text())
    if source['source_commit'] != args.source_commit or proof.get('status') != 'passed':
        raise ValueError('Immutable source and real-IEMOCAP preflight must pass first')
    if proof['source_commit'] != args.source_commit or args.job_mib < proof['job_memory_budget_mib']:
        raise ValueError('Preflight source or measured memory reservation differs')
    # Fold5 first provides a quick result comparable with earlier fold5 screens.
    jobs = [(fold, variant) for fold in (5, 1, 2, 3, 4) for variant in ('A0','A1','A2','A3')]
    ledger = args.output/'QUEUE.json'
    history = json.loads(ledger.read_text()) if ledger.exists() else {}
    for row in history.values():
        if row['status'] not in ('complete','failed'):
            raise RuntimeError('Inspect unfinished jobs instead of duplicating them')
    pending = [(f,v) for f,v in jobs if f'fold{f}_{v}' not in history]
    running = {}
    while pending or running:
        for key, item in list(running.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['log'].close()
            path = Path(history[key]['output'])/'SUMMARY.json'
            summary = json.loads(path.read_text()) if path.exists() else {}
            history[key].update(status='complete' if code == 0 and summary.get('status') == 'complete'
                                else 'failed', returncode=code, finished_at=time.time())
            del running[key]
        if pending:
            cards = admissible_gpus(inventory(allowed), job_mib=args.job_mib,
                                    reserve_mib=4096, per_gpu=args.per_gpu)
            for card in cards:
                own = sum(x['uuid'] == card['uuid'] for x in running.values())
                foreign = foreign_jobs(args.mosei_output/'QUEUE.json', card['uuid'])
                slots = min(card['slots'], max(0,args.per_gpu-own-foreign))
                for _ in range(slots):
                    if not pending:
                        break
                    fold, variant = pending.pop(0)
                    key = f'fold{fold}_{variant}'
                    output = args.output/'runs'/f'fold_{fold}'/variant
                    if output.exists():
                        raise FileExistsError(output)
                    log_path = args.output/f'{key}.log'
                    log = log_path.open('x')
                    command = [sys.executable,'-m','experiments.cam_unified_20261030.iemocap',
                        '--variant',variant,'--fold',str(fold),'--output',str(output),
                        '--gpu-uuid',card['uuid'],'--source-commit',args.source_commit]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid'],
                               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                    running[key] = dict(process=process,log=log,uuid=card['uuid'])
                    history[key] = dict(status='running',pid=process.pid,uuid=card['uuid'],
                        host_gpu=card['index'],fold=fold,variant=variant,output=str(output),
                        command=command,started_at=time.time(),log=str(log_path))
                    print(f'Started {key}: PID{process.pid} GPU{card["index"]}',flush=True)
        write_json(ledger, history)
        if pending or running:
            time.sleep(15)
    subprocess.run([sys.executable,'-m','experiments.cam_unified_20261030.iemocap_summarize',
                    '--output',str(args.output)],check=True)
    if any(row['status'] != 'complete' for row in history.values()):
        raise SystemExit('Failed IEMOCAP arms retained; no automatic retry')


if __name__ == '__main__':
    main()
