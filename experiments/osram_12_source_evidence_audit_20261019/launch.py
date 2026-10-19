"""Two finite frozen audits, no training queue or automatic scope expansion."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/data2/yb/conda_envs/cmerc-repro/bin/python'
BAD_GPU = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'


def build_jobs(output, source_commit, gpu_uuid):
    return [dict(dataset=dataset, output=str(Path(output)/dataset),
        command=[PYTHON, '-u', '-m', 'experiments.osram_12_source_evidence_audit_20261019.run',
                 '--dataset', dataset, '--gpu-uuid', gpu_uuid,
                 '--output', str(Path(output)/dataset), '--source-commit', source_commit])
            for dataset in ('MELD', 'IEMOCAPSix')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    if not args.gpu_uuid.startswith('GPU-') or args.gpu_uuid == BAD_GPU:
        raise ValueError('Require verified healthy GPU UUID, never physical GPU4')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'launcher.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(args.output).free < 4*2**30:
        raise RuntimeError('Need4GiB free; do not remove existing artifacts')
    line = subprocess.check_output(['nvidia-smi', '--id='+args.gpu_uuid,
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
    index, uuid, free = [v.strip() for v in line.split(',')]
    if index == '4' or uuid != args.gpu_uuid or int(free) < 4096:
        raise RuntimeError('GPU identity/free-memory preflight failed: '+line)
    jobs = build_jobs(args.output, args.source_commit, args.gpu_uuid)
    plan = dict(source_commit=args.source_commit, jobs=jobs, max_processes=2,
                physical_gpu=int(index), gpu_uuid=uuid, no_training=True,
                no_expansion=True, mask_protocol='gcnet-random-mask-v1')
    plan_path = args.output/'EXPERIMENT.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Immutable audit plan differs')
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    processes = []
    for job in jobs:
        path = Path(job['output'])
        path.mkdir(exist_ok=True)
        log = (path/'audit.log').open('a')
        process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, env=dict(os.environ))
        processes.append((job, process, log))
    records = []
    while processes:
        pending = []
        for job, process, log in processes:
            code = process.poll()
            if code is None:
                pending.append((job, process, log))
                continue
            log.close()
            summary_path = Path(job['output'])/'SUMMARY.json'
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            records.append(dict(dataset=job['dataset'], pid=process.pid, returncode=code,
                                status=summary.get('status', 'missing'), error=summary.get('error')))
            (args.output/'QUEUE_FINAL.json').write_text(json.dumps(records, indent=2)+'\n')
        processes = pending
        if processes:
            time.sleep(5)
    if any(r['returncode'] != 0 or r['status'] != 'complete' for r in records):
        raise RuntimeError('Incomplete frozen audit; see each independent log. No automatic rerun.')


if __name__ == '__main__':
    main()
