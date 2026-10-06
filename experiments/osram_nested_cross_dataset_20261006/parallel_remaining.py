"""Transfer only undispatched IEMOCAP folds to same-GPU parallel execution.

Training continues from the original immutable source. Only the old queue
coordinator is suspended; never signal a training child or an unrelated job.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006')
SOURCE = ROOT / 'source_33ded09'
UUID = 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'


def pending_folds(queue):
    claimed = [r['task']['fold'] for r in queue['tasks']]
    if any(f not in range(1, 6) for f in claimed) or len(set(claimed)) != len(claimed):
        raise ValueError('invalid or duplicated fold records')
    return [f for f in range(1, 6) if f not in claimed]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def process_identity(pid):
    root = Path(f'/proc/{pid}')
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(start_time=fields[19], state=fields[0],
                command=(root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                cwd=str((root / 'cwd').resolve()))


def free_memory():
    output = subprocess.check_output(['nvidia-smi', '-i', UUID,
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    number, uuid, free = [s.strip() for s in output.strip().split(',')]
    if number != '6' or uuid != UUID:
        raise ValueError('not verified healthy host GPU6')
    return int(free)


def main():
    lock = (ROOT / 'parallel_handoff.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = ROOT / 'PARALLEL_IEMOCAP.json'
    if path.exists():
        raise FileExistsError('Reconcile existing handoff; never duplicate it')
    queue_path = ROOT / 'QUEUE_iemocap.json'
    old = read(queue_path)
    pid = old['pid']
    identity = process_identity(pid)
    if (identity['cwd'] != str(SOURCE) or '--lane iemocap --gpu 6' not in identity['command']
            or '--train' in identity['command'] or old['status'] != 'running'):
        raise ValueError('old coordinator identity/state mismatch')
    if not pending_folds(old):
        print('No undispatched folds; existing jobs left untouched', flush=True)
        return
    if free_memory() < 16000 or shutil.disk_usage(ROOT).free < 50 * 2**30:
        raise ValueError('insufficient capacity for two additional folds')
    record = dict(status='handoff_preparing', pid=os.getpid(), gpu=6, gpu_uuid=UUID,
                  old_coordinator_pid=pid, old_identity=identity, old_queue=old,
                  training_source=str(SOURCE), training_commit='33ded09', tasks=[],
                  coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(path, record)
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(100):
            stopped = process_identity(pid)
            if stopped['state'] == 'T':
                break
            time.sleep(.05)
        if stopped['state'] != 'T' or stopped['start_time'] != identity['start_time']:
            raise ValueError('coordinator did not stop safely')
        # Re-read after the stop, so a just-launched fold is never duplicated.
        old = read(queue_path)
        pending = pending_folds(old)
        for fold in pending:
            output = ROOT / f'IEMOCAPSix/seed_66/fold_{fold}'
            log = ROOT / f'iemocap_fold_{fold}.log'
            if output.exists() or log.exists():
                raise FileExistsError(f'fold{fold} already has artifacts')
        record.update(status='running', old_queue=old, pending_folds=pending)
        write(path, record)
    except BaseException:
        os.kill(pid, signal.SIGCONT)
        write(path, dict(record, status='handoff_failed_original_resumed'))
        raise

    children = []
    old_retired = False
    try:
        for fold in pending:
            while free_memory() < 9000:
                time.sleep(10)
            command = [sys.executable, '-u', '-m', 'experiments.osram_nested_cross_dataset_20261006.run',
                       '--train', '--lane', 'iemocap', '--fold', str(fold), '--gpu', '6']
            log = ROOT / f'iemocap_fold_{fold}.log'
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=SOURCE, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS='1',
                             MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            row = dict(task=dict(dataset='IEMOCAPSix', seed=66, fold=fold), pid=child.pid,
                       status='running', gpu_uuid=UUID, command=command, log=str(log),
                       output=str(ROOT / f'IEMOCAPSix/seed_66/fold_{fold}'))
            record['tasks'].append(row)
            children.append((child, row))
            write(path, record)
            time.sleep(5)
        while True:
            for child, row in children:
                code = child.poll()
                if code is not None:
                    verified = read(Path(row['output']) / 'PROVENANCE.json').get('outputs_verified', False)
                    row.update(exit_code=code, status='complete' if code == 0 and verified else 'failed')
            active_old = []
            for row in old['tasks']:
                if row['status'] == 'running':
                    provenance = Path(row['output']) / 'PROVENANCE.json'
                    actual = read(provenance) if provenance.exists() else {}
                    if actual.get('status') == 'complete' and actual.get('outputs_verified'):
                        row.update(status='complete', exit_code=0)
                    elif actual.get('status') == 'failed':
                        row.update(status='failed')
                    else:
                        active_old.append(row)
            # Terminate only the suspended coordinator, after its child finished.
            # Queue SIGTERM before SIGCONT to avoid dispatching another fold.
            if not active_old and not old_retired:
                current = process_identity(pid)
                if current['start_time'] != identity['start_time'] or current['state'] != 'T':
                    raise ValueError('coordinator identity changed before retirement')
                os.kill(pid, signal.SIGTERM)
                os.kill(pid, signal.SIGCONT)
                old_retired = True
                record['old_coordinator_retired'] = True
            all_rows = old['tasks'] + record['tasks']
            done = not active_old and all(r['status'] in ('complete', 'failed') for r in record['tasks'])
            record['status'] = ('failed' if any(r['status'] == 'failed' for r in all_rows) else 'complete') if done else 'running'
            write(path, record)
            write(queue_path, dict(old, tasks=all_rows, status=record['status'],
                                   pid=os.getpid(), dispatch_mode='parallel-handoff'))
            if done:
                return
            time.sleep(10)
    except BaseException as error:
        write(path, dict(record, status='failed_needs_reconciliation', error=repr(error)))
        # Do not resume the old dispatcher once any fold was transferred.
        raise


if __name__ == '__main__':
    main()
