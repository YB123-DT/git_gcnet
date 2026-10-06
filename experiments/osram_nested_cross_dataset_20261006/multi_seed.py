"""Finite 27-run extension: Four/Six/MOSEI three seeds on healthy GPU3/6."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .run import (ROOT, LABEL, additional_tasks, configuration, gpu_info, output_dir,
                  read, reference_dir, sha, train, write, now)

INPUTS = ROOT / 'INPUTS_3SEED.json'
STATUS = ROOT / 'QUEUE_3SEED.json'
GPUS = (3, 6)
RESERVE_MIB = 8000
MAX_PER_GPU = 3


def prepare():
    if INPUTS.exists():
        raise FileExistsError(INPUTS)
    prior = read(ROOT / 'INPUTS.json')
    data = dict(prior['data'])
    # Identical feature files; only the class-label/split pickle changes.
    six = data['IEMOCAPSix']
    files = dict(six['files'])
    old_label = str(Path(six['dataset_root']) / 'IEMOCAP/IEMOCAP_features_raw_6way.pkl')
    del files[old_label]
    label = str(Path(six['dataset_root']) / 'IEMOCAP/IEMOCAP_features_raw_4way.pkl')
    files[label] = sha(label)
    data['IEMOCAPFour'] = dict(six, files=files)
    inputs = dict(label=LABEL, tasks=additional_tasks(), data=data, references={})
    for task in inputs['tasks']:
        ref = reference_dir(task)
        configuration(read(ref / 'config.json'), task)
        if read(ref / 'PROVENANCE.json').get('status') != 'complete':
            raise ValueError('incomplete same-seed Flat reference')
        if output_dir(task).exists():
            raise FileExistsError('already started/finished: ' + str(task))
        inputs['references'][str(ref)] = {name: sha(ref / name) for name in ('config.json', 'metrics.json')}
    # Verify unchanged feature/label versions before sealing the added scope.
    unique = {p: h for desc in data.values() for p, h in desc['files'].items()}
    for p, h in unique.items():
        if sha(p) != h:
            raise ValueError('data changed: ' + p)
    write(INPUTS, inputs)


def memory_use():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(p.strip()): int(m.strip()) for line in text.splitlines()
            for p, m in [line.split(',')] if p.strip().isdigit() and m.strip().isdigit()}


def dispatch():
    lock = (ROOT / 'multiseed.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATUS.exists():
        raise FileExistsError('Reconcile prior queue before attempting any restart')
    inputs = read(INPUTS)
    if inputs['tasks'] != additional_tasks():
        raise ValueError('27-run scope changed')
    from experiments.osram_core20_20261005.run import verify_snapshot
    source = Path(__file__).resolve().parents[2]
    snapshot = verify_snapshot(source)
    rows = [dict(task=t, status='pending', output=str(output_dir(t))) for t in inputs['tasks']]
    state = dict(label=LABEL, status='running', pid=os.getpid(), started_utc=now(),
                 source=str(source), code_commit=snapshot['code_commit'], gpu_whitelist=list(GPUS),
                 max_per_gpu=MAX_PER_GPU, reservation_mib=RESERVE_MIB,
                 inputs_sha256=sha(INPUTS), tasks=rows)
    children = {}
    halt = False
    write(STATUS, state)
    try:
        while True:
            for index, child in list(children.items()):
                code = child.poll()
                if code is not None:
                    row = rows[index]
                    prov = Path(row['output']) / 'PROVENANCE.json'
                    verified = read(prov).get('outputs_verified', False) if prov.exists() else False
                    row.update(exit_code=code, status='complete' if code == 0 and verified else 'failed')
                    halt |= row['status'] == 'failed'
                    del children[index]
            for gpu in GPUS:
                if halt or not any(r['status'] == 'pending' for r in rows):
                    break
                uuid, free = gpu_info(gpu)
                active = [r for r in rows if r['status'] == 'running' and r.get('gpu') == gpu]
                use = memory_use()
                # Reserve full budgets for loading children, whose VRAM is not yet visible.
                free -= sum(max(0, RESERVE_MIB - use.get(r['pid'], 0)) for r in active)
                if (len(active) >= MAX_PER_GPU or free < RESERVE_MIB or
                        shutil.disk_usage(ROOT).free < 25 * 2**30):
                    continue
                index = next(i for i, r in enumerate(rows) if r['status'] == 'pending')
                row = rows[index]
                task = row['task']
                log = ROOT / f'{task["dataset"]}_seed{task["seed"]}_fold{task["fold"]}.log'
                if Path(row['output']).exists() or log.exists():
                    raise FileExistsError('refusing duplicate task: ' + str(task))
                command = [sys.executable, '-u', '-m', 'experiments.osram_nested_cross_dataset_20261006.multi_seed',
                           '--train', '--dataset', task['dataset'], '--seed', str(task['seed']),
                           '--fold', str(task['fold']), '--gpu', str(gpu)]
                with log.open('x') as stream:
                    child = subprocess.Popen(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1',
                                 MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                children[index] = child
                row.update(status='running', pid=child.pid, gpu=gpu, gpu_uuid=uuid,
                           command=command, log=str(log), started_utc=now())
                write(STATUS, state)
            state['status'] = 'failed_draining' if halt else 'running'
            write(STATUS, state)
            if not children and (halt or not any(r['status'] == 'pending' for r in rows)):
                state.update(status='failed' if halt else 'complete', finished_utc=now())
                write(STATUS, state)
                return
            time.sleep(10)
    except BaseException as error:
        state.update(status='failed_needs_reconciliation', error=repr(error))
        write(STATUS, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dataset', choices=('IEMOCAPFour', 'IEMOCAPSix', 'CMUMOSEI'))
    parser.add_argument('--seed', type=int, choices=(66, 67, 68))
    parser.add_argument('--fold', type=int)
    parser.add_argument('--gpu', type=int, choices=GPUS)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.train:
        task = dict(dataset=args.dataset, seed=args.seed, fold=args.fold)
        if task not in additional_tasks() or args.gpu is None:
            parser.error('only one of the 27 authorized tasks is allowed')
        uuid, _ = gpu_info(args.gpu)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
            raise ValueError('visible GPU UUID mismatch')
        train(task, args.gpu, uuid, inputs_path=INPUTS)
    else:
        dispatch()


if __name__ == '__main__':
    main()
