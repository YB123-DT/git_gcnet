"""Old Nested, seed66: MOSEI and six-class IEMOCAP five-session screening."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006')
REMOTE = ROOT.parent
METHOD = 'nested_gnn_rooted_evidence'
LABEL = 'INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle; NOT A FORMAL PAPER RESULT'
REQUIRED = dict(epochs=100, training_objective='emotion-only', backbone_type='osram',
                train_rate_mode='cyclic', checkpoint_selection='test-oracle-per-rate',
                evaluate_test=True, osram_bidirectional=False, osram_write_step=.6,
                osram_readout_fusion='flat', osram_output_dim=1600, osram_num_heads=8,
                osram_key_dim=64, osram_value_dim=64, latent_dim=256,
                disable_unused_aux_modules=True, classification_completion=False,
                completion_path='none', fusion_type='mean')
FEATURE_NAMES = ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')


def tasks():
    return [dict(dataset=d, fold=f, seed=66) for d, folds in
            [('CMUMOSEI', [1]), ('IEMOCAPSix', range(1, 6))] for f in folds]


def additional_tasks():
    # Previously authorized seed66 Six/MOSEI must never be dispatched again.
    rows = [dict(dataset='CMUMOSEI', seed=s, fold=1) for s in (67, 68)]
    rows += [dict(dataset=d, seed=s, fold=f) for f in range(1, 6)
             for d, seeds in [('IEMOCAPFour', (66, 67, 68)), ('IEMOCAPSix', (67, 68))]
             for s in seeds]
    return rows


def output_dir(task):
    return ROOT / task['dataset'] / f'seed_{task["seed"]}/fold_{task["fold"]}'


def configuration(reference, task):
    for key, expected in dict(REQUIRED, **task).items():
        if reference.get(key) != expected:
            raise ValueError(f'reference mismatch {key}: {reference.get(key)!r}')
    from experiments.osram_meaningful20_20261003.run import FORBIDDEN
    if any(reference.get(k, False) for k in FORBIDDEN):
        raise ValueError('reference contains an extra intervention')
    if reference.get('osram_meaningful_block', 'none') not in (None, 'none'):
        raise ValueError('reference is not Flat')
    if reference.get('train_missing_rates') not in (None, [i / 10 for i in range(8)]):
        raise ValueError('cyclic rates changed')
    return dict(reference, osram_meaningful_block=METHOD)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def reference_dir(task):
    if task['dataset'] == 'CMUMOSEI':
        return REMOTE / f'osram_mosei_cfg84_nojepa_20260919/seed_{task["seed"]}'
    classes = {'IEMOCAPFour': 4, 'IEMOCAPSix': 6}[task['dataset']]
    return REMOTE / f'osram_iemocap_cfg84_nojepa_5session_20260919/iemocap{classes}/seed_{task["seed"]}/fold_{task["fold"]}'


def data_description(dataset):
    if dataset == 'CMUMOSEI':
        root = Path('/data2/yb/paper/GCNet_repro_cmumosei_10seed_20260819/dataset')
        folder, label = 'CMUMOSEI', 'CMUMOSEI_features_raw_2way.pkl'
    else:
        root = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
        folder, label = 'IEMOCAP', 'IEMOCAP_features_raw_6way.pkl'
    features = [root / folder / 'features' / name for name in FEATURE_NAMES]
    files = [root / folder / label]
    for feature in features:
        if not feature.is_dir():
            raise FileNotFoundError(feature)
        files += sorted(p for p in feature.rglob('*') if p.is_file())
    return dict(dataset_root=str(root), feature_roots=list(map(str, features)),
                files={str(p): sha(p) for p in files})


def prepare():
    manifest = ROOT / 'INPUTS.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    inputs = dict(label=LABEL, tasks=tasks(), data={}, references={})
    for dataset in ('CMUMOSEI', 'IEMOCAPSix'):
        inputs['data'][dataset] = data_description(dataset)
    for task in tasks():
        ref = reference_dir(task)
        configuration(read(ref / 'config.json'), task)
        if read(ref / 'PROVENANCE.json').get('status') != 'complete':
            raise ValueError(f'incomplete Flat reference {ref}')
        inputs['references'][str(ref)] = {name: sha(ref / name) for name in
                                        ('config.json', 'metrics.json')}
    write(manifest, inputs)


def gpu_info(index):
    if index == 4:
        raise ValueError('biggpu GPU4 forbidden')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    for line in rows.splitlines():
        number, uuid, free = [s.strip() for s in line.split(',')]
        if int(number) == index:
            return uuid, int(free)
    raise ValueError('GPU missing')


def train(task, gpu, uuid, *, inputs_path=None):
    source = Path(__file__).resolve().parents[2]
    from experiments.osram_core20_20261005.run import verify_snapshot
    snapshot = verify_snapshot(source)
    inputs_path = Path(inputs_path) if inputs_path is not None else ROOT / 'INPUTS.json'
    inputs = read(inputs_path)
    if task not in inputs['tasks']:
        raise ValueError('task not in pinned authorization manifest')
    data = inputs['data'][task['dataset']]
    ref = reference_dir(task)
    for name, digest in inputs['references'][str(ref)].items():
        if sha(ref / name) != digest:
            raise ValueError('Flat reference changed')
    for name, digest in data['files'].items():
        if sha(name) != digest:
            raise ValueError('data changed: ' + name)
    os.environ['GCNET_DATASET_ROOT'] = data['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(ROOT / ('cache_' + task['dataset']))
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    from gcnet_missing_m3.training_resume import TrainingState
    import torch
    torch.set_num_threads(1)
    cfg = TrainConfig(**configuration(read(ref / 'config.json'), task))
    output = output_dir(task)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(label=LABEL, status='running', task=task, server=socket.gethostname(),
                  gpu_index=gpu, gpu_uuid=uuid, pid=os.getpid(), started_utc=now(),
                  source=str(source), code_commit=snapshot['code_commit'],
                  snapshot_sha256=sha(source / 'SNAPSHOT.json'), inputs_sha256=sha(inputs_path),
                  reference=str(ref), reference_selection_metric=read(ref / 'metrics.json').get('selection_metric'),
                  configuration_delta={'osram_meaningful_block': METHOD}, config=asdict(cfg),
                  python=sys.version)
    write(output / 'PROVENANCE.json', record)
    try:
        identity = {k: record[k] for k in ('snapshot_sha256', 'inputs_sha256')}
        identity['config'] = hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()
        metrics = run_experiment(cfg, *data['feature_roots'], str(output),
                                 training_state=TrainingState(output, identity=identity))
        expected_metric = 'weighted_f1' if task['dataset'] == 'CMUMOSEI' else 'accuracy'
        if metrics.get('selection_metric') != expected_metric:
            raise ValueError('checkpoint selection metric mismatch')
        if metrics['mask_sha256'] != read(ref / 'metrics.json')['mask_sha256']:
            raise ValueError('ordered evaluation mask mismatch')
        from experiments.osram_mosi_hparam_sweep_20260918.run import canonical_mask_hashes
        if canonical_mask_hashes(output) != canonical_mask_hashes(ref):
            raise ValueError('canonical sample/mask mismatch')
        if len(read(output / 'history.json')) != 100:
            raise ValueError('epochs incomplete')
        artifacts = ['config.json', 'metrics.json', 'history.json', 'last_training.pt']
        artifacts += [f'{kind}_miss_0p{i}.{ext}' for i in range(8) for kind, ext in
                      [('best', 'pt'), ('predictions', 'npz')]]
        record.update(artifact_sha256={name: sha(output / name) for name in artifacts},
                      outputs_verified=True, status='complete', finished_utc=now(), exit_code=0)
        verify_snapshot(source)
        write(output / 'PROVENANCE.json', record)
    except BaseException as error:
        write(output / 'PROVENANCE.json', dict(record, status='failed', finished_utc=now(), error=repr(error)))
        raise


def queue(lane, gpu):
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / f'{lane}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = ROOT / f'QUEUE_{lane}.json'
    if status_path.exists():
        raise FileExistsError('Reconcile existing queue before restarting')
    selected = [t for t in tasks() if (t['dataset'] == 'CMUMOSEI') == (lane == 'mosei')]
    state = dict(label=LABEL, status='running', pid=os.getpid(), gpu=gpu, tasks=[], started_utc=now())
    write(status_path, state)
    try:
        for task in selected:
            uuid, free = gpu_info(gpu)
            needed = 26000 if lane == 'mosei' else 16000
            while free < needed or shutil.disk_usage(ROOT).free < 25 * 2**30:
                state.update(status='waiting_capacity')
                write(status_path, state)
                time.sleep(20)
                uuid, free = gpu_info(gpu)
            output = ROOT / task['dataset'] / f'seed_66/fold_{task["fold"]}'
            log = ROOT / f'{lane}_fold_{task["fold"]}.log'
            command = [sys.executable, '-u', '-m', 'experiments.osram_nested_cross_dataset_20261006.run',
                       '--train', '--lane', lane, '--fold', str(task['fold']), '--gpu', str(gpu)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with log.open('x') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                row = dict(task=task, pid=child.pid, gpu_uuid=uuid, command=command,
                           log=str(log), output=str(output), status='running')
                state['tasks'].append(row)
                state['status'] = 'running'
                write(status_path, state)
                code = child.wait()
            row.update(exit_code=code, status='complete' if code == 0 else 'failed')
            write(status_path, state)
            if code != 0 or read(output / 'PROVENANCE.json').get('status') != 'complete':
                raise RuntimeError(f'failed {task}; inspect {log}')
        state.update(status='complete', finished_utc=now())
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished_utc=now())
        raise
    finally:
        write(status_path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lane', choices=('mosei', 'iemocap'))
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.lane and args.gpu is not None:
        if args.train:
            task = next(t for t in tasks() if t['fold'] == args.fold and
                        (t['dataset'] == 'CMUMOSEI') == (args.lane == 'mosei'))
            uuid, _ = gpu_info(args.gpu)
            train(task, args.gpu, uuid)
        else:
            queue(args.lane, args.gpu)
    else:
        parser.error('use --prepare or --lane NAME --gpu INDEX')


if __name__ == '__main__':
    main()
