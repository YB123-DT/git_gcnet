"""Three-seed history-support query adapter with original cyclic random-only training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
SEEDS = (66, 67, 68)
LABEL = 'INTERNAL TEST-ORACLE DIAGNOSTIC; NOT A FORMAL PAPER RESULT'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configuration_dict(seed, reference_root):
    config = read(reference_root / f'seed_{seed}/config.json')
    expected = dict(seed=seed, dataset='CMUMOSI', epochs=100, training_objective='emotion-only',
                    completion_path='none', classification_completion=False, train_rate_mode='cyclic',
                    train_missing_rates=[i / 10 for i in range(8)], checkpoint_selection='test-oracle-per-rate',
                    osram_bidirectional=False, osram_write_step=.6, osram_readout_fusion='flat',
                    osram_ablation='full', osram_emotion_ablation='full', osram_gap_read='residual',
                    osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'reference {key}={config.get(key)!r}, expected {value!r}')
    if config.get('completion_write_to_memory', False):
        raise ValueError('completion write must be disabled')
    if config.get('osram_history_query_adapter', False):
        raise ValueError('reference already has history query adaptation')
    return dict(config, osram_history_query_adapter=True)


def feature_roots(args):
    return tuple(str(args.dataset_root / 'CMUMOSI/features' / name) for name in
                 ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT'))


def preflight(args):
    records = []
    for seed in SEEDS:
        source = args.reference_root / f'seed_{seed}'
        config = configuration_dict(seed, args.reference_root)
        if read(source / 'PROVENANCE.json').get('status') != 'complete':
            raise ValueError(f'incomplete reference seed {seed}')
        if read(source / 'metrics.json').get('selection_protocol') != 'per-rate-test-oracle':
            raise ValueError('unexpected reference selection protocol')
        records.append(dict(seed=seed, config=config, reference_config_sha256=sha(source / 'config.json'),
                            reference_metrics_sha256=sha(source / 'metrics.json'),
                            semantic_delta={'osram_history_query_adapter': True}))
    for root in feature_roots(args):
        if not Path(root).is_dir():
            raise FileNotFoundError(root)
    return records


def assert_fresh(root):
    for path in [root / 'launch.json', root / 'coordinator.log', root / 'children.json', root / 'persistent_gap',
                 *(root / f'seed_{seed}' for seed in SEEDS), *(root / f'seed_{seed}.log' for seed in SEEDS)]:
        if path.exists():
            raise FileExistsError(f'refusing overwrite: {path}')


def verify_outputs(output, reference):
    from experiments.osram_mosi_hparam_sweep_20260918.run import canonical_mask_hashes
    if read(output / 'metrics.json').get('selection_protocol') != 'per-rate-test-oracle':
        raise ValueError('unexpected selection protocol')
    if canonical_mask_hashes(output) != canonical_mask_hashes(reference):
        raise ValueError('canonical evaluation mask mismatch')
    for index in range(8):
        if not (output / f'best_miss_0p{index}.pt').is_file():
            raise FileNotFoundError(f'missing checkpoint rate {index / 10}')


def train(args):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    from experiments.osram_cfg84_conversation_mix_20260927.evaluate import evaluate_seed
    record = next(row for row in preflight(args) if row['seed'] == args.seed)
    output = args.output_root / f'seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(record, label=LABEL, status='training', from_scratch=True,
                      started_utc=datetime.now(timezone.utc).isoformat(),
                      gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      source_sha256={name: sha(REPO / name) for name in (
                          'gcnet_missing_m3/train_gcnet.py', 'gcnet_missing_m3/model.py',
                          'gcnet_missing_m3/osram.py',
                          'experiments/osram_cfg84_history_query_random_20260928/run.py',
                          'experiments/osram_cfg84_fixed_modality_ablation_20260923/run.py',
                          'experiments/osram_cfg84_conversation_mix_20260927/evaluate.py')})
    write(output / 'PROVENANCE.json', provenance)
    try:
        torch.set_num_threads(2)
        config = TrainConfig(**record['config'])
        run_experiment(config, *feature_roots(args), output_dir=str(output))
        verify_outputs(output, args.reference_root / f'seed_{args.seed}')
        evaluate_seed(args.seed, output, args.output_root / 'persistent_gap', feature_roots(args),
                      torch.device('cuda:0' if config.device == 'cuda' else 'cpu'),
                      expected_train_rate_mode='cyclic')
    except BaseException as error:
        write(output / 'PROVENANCE.json', dict(provenance, status='failed', error=repr(error)))
        raise
    write(output / 'PROVENANCE.json', dict(provenance, status='complete',
          completed_utc=datetime.now(timezone.utc).isoformat(), mask_validation='canonical_row_multiset'))


def child_command(args, mode, *values):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), mode, *values]
    for key in ('reference_root', 'output_root', 'dataset_root'):
        command += ['--' + key.replace('_', '-'), str(getattr(args, key).resolve())]
    return command + ['--gpus', *args.gpus, '--max-tasks-per-gpu', str(args.max_tasks_per_gpu)]


def coordinate(args):
    write(args.output_root / 'PREFLIGHT.json', preflight(args))
    pending = list(SEEDS)
    active, records = {}, []
    slots = [gpu for _ in range(args.max_tasks_per_gpu) for gpu in args.gpus]
    while pending or active:
        for pid, (child, row) in list(active.items()):
            code = child.poll()
            if code is not None:
                row.update(returncode=code, status='complete' if code == 0 else 'failed')
                slots.append(row['gpu'])
                del active[pid]
                write(args.output_root / 'children.json', records)
        # Failures are recorded, never retried; the other predeclared seeds still run.
        while pending and slots:
            seed, gpu = pending.pop(0), slots.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       GCNET_DATASET_ROOT=str(args.dataset_root), PYTHONPATH=str(REPO))
            with (args.output_root / f'seed_{seed}.log').open('x') as log:
                child = subprocess.Popen(child_command(args, '--seed', str(seed)), cwd=REPO,
                                         env=env, stdout=log, stderr=subprocess.STDOUT)
            row = dict(seed=seed, gpu=gpu, pid=child.pid, status='running')
            records.append(row)
            active[child.pid] = child, row
            write(args.output_root / 'children.json', records)
        if active:
            time.sleep(2)
    if any(row['returncode'] != 0 for row in records):
        raise RuntimeError(f'failed children: {records}')


def launch(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_fresh(args.output_root)
        preflight(args)
        env = dict(os.environ, HISTORY_QUERY_RANDOM_LOCK_FD=str(lock.fileno()))
        with (args.output_root / 'coordinator.log').open('x') as log:
            child = subprocess.Popen(child_command(args, '--coordinate'), cwd=REPO, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(lock.fileno(),))
        write(args.output_root / 'launch.json', dict(pid=child.pid, label=LABEL, gpus=args.gpus,
              max_tasks_per_gpu=args.max_tasks_per_gpu, started_utc=datetime.now(timezone.utc).isoformat()))
        print(f'Background coordinator PID {child.pid}; logs: {args.output_root}', flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--coordinate', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--seed', type=int, choices=SEEDS)
    result.add_argument('--output-root', type=Path, default=Path('/data2/yb/remote_experiments/osram_cfg84_history_query_random_20260928/runs'))
    result.add_argument('--reference-root', type=Path, default=Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full'))
    result.add_argument('--dataset-root', type=Path, default=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'))
    result.add_argument('--gpus', '--gpu', nargs='+', default=['0'])
    result.add_argument('--max-tasks-per-gpu', type=int, choices=range(1, 5), default=4)
    return result


def main():
    args = parser().parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError('duplicate GPUs violate capacity limits')
    if args.launch:
        launch(args)
    elif args.preflight:
        print(json.dumps(preflight(args), indent=2))
    elif args.coordinate:
        if 'HISTORY_QUERY_RANDOM_LOCK_FD' not in os.environ:
            raise RuntimeError('internal coordinator: use --launch')
        write(args.output_root / 'status.json', dict(status='running', label=LABEL))
        try:
            coordinate(args)
        except BaseException as error:
            write(args.output_root / 'status.json', dict(status='failed', label=LABEL, error=repr(error)))
            raise
        write(args.output_root / 'status.json', dict(status='complete', label=LABEL))
    else:
        train(args)


if __name__ == '__main__':
    main()
