"""Paired cfg84 flat reference versus local-conditioned evidence gate with L1 regularization; internal Test-oracle only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_cfg84_history_query_random_20260928 import run as common

SEEDS = common.SEEDS
LABEL = common.LABEL
read, write, sha = common.read, common.write, common.sha
feature_roots, verify_outputs = common.feature_roots, common.verify_outputs


def configuration_dict(seed, reference_root):
    # Reuse the existing strict original-cfg84 validator, not its query intervention.
    common.configuration_dict(seed, reference_root)
    config = read(reference_root / f'seed_{seed}/config.json')
    if any(config.get(flag, False) for flag in ('osram_local_evidence_gate', 'osram_history_input_gate', 'osram_post_grn')):
        raise ValueError('reference already enables local-conditioned evidence gate')
    return dict(config, osram_local_evidence_gate=True, osram_local_evidence_gate_reg_weight=.001, osram_local_evidence_gate_reg_type='l1')


def validate_args(args):
    if args.gpus != ['0'] or not 1 <= args.max_tasks_per_gpu <= 3:
        raise ValueError('GPU 0 only, at most three concurrent jobs')
    if args.smoke and 'smoke' not in args.output_root.name:
        raise ValueError('smoke must use a distinct output directory named with smoke')
    if not args.smoke and 'smoke' in args.output_root.name:
        raise ValueError('full run cannot use a smoke directory')


def preflight(args):
    validate_args(args)
    records = []
    for seed in SEEDS:
        source = args.reference_root / f'seed_{seed}'
        config = configuration_dict(seed, args.reference_root)
        if read(source / 'PROVENANCE.json').get('status') != 'complete':
            raise ValueError(f'incomplete flat reference: {source}')
        if read(source / 'metrics.json').get('selection_protocol') != 'per-rate-test-oracle':
            raise ValueError('unexpected reference selection protocol')
        if args.smoke:
            config['epochs'] = 1
        records.append(dict(seed=seed, config=config, smoke_only=args.smoke,
            semantic_delta={'osram_local_evidence_gate': True, 'osram_local_evidence_gate_reg_weight': .001, 'osram_local_evidence_gate_reg_type': 'l1'},
            reference_config_sha256=sha(source / 'config.json'),
            reference_metrics_sha256=sha(source / 'metrics.json')))
    for root in feature_roots(args):
        if not Path(root).is_dir():
            raise FileNotFoundError(root)
    return records


def assert_fresh(root):
    common.assert_fresh(root)
    for name in ('status.json', 'PREFLIGHT.json', 'SUMMARY.json'):
        if (root / name).exists():
            raise FileExistsError(f'refusing overwrite: {root / name}')


def rate_scores(metrics):
    values = {f'{i / 10:.1f}': float(metrics['test'][f'{i / 10:.1f}']['weighted_f1']) * 100
              for i in range(8)}
    return dict(per_rate=values, mean_8rate=statistics.mean(values.values()),
                high_missing=statistics.mean(values[f'{i / 10:.1f}'] for i in (5, 6, 7)))


def summarize(args):
    rows = []
    for seed in SEEDS:
        output = args.output_root / f'seed_{seed}'
        if read(output / 'PROVENANCE.json')['status'] != 'complete':
            raise ValueError('summary requires all three completed seeds')
        flat = rate_scores(read(args.reference_root / f'seed_{seed}/metrics.json'))
        shift = rate_scores(read(output / 'metrics.json'))
        rows.append(dict(seed=seed, flat=flat, local_evidence_gate=shift,
                         delta_pp={key: shift[key] - flat[key]
                                   for key in ('mean_8rate', 'high_missing')}))
    aggregate = {}
    for variant in ('flat', 'local_evidence_gate', 'delta_pp'):
        aggregate[variant] = {key: dict(mean=statistics.mean(row[variant][key] for row in rows),
            sample_std=statistics.stdev(row[variant][key] for row in rows))
            for key in ('mean_8rate', 'high_missing')}
    write(args.output_root / 'SUMMARY.json', dict(label=LABEL, smoke_only=args.smoke,
          seeds=rows, aggregate=aggregate, reference_retrained=False,
          diagnostics='local_evidence_gate: per-evidence means/saturation and regularization in metrics/history'))


def train(args):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    record = next(row for row in preflight(args) if row['seed'] == args.seed)
    output = args.output_root / f'seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(record, label=LABEL, status='training', from_scratch=True,
        started_utc=datetime.now(timezone.utc).isoformat(),
        gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256={name: sha(REPO / name) for name in (
            'gcnet_missing_m3/train_gcnet.py', 'gcnet_missing_m3/model.py',
            'gcnet_missing_m3/osram.py',
            'experiments/osram_evidence_gate_l1_20260930/run.py',
            'experiments/osram_cfg84_history_query_random_20260928/run.py')})
    write(output / 'PROVENANCE.json', provenance)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('training requires CUDA_VISIBLE_DEVICES=0')
        torch.set_num_threads(2)
        run_experiment(TrainConfig(**record['config']), *feature_roots(args), output_dir=str(output))
        verify_outputs(output, args.reference_root / f'seed_{args.seed}')
    except BaseException as error:
        write(output / 'PROVENANCE.json', dict(provenance, status='failed', error=repr(error)))
        raise
    write(output / 'PROVENANCE.json', dict(provenance, status='complete',
        completed_utc=datetime.now(timezone.utc).isoformat(), mask_validation='canonical_row_multiset'))


def child_command(args, mode, *values):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), mode, *values]
    for key in ('reference_root', 'output_root', 'dataset_root'):
        command += ['--' + key.replace('_', '-'), str(getattr(args, key).resolve())]
    command += ['--gpus', '0', '--max-tasks-per-gpu', str(args.max_tasks_per_gpu)]
    return command + (['--smoke'] if args.smoke else [])


def coordinate(args):
    write(args.output_root / 'PREFLIGHT.json', preflight(args))
    pending, active, records = list(SEEDS), {}, []
    while pending or active:
        for pid, (child, row) in list(active.items()):
            code = child.poll()
            if code is not None:
                row.update(returncode=code, status='complete' if code == 0 else 'failed')
                del active[pid]
                write(args.output_root / 'children.json', records)
        while pending and len(active) < args.max_tasks_per_gpu:
            gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.free',
                                           '--format=csv,noheader,nounits'], text=True).strip().split(',')
            if gpu[0].strip() != 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45':
                raise RuntimeError('GPU0 UUID mismatch')
            if int(gpu[1]) < 3000:
                time.sleep(10)
                break
            seed = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       GCNET_DATASET_ROOT=str(args.dataset_root), PYTHONPATH=str(REPO))
            with (args.output_root / f'seed_{seed}.log').open('x') as log:
                child = subprocess.Popen(child_command(args, '--seed', str(seed)), cwd=REPO,
                                         env=env, stdout=log, stderr=subprocess.STDOUT)
            row = dict(seed=seed, gpu='0', pid=child.pid, status='running')
            records.append(row)
            active[child.pid] = child, row
            write(args.output_root / 'children.json', records)
            time.sleep(20)  # Observe allocation before admitting another seed.
        if active:
            time.sleep(2)
    if any(row['returncode'] != 0 for row in records):
        raise RuntimeError(f'failed children: {records}')
    summarize(args)


def launch(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_fresh(args.output_root)
        preflight(args)
        env = dict(os.environ, EVIDENCE_GATE_L1_LOCK_FD=str(lock.fileno()))
        with (args.output_root / 'coordinator.log').open('x') as log:
            child = subprocess.Popen(child_command(args, '--coordinate'), cwd=REPO, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write(args.output_root / 'launch.json', dict(pid=child.pid, label=LABEL, smoke_only=args.smoke,
            gpus=['0'], max_tasks_per_gpu=args.max_tasks_per_gpu,
            started_utc=datetime.now(timezone.utc).isoformat()))
        print(f'Background coordinator PID {child.pid}; logs: {args.output_root}', flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--coordinate', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--seed', type=int, choices=SEEDS)
    mode.add_argument('--summarize', action='store_true')
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--output-root', type=Path, default=Path('/data1/yb/remote_experiments/osram_evidence_gate_l1_20260930/runs'))
    result.add_argument('--reference-root', type=Path, default=Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full'))
    result.add_argument('--dataset-root', type=Path, default=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'))
    result.add_argument('--gpus', '--gpu', nargs='+', default=['0'])
    result.add_argument('--max-tasks-per-gpu', type=int, choices=range(1, 4), default=3)
    return result


def main():
    args = parser().parse_args()
    validate_args(args)
    if args.preflight:
        import json
        print(json.dumps(preflight(args), indent=2))
    elif args.launch:
        launch(args)
    elif args.summarize:
        summarize(args)
    elif args.coordinate:
        if 'EVIDENCE_GATE_L1_LOCK_FD' not in os.environ:
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
