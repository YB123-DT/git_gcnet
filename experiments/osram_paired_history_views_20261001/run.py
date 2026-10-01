"""Fixed seed66 paired-view control versus InfoNCE; internal Test-oracle only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_cfg84_history_query_random_20260928 import run as common
from experiments.osram_memory_only_adapter_20261001.run import rate_scores
read, write, sha = common.read, common.write, common.sha
VARIANTS = ('control', 'contrastive')
LABEL = common.LABEL
GPU_UUID = 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'


def configuration_dict(variant, reference_root):
    if variant not in VARIANTS:
        raise ValueError('only the two fixed authorized variants are allowed')
    common.configuration_dict(66, reference_root)
    config = read(reference_root / 'seed_66/config.json')
    if any(config.get(flag, False) for flag in ('paired_history_views', 'osram_memory_only_adapter',
        'osram_local_skip_gate', 'osram_local_evidence_gate', 'osram_history_input_gate',
        'osram_hierarchical_evidence_gate', 'osram_post_grn')):
        raise ValueError('reference must be the original Flat without adaptations')
    return dict(config, paired_history_views=True, history_drop_prob=.2,
                history_contrast_weight=0. if variant == 'control' else .1,
                history_contrast_temperature=.1)


def validate_args(args):
    if args.gpu != '6':
        raise ValueError('host GPU6 only; GPU4 prohibited')
    if args.smoke != ('smoke' in args.output_root.name):
        raise ValueError('smoke requires separate output root containing smoke')


def preflight(args):
    validate_args(args)
    source = args.reference_root / 'seed_66'
    if read(source / 'PROVENANCE.json')['status'] != 'complete':
        raise ValueError('Flat reference is not complete')
    if read(source / 'metrics.json')['selection_protocol'] != 'per-rate-test-oracle':
        raise ValueError('reference protocol mismatch')
    records = []
    for variant in VARIANTS:
        config = configuration_dict(variant, args.reference_root)
        if args.smoke:
            config['epochs'] = 1
        records.append(dict(variant=variant, seed=66, config=config, smoke_only=args.smoke,
            reference_config_sha256=sha(source / 'config.json'),
            reference_metrics_sha256=sha(source / 'metrics.json')))
    for root in common.feature_roots(args):
        if not Path(root).is_dir():
            raise FileNotFoundError(root)
    return records


def train(args):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    record = next(row for row in preflight(args) if row['variant'] == args.variant)
    output = args.output_root / args.variant
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(record, label=LABEL, status='training', from_scratch=True,
        started_utc=datetime.now(timezone.utc).isoformat(), gpu='6', gpu_uuid=GPU_UUID,
        source_sha256={name: sha(REPO / name) for name in (
            'gcnet_missing_m3/train_gcnet.py', 'gcnet_missing_m3/paired_views.py',
            'gcnet_missing_m3/model.py', 'gcnet_missing_m3/osram.py',
            'experiments/osram_paired_history_views_20261001/run.py')})
    write(output / 'PROVENANCE.json', provenance)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '6':
            raise ValueError('CUDA_VISIBLE_DEVICES must be host GPU6')
        torch.set_num_threads(2)
        run_experiment(TrainConfig(**record['config']), *common.feature_roots(args), output_dir=str(output))
        common.verify_outputs(output, args.reference_root / 'seed_66')
    except BaseException as error:
        write(output / 'PROVENANCE.json', dict(provenance, status='failed', error=repr(error)))
        raise
    write(output / 'PROVENANCE.json', dict(provenance, status='complete',
        completed_utc=datetime.now(timezone.utc).isoformat(), mask_validation='canonical_row_multiset'))


def aggregate_diagnostics(history):
    fields = ('anchor_count', 'eligible_anchor_count', 'dropped_observed_count',
              'view1_observed_count', 'valid_utterance_count', 'batches_no_negatives')
    result = {key: sum(row[key] for row in history) for key in fields}
    result['actual_drop_ratio'] = result['dropped_observed_count'] / max(1, result['view1_observed_count'])
    result['count_scope'] = 'training exposures summed over all epochs, not unique utterances'
    return result


def summarize(args):
    flat = rate_scores(read(args.reference_root / 'seed_66/metrics.json'))
    rows = {}
    for variant in VARIANTS:
        output = args.output_root / variant
        if read(output / 'PROVENANCE.json')['status'] != 'complete':
            raise ValueError('both runs must complete before summary')
        scores = rate_scores(read(output / 'metrics.json'))
        history = read(output / 'history.json')
        rows[variant] = dict(scores=scores,
            delta_vs_flat={k: scores[k]-flat[k] for k in ('mean_8rate', 'high_missing')},
            training_diagnostics=[row['train']['paired_history'] for row in history])
        rows[variant]['counts'] = aggregate_diagnostics(rows[variant]['training_diagnostics'])
    left, right = (rows[v]['training_diagnostics'] for v in VARIANTS)
    if len(left) != len(right) or any(a[k] != b[k] for a,b in zip(left,right)
            for k in ('view1_mask_sha256','view2_mask_sha256')):
        raise ValueError('paired arms do not have identical per-epoch training masks')
    write(args.output_root / 'SUMMARY.json', dict(label=LABEL, seed=66, smoke_only=args.smoke,
        flat=flat, variants=rows, reference_retrained=False, paired_mask_hashes_match=True))


def child_command(args, mode, *values):
    result = [sys.executable, '-u', str(Path(__file__).resolve()), mode, *values]
    for name in ('output_root', 'reference_root', 'dataset_root'):
        result += ['--' + name.replace('_', '-'), str(getattr(args, name).resolve())]
    return result + (['--smoke'] if args.smoke else [])


def coordinate(args):
    write(args.output_root / 'PREFLIGHT.json', preflight(args))
    children, records = [], []
    for variant in VARIANTS:
        while True:
            values = subprocess.check_output(['nvidia-smi', '-i', '6', '--query-gpu=uuid,memory.free',
                '--format=csv,noheader,nounits'], text=True).strip().split(',')
            if values[0].strip() != GPU_UUID:
                raise RuntimeError('GPU6 UUID mismatch')
            if int(values[1]) >= 6000:
                break
            time.sleep(10)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   GCNET_DATASET_ROOT=str(args.dataset_root), PYTHONPATH=str(REPO))
        with (args.output_root / f'{variant}.log').open('x') as log:
            child = subprocess.Popen(child_command(args, '--variant', variant), cwd=REPO,
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        records.append(dict(variant=variant, seed=66, gpu='6', pid=child.pid, status='running'))
        write(args.output_root / 'children.json', records)
        time.sleep(20)
    for child, record in zip(children, records):
        code = child.wait()
        record.update(returncode=code, status='complete' if code == 0 else 'failed')
        write(args.output_root / 'children.json', records)
    if any(row['returncode'] != 0 for row in records):
        raise RuntimeError('one or both runs failed; inspect separate logs')
    summarize(args)


def launch(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(p.name != '.launch.lock' for p in args.output_root.iterdir()):
            raise FileExistsError('refusing to overwrite or duplicate existing run')
        preflight(args)
        env = dict(os.environ, PAIRED_HISTORY_LOCK_FD=str(lock.fileno()))
        with (args.output_root / 'coordinator.log').open('x') as log:
            child = subprocess.Popen(child_command(args, '--coordinate'), cwd=REPO, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        write(args.output_root / 'launch.json', dict(pid=child.pid, label=LABEL, gpu='6',
            gpu_uuid=GPU_UUID, started_utc=datetime.now(timezone.utc).isoformat()))
        print(f'Coordinator PID {child.pid}; output {args.output_root}', flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    for name in ('launch', 'preflight', 'coordinate', 'summarize'):
        mode.add_argument('--'+name, action='store_true')
    mode.add_argument('--variant', choices=VARIANTS)
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--gpu', default='6')
    result.add_argument('--output-root', type=Path, default=Path('/data1/yb/remote_experiments/osram_paired_history_views_20261001/runs'))
    result.add_argument('--reference-root', type=Path, default=Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full'))
    result.add_argument('--dataset-root', type=Path, default=Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset'))
    return result


def main():
    args = parser().parse_args()
    validate_args(args)
    if args.preflight:
        print(json.dumps(preflight(args), indent=2))
    elif args.launch:
        launch(args)
    elif args.summarize:
        summarize(args)
    elif args.coordinate:
        if 'PAIRED_HISTORY_LOCK_FD' not in os.environ:
            raise RuntimeError('use --launch')
        write(args.output_root / 'status.json', dict(status='running', label=LABEL))
        try:
            coordinate(args)
        except BaseException as error:
            write(args.output_root / 'status.json', dict(status='failed', error=repr(error), label=LABEL))
            raise
        write(args.output_root / 'status.json', dict(status='complete', label=LABEL))
    else:
        train(args)


if __name__ == '__main__': main()
