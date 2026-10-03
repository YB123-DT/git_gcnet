"""Authorized cfg84 seeds 66/67/68 Gap-increment-filter runs; no protocol search."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.osram_current_history_relation_20261003.run import relation_config
from experiments.osram_paired_history_rho025_20261002.sweep import (
    REFERENCE, DATASET, common, read, write, sha,
)

UUIDS = {'0': 'GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',
         '5': 'GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62'}


def gpu_check(gpu):
    values = subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    if values[0].strip() != UUIDS[gpu]:
        raise RuntimeError('GPU UUID mismatch')
    return int(values[1])


def filter_config(base, seed=66):
    if seed not in (66, 67, 68):
        raise ValueError('Only seeds 66, 67, 68 are authorized')
    relation_config(base, seed=seed)  # Validate Flat; discard Relation configuration.
    if base.get('osram_gap_increment_filter', False) or base.get('osram_relation_dual_readout', False):
        raise ValueError('Reference must not already enable a filter or dual readout')
    return dict(base, osram_gap_increment_filter=True)


def counts(config):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    result = {}
    for enabled in (False, True):
        with torch.random.fork_rng(devices=[]):
            model = _build_model(TrainConfig(**dict(config, osram_gap_increment_filter=enabled)), (512, 1024, 1024))
        result['filter' if enabled else 'flat'] = dict(
            total=sum(p.numel() for p in model.parameters()),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    result['added'] = result['filter']['total'] - result['flat']['total']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gpu', choices=tuple(UUIDS), default='5')
    parser.add_argument('--seed', type=int, choices=(66, 67, 68), default=66)
    args = parser.parse_args()
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu or gpu_check(args.gpu) < 5000:
        raise RuntimeError('Wrong GPU visibility/identity or insufficient free memory')
    common.configuration_dict(args.seed, REFERENCE)  # Validate only; discard query-adapter return.
    reference = REFERENCE / f'seed_{args.seed}'
    config = filter_config(read(reference / 'config.json'), seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status='training', label='INTERNAL DIAGNOSTIC ONLY', from_scratch=True,
        code_commit=args.commit, baseline=str(reference), config=config,
        server='biggpu', gpu=args.gpu, gpu_uuid=UUIDS[args.gpu], pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda),
        reference_sha256={name: sha(reference / name) for name in ('config.json', 'metrics.json')},
        source_sha256={name: sha(ROOT / name) for name in (
            'gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py', 'gcnet_missing_m3/train_gcnet.py',
            'experiments/osram_cfg84_fixed_modality_ablation_20260923/run.py', str(Path(__file__).relative_to(ROOT)))})
    write(args.output / 'PROVENANCE.json', record)
    try:
        torch.set_num_threads(2)
        write(args.output / 'PARAMETERS.json', counts(config))
        roots = [str(DATASET / 'CMUMOSI/features' / name) for name in
                 ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        run_experiment(TrainConfig(**config), *roots, output_dir=str(args.output))
        common.verify_outputs(args.output, reference)
        record['status'] = 'complete'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(args.output / 'PROVENANCE.json', record)


if __name__ == '__main__':
    main()
