"""Fixed three-seed evidence-centered decision readout; no hyperparameter search."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.osram_gap_increment_filter_20261003.run import (
    UUIDS, gpu_check, filter_config, REFERENCE, DATASET, common, read, write, sha,
)


def decision_config(base, seed=66):
    filter_config(base, seed)  # Validate only; never use the Filter configuration.
    expected = {'train_rate_mode': 'cyclic', 'osram_bidirectional': False,
                'osram_ablation': 'full', 'osram_emotion_ablation': 'full',
                'checkpoint_selection': 'test-oracle-per-rate',
                'emotion_loss_mode': 'sample-mean'}
    for key, value in expected.items():
        if base.get(key) != value:
            raise ValueError(f'Unexpected reference {key}: {base.get(key)}')
    if base.get('osram_decision_correction', False):
        raise ValueError('Reference must be the original Flat')
    return dict(base, osram_decision_correction=True)


def counts(config):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    result = {}
    for name, enabled in (('flat', False), ('decision', True)):
        with torch.random.fork_rng(devices=[]):
            model = _build_model(TrainConfig(**dict(config, osram_decision_correction=enabled)), (512, 1024, 1024))
        result[name] = dict(total=sum(p.numel() for p in model.parameters()),
                            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            inactive=sum(p.numel() for p in model.parameters() if not p.requires_grad))
        if enabled:
            result[name]['decision_head'] = sum(p.numel() for p in model.decision_head.parameters())
    result['trainable_delta'] = result['decision']['trainable'] - result['flat']['trainable']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gpu', choices=tuple(UUIDS), default='0')
    parser.add_argument('--seed', type=int, choices=(66, 67, 68), default=66)
    args = parser.parse_args()
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu or gpu_check(args.gpu) < 5000:
        raise RuntimeError('Wrong GPU visibility/identity or insufficient free memory')
    common.configuration_dict(args.seed, REFERENCE)  # Validation only.
    reference = REFERENCE / f'seed_{args.seed}'
    config = decision_config(read(reference / 'config.json'), args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ('gcnet_missing_m3/decision_correction.py', 'gcnet_missing_m3/osram.py',
               'gcnet_missing_m3/model.py', 'gcnet_missing_m3/train_gcnet.py',
               'experiments/osram_cfg84_fixed_modality_ablation_20260923/run.py',
               str(Path(__file__).relative_to(ROOT)))
    record = dict(status='training', label='INTERNAL DIAGNOSTIC ONLY', from_scratch=True,
        code_commit=args.commit, baseline=str(reference), config=config,
        objective='(task_local + task_local_base + task_full) / 3',
        original_flat_used=False, memory_modified=False,
        server='biggpu', gpu=args.gpu, gpu_uuid=UUIDS[args.gpu], pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda),
        reference_sha256={name: sha(reference / name) for name in ('config.json', 'metrics.json')},
        source_sha256={name: sha(ROOT / name) for name in sources})
    write(args.output / 'PROVENANCE.json', record)
    try:
        torch.set_num_threads(2)
        write(args.output / 'PARAMETERS.json', counts(config))
        roots = [str(DATASET / 'CMUMOSI/features' / name) for name in
                 ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        run_experiment(TrainConfig(**config), *roots, output_dir=str(args.output))
        common.verify_outputs(args.output, reference)
        if len(read(args.output / 'history.json')) != config['epochs']:
            raise RuntimeError('Training history does not cover the declared epochs')
        record['status'] = 'complete'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(args.output / 'PROVENANCE.json', record)


if __name__ == '__main__':
    main()
