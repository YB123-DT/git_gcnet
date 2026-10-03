"""Single authorized seed66 relation run; unchanged cfg84 task/mask protocol."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_paired_history_rho025_20261002.sweep import (
    REFERENCE, DATASET, UUIDS, gpu_check, common, read, write, sha,
)


def relation_config(base):
    expected = dict(seed=66, epochs=100, osram_readout_fusion='flat',
                    training_objective='emotion-only', completion_path='none')
    for key, value in expected.items():
        if base.get(key) != value:
            raise ValueError(f'Unexpected baseline {key}: {base.get(key)}')
    for key in ('paired_history_views', 'osram_relation_block', 'osram_post_grn',
                'osram_history_query_adapter', 'classification_completion',
                'osram_local_skip_gate', 'osram_memory_only_adapter',
                'osram_history_input_gate', 'osram_local_evidence_gate',
                'osram_hierarchical_evidence_gate', 'completion_write_to_memory'):
        if base.get(key, False):
            raise ValueError(f'Baseline already enables {key}')
    return dict(base, osram_relation_block=True, osram_relation_mode='pairwise',
                osram_relation_dim=128, osram_relation_out_dim=64)


def parameter_counts(config):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    result = {}
    for name in ('flat', 'pairwise', 'control'):
        cfg = dict(config, osram_relation_block=name != 'flat',
                   osram_relation_mode='pairwise' if name == 'flat' else name)
        with torch.random.fork_rng(devices=[]):
            model = _build_model(TrainConfig(**cfg), (512, 1024, 1024))
        result[name] = dict(total=sum(p.numel() for p in model.parameters()),
                            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    for entry in result.values():
        entry['added'] = entry['total'] - result['flat']['total']
    return result


def train(args):
    import torch
    from gcnet_missing_m3.train_gcnet import TrainConfig, run_experiment
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu:
        raise ValueError('Visible device must match recorded host GPU')
    if gpu_check(args.gpu) < 4000:
        raise RuntimeError('Insufficient free memory; do not change the protocol')
    common.configuration_dict(66, REFERENCE)  # Validate only: returned query variant is NOT used.
    reference = REFERENCE / 'seed_66'
    config = relation_config(read(reference / 'config.json'))
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status='training', label='INTERNAL DIAGNOSTIC ONLY',
                  started_at=datetime.now(timezone.utc).isoformat(), from_scratch=True,
                  code_commit=args.commit, baseline_implementation_commit='7d56881f13e50b98a65a8fc9ca19f31222b39a55',
                  baseline=str(reference), config=config, server='biggpu',
                  gpu=args.gpu, gpu_uuid=UUIDS[args.gpu], pid=os.getpid(),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   cuda=torch.version.cuda),
                  reference_sha256={name: sha(reference / name) for name in ('config.json', 'metrics.json')},
                  source_sha256={name: sha(REPO / name) for name in (
                      'gcnet_missing_m3/osram.py', 'gcnet_missing_m3/model.py',
                      'gcnet_missing_m3/train_gcnet.py', str(Path(__file__).relative_to(REPO)))})
    write(args.output / 'PROVENANCE.json', record)
    try:
        torch.set_num_threads(2)
        write(args.output / 'PARAMETERS.json', parameter_counts(config))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=('5', '6'), default='5')
    parser.add_argument('--commit', required=True)
    train(parser.parse_args())
