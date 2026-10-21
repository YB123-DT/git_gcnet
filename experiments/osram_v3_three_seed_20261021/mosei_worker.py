"""Supplement MOSEI V3 seeds67/68 with the matched 60-epoch 1e-3 Flat protocol."""
import argparse
from dataclasses import replace
from pathlib import Path

from experiments.osram_source_role_v3_20261021 import worker as v3
from experiments.osram_mosei_nested_gcnet_3seed_20261021 import worker as matched

FLAT_ROOT = Path('/data2/yb/remote_experiments/osram_mosei_nested_gcnet_3seed_20261021'
                 '/execution_v1/runs/CMUMOSEI')


def flat_reference(seed):
    if seed not in (67, 68):
        raise ValueError('Supplement only MOSEI V3 seeds67/68; retain running seed66')
    return FLAT_ROOT / f'seed_{seed}' / 'flat'


def candidate_config(cfg, seed):
    flat_reference(seed)
    if cfg.osram_source_role_v3:
        raise ValueError('Require the original Flat before enabling V3')
    return replace(matched.candidate_config(cfg, seed, 'flat'), osram_source_role_v3=True)


def setup(dataset, uuid, output, seed):
    if dataset != 'CMUMOSEI':
        raise ValueError('This supplement is restricted to MOSEI')
    flat_reference(seed)
    import torch
    from experiments.osram_mosei_gcnet_router_20261020 import worker as old
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    # Reuse verified seed66 data/split creation, then replace its unused model.
    bundle = old.setup(uuid, output, 66)
    del bundle['model']
    cfg = candidate_config(bundle['cfg'], seed)
    bundle['cfg'] = cfg
    bundle['seed_protocol'] = seed_training_loader(bundle, seed)
    baseline = old.fresh_backbone(replace(cfg, osram_source_role_v3=False),
                                  bundle['dimensions'], torch.device('cuda:0')).eval()
    initial = v3.original_digest(baseline)
    model = old.fresh_backbone(cfg, bundle['dimensions'], torch.device('cuda:0')).train()
    if v3.original_digest(model) != initial:
        raise ValueError('V3 changed same-seed original initialization')
    bundle.update(model=model, initial_backbone_sha256=initial)
    return bundle, baseline


def source_catalog(prior_sources):
    return tuple(dict.fromkeys(v3.source_catalog(prior_sources) + (
        'experiments/osram_v3_three_seed_20261021/mosei_worker.py',
        'experiments/osram_mosei_nested_gcnet_3seed_20261021/worker.py',
        'experiments/osram_mosei_gcnet_router_20261020/worker.py',
        'experiments/osram_text_path_router_joint_20261019/run.py')))


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(67, 68), required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('train', 'smoke'), required=True)
    parser.add_argument('--resume', action='store_true')
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    forwarded = ['--dataset', 'CMUMOSEI', '--seed', str(args.seed),
        '--gpu-uuid', args.gpu_uuid, '--source-commit', args.source_commit,
        '--output', str(args.output), '--phase', args.phase]
    if args.resume:
        forwarded.append('--resume')
    return v3.main(forwarded, setup_factory=setup,
        references={'CMUMOSEI': flat_reference(args.seed)}, seeds=(67, 68),
        source_catalog_factory=source_catalog)


if __name__ == '__main__':
    main()
