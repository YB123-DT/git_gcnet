"""Fixed Flat128 and existing Nested21 configurations on the original cmerc loop."""
import argparse
from dataclasses import asdict, replace
import sys

from experiments.osram_nested_cmerc_tuning_20261023 import worker as shared
from .config import VARIANTS, candidate_config, run_spec


def validate_scope(variant, dataset, method, seed, fold):
    spec = run_spec(variant)
    if (dataset, method, seed, fold) != ('CMUMOSEI', spec['method'], spec['seed'], 1):
        raise ValueError('Only the approved MOSEI Flat128 seeds/Nested21 slots are authorized')


def setup(dataset, method, uuid, output, *, seed, fold, variant):
    import torch
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_mosei_v3_capacity_20261022.worker import digest_without
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader

    validate_scope(variant, dataset, method, seed, fold)
    # Load accepted data/split/test bank using the unchanged validated seed66
    # helper. Only then choose this slot's initialization/training RNG seed.
    bundle, large = shared.setup(dataset, 'none', uuid, output, seed=66, fold=1)
    old_cfg = bundle['cfg']
    cfg = candidate_config(old_cfg, variant)
    if seed != 66:
        large = _build_model(replace(old_cfg, seed=seed), bundle['dimensions'], torch.device('cuda:0'))
    plain = _build_model(replace(cfg, osram_meaningful_block='none'),
                         bundle['dimensions'], torch.device('cuda:0'))
    model = plain if method == 'none' else _build_model(cfg, bundle['dimensions'], torch.device('cuda:0'))
    ignored = ('osram.emotion_adapter.', 'osram.meaningful_block.')
    if digest_without(large, ignored) != digest_without(model, ignored):
        raise RuntimeError('Untouched original parameters changed at this seed')
    initial = h.backbone_digest(plain)
    if digest_without(model, ('osram.meaningful_block.',)) != initial:
        raise RuntimeError('Nested changed the same-width/same-seed original initialization')
    if cfg.osram_adapter_hidden_dim != 128 or model.osram.write_step != .6:
        raise RuntimeError('Wrong Flat width or Memory write strength')
    model.requires_grad_(True)
    bundle.update(model=model, cfg=cfg, initial_backbone_sha256=initial,
                  config_reference_delta={k:v for k,v in asdict(cfg).items() if v != asdict(old_cfg)[k]})
    bundle['seed_protocol'] = seed_training_loader(bundle, seed)
    return bundle, plain


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant', choices=('F67','F68', *VARIANTS), required=True)
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    spec = run_spec(args.variant)

    def validate(dataset, method, seed, fold):
        validate_scope(args.variant, dataset, method, seed, fold)

    def prepare(*positional, **keywords):
        return setup(*positional, **keywords, variant=args.variant)

    shared.main(['--dataset','CMUMOSEI','--method',spec['method'],'--fold','1']+remaining,
        setup_function=prepare, scope_validator=validate, method_choices=(spec['method'],),
        additional_source_files=('experiments/osram_mosei_flat128_nested_20261024/config.py',
                                 'experiments/osram_mosei_flat128_nested_20261024/worker.py'))


if __name__ == '__main__':
    main()
