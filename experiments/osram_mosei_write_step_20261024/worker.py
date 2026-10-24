"""Four MOSEI Flat write-step variants; unchanged accepted cmerc training loop."""
import argparse
from dataclasses import asdict, replace
import sys

from experiments.osram_nested_cmerc_tuning_20261023 import worker as shared

STEPS = (.2, .4, .8, 1.)


def candidate_config(cfg, step):
    if step not in STEPS or cfg.osram_adapter_hidden_dim != 0:
        raise ValueError('Only four new write strengths on original Flat1600 are authorized')
    shared.candidate_config(cfg, 'none', seed=66, fold=1)
    return replace(cfg, osram_write_step=step)


def setup(dataset, method, uuid, output, *, seed, fold, step):
    import torch
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosei_capacity_cmerc_20261024.report import CONTROL
    import json

    bundle, baseline = shared.setup(dataset, method, uuid, output, seed=seed, fold=fold)
    cfg = candidate_config(bundle['cfg'], step)
    model = _build_model(cfg, bundle['dimensions'], torch.device('cuda:0'))
    initial = h.backbone_digest(model)
    reference = json.loads((CONTROL/'SUMMARY.json').read_text())
    if initial != h.backbone_digest(baseline) or initial != reference['initial_backbone_sha256']:
        raise RuntimeError('A scalar write-step change must preserve every initial parameter')
    if model.osram.write_step != step:
        raise RuntimeError('Configured write step is not active in the real Memory writer')
    model.requires_grad_(True)
    bundle.update(model=model, cfg=cfg, initial_backbone_sha256=initial,
                  config_reference_delta={'osram_write_step': step})
    return bundle, baseline


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--write-step', type=float, choices=STEPS, required=True)
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    def validate(dataset, method, seed, fold):
        if (dataset, method, seed, fold) != ('CMUMOSEI', 'none', 66, 1):
            raise ValueError('Only MOSEI Flat seed66/fold1 is authorized')

    def prepare(*positional, **keywords):
        return setup(*positional, **keywords, step=args.write_step)

    shared.main(['--dataset', 'CMUMOSEI', '--method', 'none', '--fold', '1'] + remaining,
        setup_function=prepare, scope_validator=validate, method_choices=('none',),
        additional_source_files=('experiments/osram_mosei_write_step_20261024/worker.py',))


if __name__ == '__main__':
    main()
