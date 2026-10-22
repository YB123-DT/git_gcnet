"""Reuse the grid loop, supplement only head48 seeds67/68 from scratch."""
from dataclasses import replace
from pathlib import Path

from experiments.osram_mosi_memory_grid_20261022 import worker as base

SEEDS=(67,68)
MODE='head_48'
REFERENCE_ROOT=Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_3seed_20261021/execution_v1/runs/MOSI')


def candidate_config(cfg,mode):
    if cfg.seed not in SEEDS or mode!=MODE:
        raise ValueError('Only supplemental head48 seeds67/68 are authorized')
    return replace(base.candidate_config(replace(cfg,seed=66),mode),seed=cfg.seed)


def reference_for_seed(seed):
    if seed not in SEEDS:
        raise ValueError('Only supplemental seeds67/68; do not repeat66')
    return REFERENCE_ROOT/f'seed_{seed}'/'flat'


def source_catalog(prior_sources):
    additions=('experiments/osram_mosi_head48_3seed_20261022/'+name
               for name in ('worker.py','run.py','report.py'))
    return sorted(set(base.source_catalog(prior_sources)).union(additions))


def main(argv=None):
    return base.main(argv,allowed_seeds=SEEDS,allowed_modes=(MODE,),
        config_factory=candidate_config,reference_factory=reference_for_seed,
        source_catalog_factory=source_catalog)


if __name__=='__main__':
    main()
