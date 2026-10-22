"""One fresh MOSI66 joint configuration, not another parameter sweep."""
from dataclasses import replace

from experiments.osram_mosi_memory_grid_20261022 import worker as base

MODE='head48_step04'


def candidate_config(cfg,mode):
    if mode!=MODE or cfg.seed!=66:
        raise ValueError('Only head48/write_step0.4 seed66 is authorized')
    return replace(base.candidate_config(cfg,'head_48'),osram_write_step=.4)


def source_catalog(prior):
    return sorted(set(base.source_catalog(prior)).union(
        'experiments/osram_mosi_head48_step04_20261022/'+name
        for name in ('worker.py','run.py','report.py')))


def main(argv=None):
    return base.main(argv,allowed_seeds=(66,),allowed_modes=(MODE,),
        config_factory=candidate_config,source_catalog_factory=source_catalog)


if __name__=='__main__':
    main()
