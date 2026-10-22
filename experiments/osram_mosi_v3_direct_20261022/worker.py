"""One fresh MOSI66 run replacing active read inputs with V3 decoded evidence."""
from dataclasses import replace

import numpy as np

from experiments.osram_mosi_v3_capacity_20261022 import worker as base

MODE = 'f512_v128'


def candidate_config(cfg, mode):
    if mode not in (MODE, 'f512_plain'):
        raise ValueError('Only Flat512/V3-128 direct and its internal plain initializer')
    result = base.candidate_config(cfg, mode)
    return replace(result, epochs=100,
                   osram_source_role_v3_mode='direct' if mode == MODE else 'residual')


def schedule(dataset):
    if dataset != 'CMUMOSI':
        raise ValueError('Only MOSI seed66 is approved')
    return dict(epochs=100, first_lr=3e-4, final_lr=3e-4)


def initialization_check(expected, actual):
    # The original Flat adapter is zero-initialized, so initial task predictions
    # can coincide even though decoded history is random and NOT an identity.
    if expected.keys() != actual.keys():
        raise ValueError('Initial prediction schema differs')
    for key in expected:
        if not np.isfinite(actual[key]).all():
            raise ValueError('Nonfinite direct-initialization output: ' + key)
    return dict(decoder_initialization='random', readout_identity_initialization=False,
                initial_predictions_exact_same_width_flat=all(
                    np.array_equal(expected[key], actual[key]) for key in expected))


def source_catalog(prior):
    return sorted(set(base.source_catalog(prior)).union(
        'experiments/osram_mosi_v3_direct_20261022/' + name
        for name in ('worker.py', 'run.py', 'report.py')))


def main(argv=None):
    return base.main(argv, allowed_modes=(MODE,), config_factory=candidate_config,
                     source_catalog_factory=source_catalog, schedule_factory=schedule,
                     initialization_check=initialization_check)


if __name__ == '__main__':
    main()
