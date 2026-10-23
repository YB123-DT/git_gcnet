"""Extend only the selected Nested128 to the authorized seed/fold matrix."""
from dataclasses import asdict, replace
import importlib

import pytest

from test_nested_cmerc_worker import config


def worker():
    return importlib.import_module('experiments.osram_nested_cmerc_tuning_20261023.worker')


def test_selected_model_supports_exactly_three_seeds_and_five_folds():
    w = worker()
    for seed in (66, 67, 68):
        for fold in range(1, 6):
            base = replace(config('IEMOCAPSix'), seed=seed, fold=fold)
            actual = w.candidate_config(base, 'nested_dim128', seed=seed, fold=fold)
            expected = asdict(base); expected['osram_meaningful_block'] = 'nested_dim128'
            assert asdict(actual) == expected
            assert f'fold_{fold}' in str(w.reference_path('IEMOCAPSix', fold=fold))


@pytest.mark.parametrize('dataset,method,seed,fold', [
    ('CMUMOSEI', 'nested_dim128', 67, 1), ('IEMOCAPSix', 'nested_dim32', 67, 5),
    ('IEMOCAPSix', 'none', 67, 1), ('IEMOCAPSix', 'nested_dim128', 65, 5),
    ('IEMOCAPSix', 'nested_dim128', 66, 6), ('IEMOCAPFour', 'nested_dim128', 66, 5)])
def test_extended_scope_rejects_unapproved_models_and_seeds(dataset, method, seed, fold):
    with pytest.raises(ValueError):
        worker().validate_scope(dataset, method, seed, fold)


def test_legacy_seed66_fold5_configuration_remains_identical():
    w = worker()
    base = config('IEMOCAPSix')
    assert w.candidate_config(base, 'nested_dim128') == w.candidate_config(
        base, 'nested_dim128', seed=66, fold=5)
