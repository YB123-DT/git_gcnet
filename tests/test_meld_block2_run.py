import importlib

import pytest


def module():
    return importlib.import_module('experiments.osram_meld_block2_auto_20261014.run')


def test_finite_cli_bounds():
    args = module().parse_args(['--phase', 'all', '--gpus', '0,2,3', '--per-gpu', '4',
                              '--seed', '66', '--top-k', '2', '--trials-per-model', '20'])
    assert args.gpus == [0, 2, 3] and args.trials_per_model == 20
    for tail in (['--gpus', '4'], ['--per-gpu', '5'], ['--trials-per-model', '21'], ['--seed', '67']):
        with pytest.raises((ValueError, SystemExit)):
            module().parse_args(tail)


def test_hpo_export_preserves_failures_and_best_actual_config():
    rows = [dict(method='B1', number=0, state='FAIL', params={'lr': .001}, value=None, user_attrs={}),
            dict(method='B1', number=1, state='COMPLETE', params={'lr': .0005}, value=.2,
                 user_attrs={'queue_record': {'config': {'learning_rate': .0005, 'epochs': 50}}})]
    export = module().study_export(rows, ['B1'])
    assert export['studies'][0]['best_trial'] == 1
    assert export['studies'][0]['best_config']['epochs'] == 50
    assert export['studies'][0]['trials'][0]['state'] == 'FAIL'
