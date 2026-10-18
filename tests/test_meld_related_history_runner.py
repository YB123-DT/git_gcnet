import importlib
import inspect
import types

import pytest


def runner():
    try:
        return importlib.import_module('experiments.osram_meld_related_history_20261018.run')
    except ModuleNotFoundError:
        pytest.fail('B1/B2 finite runner is missing')


def test_two_configs_preserve_entire_o6_training_contract():
    r = runner()
    expected = r.screen.make_config(r.screen.source_config(), 'O6')
    for kind in ('B1', 'B2'):
        assert r.make_config(kind) == expected
    assert expected['learning_rate'] == 1e-5
    assert expected['weight_decay'] == 2e-4
    assert expected['epochs'] == 50 and expected['seed'] == 66
    assert expected['missing_mask_protocol'] == 'gcnet-random-mask-v1'
    assert expected['osram_output_dim'] == 1600
    assert expected['osram_readout_fusion'] == 'flat'
    assert expected['training_objective'] == 'emotion-only'


def test_branch_metadata_are_explicit_not_hidden_training_config_changes():
    r = runner()
    assert r.branch_config('B1') == dict(kind='B1', window=8, attention_dim=128,
        input_dim=2435, relation_weight=0.)
    assert r.branch_config('B2') == dict(kind='B2', window=8, attention_dim=128,
        input_dim=2564, relation_weight=.05)
    with pytest.raises(ValueError):
        r.make_config('B3')


def test_render_has_all_rates_and_separate_ua_from_acc_same_epoch():
    r = runner()
    metrics = dict(ACC=61., UA=38., WF1=58.)
    scores = dict(mean=metrics, high=metrics,
        rates={str(i/10): dict(metrics, epoch=i+1) for i in range(8)})
    rows = {kind: dict(status='complete',testbest=scores,parameters=14000000,
        added_parameters=700000) for kind in ('Flat', 'B1', 'B2')}
    rendered = r.render(rows)
    assert 'INTERNAL TEST-ORACLE SCREEN' in rendered
    assert '0.7' in rendered and 'UA' in rendered and 'Flat' in rendered
    assert 'B1' in rendered and 'B2' in rendered


def test_smoke_evaluator_keeps_capture_signature(tmp_path):
    def train(model, loader, optimizer, epoch):
        pass

    def evaluate(model, loader, schedule, collect=False):
        return model, list(loader), schedule, collect

    trainer = types.SimpleNamespace(train_epoch=train, evaluate_rate=evaluate)
    runner().observe_training(trainer, tmp_path, None, smoke=True)
    assert inspect.signature(trainer.evaluate_rate) == inspect.signature(evaluate)
    assert trainer.evaluate_rate('model', [1, 2], 'schedule') == ('model', [1], 'schedule', False)
