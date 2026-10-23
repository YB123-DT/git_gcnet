"""Guard the finite, existing-Nested-only cmerc tuning protocol."""
from dataclasses import make_dataclass, replace
import importlib

import pytest


def worker():
    return importlib.import_module('experiments.osram_nested_cmerc_tuning_20261023.worker')


def config(dataset):
    fields = dict(dataset=dataset, seed=66, fold=1 if dataset == 'CMUMOSEI' else 5,
        epochs=60 if dataset == 'CMUMOSEI' else 100,
        learning_rate=1e-4 if dataset == 'CMUMOSEI' else 1e-3,
        weight_decay=1e-5, optimizer='adam', lr_schedule='constant', batch_size=32,
        missing_mask_protocol='gcnet-random-mask-v1', train_rate_mode='cyclic',
        train_missing_rates=tuple(i/10 for i in range(8)), training_objective='emotion-only',
        checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1' if dataset == 'CMUMOSEI' else 'accuracy',
        evaluate_test=True, osram_meaningful_block='none', osram_output_dim=1600,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
        osram_write_step=.6, osram_bidirectional=False, osram_readout_fusion='flat',
        osram_disable_local_skip=False, paired_history_views=False,
        osram_source_role_v3=False, osram_tasr=False, osram_source_role=False,
        osram_source_relation=False, osram_source_tracking=False, completion_path='none',
        initial_backbone_checkpoint=None, preserve_selection_artifacts=True)
    cls = make_dataclass('Config', [(k, type(v)) for k, v in fields.items()])
    return cls(**fields)


@pytest.mark.parametrize('dataset', ['CMUMOSEI', 'IEMOCAPSix'])
def test_existing_nested_only_changes_module_switch(dataset):
    from dataclasses import asdict
    w = worker()
    original = config(dataset)
    for method in w.METHODS:
        actual = w.candidate_config(original, method)
        expected = asdict(original)
        expected['osram_meaningful_block'] = method
        assert asdict(actual) == expected


@pytest.mark.parametrize('change', [dict(learning_rate=.001), dict(epochs=100),
    dict(missing_mask_protocol='conversation-mask-v2'), dict(checkpoint_selection='validation'),
    dict(osram_source_role_v3=True), dict(paired_history_views=True),
    dict(osram_write_step=.4), dict(osram_disable_local_skip=True), dict(seed=67)])
def test_reject_unapproved_protocol_changes(change):
    with pytest.raises(ValueError):
        worker().candidate_config(replace(config('CMUMOSEI'), **change), 'nested_dim32')


def test_scope_never_includes_mosi_meld_allfold_or_direct():
    w = worker()
    assert len(w.METHODS) == 7 and 'none' in w.METHODS
    assert 'nested_gnn_rooted_evidence' in w.METHODS
    assert all('direct' not in name for name in w.METHODS)
    for changes in [dict(dataset='CMUMOSI'), dict(dataset='IEMOCAPFour'), dict(fold=1)]:
        with pytest.raises(ValueError):
            w.candidate_config(replace(config('IEMOCAPSix'), **changes), 'none')


def test_gradient_guard_rejects_only_decoder_connections():
    w = worker()
    with pytest.raises(RuntimeError):
        w.check_nested_gradients({'local_decoder.weight': True})
    w.check_nested_gradients({name: True for name in ('tokenizer.local.weight',
        'core.layers.0.0.weight', 'core.pool.0.weight', 'core.readout.0.weight',
        'local_decoder.weight', 'memory_decoders.0.weight')})


def test_parity_reports_roundoff_honestly_and_rejects_prediction_changes():
    import numpy as np
    w = worker()
    old = np.array([[-1., -2., -3.]], dtype=np.float32)
    new = old.copy(); new[0, 0] += 7e-7
    parity = w.prediction_parity(old, new, scalar=False)
    assert parity['numerically_equal'] and not parity['bitwise_equal']
    assert 0 < parity['max_absolute_error'] < 1e-6
    with pytest.raises(AssertionError):
        w.prediction_parity(old, old + .01, scalar=False)
    with pytest.raises(AssertionError):
        w.prediction_parity(np.array([[1e-8]]), np.array([[-1e-8]]), scalar=True)
