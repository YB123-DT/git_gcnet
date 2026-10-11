"""Contracts for the two approved seed66 counterfactual-readout screens."""

import pytest


def reference():
    return dict(dataset='CMUMOSI', seed=66, epochs=100, learning_rate=.001,
                lr_schedule='constant', backbone_type='osram',
                training_objective='emotion-only', checkpoint_selection='test-oracle-per-rate',
                train_rate_mode='cyclic', osram_readout_fusion='flat',
                osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64,
                osram_value_dim=64, osram_write_step=.6, osram_bidirectional=False,
                disable_unused_aux_modules=True)


@pytest.mark.parametrize('mode', ['marginal', 'coalition'])
def test_candidate_changes_only_requested_readout(mode):
    from experiments.osram_counterfactual_readouts_20261011.run import candidate_config
    base = reference()
    result = candidate_config(base, mode)
    assert {key: value for key, value in result.items() if base.get(key) != value} == {
        'osram_counterfactual_readout': mode}


@pytest.mark.parametrize('field,value', [
    ('seed', 67), ('epochs', 150), ('checkpoint_selection', 'validation'),
    ('osram_meaningful_block', 'nested_gnn_rooted_evidence'),
    ('paired_history_views', True), ('osram_relation_block', True),
    ('osram_base_first_gap', True), ('training_objective', 'joint'),
])
def test_candidate_rejects_protocol_or_competing_method(field, value):
    from experiments.osram_counterfactual_readouts_20261011.run import candidate_config
    with pytest.raises(ValueError):
        candidate_config(dict(reference(), **{field: value}), 'marginal')


def test_only_two_named_tasks_and_fixed_schedule():
    from experiments.osram_counterfactual_readouts_20261011.run import specs, schedule
    assert specs() == [
        dict(id='marginal_seed66', dataset='CMUMOSI', model='marginal', seed=66, fold=1),
        dict(id='coalition_seed66', dataset='CMUMOSI', model='coalition', seed=66, fold=1),
    ]
    assert schedule() == ((100, .001), (150, .0001))


def test_invalid_mode_is_rejected():
    from experiments.osram_counterfactual_readouts_20261011.run import candidate_config
    with pytest.raises(ValueError, match='mode'):
        candidate_config(reference(), 'other')
