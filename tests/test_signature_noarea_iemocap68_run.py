"""The no-area training control changes the descriptor, not the experiment."""
from copy import deepcopy
import json

import numpy as np
import pytest


def fixture_reference():
    from experiments.osram_signature_noarea_iemocap68_20261025.run import NUMERICAL_SOURCE
    parent = {'parent_id': 'parent', 'checkpoint_sha256': 'parent'}
    config = {'dataset': 'IEMOCAPSix', 'batch_size': 32, 'learning_rate': 3e-4}
    reference = dict(status='complete', epoch=100, total_parameters=281856,
        parent_id='parent', identity=dict(mode='signature', dataset='IEMOCAPSix', seed=68,
            fold=5, evaluation_rate='0.3', source_commit=NUMERICAL_SOURCE, smoke=False,
            stage2_epochs=100, initial_block_sha256='init', parent=parent,
            effective_stage2_config=config, test_mask_sha256='mask',
            cyclic_training=True, mask_protocol='gcnet-random-mask-v1'))
    return reference, parent, config


def test_only_seed68_and_original_finite_budget():
    from experiments.osram_signature_noarea_iemocap68_20261025.run import validate_scope, budget, learning_rate
    for fold in range(1, 6):
        validate_scope(68, fold)
    for seed, fold in [(66, 1), (67, 5), (68, 0), (68, 6), (True, 1), (68, True)]:
        with pytest.raises(ValueError):
            validate_scope(seed, fold)
    assert budget() == 100
    assert learning_rate(0) == learning_rate(99) == 3e-4
    with pytest.raises(ValueError):
        learning_rate(100)


def test_reference_requires_same_parent_initialization_config_and_mask():
    from experiments.osram_signature_noarea_iemocap68_20261025.run import validate_reference
    reference, parent, config = fixture_reference()
    validate_reference(reference, parent, config, 'init', 'mask', 5, '0.3')
    for kwargs in [dict(parent={'parent_id': 'wrong'}), dict(initial_sha='wrong'),
                   dict(mask_sha='wrong'), dict(config={'learning_rate': 1e-3})]:
        args = dict(reference=reference, parent=parent, config=config,
                    initial_sha='init', mask_sha='mask', fold=5, rate='0.3')
        args.update(kwargs)
        with pytest.raises(ValueError):
            validate_reference(**args)


def test_json_reference_accepts_tuple_config_sequences_but_not_different_values():
    from experiments.osram_signature_noarea_iemocap68_20261025.run import validate_reference
    reference, parent, config = fixture_reference()
    config['train_missing_rates'] = (0., .1, .2)
    parent['config'] = dict(train_missing_rates=(0., .1, .2), feature_names=('a', 't', 'v'))
    reference['identity']['parent'] = parent
    reference['identity']['effective_stage2_config'] = config
    stored = json.loads(json.dumps(reference))
    validate_reference(stored, parent, config, 'init', 'mask', 5, '0.3')
    changed = deepcopy(parent)
    changed['config']['train_missing_rates'] = (0., .1, .7)
    with pytest.raises(ValueError):
        validate_reference(stored, changed, config, 'init', 'mask', 5, '0.3')


@pytest.mark.parametrize('field,value', [('status', 'running'), ('epoch', 99),
    ('total_parameters', 123)])
def test_reference_rejects_unfinished_or_different_capacity(field, value):
    from experiments.osram_signature_noarea_iemocap68_20261025.run import validate_reference
    reference, parent, config = fixture_reference()
    reference[field] = value
    with pytest.raises(ValueError):
        validate_reference(reference, parent, config, 'init', 'mask', 5, '0.3')


def test_epoch_trace_requires_identical_masks_and_optimizer_budget():
    from experiments.osram_signature_noarea_iemocap68_20261025.run import validate_epoch_trace
    row = dict(epoch=1, optimizer_steps=4, training_rates=[0., .1, .2, .3],
               mask_trace_sha256='trace', N=12)
    validate_epoch_trace(row, deepcopy(row))
    for key, value in [('optimizer_steps', 5), ('training_rates', [.7]),
                       ('mask_trace_sha256', 'other'), ('N', 13), ('epoch', 2)]:
        wrong = dict(row, **{key: value})
        with pytest.raises(ValueError):
            validate_epoch_trace(wrong, row)


def test_original_metric_and_selection_functions_are_reused_including_class_zero():
    from experiments.osram_signature_noarea_iemocap68_20261025 import run
    from experiments.osram_signature_iemocap_cyclic_20261025 import run as original
    assert run.evaluate is original.evaluate
    assert run.measured_metrics is original.measured_metrics
    assert run.improves is original.improves
    labels = np.arange(6)
    logits = np.eye(6) * 4
    score = run.measured_metrics(labels, logits, logits)
    assert score['ACC'] == score['UA'] == 100
    assert score['N'] == 6 and score['support'] == [1] * 6
    assert not run.improves({'ACC': 60., 'UA': 80.}, {'ACC': 60., 'UA': 10.})


def test_reference_history_must_be_contiguous_and_complete():
    from experiments.osram_signature_noarea_iemocap68_20261025.run import validate_history
    history = [dict(epoch=n) for n in range(1, 101)]
    validate_history(history, 100)
    with pytest.raises(ValueError):
        validate_history(history[:-1], 100)
    with pytest.raises(ValueError):
        validate_history([dict(epoch=1), dict(epoch=1)], 2)
