"""MOSI NoArea changes one descriptor while matching the completed seed68 run."""
from copy import deepcopy
import json

import pytest


def fixture_reference():
    from experiments.osram_signature_noarea_mosi68_20261025.run import NUMERICAL_SOURCE
    parent = {'parent_id': 'parent', 'checkpoint_sha256': 'parent'}
    config = dict(dataset='CMUMOSI', batch_size=32, learning_rate=3e-4)
    reference = dict(status='complete', epoch=150, total_parameters=281856,
        parent_id='parent', identity=dict(mode='signature', dataset='CMUMOSI', seed=68,
            evaluation_rate='0.3', source_commit=NUMERICAL_SOURCE, smoke=False,
            stage2_epochs=150, initial_block_sha256='init', parent=parent,
            effective_stage2_config=config, test_mask_sha256='mask',
            cyclic_training=True, mask_protocol='gcnet-random-mask-v1'))
    return reference, parent, config


def test_only_seed68_one_fold_and_original_budget_and_lr():
    from experiments.osram_signature_noarea_mosi68_20261025 import run
    from experiments.osram_signature_cyclic68_20261025 import run as original
    run.validate_scope(68, 1)
    for seed, fold in [(66, 1), (67, 1), (68, 2), (True, 1), (68, True)]:
        with pytest.raises(ValueError):
            run.validate_scope(seed, fold)
    assert run.budget is original.budget
    assert run.learning_rate is original.learning_rate
    assert run.budget('CMUMOSI') == 150
    assert run.learning_rate('CMUMOSI', 0) == 3e-4
    assert run.learning_rate('CMUMOSI', 99) == 3e-4
    assert run.learning_rate('CMUMOSI', 100) == 3e-4 / 10
    assert run.learning_rate('CMUMOSI', 149) == 3e-4 / 10
    with pytest.raises(ValueError):
        run.learning_rate('CMUMOSI', 150)


def test_reference_requires_same_parent_initialization_config_and_mask():
    from experiments.osram_signature_noarea_mosi68_20261025.run import validate_reference
    reference, parent, config = fixture_reference()
    validate_reference(reference, parent, config, 'init', 'mask', '0.3')
    for updates in [dict(parent={'parent_id': 'wrong'}), dict(initial_sha='wrong'),
                    dict(mask_sha='wrong'), dict(config={'learning_rate': 1e-3}),
                    dict(rate='0.7')]:
        args = dict(reference=reference, parent=parent, config=config,
            initial_sha='init', mask_sha='mask', rate='0.3')
        args.update(updates)
        with pytest.raises(ValueError):
            validate_reference(**args)


def test_reference_json_tuple_canonicalization_preserves_values():
    from experiments.osram_signature_noarea_mosi68_20261025.run import validate_reference
    reference, parent, config = fixture_reference()
    config['train_missing_rates'] = (0., .1, .2)
    parent['config'] = dict(feature_names=('a', 't', 'v'))
    stored = json.loads(json.dumps(reference))
    validate_reference(stored, parent, config, 'init', 'mask', '0.3')
    config['train_missing_rates'] = (0., .1, .7)
    with pytest.raises(ValueError):
        validate_reference(stored, parent, config, 'init', 'mask', '0.3')


@pytest.mark.parametrize('field,value', [('status', 'running'), ('epoch', 149),
    ('total_parameters', 123)])
def test_reference_rejects_unfinished_or_different_capacity(field, value):
    from experiments.osram_signature_noarea_mosi68_20261025.run import validate_reference
    reference, parent, config = fixture_reference()
    reference[field] = value
    with pytest.raises(ValueError):
        validate_reference(reference, parent, config, 'init', 'mask', '0.3')


def test_epoch_trace_requires_same_masks_steps_data_and_exact_lr():
    from experiments.osram_signature_noarea_mosi68_20261025.run import validate_epoch_trace
    row = dict(epoch=101, optimizer_steps=4, training_rates=[0., .1, .2, .3],
        mask_trace_sha256='trace', N=12, learning_rate=3e-4 / 10)
    validate_epoch_trace(row, deepcopy(row))
    for key, value in [('optimizer_steps', 5), ('training_rates', [.7]),
                       ('mask_trace_sha256', 'other'), ('N', 13), ('epoch', 102),
                       ('learning_rate', 3e-4)]:
        with pytest.raises(ValueError):
            validate_epoch_trace(dict(row, **{key: value}), row)


def test_original_evaluation_is_reused_and_selection_is_strict_wf1():
    from experiments.osram_signature_noarea_mosi68_20261025 import run
    from experiments.osram_signature_cyclic68_20261025 import run as original
    assert run.evaluate is original.evaluate
    assert run.improves({'WF1': 80., 'ACC': 75.}, None)
    assert not run.improves({'WF1': 80., 'ACC': 99.}, {'WF1': 80., 'ACC': 75.})
    assert run.improves({'WF1': 80.1, 'ACC': 75.}, {'WF1': 80., 'ACC': 99.})


def test_training_history_contiguous_and_complete():
    from experiments.osram_signature_noarea_mosi68_20261025.run import validate_history
    validate_history([dict(epoch=e) for e in range(1, 151)], 150)
    with pytest.raises(ValueError):
        validate_history([dict(epoch=e) for e in range(1, 150)], 150)
    with pytest.raises(ValueError):
        validate_history([dict(epoch=1), dict(epoch=1)], 2)
