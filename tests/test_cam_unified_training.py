"""The unified runner must not silently change its paired experiment contract."""
from copy import deepcopy
import importlib

import numpy as np
import pytest
import torch


def protocol():
    return importlib.import_module('experiments.cam_unified_20261030.protocol')


def baseline_payload():
    return dict(protocol().REQUIRED_CONFIG, osram_emotion_ablation='full',
                osram_gap_read='residual', gap_residual_strength=1.0,
                train_missing_rates=[i / 10 for i in range(8)])


def test_variants_change_only_the_two_approved_switches():
    base = baseline_payload()
    expected = {'A0': ('local-only', 'residual'), 'A1': ('local-base', 'residual'),
                'A2': ('full', 'raw'), 'A3': ('full', 'residual')}
    for variant, switches in expected.items():
        actual = protocol().variant_config(base, variant)
        assert (actual['osram_emotion_ablation'], actual['osram_gap_read']) == switches
        assert {k: v for k, v in actual.items() if k not in ('osram_emotion_ablation', 'osram_gap_read')} == {
            k: v for k, v in base.items() if k not in ('osram_emotion_ablation', 'osram_gap_read')}
    assert base == baseline_payload()


@pytest.mark.parametrize('key,value', [('missing_mask_protocol', 'conversation-mask-v2'),
    ('fold', 5), ('learning_rate', 1e-3), ('osram_write_step', .4),
    ('checkpoint_selection_metric', 'accuracy'), ('osram_meaningful_block', 'other')])
def test_protocol_rejects_unapproved_parent_changes(key, value):
    raw = baseline_payload()
    raw[key] = value
    with pytest.raises(ValueError, match=key):
        protocol().variant_config(raw, 'A3')


def test_best_selection_preserves_earliest_tie_and_ignores_mse():
    p = protocol()
    assert p.is_improvement({'WF1': 80., 'MSE': 2.}, None)
    assert not p.is_improvement({'WF1': 80., 'MSE': .1}, {'WF1': 80., 'MSE': 2.})
    assert p.is_improvement({'WF1': 80.001, 'MSE': 3.}, {'WF1': 80., 'MSE': 2.})
    with pytest.raises(ValueError):
        p.is_improvement({'WF1': float('nan'), 'MSE': 1.}, None)


def test_aggregate_is_unweighted_eight_rate_mean_and_three_high_rates():
    rows = {r: {'ACC': i, 'WF1': 2 * i, 'MSE': .1 * i}
            for i, r in enumerate(protocol().RATES)}
    result = protocol().aggregate(rows)
    assert result['mean']['ACC'] == 3.5
    assert result['high']['ACC'] == 6.
    assert result['high']['WF1'] == 12.


def test_rng_isolation_restores_numpy_and_torch():
    r = importlib.import_module('experiments.cam_unified_20261030.runtime')
    before = deepcopy(r.rng_state())
    expected_numpy = np.random.randint(0, 10000, 4)
    expected_torch = torch.rand(4)
    r.restore_rng(before)
    with r.isolated_rng(123):
        np.random.randint(0, 10000, 10)
        torch.rand(10)
    np.testing.assert_array_equal(np.random.randint(0, 10000, 4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)


def test_model_digest_and_strict_checkpoint_do_not_allow_shape_fallback(tmp_path):
    r = importlib.import_module('experiments.cam_unified_20261030.runtime')
    model = torch.nn.Linear(3, 1)
    expected = r.state_digest(model)
    path = tmp_path / 'best.pt'
    r.atomic_save(path, {'model': deepcopy(model.state_dict()), 'epoch': 7})
    with torch.no_grad():
        model.weight.add_(1)
    assert r.state_digest(model) != expected
    state = r.load_strict(model, path)
    assert state['epoch'] == 7
    assert r.state_digest(model) == expected
    with pytest.raises(RuntimeError):
        r.load_strict(torch.nn.Linear(2, 1), path)


def test_metadata_keeps_neutral_regression_labels_and_batch_major_order():
    r = importlib.import_module('experiments.cam_unified_20261030.runtime')
    data = [None] * 10
    data[6] = torch.tensor([[0, 1, 0], [1, 0, 0]])
    data[7] = torch.tensor([[1, 1, 0], [1, 0, 0]])
    data[8] = torch.tensor([[0., -.7, 8.], [1.2, 8., 8.]])
    data[-1] = ['c1', 'c2']
    result = r.metadata(data)
    np.testing.assert_array_equal(result['labels'], np.asarray([0., -.7, 1.2], dtype=np.float32))
    np.testing.assert_array_equal(result['conversation_ids'], ['c1', 'c1', 'c2'])
    np.testing.assert_array_equal(result['utterance_indices'], [0, 1, 0])


def test_selected_weights_and_same_call_predictions_are_bound(tmp_path):
    r = importlib.import_module('experiments.cam_unified_20261030.runtime')
    t = importlib.import_module('experiments.cam_unified_20261030.train')
    model = torch.nn.Linear(3, 1)
    availability = np.ones((3, 3), dtype=np.int64)
    arrays = dict(prediction=np.asarray([[1.], [-1.], [0.]]), labels=np.asarray([1., -1., 0.]),
                  availability=availability)
    metrics = dict(ACC=100., WF1=100., MSE=0., mask_sha256=__import__('hashlib').sha256(availability.tobytes()).hexdigest())
    best = {}
    identity = {'variant': 'A3', 'config': {}}
    t.persist_selected(tmp_path, model, epoch=2, identity=identity, best=best,
                       evaluated={rate: (dict(metrics), dict(arrays)) for rate in protocol().RATES})
    assert set(best) == set(protocol().RATES)
    for rate, row in best.items():
        state = torch.load(tmp_path / row['checkpoint'], map_location='cpu', weights_only=False)
        assert state['kind'] == 'weights-only'
        assert state['selection']['rate'] == rate
        assert state['epoch'] == 2
        assert row['prediction_sha256'] == r.sha(tmp_path / row['predictions'])
    # A tied epoch must not overwrite the checkpoint or its selected predictions.
    before = deepcopy(best)
    t.persist_selected(tmp_path, model, epoch=3, identity=identity, best=best,
                       evaluated={rate: (dict(metrics), dict(arrays)) for rate in protocol().RATES})
    assert best == before


def test_restore_rejects_weights_only_or_different_identity(tmp_path):
    t = importlib.import_module('experiments.cam_unified_20261030.train')
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match='full-resume'):
        t.restore(model, optimizer, {'kind': 'weights-only', 'identity': {}}, {})
    with pytest.raises(ValueError, match='identity'):
        t.restore(model, optimizer, {'kind': 'full-resume', 'identity': {'variant': 'A1'}},
                  {'variant': 'A3'})
