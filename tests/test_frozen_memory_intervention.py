import importlib
import inspect
import numpy as np
import pytest


def implementation():
    return importlib.import_module('experiments.osram_frozen_memory_audit_20261009.intervention')


def test_strict_history_subset_nonempty_and_equal_one_bit():
    api = implementation()
    masks = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
    original = masks.copy()
    pairs, skipped = api.select_pairs(masks, 4)
    assert pairs and skipped['same_vs_other_speaker_A'] == 'speaker_unavailable'
    for pair in pairs:
        for arm in ('delete', 'control'):
            changed = api.delete_bit(masks, 4, pair[arm])
            assert np.array_equal(changed[4:], masks[4:])
            assert np.all(changed <= masks) and np.all(changed.sum(-1) > 0)
            assert int((masks - changed).sum()) == 1
    np.testing.assert_array_equal(original, masks)


def test_temporal_pairing_and_recent_earlier_definition():
    api = implementation()
    masks = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    pairs, _ = api.select_pairs(masks, 3)
    by_name = {pair['family']: pair for pair in pairs}
    assert by_name['remove_T_vs_A']['delete'] == (0, 1)
    assert by_name['remove_T_vs_A']['control'] == (0, 0)
    assert by_name['remove_T_vs_V']['delete'] == (1, 1)
    assert by_name['recent_vs_earlier_A']['delete'] == (2, 0)
    assert by_name['recent_vs_earlier_A']['control'] == (0, 0)
    assert not api.select_pairs(masks, 0)[0]


def test_closest_available_pair_records_mismatched_lag():
    api = implementation()
    masks = np.array([[0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]])
    pairs, _ = api.select_pairs(masks, 3)
    pair = next(p for p in pairs if p['family'] == 'remove_T_vs_A')
    assert pair['delete'] == (0, 1) and pair['control'] == (1, 0)
    assert pair['lag_mismatch'] == 1


def test_speaker_requires_two_ids_and_choice_is_label_free():
    api = implementation()
    masks = np.ones((5, 3), dtype=int)
    assert 'labels' not in inspect.signature(api.select_pairs).parameters
    single, skipped = api.select_pairs(masks, 4, np.zeros(5))
    assert not any(p['family'].startswith('same_vs') for p in single)
    assert skipped['same_vs_other_speaker_A'] == 'speaker_unavailable'
    pairs, _ = api.select_pairs(masks, 4, np.array([0, 1, 0, 1, 0]))
    same = next(p for p in pairs if p['family'] == 'same_vs_other_speaker_T')
    assert same['delete'] == (2, 1) and same['control'] == (3, 1)


def test_invalid_deletions_rejected():
    api = implementation()
    masks = np.array([[1, 0, 0], [1, 1, 1]])
    for position in ((0, 0), (0, 1), (1, 0)):
        with pytest.raises(ValueError):
            api.delete_bit(masks, 1, position)


def test_independent_batched_scans_preserve_frozen_state_and_current_local():
    torch = pytest.importorskip('torch')
    from experiments.osram_history_drift_20261002.diagnostic import capture

    class CausalFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.osram = torch.nn.Module()
            self.osram.emotion_adapter = torch.nn.Linear(4352, 1, bias=False)
            self.osram.osram_readout_fusion = 'flat'
            self.osram.bidirectional = False
            self.osram.latent_dim = 256
            self.scans = 0
            with torch.no_grad():
                self.osram.emotion_adapter.weight.fill_(1 / 4352)

        def forward(self, inputs, availability, qmask, umask, lengths, predict_missing=False):
            self.scans += 1
            current = inputs[0].sum(-1, keepdim=True)
            local = current.expand(-1, -1, 256)
            past = current.cumsum(0) - current
            forward = past[..., None].expand(-1, -1, 4, 512)
            memory = torch.cat((forward, torch.zeros_like(forward)), -1).flatten(-2)
            hidden = torch.cat((local, memory), -1)
            return self.osram.emotion_adapter(hidden), hidden

    model = CausalFixture().double().eval().requires_grad_(False)
    mask = torch.ones(5, 2, 3)
    mask[3:, 1] = 0
    view = dict(incomplete=torch.arange(30, dtype=torch.float64).reshape(5, 2, 3),
                availability=mask, qmask=torch.zeros(2, 5, dtype=torch.long),
                umask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
                labels=torch.ones(2, 5), lengths=[5, 3], conversation_ids=['a', 'b'])
    baseline = capture(model, lambda: model([view['incomplete']], mask, view['qmask'],
                                           view['umask'], view['lengths']))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    scans_before = model.scans
    result = implementation().evaluate_interventions(model, view, (1, 1, 1), baseline, .7, 'test')
    assert model.scans - scans_before <= 3
    assert result['coverage']['scans'] == model.scans - scans_before
    assert result['coverage']['unique_deletions'] < 2 * len(result['rows'])
    assert result['coverage']['targets'] == 8
    assert result['coverage']['eligible'] == len(result['rows']) > 16
    assert all(row['current_local_max_error'] == 0 for row in result['rows'])
    assert any(row['base_drift_delete']['abs'] > 0 for row in result['rows'])
    assert all(row['position_delete'] < row['utterance_index'] for row in result['rows'])
    for row in result['rows']:
        b = view['conversation_ids'].index(row['conversation_id'])
        for arm in ('delete', 'control'):
            pos = row[f'position_{arm}']
            mod = ('A', 'T', 'V').index(row[f'modality_{arm}'])
            # Closed-form fixture output verifies cache keys and target reuse.
            expected = row['pred_real'] - float(view['incomplete'][pos, b, mod]) * 2048 / 4352
            assert row[f'pred_{arm}'] == pytest.approx(expected, abs=1e-5)
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    repeat = implementation().evaluate_interventions(model, view, (1, 1, 1), baseline, .7, 'test')
    assert result == repeat
