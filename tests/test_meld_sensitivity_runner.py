"""Contracts that protect paired diagnostic identities and fixed-class metrics."""
import importlib

import numpy as np
import pytest
import torch


def runner():
    try:
        return importlib.import_module('experiments.osram_meld_memory_sensitivity_20261017.run')
    except ModuleNotFoundError:
        pytest.fail('sensitivity runner has not been implemented')


def test_sample_rows_follow_metric_batch_major_order_and_actual_positions():
    rows = runner().sample_rows(['a', 'b'], torch.tensor([[1, 0, 1], [1, 1, 0]]))
    assert rows == [('a', 0), ('a', 2), ('b', 0), ('b', 1)]


def test_metrics_keep_absent_classes_and_paired_changes():
    report = runner().metrics(np.array([0, 0, 1]), np.array([[2., 0., 0, 0, 0, 0, 0],
                             [0., 2., 0, 0, 0, 0, 0], [0., 2., 0, 0, 0, 0, 0]]),
                             np.array([1, 0, 1]))
    assert report['accuracy'] == pytest.approx(2 / 3)
    assert report['unweighted_accuracy'] == pytest.approx(1.5 / 7)
    assert report['corrections'] == 1
    assert report['harms'] == 1
    assert np.isfinite(report['cross_entropy'])


def test_history_bins_use_number_of_prior_valid_utterances():
    assert [runner().history_bin(i) for i in (0, 1, 3, 4, 7, 8, 15, 16)] == [
        'first', '1-3', '1-3', '4-7', '4-7', '8-15', '8-15', '16+']


def test_readout_replay_preserves_skip_and_masks_padding():
    import types
    run = runner()
    osram = types.SimpleNamespace(local_skip=torch.nn.Linear(2, 3),
        emotion_adapter=torch.nn.Linear(6, 3), emotion_norm=torch.nn.LayerNorm(3))
    model = types.SimpleNamespace(osram=osram, smax_fc=torch.nn.Linear(3, 7))
    local, flat = torch.randn(2, 1, 2), torch.randn(2, 1, 6)
    valid = torch.tensor([[True], [False]])
    actual = run.readout_logits(model, local, flat, valid)
    expected = model.smax_fc(torch.where(valid[..., None], osram.emotion_norm(
        osram.local_skip(local) + osram.emotion_adapter(flat)), 0))
    torch.testing.assert_close(actual, expected)
