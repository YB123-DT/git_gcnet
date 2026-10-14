"""Selection artifacts must come from the evaluated epoch without another draw."""
from types import SimpleNamespace
import json

import numpy as np
import torch

from tests.test_meld_query_decision_trace import fixture
from gcnet_modality_jepa.gcnet_legacy_random_mask import GCNetLegacyRandomMaskSchedule
from gcnet_modality_jepa.gcnet_legacy_random_mask import random_mask
from gcnet_modality_jepa.train_gcnet import random_mask as original_random_mask
import pytest
from gcnet_missing_m3 import train_gcnet as trainer


def test_collection_keeps_sampler_and_model_rng_identical():
    model, raw, *_ = fixture()
    schedule = GCNetLegacyRandomMaskSchedule('MELD', 'test', 1, .4, 66)
    results = []
    for collect in (False, True):
        np.random.seed(91)
        torch.manual_seed(75)
        metrics, artifacts = trainer.evaluate_rate(model, [raw], schedule, 'MELD', (4, 4, 4),
            torch.device('cpu'), collect=collect, collect_identity=collect)
        results.append((metrics, artifacts, np.random.get_state(), torch.get_rng_state()))
    assert results[0][0]['accuracy'] == results[1][0]['accuracy']
    assert np.array_equal(results[0][2][1], results[1][2][1])
    assert results[0][2][2:] == results[1][2][2:]
    assert torch.equal(results[0][3], results[1][3])
    a = results[1][1]
    assert a['conversation_ids'].tolist() == ['a'] * 4 + ['b'] * 2
    assert a['utterance_indices'].tolist() == [0, 1, 2, 3, 0, 1]
    assert a['speakers'].tolist() == [0, 1, 0, 1, 1, 0]
    assert a['logits'].shape == (6, 7)
    np.testing.assert_array_equal(a['logits'].argmax(1), a['predictions'])


def test_default_disabled_and_selected_artifact_identity(tmp_path):
    assert trainer.TrainConfig().preserve_selection_artifacts is False
    artifact = dict(labels=np.arange(7), predictions=np.arange(7),
                    availability=np.ones((7, 3), np.float32), logits=np.eye(7, dtype=np.float32))
    reference = trainer._save_selection_artifacts(tmp_path, '0.7', 4, artifact)
    path = tmp_path / reference['path']
    with np.load(path) as saved:
        assert saved['selected_epoch'].item() == 4
        np.testing.assert_array_equal(saved['predictions'], artifact['predictions'])
    assert reference['sha256']


@pytest.mark.parametrize('rate', [i / 10 for i in range(8)])
def test_sampler_bitwise_equivalent_to_original_gcnet(rate):
    np.random.seed(66)
    expected = original_random_mask(3, 97, rate)
    old_state = np.random.get_state()
    np.random.seed(66)
    actual = random_mask(3, 97, rate)
    new_state = np.random.get_state()
    np.testing.assert_array_equal(actual, expected)
    assert np.array_equal(old_state[1], new_state[1]) and old_state[2:] == new_state[2:]
    assert np.all(actual.sum(-1) >= 1)
    if rate == 0:
        assert np.all(actual == 1)
    if rate == .7:
        assert np.all(actual.sum(-1) == 1)


def test_padded_rectangle_independent_sides_and_original_speaker_selector():
    model, raw, *_ = fixture()
    np.random.seed(67)
    expected_sides = [original_random_mask(3, 8, .5).reshape(2, 4, 3) for _ in range(2)]
    expected = np.where(raw[6].numpy()[..., None] == 0, expected_sides[0], expected_sides[1])
    np.random.seed(67)
    schedule = GCNetLegacyRandomMaskSchedule('MELD', 'train', 1, .5, 66)
    view = trainer._prepare_view(raw, schedule, epoch=0, dimensions=(4, 4, 4))
    valid = raw[7].bool().numpy()
    np.testing.assert_array_equal(view['availability'].transpose(0, 1).numpy()[valid], expected[valid])
    assert not np.array_equal(expected_sides[0], expected_sides[1])
    assert not view['availability'][~raw[7].T.bool()].any()
