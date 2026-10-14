from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.osram_meld_information_availability_20261014.run import (
    flatten_batch_cache,
    model_state_hash,
    source_epoch_metrics,
)


def _batch():
    # time=3, batch=2, valid positions: c1[0:3], c2[0:2]
    umask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    view = {
        "umask": umask,
        "qmask": torch.tensor([[0, 1, 0], [2, 2, 0]]),
        "labels": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "availability": torch.tensor(
            [[[1, 1, 1], [1, 0, 1]], [[1, 0, 1], [0, 1, 1]], [[0, 1, 1], [0, 0, 0]]]
        ),
        "conversation_ids": ["c1", "c2"],
    }
    local = torch.arange(3 * 2 * 256, dtype=torch.float32).reshape(3, 2, 256)
    flat = torch.zeros(3, 2, 256 + 4 * 1024)
    flat[..., :256] = local
    flat[..., 256 : 256 + 1024] = 2
    gaps = flat[..., 256 + 1024 :].reshape(3, 2, 3, 1024)
    gaps[:] = 3
    logits = torch.randn(3, 2, 7)
    return view, SimpleNamespace(local=local, flat_input=flat), logits


def test_flatten_cache_keeps_forward_memory_and_temporal_identity():
    view, capture, logits = _batch()
    cache = flatten_batch_cache(view, capture, logits, local_dim=256, context_dim=1024)
    assert cache["local"].shape == (5, 256)
    assert cache["base"].shape == (5, 512)
    assert cache["gap"].shape == (5, 3, 512)
    assert cache["prediction"].shape == (5, 7)
    np.testing.assert_array_equal(cache["conversation_ids"], ["c1", "c1", "c1", "c2", "c2"])
    np.testing.assert_array_equal(cache["utterance_indices"], [0, 1, 2, 0, 1])
    # Current ATV means no active Gap; c1 position1 lacks Text so only Gap-T survives.
    assert np.count_nonzero(cache["gap"][0]) == 0
    assert np.count_nonzero(cache["gap"][1, 1]) == 512
    assert np.count_nonzero(cache["gap"][1, [0, 2]]) == 0


def test_flatten_cache_rejects_flat_local_mismatch():
    view, capture, logits = _batch()
    capture.flat_input[..., 0] += 1
    with pytest.raises(AssertionError):
        flatten_batch_cache(view, capture, logits, local_dim=256, context_dim=1024)


def test_model_state_hash_changes_only_when_parameters_change():
    model = torch.nn.Linear(3, 2)
    first = model_state_hash(model)
    second = model_state_hash(model)
    assert first == second
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert model_state_hash(model) != first


def test_source_epoch_metrics_select_exact_checkpoint_epoch_and_rate():
    history = [
        {"epoch": 1, "test_oracle": {"0.0": {"accuracy": 0.4, "weighted_f1": 0.3}}},
        {"epoch": 2, "test_oracle": {"0.0": {"accuracy": 0.6, "weighted_f1": 0.5}}},
    ]
    assert source_epoch_metrics(history, epoch=2, rate=0.0) == {"accuracy": 0.6, "weighted_f1": 0.5}
    with pytest.raises(ValueError, match="epoch 3"):
        source_epoch_metrics(history, epoch=3, rate=0.0)
