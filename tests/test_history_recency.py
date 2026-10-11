"""Strict-past modality history metadata for the original Flat readout."""

import importlib

import pytest
import torch


def metadata(availability, umask):
    module = importlib.import_module('gcnet_missing_m3.history_recency')
    return module.strict_past_history_recency(availability, umask)


def test_history_recency_is_strictly_past_and_marks_never_seen():
    availability = torch.tensor([
        [[1, 0, 0]],
        [[0, 1, 0]],
        [[0, 0, 1]],
        [[1, 0, 0]],
        [[0, 0, 0]],
    ])
    actual = metadata(availability, torch.ones(1, 5))[:, 0]
    expected = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 1, 0, 2, 1, 0],
        [1, 1, 1, 3, 2, 1],
        [1, 1, 1, 1, 3, 2],
    ], dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_current_availability_cannot_change_current_metadata():
    prefix = torch.tensor([[[1, 0, 1]], [[0, 1, 0]]])
    left = torch.cat((prefix, torch.tensor([[[1, 1, 1]]])), 0)
    right = torch.cat((prefix, torch.tensor([[[0, 0, 1]]])), 0)
    umask = torch.ones(1, 3)
    torch.testing.assert_close(metadata(left, umask)[2], metadata(right, umask)[2])


def test_padding_is_zero_and_does_not_update_history():
    availability = torch.tensor([
        [[1, 0, 0]],
        [[0, 1, 1]],  # padded and must not enter history
        [[0, 0, 0]],
    ])
    umask = torch.tensor([[1., 0., 1.]])
    actual = metadata(availability, umask)[:, 0]
    torch.testing.assert_close(actual[1], torch.zeros(6))
    torch.testing.assert_close(actual[2], torch.tensor([1., 0., 0., 2., 0., 0.]))


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError):
        metadata(torch.ones(2, 3), torch.ones(1, 2))
    with pytest.raises(ValueError):
        metadata(torch.ones(2, 1, 3), torch.ones(2, 1))

