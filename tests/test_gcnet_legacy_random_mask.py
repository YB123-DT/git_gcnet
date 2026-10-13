import numpy as np
import pytest
import torch

from gcnet_modality_jepa.gcnet_legacy_random_mask import (
    GCNetLegacyRandomMaskSchedule,
    random_mask,
)
from gcnet_modality_jepa.train_gcnet import build_primary_mask_tensors


def test_rate_point_seven_is_exactly_one_observed_modality():
    matrix = random_mask(3, 97, 0.7, np.random.RandomState(7))
    assert np.all(matrix.sum(axis=1) == 1)


@pytest.mark.parametrize("rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
def test_intermediate_rates_match_gcnet_global_tolerance(rate):
    matrix = random_mask(3, 128, rate, np.random.RandomState(11))
    assert np.all(matrix.sum(axis=1) >= 1)
    assert abs(matrix.mean() - (1.0 - rate)) < 0.005


def test_batch_adapter_draws_independent_host_and_guest_masks():
    np.random.seed(66)
    schedule = GCNetLegacyRandomMaskSchedule("MELD", "train", 1, 0.5, 123)
    umask = torch.ones(4, 12)
    host, guest = build_primary_mask_tensors(schedule, ["a", "b", "c", "d"], umask, 0)
    assert host.shape == guest.shape == (12, 4, 3)
    assert torch.all(host.sum(dim=-1) >= 1)
    assert torch.all(guest.sum(dim=-1) >= 1)
    assert not torch.equal(host, guest)


def test_batch_adapter_zeros_only_padding_rows():
    np.random.seed(67)
    schedule = GCNetLegacyRandomMaskSchedule("MELD", "train", 1, 0.7, 123)
    umask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    host, guest = build_primary_mask_tensors(schedule, ["a", "b"], umask, 0)
    valid = umask.T.bool()
    assert torch.all(host[valid].sum(dim=-1) == 1)
    assert torch.all(guest[valid].sum(dim=-1) == 1)
    assert torch.count_nonzero(host[~valid]) == 0
    assert torch.count_nonzero(guest[~valid]) == 0
