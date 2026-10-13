import torch

from experiments.osram_iemocap_flat_source_combinations_20261013.evaluate import (
    authoritative_combination_prediction,
    flatten_metric_availability,
)


def test_flatten_metric_availability_matches_batch_major_metric_order():
    availability = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 1, 0]],
            [[1, 0, 1], [0, 1, 1]],
        ],
        dtype=torch.float32,
    )
    umask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)

    actual = flatten_metric_availability(availability, umask)

    expected = torch.tensor(
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32
    )
    torch.testing.assert_close(actual, expected)


def test_original_forward_is_authoritative_for_full_combination():
    replayed = torch.tensor([1, 2, 3]).numpy()
    original = torch.tensor([1, 4, 3]).numpy()

    assert authoritative_combination_prediction("LB", replayed, original) is replayed
    assert authoritative_combination_prediction("LBG", replayed, original) is original
