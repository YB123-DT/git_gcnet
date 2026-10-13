import torch

from experiments.osram_meld_flat_history_scale_20261013.evaluate import (
    scale_flat_history,
)


def test_scale_flat_history_changes_only_base_and_gap_slots():
    flat_input = torch.tensor(
        [[[1.0, 2.0, 10.0, 20.0, 30.0, 40.0]],
         [[3.0, 4.0, 11.0, 21.0, 31.0, 41.0]]]
    )
    umask = torch.tensor([[1.0, 0.0]])
    actual = scale_flat_history(
        flat_input, umask, local_dim=2, context_dim=1, alpha=1.5
    )
    torch.testing.assert_close(
        actual[0, 0], torch.tensor([1.0, 2.0, 15.0, 30.0, 45.0, 60.0])
    )
    assert torch.count_nonzero(actual[1]) == 0


def test_alpha_one_preserves_valid_flat_input_exactly():
    torch.manual_seed(5)
    flat_input = torch.randn(3, 2, 10)
    umask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    actual = scale_flat_history(
        flat_input, umask, local_dim=2, context_dim=2, alpha=1.0
    )
    valid = umask.T.bool().unsqueeze(-1)
    expected = torch.where(valid, flat_input, torch.zeros_like(flat_input))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
