import torch

from experiments.osram_meld_local_skip_ablation_20261014.evaluate import (
    replay_flat_readout,
)


def test_shortcut_off_removes_only_local_skip_and_keeps_adapter_local():
    torch.manual_seed(3)
    local = torch.randn(3, 2, 4)
    flat_input = torch.randn(3, 2, 4 + 4 * 6)
    flat_input[..., :4] = local
    umask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    local_skip = torch.nn.Linear(4, 8)
    adapter = torch.nn.Sequential(torch.nn.Linear(28, 8), torch.nn.GELU(), torch.nn.Linear(8, 8))
    norm = torch.nn.LayerNorm(8)

    original = replay_flat_readout(
        local_skip, adapter, norm, local, flat_input, umask, keep_local_shortcut=True,
    )
    shortcut_off = replay_flat_readout(
        local_skip, adapter, norm, local, flat_input, umask, keep_local_shortcut=False,
    )

    valid = umask.T.bool().unsqueeze(-1)
    expected_original = norm(local_skip(local) + adapter(flat_input))
    expected_off = norm(adapter(flat_input))
    torch.testing.assert_close(original[valid.expand_as(original)], expected_original[valid.expand_as(original)])
    torch.testing.assert_close(shortcut_off[valid.expand_as(shortcut_off)], expected_off[valid.expand_as(shortcut_off)])
    assert torch.count_nonzero(original[~valid.expand_as(original)]) == 0
    assert torch.count_nonzero(shortcut_off[~valid.expand_as(shortcut_off)]) == 0
    # The adapter still receives the unmodified Local prefix.
    torch.testing.assert_close(flat_input[..., :4], local)


def test_shortcut_off_differs_when_skip_is_nonzero():
    local = torch.ones(1, 1, 2)
    flat_input = torch.cat((local, torch.zeros(1, 1, 8)), dim=-1)
    umask = torch.ones(1, 1)
    skip = torch.nn.Linear(2, 3, bias=False)
    adapter = torch.nn.Linear(10, 3, bias=False)
    norm = torch.nn.LayerNorm(3)
    with torch.no_grad():
        skip.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]))
        adapter.weight.zero_()
    original = replay_flat_readout(skip, adapter, norm, local, flat_input, umask, keep_local_shortcut=True)
    off = replay_flat_readout(skip, adapter, norm, local, flat_input, umask, keep_local_shortcut=False)
    assert not torch.equal(original, off)
