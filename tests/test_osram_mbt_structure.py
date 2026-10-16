"""B3 structural controls preserve its legacy default and original Flat path."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
from gcnet_missing_m3.meaningful_input_representation import (
    _MultimodalBottleneck, _TransformerBlock,
)
from gcnet_missing_m3.train_gcnet import TrainConfig


class LegacyBottleneck(_MultimodalBottleneck):
    def __init__(self):
        nn.Module.__init__(self)
        self.initial_bottleneck = nn.Parameter(torch.empty(1, 4, 128))
        nn.init.normal_(self.initial_bottleneck, std=.02)
        self.private = nn.ModuleList([_TransformerBlock(128) for _ in range(5)])
        self.exchange = nn.ModuleList([
            nn.ModuleList([_TransformerBlock(128) for _ in range(5)]) for _ in range(2)
        ])
        self.final_norm = nn.LayerNorm(128, eps=1e-6)


def test_default_b3_exact_legacy_weights_rng_and_output():
    torch.set_num_threads(1)
    torch.manual_seed(66)
    legacy = LegacyBottleneck()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    actual = _MultimodalBottleneck(bottlenecks=4, rounds=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert actual.state_dict().keys() == legacy.state_dict().keys()
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in legacy.state_dict().items())
    tokens = torch.randn(2, 17, 128)
    columns = torch.tensor([0, *range(1, 9), *range(17, 25)])
    assert torch.equal(actual(tokens, columns, 8), legacy(tokens, columns, 8))


@pytest.mark.parametrize('tokens,rounds', [(4, 2), (2, 2), (8, 2), (4, 1), (4, 3)])
def test_b3_structure_identity_masks_and_delayed_core_gradients(tokens, rounds):
    torch.set_num_threads(1)
    torch.manual_seed(66)
    block = MeaningfulInputAdapter(256, 1024, 1600, 'mbt_role_bottleneck', 8, 64,
                                   osram_mbt_bottlenecks=tokens, osram_mbt_rounds=rounds)
    assert block.core.core.initial_bottleneck.shape == (1, tokens, 128)
    assert len(block.core.core.exchange) == rounds
    local = torch.randn(4, 1, 256)
    base = torch.cat((torch.randn(4, 1, 512), torch.zeros(4, 1, 512)), -1)
    gap = torch.cat((torch.randn(4, 1, 3, 512), torch.zeros(4, 1, 3, 512)), -1)
    av = torch.tensor([[[1., 0., 1.]]] * 4)
    mask = torch.tensor([[1., 1., 1., 0.]])
    valid = mask.T.bool()
    local[~valid] = 0
    base[~valid] = 0
    gap = torch.where((valid[..., None] & ~av.bool())[..., None], gap, 0.)
    dirty = gap.masked_fill(av.bool()[..., None], float('nan'))
    dirty[~valid] = float('nan')
    outputs = block(local, base, dirty, av, mask)
    assert all(torch.equal(a, b) for a, b in zip(outputs, (local, base, gap)))
    opt = torch.optim.Adam(block.parameters(), lr=.001)
    core_gradient = False
    for _ in range(3):
        opt.zero_grad()
        outputs = block(local, base, dirty, av, mask)
        sum(x.square().mean() for x in outputs).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())
        core_gradient |= any(p.grad is not None and bool(p.grad.count_nonzero())
                             for p in block.core.core.parameters())
        opt.step()
    assert core_gradient, 'zero-init decoder must not hide a disconnected MBT core'
    outputs = block(local, base, dirty, av, mask)
    assert all(torch.equal(x[0], original[0]) for x, original in zip(outputs, (local, base, gap)))
    assert all(not x[~valid].count_nonzero() for x in outputs)
    assert not outputs[2][av.bool()].count_nonzero()
    assert not outputs[1][..., 512:].count_nonzero()
    assert not outputs[2][..., 512:].count_nonzero()


def test_config_fields_validate_positive_integers():
    cfg = TrainConfig()
    assert cfg.osram_mbt_bottlenecks == 4 and cfg.osram_mbt_rounds == 2
    for name in ('osram_mbt_bottlenecks', 'osram_mbt_rounds'):
        for invalid in (0, -1, True, 2.5, '2'):
            with pytest.raises(ValueError, match='positive integer'):
                replace(cfg, **{name: invalid})
    replace(cfg, osram_mbt_bottlenecks=8, osram_mbt_rounds=3)


def test_full_model_wiring_flat_off_and_skip_unchanged():
    pytest.importorskip('torch_geometric')
    from tests.test_meaningful_block_integration import config
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    device = torch.device('cpu')
    torch.set_num_threads(1)
    cfg = config()
    original = _build_model(cfg, (3, 4, 5), device).eval()
    rng = torch.get_rng_state().clone()
    off = _build_model(replace(cfg, osram_mbt_bottlenecks=8, osram_mbt_rounds=3), (3, 4, 5), device).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, off.state_dict()[k]) for k, v in original.state_dict().items())
    changed = _build_model(replace(cfg, osram_meaningful_block='mbt_role_bottleneck',
                                  osram_mbt_bottlenecks=2, osram_mbt_rounds=3), (3, 4, 5), device).eval()
    assert len(changed.osram.meaningful_block.core.core.exchange) == 3
    assert changed.osram.meaningful_block.core.core.initial_bottleneck.shape[1] == 2
    seen = {}
    handles = [changed.osram.meaningful_block.register_forward_pre_hook(
        lambda m, args: seen.update(local=args[0].detach().clone())),
        changed.osram.local_skip.register_forward_pre_hook(
        lambda m, args: seen.update(skip=args[0].detach().clone()))]
    umask = torch.tensor([[1., 1., 1.], [1., 0., 0.]])
    av = torch.tensor([[[1., 0., 1.], [1., 1., 1.]]] * 3)
    av[~umask.T.bool()] = 0.
    args = ([torch.randn(3, 2, 12)], av, torch.zeros(2, 3, dtype=torch.long), umask, [3, 1])
    try:
        with torch.no_grad():
            expected = original(*args, predict_missing=False)[0]
            assert torch.equal(expected, off(*args, predict_missing=False)[0])
            assert torch.equal(expected, changed(*args, predict_missing=False)[0])
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(seen['local'], seen['skip'])
