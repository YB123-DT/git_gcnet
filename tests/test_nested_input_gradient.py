"""Analytic checks for frozen activation-gradient comparisons."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


_path = Path(__file__).resolve().parents[1] / 'experiments/osram_nested_input_gradient_20261010/gradient.py'
_spec = importlib.util.spec_from_file_location('nested_input_gradient', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
measure = _module.measure_input_gradients


class Skip(nn.Module):
    def forward(self, x):
        return 3 * x[..., :1]


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.scale * (2 * x[..., :1] + 7 * x[..., 256:257]
                             + 11 * x[..., 1280:1281] + 13 * x[..., 2304:2305]
                             + 17 * x[..., 3328:3329]
                             + x[..., 768:769] + x[..., 1792:1793])


class InputBlock(nn.Module):
    def __init__(self, local_scale=1., memory_scale=1.):
        super().__init__()
        self.local_scale, self.memory_scale = local_scale, memory_scale

    def forward(self, local, base, gap, availability, umask):
        return self.local_scale * local, self.memory_scale * base, self.memory_scale * gap


def model(block=None):
    result = nn.Module()
    result.osram = nn.Module()
    result.osram.local_skip = Skip()
    result.osram.emotion_adapter = Adapter()
    result.osram.emotion_norm = nn.Identity()
    result.osram.meaningful_block = block or InputBlock()
    result.smax_fc = nn.Identity()
    return result


def inputs():
    local = torch.zeros(2, 1, 256)
    base = torch.zeros(2, 1, 1024)
    gap = torch.zeros(2, 1, 3, 1024)
    local[..., 0], base[..., 0], gap[..., 0] = .1, .2, .3
    availability = torch.tensor([[[1., 0., 1.]], [[0., 1., 1.]]])
    return local, base, gap, availability, torch.ones(1, 2), torch.zeros(1, 2)


def test_known_sum_mse_gradient_and_output_jacobian():
    result = measure(model(), *inputs())
    assert torch.allclose(result['prediction'], torch.tensor([[5.8], [5.2]]))
    assert torch.equal(result['jacobian_local'][..., 0], torch.full((2, 1), 5.))
    expected = torch.tensor([[[7., 0., 13., 0.]], [[7., 11., 0., 0.]]])
    assert torch.equal(result['jacobian_memory'][..., 0], expected)
    assert torch.allclose(result['grad_local'][..., 0], 10 * result['prediction'])
    assert torch.allclose(result['grad_memory'][..., 0],
                          2 * result['prediction'][..., None] * expected)
    assert torch.allclose(result['loss_grad_rms_memory'],
                          result['loss_grad_norm_memory'] / 512 ** .5)


def test_inactive_gap_backward_and_padding_cannot_leak():
    values = list(inputs())
    values[4][0, 1] = 0
    for value in values[:4]:
        value[1] = float('nan')
    values[5][0, 1] = float('nan')
    values[1][0, ..., 512:] = float('nan')
    values[2][0, ..., 512:] = float('nan')
    values[2][0, 0, [0, 2]] = float('nan')
    result = measure(model(), *values)
    assert torch.isfinite(result['prediction']).all()
    assert result['prediction'][1].eq(0).all()
    assert result['grad_local'][1].eq(0).all()
    assert result['grad_memory'][~result['active']].eq(0).all()
    assert result['jacobian_memory'][~result['active']].eq(0).all()


def test_duplication_does_not_rescale_per_utterance_gradient():
    values = inputs()
    single = measure(model(), *values)
    doubled = tuple(torch.cat((x, x), dim=1 if i < 4 else 0) for i, x in enumerate(values))
    repeated = measure(model(), *doubled)
    for name in ('grad_local', 'grad_memory', 'jacobian_local', 'jacobian_memory'):
        assert torch.equal(repeated[name][:, :1], single[name])


def test_identity_nested_matches_flat_and_preserves_parameters():
    net = model()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    flat = measure(net, *inputs())
    nested = measure(net, *inputs(), kind='nested')
    for name in ('prediction', 'grad_local', 'grad_memory', 'jacobian_local', 'jacobian_memory'):
        assert torch.equal(flat[name], nested[name])
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in net.parameters())
    assert not net.training


def test_nested_local_skip_uses_original_local():
    result = measure(model(InputBlock(local_scale=4., memory_scale=2.)), *inputs(), kind='nested')
    # The raw Local skip contributes 3, transformed adapter Local contributes 2*4.
    assert result['jacobian_local'][..., 0].eq(11).all()
    assert result['jacobian_memory'][..., 0, 0].eq(14).all()


def test_empty_batch_rejected():
    values = list(inputs())
    values[4].zero_()
    with pytest.raises(ValueError, match='valid utterance'):
        measure(model(), *values)
