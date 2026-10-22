"""Single-factor V3 connection pairs preserve initialization and masking."""
import copy

import pytest
import torch

from gcnet_missing_m3.source_role_v3 import HierarchicalSourceRoleV3
from gcnet_missing_m3.source_role_v3_paired import PairedV3Block


VARIANTS = ('topology-full', 'topology-diagonal', 'local-adapter',
            'local-shared', 'task-pre-norm', 'task-post-norm')


def make(variant):
    return PairedV3Block(local_dim=5, forward_dim=7, width=8,
                         pair_hidden=16, output_dim=11, variant=variant)


def inputs():
    trajectory = torch.ones(4, 2, 3, 4)
    trajectory[..., 0] = 3
    trajectory[..., 1] = .5
    trajectory[..., 2] = 2
    return [torch.randn(4, 2, 5), torch.randn(4, 2, 3, 7),
            torch.randn(4, 2, 3, 3, 7), trajectory,
            torch.tensor([1., 0., 0.]).expand(4, 2, 3).clone(), torch.ones(2, 4)]


@pytest.mark.parametrize('left,right', [VARIANTS[:2], VARIANTS[2:4], VARIANTS[4:]])
def test_pair_initialization_parameters_and_rng_are_identical(left, right):
    torch.manual_seed(42)
    a = make(left)
    state = torch.get_rng_state()
    torch.manual_seed(42)
    b = make(right)
    assert torch.equal(state, torch.get_rng_state())
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_full_topology_exactly_matches_old_core_and_gradients():
    torch.manual_seed(43)
    old = HierarchicalSourceRoleV3(local_dim=5, forward_dim=7, width=8, pair_hidden=16)
    state = torch.get_rng_state()
    torch.manual_seed(43)
    new = make('topology-full')
    assert torch.equal(state, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    with torch.no_grad():
        for decoder in old.decoders:
            decoder.weight.normal_(); decoder.bias.normal_()
    new.load_state_dict(old.state_dict(), strict=True)
    args = inputs()
    a, b = old(*args), new(*args)
    assert torch.equal(a, b)
    a.square().mean().backward(); b.square().mean().backward()
    for name, p in old.named_parameters():
        assert torch.equal(p.grad, new.get_parameter(name).grad), name


def test_diagonal_attention_is_source_isolated_and_gradients_finite():
    model = make('topology-diagonal')
    tokens = torch.randn(1, 1, 4, 3, 8, requires_grad=True)
    active = torch.ones(1, 1, 4, 3, dtype=torch.bool)
    active[:, :, 2, 2] = False  # inactive query inside a multi-source row
    output = model._mix_sources(tokens, active)
    changed = tokens.detach().clone()
    changed[:, :, :, 1] += 7
    actual = model._mix_sources(changed, active)
    assert torch.equal(output[:, :, :, [0, 2]], actual[:, :, :, [0, 2]])
    assert not torch.equal(output[:, :, :, 1], actual[:, :, :, 1])
    (output * torch.randn_like(output)).sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    grad = model.source_attention.in_proj_weight.grad
    assert grad[:16].abs().max() < 1e-7  # singleton softmax has no Q/K derivative
    assert grad[16:].abs().sum() > 0
    assert model.source_attention.out_proj.weight.grad.abs().sum() > 0


def test_diagonal_retains_original_zero_and_single_source_rules():
    model = make('topology-diagonal')
    tokens = torch.randn(2, 1, 4, 3, 8)
    active = torch.zeros(2, 1, 4, 3, dtype=torch.bool)
    active[0, :, :, 1] = True
    result = model._mix_sources(tokens, active)
    assert torch.equal(result[0, :, :, 1], tokens[0, :, :, 1])
    assert not result[1].any()
    assert not result[0, :, :, [0, 2]].any()


@pytest.mark.parametrize('variant', VARIANTS)
def test_zero_start_masks_nan_safety_training_and_strict_restore(variant):
    model = make(variant)
    args = inputs()
    args[3][0, ..., 3] = 0  # no history
    args[4][1] = 1  # all currently visible
    args[5][1, 3] = 0  # padding
    args[3][:, :, 2, 3] = 0  # historical source V never seen
    out = model(*args)
    assert not out.any()
    target = torch.randn_like(out)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(4):
        optimizer.zero_grad()
        (model(*args) - target).square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model.memory_projection.weight.grad.abs().sum() > 0
    expected = model(*args)
    assert not expected[0].any() and not expected[1].any() and not expected[3, 1].any()
    if variant.startswith('topology'):
        assert not expected[:, :, 1].any()
    args[1][:, :, 2] = float('nan')
    args[2][:, :, :, 2] = float('nan')
    args[2][:, :, 0] = float('nan')
    assert torch.equal(expected, model(*args))
    restored = make(variant)
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    assert torch.equal(expected, restored(*args))
