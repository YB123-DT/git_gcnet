"""A second memory access differs only by permitted query address space."""
import copy

import pytest
import torch

from gcnet_missing_m3.conditional_reread import ConditionalMemoryReread


def make(mode='free'):
    return ConditionalMemoryReread(5, 2, 6, 4, mode=mode)


def inputs():
    return [torch.randn(3, 5), torch.randn(3, 8),
            torch.tensor([[1., 0., 1.]]).expand(3, 3),
            torch.randn(3, 2, 4, 6), torch.randn(3, 2, 6, 4),
            torch.tensor([[True, False, True, False]]).expand(3, 4),
            torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool)]


def open_output(model):
    with torch.no_grad():
        model.output_projection.weight.copy_(torch.eye(8))


def test_modes_have_identical_initial_parameters_and_rng():
    torch.manual_seed(9)
    a = make('free')
    rng = torch.get_rng_state()
    torch.manual_seed(9)
    b = make('span')
    assert torch.equal(rng, torch.get_rng_state())
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(value, b.state_dict()[key])
               for key, value in a.state_dict().items())
    assert not a.output_projection.weight.any()
    assert not a.output_projection.bias.any()
    for layer in a.query_net:
        if isinstance(layer, torch.nn.Linear):
            assert layer.weight.abs().sum() > 0
            assert not layer.bias.any()


@pytest.mark.parametrize('mode', ['free', 'span'])
def test_zero_initial_output_and_inactive_safety(mode):
    model = make(mode)
    args = inputs()
    assert torch.equal(model(*args), torch.zeros(3, 8))
    open_output(model)
    args[-1][0] = False
    args[-2][1] = False
    for index in (0, 1, 3, 4):
        args[index][0:2] = float('nan')
    output = model(*args)
    assert torch.isfinite(output).all()
    assert not output[:2].any()
    output.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_span_ignores_inactive_gap_columns_even_nan():
    model = make('span')
    open_output(model)
    args = inputs()
    original = model(*args)
    args[4][..., [1, 3]] = float('nan')
    changed = model(*args)
    assert torch.equal(original, changed)
    assert model.last_diagnostics['query_rank_mean'] <= 2


@pytest.mark.parametrize('mode', ['free', 'span'])
def test_all_visible_still_allows_base_reread_without_rng_draws(mode):
    model = make(mode)
    open_output(model)
    args = inputs()
    args[2] = torch.ones_like(args[2])
    args[5] = torch.tensor([[True, False, False, False]]).expand(3, 4)
    before = torch.get_rng_state()
    output = model(*args)
    assert torch.equal(before, torch.get_rng_state())
    assert output.abs().sum() > 0
    assert model.last_diagnostics['active_count'] == 3


@pytest.mark.parametrize('mode', ['free', 'span'])
def test_rank_deficiency_finite_gradient_into_old_query_space(mode):
    model = make(mode)
    open_output(model)
    args = inputs()
    args[4].zero_()
    args[4][..., 0, 0] = 1
    args[4][..., 0, 2] = 1
    args[4].requires_grad_()
    args[3].requires_grad_()
    model(*args).square().sum().backward()
    if mode == 'span':
        assert torch.isfinite(args[4].grad).all()
        assert args[4].grad.abs().sum() > 0
    else:
        assert args[4].grad is None
    assert torch.isfinite(args[3].grad).all()
    assert args[3].grad.abs().sum() > 0
    assert model.last_diagnostics['query_rank_mean'] == 1
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_zero_rank_projection_has_finite_backward():
    model = make('span')
    open_output(model)
    args = inputs()
    args[4].zero_().requires_grad_()
    output = model(*args)
    assert not output.any()
    output.sum().backward()
    assert torch.isfinite(args[4].grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_second_access_distinguishes_states_with_identical_old_reads():
    free = make('free')
    span = make('span')
    with torch.no_grad():
        free.query_net[-1].weight.zero_()
        free.query_net[-1].bias.copy_(torch.tensor([1., 1., 0., 0., 0., 0.] * 2))
    open_output(free)
    span.load_state_dict(free.state_dict())
    args = inputs()
    args[4].zero_()
    args[4][..., 0, 0] = 1
    args[5] = torch.tensor([[True, False, False, False]]).expand(3, 4)
    changed = [value.clone() for value in args]
    changed[3][..., 1] += 3
    assert torch.equal(args[3] @ args[4], changed[3] @ changed[4])
    assert not torch.equal(free(*args), free(*changed))
    assert torch.equal(span(*args), span(*changed))
    assert span.last_diagnostics['query_out_of_span_norm_mean'] < 1e-6


@pytest.mark.parametrize('mode', ['free', 'span'])
def test_updates_open_query_gradient_and_strict_state_restore(mode):
    model = make(mode)
    args = inputs()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    target = torch.randn(3, 8)
    gradient_groups = set()
    for _ in range(3):
        optimizer.zero_grad()
        (model(*args) - target).square().mean().backward()
        for group in model.expected_gradient_groups():
            parameters = [p for name, p in model.named_parameters() if name.startswith(group)]
            if any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters):
                gradient_groups.add(group)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert gradient_groups == set(model.expected_gradient_groups())
    restored = make(mode)
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    assert torch.equal(model(*args), restored(*args))


def test_bad_mode_and_rank_tolerances_rejected():
    with pytest.raises(ValueError):
        make('other')
    with pytest.raises(ValueError):
        ConditionalMemoryReread(5, 2, 6, 4, rank_rtol=-1)
