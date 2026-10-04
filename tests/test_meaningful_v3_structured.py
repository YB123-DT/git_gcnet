import pytest
import torch

from gcnet_missing_m3.meaningful_v3_structured import build


METHODS = ['survey80_struct_nbfnet', 'survey80_struct_neural_lp', 'survey80_struct_qre_game']


@pytest.mark.parametrize('method', METHODS)
def test_raw_shapes_zero_init_and_inactive_nan(method):
    torch.manual_seed(8)
    model = build(method)
    local = torch.randn(3, 256)
    evidence = torch.randn(3, 4, 512)
    active = torch.tensor([[True, True, True, True], [True, False, True, False], [False]*4])
    poisoned = torch.where(active[..., None], evidence, torch.full_like(evidence, float('nan')))
    actual_local, actual_evidence = model(local, poisoned, active, torch.ones(3, 3))
    torch.testing.assert_close(actual_local, local)
    torch.testing.assert_close(actual_evidence, torch.where(active[..., None], evidence, torch.zeros_like(evidence)))
    (actual_local.square().mean() + actual_evidence.square().mean()).backward()
    assert model.decoders[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('method', METHODS)
def test_open_residual_core_gradients_and_mask_invariance(method):
    torch.manual_seed(12)
    model = build(method, latent_dim=12, num_heads=2, value_dim=8)
    for decoder in model.decoders:
        torch.nn.init.normal_(decoder.weight, std=0.03)
    local = torch.randn(2, 12, requires_grad=True)
    evidence = torch.randn(2, 4, 16, requires_grad=True)
    active = torch.tensor([[True, False, True, False], [False]*4])
    availability = torch.zeros(2, 3)
    a = model(local, evidence, active, availability)
    poisoned = torch.where(active[..., None], evidence, torch.full_like(evidence, float('nan')))
    b = model(local, poisoned, active, availability)
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second)
        assert torch.isfinite(first).all()
    sum(t.square().mean() for t in a).backward()
    assert torch.count_nonzero(evidence.grad[~active]) == 0
    core_grads = [p.grad for p in model.core.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in core_grads)
    assert sum(g.abs().sum() for g in core_grads) > 0


def test_qre_implicit_derivative_and_saddle():
    from gcnet_missing_m3.meaningful_v3_structured import _EntropyGame
    torch.manual_seed(21)
    cost = (4 * torch.rand(2, 3, 3, dtype=torch.double)).requires_grad_()
    u, v = _EntropyGame.apply(cost)
    torch.testing.assert_close(u, (-torch.einsum('bij,bj->bi', cost, v)).softmax(-1))
    torch.testing.assert_close(v, torch.einsum('bij,bi->bj', cost, u).softmax(-1))
    assert torch.autograd.gradcheck(_EntropyGame.apply, (cost,))
