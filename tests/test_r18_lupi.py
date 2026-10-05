import torch

from gcnet_missing_m3.r18_lupi import PrivilegedNoise


def test_eval_skips_privileged_branch_and_returns_identity():
    module = PrivilegedNoise(5, 3).eval()
    hidden = torch.randn(2, 2, 3)
    valid = torch.tensor([[True, False], [True, True]])
    hidden[~valid] = float("nan")
    def fail_if_called(*args):
        raise AssertionError("evaluation entered the privileged branch")
    hook = module.variance_head.register_forward_pre_hook(fail_if_called)
    try:
        for privileged in (None, object(), torch.full((2, 2, 5), float("nan"))):
            output, penalty = module(hidden, privileged, valid)
            torch.testing.assert_close(output[valid], hidden[valid])
            assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))
            assert penalty.item() == 0
    finally:
        hook.remove()


def test_training_formula_padding_and_gradients():
    torch.manual_seed(3)
    module = PrivilegedNoise(5, 3).train()
    with torch.no_grad():
        for layer in (module.variance_head[0], module.variance_head[2]):
            layer.weight.fill_(0.01)
            layer.bias.fill_(0.2)
    hidden = torch.ones(2, 2, 3, requires_grad=True)
    privileged = torch.ones(2, 2, 5)
    valid = torch.tensor([[True, False], [True, True]])
    privileged[~valid] = float("nan")
    with torch.no_grad():
        sigma = (0.5 * module.variance_head(privileged[valid])).exp()
    torch.manual_seed(7)
    expected = hidden[valid].detach() * (1 + torch.randn_like(sigma) * sigma)
    torch.manual_seed(7)
    output, penalty = module(hidden, privileged, valid)
    torch.testing.assert_close(output[valid], expected)
    torch.testing.assert_close(penalty, sigma.square().mean().sqrt())
    assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))
    (output.square().mean() + 0.001 * penalty).backward()
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad[valid].abs().sum() > 0
    assert hidden.grad[~valid].abs().sum() == 0
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_all_padding_is_finite_and_has_no_privileged_forward():
    module = PrivilegedNoise(5, 3).train()
    hidden = torch.full((2, 3), float("nan"), requires_grad=True)
    privileged = torch.full((2, 5), float("nan"))
    output, penalty = module(hidden, privileged, torch.zeros(2, dtype=torch.bool))
    assert torch.equal(output, torch.zeros_like(output))
    assert penalty.item() == 0
    (output.sum() + penalty).backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden))
