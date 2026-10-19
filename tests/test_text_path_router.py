"""Tests of frozen three-candidate probability routing (not a new backbone)."""

import io

import pytest
import torch
from torch.nn import functional as F

from experiments.osram_text_path_router_20261019.router import TextPathRouter


def inputs(classes=7, requires_grad=False):
    generator = torch.Generator().manual_seed(81)
    values = [
        torch.randn(5, 256, generator=generator),
        torch.randn(5, 512, generator=generator),
        torch.randn(5, 512, generator=generator),
        torch.tensor([[3., .5, 2., 1.]]).repeat(5, 1),
        torch.randn(5, 3, classes, generator=generator),
    ]
    return [value.requires_grad_(requires_grad) for value in values]


def test_initial_weights_are_uniform_probability_mixture_not_logit_average():
    router = TextPathRouter()
    values = inputs()
    log_probs, weights = router(*values, torch.ones(5, dtype=torch.bool))
    assert torch.equal(weights, torch.full_like(weights, 1 / 3))
    expected = values[-1].softmax(-1).mean(1)
    torch.testing.assert_close(log_probs.exp(), expected)
    assert not torch.allclose(log_probs.exp(), values[-1].mean(1).softmax(-1))


def test_bypass_is_exact_original_logsoftmax_and_full_only_weights():
    router = TextPathRouter()
    values = inputs()
    eligible = torch.tensor([True, False, True, False, False])
    log_probs, weights = router(*values, eligible)
    assert torch.equal(log_probs[~eligible], F.log_softmax(values[-1][:, 0], -1)[~eligible])
    assert torch.equal(weights[~eligible], torch.tensor([[1., 0., 0.]]).expand(3, -1))


@pytest.mark.parametrize('classes', [6, 7])
def test_large_logits_and_zero_evidence_produce_finite_probabilities(classes):
    router = TextPathRouter()
    values = inputs(classes)
    values[1].zero_()
    values[2].zero_()
    values[3][:, 2] = -1
    values[-1] *= 1e5
    log_probs, weights = router(*values, torch.ones(5, dtype=torch.bool))
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(5))


def test_only_router_receives_gradients_and_parameters_update():
    router = TextPathRouter()
    values = inputs(requires_grad=True)
    previous = {name: value.clone() for name, value in router.named_parameters()}
    optimizer = torch.optim.Adam(router.parameters(), lr=.001)
    labels = torch.arange(5)
    for _ in range(3):
        optimizer.zero_grad()
        log_probs, _ = router(*values, torch.ones(5, dtype=torch.bool))
        F.nll_loss(log_probs, labels).backward()
        assert all(value.grad is None for value in values)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in router.parameters())
        optimizer.step()
    assert any(not torch.equal(previous[name], value) for name, value in router.named_parameters())
    assert router.local_projection.weight.grad.abs().sum() > 0
    assert router.base_projection.weight.grad.abs().sum() > 0
    assert router.gap_projection.weight.grad.abs().sum() > 0


def test_empty_eligibility_is_exact_bypass_and_allows_zero_training_gradient():
    router = TextPathRouter()
    values = inputs()
    log_probs, weights = router(*values, torch.zeros(5, dtype=torch.bool))
    assert torch.equal(log_probs, F.log_softmax(values[-1][:, 0], -1))
    F.nll_loss(log_probs, torch.arange(5)).backward()
    assert all(parameter.grad is None or parameter.grad.count_nonzero() == 0
               for parameter in router.parameters())
    assert torch.equal(weights[:, 0], torch.ones(5))


def test_checkpoint_round_trip_preserves_candidate_order_and_output():
    router = TextPathRouter()
    with torch.no_grad():
        router.network[-1].bias.copy_(torch.tensor([-3., 2., .25]))
    values = inputs()
    eligible = torch.ones(5, dtype=torch.bool)
    expected, weights = router(*values, eligible)
    torch.testing.assert_close(weights, torch.tensor([-3., 2., .25]).softmax(0).expand(5, -1))
    torch.testing.assert_close(expected.exp(), (values[-1].softmax(-1) * weights[..., None]).sum(1))
    stream = io.BytesIO()
    torch.save(router.state_dict(), stream)
    stream.seek(0)
    restored = TextPathRouter()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    actual, actual_weights = restored(*values, eligible)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_weights, weights)


def test_initialization_and_forward_do_not_advance_global_rng():
    torch.manual_seed(29)
    before = torch.random.get_rng_state().clone()
    router = TextPathRouter()
    assert torch.equal(torch.random.get_rng_state(), before)
    values = inputs()
    router(*values, torch.ones(5, dtype=torch.bool))
    assert torch.equal(torch.random.get_rng_state(), before)
