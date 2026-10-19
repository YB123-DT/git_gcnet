"""MOSI routing preserves the original scalar regression task and head."""

import io

import pytest
import torch
from torch.nn import functional as F

from experiments.osram_text_path_router_20261019.router import TextPathRouter


def router_class():
    from experiments.osram_text_path_router_joint_mosi_20261019.router import RegressionTextPathRouter
    return RegressionTextPathRouter


def inputs(dtype=torch.float32, requires_grad=False):
    generator = torch.Generator().manual_seed(93)
    values = [torch.randn(5, 256, generator=generator, dtype=dtype),
              torch.randn(5, 512, generator=generator, dtype=dtype),
              torch.randn(5, 512, generator=generator, dtype=dtype),
              torch.tensor([[3., .5, 2., 1.]], dtype=dtype).repeat(5, 1),
              torch.tensor([[[-2.], [1.], [4.]], [[-.8], [-1.2], [.1]],
                            [[2.], [.5], [-1.]], [[0.], [0.], [0.]],
                            [[-3.], [-2.], [-1.]]], dtype=dtype)]
    return [value.requires_grad_(requires_grad) for value in values]


def test_uniform_scalar_mixture_is_not_single_class_softmax():
    router = router_class()()
    values = inputs()
    prediction, weights = router(*values, torch.ones(5, dtype=torch.bool))
    assert prediction.shape == (5, 1)
    torch.testing.assert_close(prediction, values[-1].mean(1))
    assert torch.equal(weights, torch.full_like(weights, 1 / 3))
    assert prediction[0, 0] == 1
    assert prediction[-1, 0] == -2
    assert not torch.equal(prediction, values[-1].softmax(-1).mean(1))


def test_scalar_bypass_is_exact_full_candidate_and_dtype_is_preserved():
    router = router_class()().double()
    values = inputs(torch.float64)
    eligible = torch.tensor([True, False, False, True, False])
    prediction, weights = router(*values, eligible)
    assert prediction.dtype == weights.dtype == torch.float64
    assert torch.equal(prediction[~eligible], values[-1][~eligible, 0])
    assert torch.equal(weights[~eligible], torch.tensor([[1., 0., 0.]], dtype=torch.float64).expand(3, -1))


def test_features_and_candidates_have_joint_gradients_after_few_updates():
    router = router_class()()
    values = inputs(requires_grad=True)
    optimizer = torch.optim.Adam(router.parameters(), lr=.001)
    previous = {name: value.detach().clone() for name, value in router.named_parameters()}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        for value in values:
            value.grad = None
        prediction, _ = router(*values, torch.ones(5, dtype=torch.bool))
        F.mse_loss(prediction, torch.full_like(prediction, .7)).backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in router.parameters())
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in values)
        assert values[-1].grad.abs().sum() > 0
        optimizer.step()
    assert router.local_projection.weight.grad.abs().sum() > 0
    assert router.base_projection.weight.grad.abs().sum() > 0
    assert router.gap_projection.weight.grad.abs().sum() > 0
    assert any(not torch.equal(previous[name], value) for name, value in router.named_parameters())


def test_no_eligible_rows_still_train_the_original_scalar_head():
    router = router_class()()
    scalar_head = torch.nn.Linear(4, 1)
    hidden = torch.randn(5, 4)
    full = scalar_head(hidden)
    values = inputs()
    values[-1] = torch.stack((full, full + 1, full - 1), 1)
    original = scalar_head.weight.detach().clone()
    optimizer = torch.optim.Adam(list(scalar_head.parameters()) + list(router.parameters()), lr=.01)
    prediction, weights = router(*values, torch.zeros(5, dtype=torch.bool))
    assert torch.equal(prediction, full)
    F.mse_loss(prediction, torch.zeros_like(prediction)).backward()
    assert torch.isfinite(scalar_head.weight.grad).all()
    assert scalar_head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None or parameter.grad.count_nonzero() == 0
               for parameter in router.parameters())
    optimizer.step()
    assert not torch.equal(scalar_head.weight, original)
    assert torch.equal(weights[:, 0], torch.ones(5))


def test_zero_evidence_and_unseen_history_are_numerically_safe():
    router = router_class()()
    values = inputs(requires_grad=True)
    with torch.no_grad():
        values[1].zero_()
        values[2].zero_()
        values[3].zero_()
        values[3][:, 2] = -1
    prediction, weights = router(*values, torch.ones(5, dtype=torch.bool))
    prediction.square().mean().backward()
    assert torch.isfinite(prediction).all() and torch.isfinite(weights).all()
    assert all(torch.isfinite(value.grad).all() for value in values)


def test_default_reuses_classification_network_parameters_without_advancing_rng():
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    router = router_class()()
    assert not router.detach_inputs
    assert torch.equal(torch.random.get_rng_state(), before)
    classification = TextPathRouter(detach_inputs=False)
    assert sum(parameter.numel() for parameter in router.parameters()) == 95439
    assert router.state_dict().keys() == classification.state_dict().keys()
    assert all(torch.equal(value, classification.state_dict()[name])
               for name, value in router.state_dict().items())
    router(*inputs(), torch.ones(5, dtype=torch.bool))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_checkpoint_restores_nonuniform_scalar_mixture():
    router = router_class()()
    with torch.no_grad():
        router.network[-1].bias.copy_(torch.tensor([-2., 1., .2]))
    values = inputs()
    eligible = torch.ones(5, dtype=torch.bool)
    expected, weights = router(*values, eligible)
    torch.testing.assert_close(expected, (values[-1] * weights[..., None]).sum(1))
    stream = io.BytesIO()
    torch.save(router.state_dict(), stream)
    stream.seek(0)
    restored = router_class()()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    actual, actual_weights = restored(*values, eligible)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_weights, weights)


@pytest.mark.parametrize('position', range(6))
def test_rejects_misaligned_input_rows_or_nonscalar_candidates(position):
    router = router_class()()
    values = inputs() + [torch.ones(5, dtype=torch.bool)]
    if position == 4:
        values[position] = torch.zeros(5, 3, 2)
    else:
        values[position] = values[position][:-1]
    with pytest.raises(ValueError, match='scalar|aligned'):
        router(*values)


def test_rejects_nonboolean_eligibility():
    router = router_class()()
    with pytest.raises(ValueError, match='scalar|aligned'):
        router(*inputs(), torch.ones(5))


def test_empty_rows_have_well_defined_scalar_shapes():
    router = router_class()()
    prediction, weights = router(*(value[:0] for value in inputs()), torch.zeros(0, dtype=torch.bool))
    assert prediction.shape == (0, 1)
    assert weights.shape == (0, 3)
