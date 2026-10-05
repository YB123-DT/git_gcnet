import importlib
import math

import pytest
import torch


def _loss():
    return importlib.import_module('gcnet_missing_m3.r12_rnc').rnc_loss


def _reference(features, labels, temperature=2.0):
    terms = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            distance = (labels[i] - labels[j]).abs()
            candidates = [
                -(features[i] - features[k]).norm() / temperature
                for k in range(len(labels))
                if k != i and (labels[i] - labels[k]).abs() >= distance
            ]
            terms.append(
                (features[i] - features[j]).norm() / temperature
                + torch.logsumexp(torch.stack(candidates), dim=0)
            )
    return torch.stack(terms).mean()


def test_matches_reference_values_and_gradients_with_distance_ties():
    features = torch.tensor([[1., 2.], [0., 3.], [-2., 1.], [4., 0.]],
                            dtype=torch.double, requires_grad=True)
    labels = torch.tensor([0., 1., -1., 0.], dtype=torch.double)
    actual = _loss()(features, labels)
    expected = _reference(features, labels)
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual, features, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, features)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()


@pytest.mark.parametrize('count', [0, 1])
def test_small_batches_produce_differentiable_zero(count):
    features = torch.randn(count, 3, requires_grad=True)
    loss = _loss()(features, torch.zeros(count, 1))
    assert loss.item() == 0
    loss.backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad) == 0


def test_self_exclusion_and_equal_label_ties_have_known_values():
    features = torch.zeros(4, 3, requires_grad=True)
    torch.testing.assert_close(_loss()(features, torch.zeros(4)),
                               torch.tensor(math.log(3)))
    loss = _loss()(features[:2], torch.tensor([0., 3.]))
    assert loss.item() == 0


def test_large_distances_and_permutations_are_finite_and_equivalent():
    features = torch.tensor([[0., 0.], [1.e5, 0.], [-1.e5, 1.e5], [0., 2.e5]],
                            requires_grad=True)
    labels = torch.tensor([0., 1., -1., 2.])
    loss = _loss()(features, labels)
    order = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(loss, _loss()(features[order], labels[order, None]))
    torch.testing.assert_close(loss, _reference(features, labels))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf')])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match='temperature'):
        _loss()(torch.zeros(2, 3), torch.zeros(2), temperature)
