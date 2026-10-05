"""Numerical primal/dual checks for the C20 CAGrad shared direction."""

import numpy as np
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scipy.optimize import minimize

from gcnet_missing_m3.core20_cagrad import conflict_averse_direction


def test_conflicting_gradients_match_independent_primal_optimum():
    # Independently solve the original max-min ball problem (paper Eq. 3),
    # rather than rebuilding the implementation's dual objective.
    grads = torch.tensor([[2., -1., .2], [.1, 1., -.8]], dtype=torch.float64)
    direction, diagnostics = conflict_averse_direction(grads, rescale=0)
    g = grads.numpy()
    mean = g.mean(1)
    radius = .4 * np.linalg.norm(mean)
    result = minimize(
        lambda z: -z[-1], np.r_[mean, np.min(g.T @ mean)],
        method="SLSQP", constraints=[
            {"type": "ineq", "fun": lambda z: g.T @ z[:-1] - z[-1]},
            {"type": "ineq", "fun": lambda z: radius ** 2 - np.sum((z[:-1] - mean) ** 2)},
        ], options={"ftol": 1e-12, "maxiter": 500},
    )
    assert result.success
    assert abs(np.min(g.T @ direction.numpy()) - result.x[-1]) < 2e-7
    assert np.linalg.norm(direction.numpy() - mean) <= radius + 1e-10
    assert diagnostics["normalized_dual_objective"] <= diagnostics["normalized_uniform_objective"] + 1e-10
    assert diagnostics["simplex_residual"] < 1e-8
    assert diagnostics["kkt_residual"] < 1e-6
    assert diagnostics["normalized_duality_gap"] < 1e-6


def _check_aligned_and_alpha_zero_limits(rescale):
    grads = torch.tensor([[1., 2., 3.], [2., 4., 6.]], dtype=torch.float64)
    direction, _ = conflict_averse_direction(grads, alpha=0, rescale=rescale)
    torch.testing.assert_close(direction, grads.mean(1))
    direction, diagnostics = conflict_averse_direction(grads, rescale=rescale)
    divisor = (1., 1.16, 1.4)[rescale]
    torch.testing.assert_close(direction, 1.4 * grads.mean(1) / divisor)
    assert diagnostics["status"] == "converged"


def _check_zero_mean_is_exact_finite_closed_form(grads):
    before = grads.clone()
    direction, diagnostics = conflict_averse_direction(grads)
    assert torch.equal(direction, grads.mean(1))
    assert torch.equal(grads, before)
    assert diagnostics["status"] == "mean_zero"


def test_scale_and_task_permutation_invariance_and_no_retained_graph():
    grads = torch.tensor([[2., -1., .2], [.1, 1., -.8]], dtype=torch.float64, requires_grad=True)
    direction, _ = conflict_averse_direction(grads)
    assert not direction.requires_grad
    for scale in (1e-15, 1e15):
        actual, _ = conflict_averse_direction(grads[:, [2, 0, 1]] * scale)
        torch.testing.assert_close(actual / scale, direction, atol=2e-7, rtol=2e-7)


def test_failure_is_rejected_and_nonfinite_inputs_are_rejected():
    grads = torch.tensor([[2., -1., .2], [.1, 1., -.8]])
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "solve failed"):
        conflict_averse_direction(grads, maxiter=1)
    with unittest.TestCase().assertRaisesRegex(ValueError, "finite"):
        conflict_averse_direction(torch.tensor([[float("nan"), 1.]]))
    with unittest.TestCase().assertRaisesRegex(ValueError, "alpha"):
        conflict_averse_direction(grads, alpha=1.)


def test_direction_can_be_delivered_to_existing_adam():
    param = torch.nn.Parameter(torch.tensor([.2, -.3]))
    optimizer = torch.optim.Adam([param], lr=.01)
    grads = torch.tensor([[2., -1., .2], [.1, 1., -.8]])
    direction, _ = conflict_averse_direction(grads)
    param.grad = direction.clone()
    before = param.detach().clone()
    torch.nn.utils.clip_grad_norm_([param], 1.)
    optimizer.step()
    assert torch.isfinite(param).all()
    assert not torch.equal(before, param)
    assert optimizer.state[param]["step"].item() == 1


class CAGradTests(unittest.TestCase):
    def test_independent_primal(self):
        test_conflicting_gradients_match_independent_primal_optimum()

    def test_limits(self):
        for rescale in (0, 1, 2):
            with self.subTest(rescale=rescale):
                _check_aligned_and_alpha_zero_limits(rescale)

    def test_zero_mean(self):
        for grads in (torch.zeros(4, 3), torch.tensor([[1., -1.], [2., -2.]])):
            with self.subTest(grads=grads):
                _check_zero_mean_is_exact_finite_closed_form(grads)

    def test_invariance(self):
        test_scale_and_task_permutation_invariance_and_no_retained_graph()

    def test_rejections(self):
        test_failure_is_rejected_and_nonfinite_inputs_are_rejected()

    def test_existing_adam(self):
        test_direction_can_be_delivered_to_existing_adam()

    def test_success_flag_cannot_hide_nonoptimal_solution(self):
        fake = SimpleNamespace(success=True, x=np.full(3, 1/3), nit=1, message="success")
        with patch("gcnet_missing_m3.core20_cagrad.minimize", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "not optimal"):
                conflict_averse_direction(torch.tensor([[2., -1., .2], [.1, 1., -.8]]))

    def test_zero_task_gradient_has_valid_optimum(self):
        grads = torch.tensor([[0., 1.], [0., 2.]])
        direction, diagnostics = conflict_averse_direction(grads)
        self.assertTrue(torch.isfinite(direction).all())
        self.assertEqual(diagnostics["normalized_duality_gap"], 0.)
        self.assertEqual(diagnostics["weights"], [1., 0.])


if __name__ == "__main__":
    unittest.main()
