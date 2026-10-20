"""Standalone contracts for bounded coupling of source-specific Base/Gap reads."""

import importlib
from pathlib import Path
import unittest

import torch


class CoupledSourceRoleTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'gcnet_missing_m3/source_role.py'
        self.assertTrue(path.exists(), 'CoupledSourceRole is not implemented')
        self.cls = importlib.import_module('gcnet_missing_m3.source_role').CoupledSourceRole
        torch.manual_seed(9)

    def inputs(self, t=2, b=2, d=5, f=7):
        trajectory = torch.ones(t, b, 3, 4)
        return [torch.randn(t, b, d), torch.randn(t, b, 3, f),
                torch.randn(t, b, 3, 3, f), trajectory,
                torch.zeros(t, b, 3), torch.ones(b, t)]

    def model(self):
        return self.cls(local_dim=5, forward_dim=7, width=6, bound=.25)

    def activate(self, model):
        with torch.no_grad():
            model.net[-1].weight.normal_(std=.1)
            model.net[-1].bias.fill_(.5)

    def test_zero_initialization_shapes_and_parameter_budget(self):
        model = self.model()
        delta, gamma = model(*self.inputs())
        self.assertEqual(delta.shape, (2, 2, 4, 7))
        self.assertEqual(gamma.shape, (2, 2, 4, 3))
        self.assertEqual(torch.count_nonzero(delta).item(), 0)
        self.assertEqual(torch.count_nonzero(gamma).item(), 0)
        full = self.cls()
        self.assertEqual(full.net[0].normalized_shape, (215,))
        self.assertLess(sum(p.numel() for p in full.parameters()), 100_000)

    def test_same_source_partner_is_mean_of_only_missing_targets(self):
        model = self.model()
        args = self.inputs(t=1, b=1)
        args[1][:] = torch.tensor([10., 20., 30.]).view(1, 1, 3, 1)
        args[2][:, :, 0] = float('nan')  # observed target cannot enter partner mean
        args[2][:, :, 1] = torch.tensor([2., 4., 6.]).view(1, 1, 3, 1)
        args[2][:, :, 2] = torch.tensor([6., 8., 10.]).view(1, 1, 3, 1)
        args[4][:] = torch.tensor([1., 0., 0.])
        captured = []
        handle = model.partner_proj.register_forward_pre_hook(
            lambda _, values: captured.append(values[0].detach().clone()))
        model(*args)
        handle.remove()
        expected = torch.tensor([[4., 6., 8.], [0., 0., 0.],
                                 [10., 20., 30.], [10., 20., 30.]])
        torch.testing.assert_close(captured[0][0, 0], expected.unsqueeze(-1).expand(4, 3, 7))

    def test_no_missing_targets_gives_zero_base_partner(self):
        model = self.model()
        self.activate(model)
        args = self.inputs()
        args[4].fill_(1.)
        args[2].fill_(float('nan'))
        captured = []
        handle = model.partner_proj.register_forward_pre_hook(
            lambda _, values: captured.append(values[0].detach().clone()))
        delta, gamma = model(*args)
        handle.remove()
        self.assertEqual(torch.count_nonzero(captured[0]).item(), 0)
        self.assertGreater(delta[:, :, 0].abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(delta[:, :, 1:]).item(), 0)
        self.assertEqual(torch.count_nonzero(gamma[:, :, 1:]).item(), 0)

    def test_constant_gate_gives_bounded_weighted_source_sum(self):
        model = self.model()
        args = self.inputs()
        with torch.no_grad():
            model.net[-1].bias.fill_(1.)
        delta, gamma = model(*args)
        coefficient = .25 * torch.tanh(torch.tensor(1.))
        reads = torch.cat((args[1].unsqueeze(2), args[2]), dim=2)
        torch.testing.assert_close(gamma, torch.full_like(gamma, coefficient))
        torch.testing.assert_close(delta, coefficient * reads.sum(dim=3))
        self.assertLessEqual(gamma.abs().max().item(), .25)

    def test_no_history_and_padding_are_exact_zero_even_with_bias(self):
        model = self.model()
        self.activate(model)
        args = self.inputs()
        args[3][0, :, :, 3] = 0  # strict-past first turn has no seen source
        args[5][1, 1] = 0
        delta, gamma = model(*args)
        self.assertEqual(torch.count_nonzero(delta[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(gamma[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(delta[1, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(gamma[1, 1]).item(), 0)
        self.assertGreater(delta[1, 0].abs().sum().item(), 0)

    def test_unseen_source_never_contributes_even_with_biased_gate(self):
        model = self.model()
        self.activate(model)
        args = self.inputs()
        args[3][:, :, 1, 3] = 0
        delta, gamma = model(*args)
        self.assertEqual(torch.count_nonzero(gamma[:, :, :, 1]).item(), 0)
        args[1][:, :, 1] = float('nan')
        args[2][:, :, :, 1] = float('nan')
        args[3][:, :, 1, :3] = float('nan')
        poisoned_delta, poisoned_gamma = model(*args)
        torch.testing.assert_close(poisoned_delta, delta)
        torch.testing.assert_close(poisoned_gamma, gamma)

    def test_inactive_nan_inputs_do_not_change_outputs_or_poison_gradients(self):
        model = self.model()
        self.activate(model)
        args = self.inputs()
        args[4][:, :, 0] = 1
        args[3][0, :, :, 3] = 0
        args[5][1, 1] = 0
        clean = [x.clone() for x in args]
        clean_delta, clean_gamma = model(*clean)
        args[2][:, :, 0] = float('nan')
        args[0][0] = float('nan')
        args[1][0] = float('nan')
        args[2][0] = float('nan')
        args[3][0, :, :, :3] = float('nan')
        for i in (0, 1, 2, 3, 4):
            args[i][1, 1] = float('nan')
        for i in (0, 1, 2, 3):
            args[i].requires_grad_()
        delta, gamma = model(*args)
        torch.testing.assert_close(delta, clean_delta)
        torch.testing.assert_close(gamma, clean_gamma)
        (delta.square().sum() + gamma.square().sum()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for x in args[:4]:
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_final_layer_learns_first_then_projection_gradients_flow(self):
        model = self.model()
        args = self.inputs()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        delta, _ = model(*args)
        delta.sum().backward()
        for parameter in model.net[-1].parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(model.local_proj.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        delta, _ = model(*args)
        delta.square().sum().backward()
        for projection in (model.local_proj, model.read_proj, model.partner_proj):
            self.assertTrue(torch.isfinite(projection.weight.grad).all())
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)

    def test_diagnostics_are_detached_and_small(self):
        model = self.model()
        self.activate(model)
        model(*self.inputs())
        self.assertEqual(model.last_gamma.shape, (4, 3))
        self.assertEqual(model.last_diagnostics['gamma_mean'].shape, (4, 3))
        self.assertEqual(model.last_diagnostics['delta_norm'].shape, (4,))
        for value in (model.last_gamma, *model.last_diagnostics.values()):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)

    def test_constructor_rejects_invalid_dimensions_and_bound(self):
        for kwargs in ({'local_dim': 0}, {'forward_dim': -1}, {'width': 0},
                       {'width': 2.5}, {'bound': 0}, {'bound': -.1},
                       {'bound': float('nan')}, {'bound': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.cls(**kwargs)

    def test_forward_rejects_wrong_shapes(self):
        model = self.model()
        replacements = {0: torch.zeros(2, 2, 6), 1: torch.zeros(2, 2, 4, 7),
                        2: torch.zeros(2, 2, 3, 7), 3: torch.zeros(2, 2, 3, 3),
                        4: torch.zeros(2, 2, 4), 5: torch.zeros(2, 3)}
        for index, value in replacements.items():
            args = self.inputs()
            args[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                model(*args)


if __name__ == '__main__':
    unittest.main()
