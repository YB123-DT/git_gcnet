import importlib
from pathlib import Path
import unittest
import torch

PATH = Path(__file__).resolve().parents[1] / 'gcnet_missing_m3/meaningful_blocks_optimization.py'
METHODS = ('hamburger_nmf_full', 'crate_mssa_ista_full', 'equilibrium_aggregation')


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.exists(), 'Optimization blocks not implemented')
        self.m = importlib.import_module('gcnet_missing_m3.meaningful_blocks_optimization')

    def fixture(self):
        local = torch.randn(3, 7)
        evidence = torch.randn(3, 4, 6)
        active = torch.tensor([[1, 1, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
        availability = ~active[:, 1:]
        return local, evidence, active, availability

    def test_all_methods_masks_gradients_and_shape(self):
        for method in METHODS:
            with self.subTest(method=method):
                block = self.m.build_optimization(method, 7, 2, 3).train()
                inputs = self.fixture()
                inputs[0].requires_grad_()
                inputs[1].requires_grad_()
                output = block(*inputs)
                self.assertEqual(output.shape, (3, 128))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for tensor in inputs[:2]:
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                gradients = [p.grad for p in block.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_mask_nan_batch_independence_and_immutable_eval(self):
        for method in METHODS:
            with self.subTest(method=method):
                block = self.m.build_optimization(method, 7, 2, 3).eval()
                inputs = self.fixture()
                before = {k: v.clone() for k, v in block.state_dict().items()}
                rng = torch.random.get_rng_state()
                with torch.inference_mode():
                    reference = block(*inputs)
                    bad = inputs[1].clone()
                    bad[~inputs[2]] = float('nan')
                    altered = block(inputs[0], bad, *inputs[2:])
                    alone = block(*(x[:1] for x in inputs))
                torch.testing.assert_close(altered, reference, atol=0, rtol=0)
                torch.testing.assert_close(alone, reference[:1], atol=3e-5, rtol=3e-5)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                for key, value in block.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]))
                self.assertTrue(all(p.grad is None for p in block.parameters()))

    def test_crate_complete_step_matches_independent_chain(self):
        step = self.m.CRATEIteration(8, heads=2).double()
        x = torch.randn(2, 3, 8, dtype=torch.double)
        z = step.norm_attention(x)
        w = torch.nn.functional.linear(z, step.subspace.weight).reshape(2, 3, 2, 4).transpose(1, 2)
        scores = (w @ w.transpose(-1, -2) / 2).softmax(-1)
        compression = (scores @ w).transpose(1, 2).reshape(2, 3, 8)
        z = step.norm_sparse(x + step.output(compression))
        d = step.dictionary
        expected = (z + .1 * (z @ d - (z @ d.T) @ d) - .01).relu()
        torch.testing.assert_close(step(x), expected)
        self.assertGreater((step(x) - (x + expected)).abs().max().item(), .1)

    def test_hamburger_factorization_matches_reference_updates(self):
        block = self.m.HamburgerBlock(7, 2, 3).double()
        x = torch.rand(2, 5, 128, dtype=torch.double)
        basis = block.initial_basis.unsqueeze(0).expand(2, -1, -1).clone()
        coefficient = (x @ basis).softmax(-1)
        for _ in range(6):
            coefficient = coefficient * (x @ basis) / (coefficient @ (basis.transpose(1, 2) @ basis) + 1e-6)
            basis = basis * (x.transpose(1, 2) @ coefficient) / (basis @ (coefficient.transpose(1, 2) @ coefficient) + 1e-6)
        coefficient = coefficient * (x @ basis) / (coefficient @ (basis.transpose(1, 2) @ basis) + 1e-6)
        torch.testing.assert_close(block.factorize(x), coefficient @ basis.transpose(1, 2))

    def test_equilibrium_nesterov_is_exact_for_quadratic_energy(self):
        block = self.m.EquilibriumBlock(7, 2, 3).double()
        target = torch.randn(2, 128, dtype=torch.double)
        energy = lambda y: .5 * (y - target).square().sum(-1)
        actual = block.solve(energy, torch.zeros_like(target), create_graph=True)
        y, v = torch.zeros_like(target), torch.zeros_like(target)
        eta, mu = block.learning_rate, block.momentum
        for _ in range(10):
            grad = y + mu * v - target
            v = mu * v - eta * grad
            y = y + (grad.abs().amax(-1) >= .001).unsqueeze(-1) * v
        torch.testing.assert_close(actual, y)

    def test_zero_active_empty_batch_and_private_initialization(self):
        for method in METHODS:
            with self.subTest(method=method):
                rng = torch.random.get_rng_state()
                block = self.m.build_optimization(method, 7, 2, 3).eval()
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                with torch.inference_mode():
                    result = block(torch.full((2, 7), float('nan')),
                                   torch.full((2, 4, 6), float('nan')),
                                   torch.zeros(2, 4, dtype=torch.bool), torch.ones(2, 3))
                    empty = block(torch.empty(0, 7), torch.empty(0, 4, 6),
                                  torch.empty(0, 4, dtype=torch.bool), torch.empty(0, 3))
                self.assertTrue(torch.equal(result, torch.zeros_like(result)))
                self.assertEqual(empty.shape, (0, 128))

    def test_hamburger_backward_only_last_coefficient_step(self):
        block = self.m.HamburgerBlock(7, 2, 3).double()
        x = torch.rand(2, 4, 128, dtype=torch.double, requires_grad=True)
        with torch.no_grad():
            b = block.initial_basis[None].expand(2, -1, -1).clone()
            c = (x @ b).softmax(-1)
            for _ in range(6):
                c = c * (x @ b) / (c @ (b.transpose(1, 2) @ b) + 1e-6)
                b = b * (x.transpose(1, 2) @ c) / (b @ (c.transpose(1, 2) @ c) + 1e-6)
        expected = (c * (x @ b) / (c @ (b.transpose(1, 2) @ b) + 1e-6)) @ b.transpose(1, 2)
        g1 = torch.autograd.grad(expected.square().sum(), x)[0]
        g2 = torch.autograd.grad(block.factorize(x).square().sum(), x)[0]
        torch.testing.assert_close(g1, g2, atol=1e-12, rtol=1e-12)

    def test_equilibrium_potential_and_scaled_energy_independent(self):
        block = self.m.EquilibriumBlock(7, 2, 3).double()
        tokens = torch.randn(2, 5, 128, dtype=torch.double)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        y = torch.randn(2, 128, dtype=torch.double)
        h = torch.cat((tokens, y[:, None].expand(-1, 5, -1)), -1)
        for stage in block.potential:
            main = torch.nn.functional.linear(h, stage.main.weight, stage.main.bias)
            if stage.norm is not None:
                main = torch.nn.functional.layer_norm(main, stage.norm.normalized_shape,
                            stage.norm.weight, stage.norm.bias, stage.norm.eps).tanh()
            skip = h if isinstance(stage.skip, torch.nn.Identity) else torch.nn.functional.linear(
                h, stage.skip.weight, stage.skip.bias)
            h = main + skip
        n = mask.sum(-1)
        value = torch.where(mask, h.square().mean(-1), 0).sum(-1)
        expected = (value + torch.nn.functional.softplus(block.raw_regularizer)*y.square().sum(-1)) * torch.log2(n+1)/(n+1e-8)
        torch.testing.assert_close(block.energy(tokens, mask, y), expected)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
