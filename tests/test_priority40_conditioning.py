"""Small CPU contracts for the five fixed conditioning candidates."""
import unittest

import torch

from gcnet_missing_m3.priority40_conditioning import build, build_input


class ConditioningContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_readout_mask_gradient_and_row_independence(self):
        mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        for method in ('m12_cross_stitch', 'm14_gct', 'm15_dynamic_relu'):
            with self.subTest(method=method):
                model = build(method, dim=8)
                x = torch.randn(3, 5, 8, requires_grad=True)
                poisoned = x.detach().masked_fill(~mask[..., None], float('nan'))
                out = model(x, mask)
                self.assertEqual(out.shape, (3, 8))
                torch.testing.assert_close(out, model(poisoned, mask))
                torch.testing.assert_close(out[2], torch.zeros(8))
                torch.testing.assert_close(out[:1], model(x[:1], mask[:1]))
                before = [p.detach().clone() for p in model.parameters()]
                optimizer = torch.optim.SGD(model.parameters(), lr=.05)
                out.square().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                optimizer.step()
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
                with torch.inference_mode():
                    self.assertTrue(torch.isfinite(model(poisoned, mask)).all())

    def test_input_identity_mask_and_training(self):
        local = torch.randn(3, 8)
        evidence = torch.randn(3, 4, 12)
        active = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
        av = torch.ones(3, 3)
        clean = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
        poison = evidence.masked_fill(~active[..., None], float('nan'))
        for method in ('m11_film', 'm13_mmtm'):
            with self.subTest(method=method):
                model = build_input(method, latent_dim=8, forward_dim=12)
                out = model(local, poison, active, av)
                torch.testing.assert_close(out[0], local)
                torch.testing.assert_close(out[1], clean)
                optimizer = torch.optim.SGD(model.parameters(), lr=.02)
                for _ in range(2):
                    optimizer.zero_grad()
                    a, b = model(local, poison, active, av)
                    (a.square().mean() + b.square().mean()).backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                    optimizer.step()
                changed = model(local, poison, active, av)
                self.assertFalse(torch.equal(changed[0], local))
                if method == 'm11_film':
                    torch.testing.assert_close(changed[1], clean)
                    torch.testing.assert_close(changed[0][2], local[2])
                with torch.inference_mode():
                    for actual, expected in zip(model(local, clean, active, av), changed):
                        torch.testing.assert_close(actual, expected)

    def test_gct_initial_mean_and_dynamic_relu_initial_relu(self):
        x = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        torch.testing.assert_close(build('m14_gct', dim=8)(x, mask), x.mean(1))
        torch.testing.assert_close(build('m15_dynamic_relu', dim=8)(x, mask), x.relu().mean(1))


if __name__ == '__main__':
    unittest.main()
