"""Small CPU contracts, not benchmark or paper reproduction tests."""
import importlib.util
import unittest

import torch


IDS = ('survey40_adf_gaussian_moment_propagation',
       'survey80_recurrent_kalman_network',
       'survey80_robust_gnc_tls_consensus', 'survey80_particle_filter_rnn')


class ProbabilisticContracts(unittest.TestCase):
    def test_factory_exists(self):
        self.assertIsNotNone(importlib.util.find_spec(
            'gcnet_missing_m3.meaningful_v3_probabilistic'))

    def test_contracts(self):
        from gcnet_missing_m3.meaningful_v3_probabilistic import build
        torch.set_num_threads(1)
        for name in IDS:
            with self.subTest(method=name):
                model = build(name, 8, 2, 4)
                x, e = torch.randn(3, 8), torch.randn(3, 4, 8)
                mask = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]]).bool()
                e[~mask] = float('nan')
                av = torch.ones(3, 3)
                clean = torch.where(mask[..., None], e, 0)
                out = model(x, e, mask, av)
                self.assertTrue(torch.equal(out[0], x))
                self.assertTrue(torch.equal(out[1], clean))
                with torch.no_grad():
                    model.bridge.weight.copy_(.005*torch.sin(torch.arange(
                        model.bridge.weight.numel()).reshape_as(model.bridge.weight)))
                rng = torch.random.get_rng_state().clone()
                out = model(x, e, mask, av)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertTrue(all(torch.isfinite(t).all() for t in out))
                self.assertTrue(torch.equal(out[0][1], x[1]))
                self.assertTrue(torch.equal(out[1][~mask], clean[~mask]))
                sum(t.square().mean() for t in out).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in model.named_parameters() if 'bridge' not in n))
                model.eval()
                a, b = model(x, e, mask, av), model(x, e, mask, av)
                self.assertTrue(all(torch.equal(u, v) for u, v in zip(a, b)))
                empty = model(x[:0], e[:0], mask[:0], av[:0])
                self.assertEqual(empty[1].shape, (0, 4, 8))


if __name__ == '__main__':
    unittest.main()
