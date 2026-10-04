"""CPU-only five-role pooling contracts and selected algorithm identities."""
import importlib.util
import itertools
import unittest

import torch


IDS = ('m16_deep_sets', 'm17_janossy', 'm18_fspool', 'm19_netvlad',
       'm20_attentive_statistics', 'm26_dolg', 'm27_isqrt_cov',
       'm28_xcit_xca', 'm29_set_norm')


class PoolingGeometryContracts(unittest.TestCase):
    def test_factory_exists(self):
        self.assertIsNotNone(importlib.util.find_spec(
            'gcnet_missing_m3.priority40_pooling_geometry'))

    def test_all_contracts(self):
        from gcnet_missing_m3.priority40_pooling_geometry import build
        torch.set_num_threads(1)
        mask = torch.tensor([[1, 1, 0, 1, 0], [0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).bool()
        for method in IDS:
            with self.subTest(method=method):
                core = build(method, dim=64)
                raw = torch.randn(4, 5, 64)
                raw[~mask] = float('nan')
                raw.requires_grad_()
                out = core(raw, mask)
                self.assertEqual(out.shape, (4, 64))
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.equal(out[1], torch.zeros(64)))
                clean = torch.where(mask[..., None], raw, 0)
                torch.testing.assert_close(out, core(clean, mask))
                torch.testing.assert_close(out[:1], core(raw[:1], mask[:1]))
                self.assertEqual(core(raw[:0], mask[:0]).shape, (0, 64))
                out.square().mean().backward()
                self.assertTrue(torch.isfinite(raw.grad).all())
                self.assertTrue(torch.equal(raw.grad[~mask], torch.zeros_like(raw.grad[~mask])))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in core.parameters()
                                    if p.grad is not None))
                core.eval()
                torch.testing.assert_close(core(raw, mask), core(raw, mask), rtol=0, atol=0)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    mixed = core(raw, mask)
                self.assertTrue(torch.isfinite(mixed).all())
                order = torch.tensor([0, 3, 4, 1, 2])  # DOLG anchor stays at index 0.
                torch.testing.assert_close(core(raw[:, order], mask[:, order]), out,
                                           rtol=2e-4, atol=2e-5)

    def test_full_janossy_and_sort_knots(self):
        from gcnet_missing_m3.priority40_pooling_geometry import build
        core = build('m17_janossy')
        x = torch.randn(1, 5, 64)
        permutations = torch.tensor(list(itertools.permutations(range(5))))
        expected = core.ordered(x[:, permutations].reshape(120, 320)).mean(0)
        torch.testing.assert_close(core(x, torch.ones(1, 5, dtype=torch.bool))[0], expected)
        self.assertEqual(core.permutations_5.shape, (120, 5))
        sort = build('m18_fspool')
        self.assertEqual(sort.pool.weight.shape, (64, 5))
        with torch.no_grad():
            sort.pool.weight.fill_(1)
        mask = torch.tensor([[1, 0, 1, 0, 1]]).bool()
        torch.testing.assert_close(sort(x, mask), torch.where(mask[..., None], x, 0).sum(1))

    def test_set_norm_and_covariance_degeneracy(self):
        from gcnet_missing_m3.priority40_pooling_geometry import build
        core = build('m29_set_norm')
        x = torch.randn(2, 5, 64)
        mask = torch.tensor([[1, 1, 0, 0, 1], [1, 0, 0, 0, 0]]).bool()
        z = core.normalize(x, mask)
        for row in range(2):
            valid = x[row, mask[row]]
            expected = (valid-valid.mean())/torch.sqrt(valid.var(unbiased=False)+core.eps)
            torch.testing.assert_close(z[row, mask[row]], expected)
        covariance = build('m27_isqrt_cov')
        constant = torch.ones(2, 5, 64, requires_grad=True)
        covariance(constant, mask).square().sum().backward()
        self.assertTrue(torch.isfinite(constant.grad).all())


if __name__ == '__main__':
    unittest.main()
