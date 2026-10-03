"""Readout mechanisms: masked safety, independent examples and equation checks."""
import importlib.util
import unittest

import torch


MODULE = 'gcnet_missing_m3.readout_candidates_interactions'
METHODS = ('dcnv2', 'cin', 'autoint', 'din', 'dlrm', 'aff', 'nonlocal')


class InteractionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.assertIsNotNone(importlib.util.find_spec(MODULE), 'interaction implementations missing')
        self.module = __import__(MODULE, fromlist=['build_interaction'])

    def test_mask_nan_gradients_and_no_forward_rng(self):
        mask = torch.tensor([[1, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        for name in METHODS:
            with self.subTest(method=name):
                net = self.module.build_interaction(name, dim=8).double()
                x = torch.randn(2, 5, 8, dtype=torch.double)
                x[~mask] = float('nan')
                x.requires_grad_()
                rng = torch.get_rng_state()
                y = net(x, mask)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(y.shape, (2, 8))
                self.assertTrue(torch.isfinite(y).all())
                self.assertEqual(y[1].count_nonzero(), 0)
                y.square().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertEqual(x.grad[~mask].count_nonzero(), 0)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
                clean = torch.where(mask[..., None], x.detach(), 0.)
                torch.testing.assert_close(y, net(clean, mask), rtol=0, atol=0)

    def test_examples_are_independent_and_methods_distinct(self):
        x = torch.randn(3, 5, 8, dtype=torch.double)
        mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        outputs = []
        for name in METHODS:
            net = self.module.build_interaction(name, dim=8).double().train()
            together = net(x, mask)
            apart = torch.cat([net(x[i:i+1], mask[i:i+1]) for i in range(3)])
            torch.testing.assert_close(together, apart, rtol=1e-10, atol=1e-10)
            outputs.append(together)
        for i in range(len(outputs)):
            for j in range(i):
                self.assertFalse(torch.allclose(outputs[i], outputs[j]))

    def test_reference_equations(self):
        x = torch.randn(1, 5, 8, dtype=torch.double)
        active = torch.tensor([[True, True, False, True, False]])
        x = torch.where(active[..., None], x, 0.)
        e = x[0]
        ids = [0, 1, 3]
        for name in METHODS:
            with self.subTest(method=name):
                m = self.module.build_interaction(name, 8).double()
                if name == 'dcnv2':
                    z = e.flatten()
                    original = z
                    for down, up in zip(m.down, m.up):
                        z = z + original * up(down(z))
                    expected = m.out(z)
                elif name == 'cin':
                    z, pooled = e, []
                    for weight in m.weights:
                        z = torch.stack([sum(weight[h, i, j] * z[i] * e[j]
                                             for i in range(z.shape[0]) for j in range(5))
                                         for h in range(16)])
                        pooled.append(z.sum(-1))
                    expected = m.out(torch.cat(pooled))
                elif name == 'autoint':
                    z = e
                    for layer in m.layers:
                        q, k, v = (p(z).reshape(5, 2, 4) for p in (layer.query, layer.key, layer.value))
                        rows = []
                        for i in range(5):
                            heads = [sum(torch.softmax(torch.stack([q[i,h] @ k[j,h] for j in ids]), 0)[s] * v[j,h]
                                         for s,j in enumerate(ids)) for h in range(2)]
                            rows.append(torch.relu(torch.cat(heads) + layer.residual(z[i])) if i in ids else torch.zeros(8, dtype=z.dtype))
                        z = torch.stack(rows)
                    expected = z[ids].mean(0)
                elif name == 'din':
                    q = e[0]
                    history = sum(m.score(torch.cat((q, e[j], q-e[j], q*e[j]))).squeeze() * e[j] for j in ids if j)
                    expected = m.out(torch.cat((q, history)))
                elif name == 'dlrm':
                    pair = torch.stack([e[i] @ e[j] for i in range(5) for j in range(i+1, 5)])
                    expected = m.out(torch.cat((e[0], pair)))
                elif name == 'aff':
                    history = e[[1, 3]].mean(0)
                    weight = torch.sigmoid(m.local(e[0]+history) + m.global_context(e[ids].mean(0)))
                    expected = weight*e[0] + (1-weight)*history
                else:
                    q, k, v = m.query(e), m.key(e), m.value(e)
                    expected = torch.stack([e[i] + m.output(sum((q[i] @ k[j])*v[j] for j in ids)/len(ids)) for i in ids]).mean(0)
                torch.testing.assert_close(m(x, active)[0], expected, rtol=1e-10, atol=1e-10)

    def test_invalid_contract(self):
        with self.assertRaises(ValueError):
            self.module.build_interaction('unknown')
        with self.assertRaises(ValueError):
            self.module.build_interaction('autoint', 7)
        with self.assertRaises(ValueError):
            self.module.build_interaction('din', 8)(torch.zeros(2, 4, 8), torch.ones(2, 4, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
