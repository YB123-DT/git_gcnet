"""Masked bilinear readouts: operator equations and numerical behavior."""
import importlib.util
import unittest

import torch


MODULE = 'gcnet_missing_m3.readout_candidates_bilinear'
METHODS = ('mlb', 'mfb', 'mutan', 'block', 'mcb', 'ban')


class BilinearTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(66)
        self.assertIsNotNone(importlib.util.find_spec(MODULE), 'bilinear implementations missing')
        self.module = __import__(MODULE, fromlist=['build_bilinear'])

    def test_masked_nan_finite_gradients_and_no_forward_rng(self):
        active = torch.tensor([[1, 1, 0, 1, 0], [0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
        for name in METHODS:
            with self.subTest(method=name):
                m = self.module.build_bilinear(name, dim=8).double().train()
                x = torch.randn(4, 5, 8, dtype=torch.double)
                x[~active] = float('nan')
                x.requires_grad_()
                rng = torch.get_rng_state().clone()
                y = m(x, active)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(y.shape, (4, 8))
                self.assertTrue(torch.isfinite(y).all())
                self.assertEqual(y[1:].count_nonzero().item(), 0)
                y.square().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertEqual(x.grad[~active].count_nonzero().item(), 0)
                self.assertGreater(x.grad[0, 0].abs().sum().item(), 0)
                self.assertGreater(x.grad[0, 1].abs().sum().item(), 0)
                for p in m.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                clean = torch.where(active[..., None], x.detach(), 0.)
                torch.testing.assert_close(y, m(clean, active), rtol=0, atol=0)

    def test_zero_inputs_empty_batch_and_example_independence(self):
        for name in METHODS:
            with self.subTest(method=name):
                m = self.module.build_bilinear(name, 8).double()
                zero = torch.zeros(2, 5, 8, dtype=torch.double, requires_grad=True)
                active = torch.ones(2, 5, dtype=torch.bool)
                out = m(zero, active)
                out.square().sum().backward()
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.isfinite(zero.grad).all())
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
                self.assertEqual(m(zero[:0], active[:0]).shape, (0, 8))
                x = torch.randn(2, 5, 8, dtype=torch.double)
                together = m(x, active)
                apart = torch.cat([m(x[i:i+1], active[i:i+1]) for i in range(2)])
                torch.testing.assert_close(together, apart, rtol=1e-10, atol=1e-10)

    def test_equations_with_independent_factor_sums(self):
        x = torch.randn(1, 5, 8, dtype=torch.double)
        active = torch.tensor([[True, True, False, True, False]])
        local, context = x[0, 0], x[0, [1, 3]].mean(0)

        def norm(v):
            v = v / (v.abs() + 1e-8).sqrt()
            return v / v.norm().clamp_min(1e-8)

        m = self.module.build_bilinear('mlb', 8).double()
        expected = m.out(torch.tanh(m.local(local)) * torch.tanh(m.context(context)))
        torch.testing.assert_close(m(x, active)[0], expected)

        m = self.module.build_bilinear('mfb', 8).double()
        a, b = m.local(local), m.context(context)
        pooled = torch.stack([sum(a[4*i+j] * b[4*i+j] for j in range(4)) for i in range(8)])
        torch.testing.assert_close(m(x, active)[0], m.out(norm(pooled)))

        m = self.module.build_bilinear('mutan', 8).double()
        a, b = torch.tanh(m.local(local)), torch.tanh(m.context(context))
        expected = m.out(sum(left(a) * right(b) for left, right in zip(m.local_factors, m.context_factors)))
        self.assertEqual(len(m.local_factors), 4)
        torch.testing.assert_close(m(x, active)[0], expected)

        m = self.module.build_bilinear('block', 8).double()
        a, b = m.local(local).reshape(4, 2), m.context(context).reshape(4, 2)
        chunks = []
        self.assertEqual(len(m.local_blocks), 4)
        for i in range(4):
            left, right = m.local_blocks[i](a[i]), m.context_blocks[i](b[i])
            value = torch.stack([sum(left[2*r+j] * right[2*r+j] for r in range(4)) for j in range(2)])
            chunks.append(norm(value))
        torch.testing.assert_close(m(x, active)[0], m.out(torch.cat(chunks)))

    def test_mcb_matches_explicit_outer_product_hash(self):
        m = self.module.build_bilinear('mcb', 8).double()
        x = torch.randn(1, 5, 8, dtype=torch.double)
        active = torch.tensor([[True, True, False, True, False]])
        local, context = x[0, 0], x[0, [1, 3]].mean(0)
        hashed = torch.zeros(1024, dtype=torch.double)
        for i in range(8):
            for j in range(8):
                index = (m.local_hash[i] + m.context_hash[j]) % 1024
                hashed[index] += m.local_sign[i] * m.context_sign[j] * local[i] * context[j]
        hashed = hashed / (hashed.abs() + 1e-8).sqrt()
        hashed = hashed / hashed.norm().clamp_min(1e-8)
        torch.testing.assert_close(m(x, active)[0], m.out(hashed), rtol=1e-9, atol=1e-9)

    def test_mcb_buffers_are_independent_fixed_and_reloadable(self):
        torch.manual_seed(31)
        torch.nn.Linear(1024, 8)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(31)
        first = self.module.build_bilinear('mcb', 8)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        torch.manual_seed(92)
        second = self.module.build_bilinear('mcb', 8)
        for name, value in first.named_buffers():
            self.assertTrue(torch.equal(value, dict(second.named_buffers())[name]))
        self.assertFalse(torch.equal(first.local_hash, first.context_hash))
        state = {name: value.clone() for name, value in first.state_dict().items()}
        second.load_state_dict(state)
        x, active = torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool)
        torch.testing.assert_close(first(x, active), second(x, active), rtol=0, atol=0)

    def test_ban_scores_and_values_use_only_active_evidence(self):
        m = self.module.build_bilinear('ban', 8).double()
        x = torch.randn(1, 5, 8, dtype=torch.double)
        active = torch.tensor([[True, True, False, True, False]])
        local, history = x[0, 0], x[0, [1, 3]]
        scores = torch.stack([m.score(torch.relu(m.query(local)) * torch.relu(m.key(e))).squeeze() for e in history])
        weights = scores.softmax(0)
        expected = m.out(sum(weights[j] * torch.relu(m.value_local(local)) * torch.relu(m.value_context(e)) for j, e in enumerate(history)))
        torch.testing.assert_close(m(x, active)[0], expected)

    def test_contract_errors(self):
        with self.assertRaises(ValueError):
            self.module.build_bilinear('unknown')
        with self.assertRaises(ValueError):
            self.module.build_bilinear('mlb', 0)
        with self.assertRaises(ValueError):
            self.module.build_bilinear('block', 7)
        m = self.module.build_bilinear('mlb', 8)
        for tokens, active in ((torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)),
                               (torch.zeros(1, 5, 8), torch.ones(1, 5)),
                               (torch.zeros(1, 5, 7), torch.ones(1, 5, dtype=torch.bool))):
            with self.assertRaises(ValueError):
                m(tokens, active)


if __name__ == '__main__':
    unittest.main()
