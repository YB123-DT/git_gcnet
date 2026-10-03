"""Equation-level and safety contracts for the four hypergraph adaptations."""
import importlib
import importlib.util
import unittest

import torch
from torch.nn import functional as F


MODULE = 'gcnet_missing_m3.meaningful_blocks_hypergraph'
METHODS = ('allset_transformer', 'ed_hnn', 'hyper_sagnn', 'sheaf_hypergnn_diag')


class HypergraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE), 'hypergraph family is not implemented')
        return importlib.import_module(MODULE)

    def test_incidence_exact_membership(self):
        m = self.module()
        for missing in (0, 1, 2):
            roles = torch.tensor([0] + [1] * 8 + sum(([k + 2] * 8 for k in range(missing)), []))
            heads = torch.tensor([-1] + list(range(8)) * (1 + missing))
            h = m.make_incidence(roles, heads, 8)
            self.assertEqual(h.shape, (9 + 8 * missing, 9 + missing))
            self.assertTrue(h[0].all())
            self.assertTrue((h[1:].sum(-1) == 2).all())
            self.assertTrue((h.sum(0)[:8] == 2 + missing).all())
            self.assertTrue((h.sum(0)[8:] == 9).all())

    def test_ed_recipient_conditioned_sum_and_restart(self):
        m = self.module()
        torch.manual_seed(19)
        op = m.EquivariantDiffusion(4).double()
        x = torch.randn(2, 3, 4, dtype=torch.double)
        x0 = torch.randn_like(x)
        h = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool)
        expected = []
        for batch in range(2):
            edges = [sum(op.phi(x[batch, u]) for u in range(3) if h[u, e]) for e in range(2)]
            nodes = [sum(op.rho(torch.cat((x[batch, v], edges[e]))) for e in range(2) if h[v, e]) for v in range(3)]
            expected.append(op.psi(.9 * torch.stack(nodes) + .1 * x0[batch]).relu())
        torch.testing.assert_close(op(x, x0, h), torch.stack(expected), rtol=1e-12, atol=1e-12)

    def test_pma_seed_softmax_both_directions(self):
        m = self.module()
        torch.manual_seed(20)
        op = m.SeedPooling(8, 2).double()
        x = torch.randn(2, 3, 8, dtype=torch.double)
        adjacency = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
        k = op.key(x).reshape(2, 3, 2, 4)
        v = op.value(x).reshape(2, 3, 2, 4)
        scores = F.leaky_relu((k * op.seed).sum(-1), .2)
        groups = []
        for members in adjacency:
            weights = scores[:, members].softmax(1)
            pooled = (weights[..., None] * v[:, members]).sum(1) + op.seed
            y = op.norm1(pooled.flatten(-2))
            groups.append(op.norm2(y + op.ffn(y).relu()))
        expected = torch.stack(groups, 1)
        torch.testing.assert_close(op(x, adjacency), expected, rtol=1e-12, atol=1e-12)
        reverse = op(expected, adjacency.T)
        self.assertEqual(reverse.shape, x.shape)

    def test_sheaf_transport_and_negative_diagonal(self):
        m = self.module()
        h = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool)
        maps = torch.tensor([[[[.2, -.3], [.4, .7]], [[.6, .8], [0., 0.]], [[0., 0.], [-.1, .9]]]], dtype=torch.double)
        got = m.sheaf_propagation(maps, h)
        lifted = torch.zeros(6, 4, dtype=torch.double)
        for v in range(3):
            for e in range(2):
                lifted[v*2:(v+1)*2, e*2:(e+1)*2] = torch.diag(maps[0, v, e])
        degree = lifted.square().sum(1).clamp_min(1e-6)
        norm = torch.diag(degree.rsqrt())
        edge_inv = torch.diag(h.sum(0).repeat_interleave(2).double().reciprocal())
        a = norm @ lifted @ edge_inv @ lifted.T @ norm
        expected = norm + a
        for v in range(3):
            expected[v*2:(v+1)*2, v*2:(v+1)*2] -= 2*a[v*2:(v+1)*2, v*2:(v+1)*2]
        torch.testing.assert_close(got[0], expected, rtol=1e-12, atol=1e-12)

    def test_sheaf_source_input_normalization_counterexample(self):
        m = self.module()
        maps = torch.full((1, 2, 1, 1), 2., dtype=torch.double)
        incidence = torch.ones(2, 1, dtype=torch.bool)
        z = torch.tensor([[[1.], [3.]]], dtype=torch.double)
        actual = m.sheaf_propagation(maps, incidence) @ z
        # Source first scales z by D^-1/2=.5, THEN applies its I+Q-2diag(Q).
        torch.testing.assert_close(actual, torch.tensor([[[1.5], [.5]]], dtype=torch.double))

    def test_sheaf_source_order_reference_forward_and_gradients(self):
        m = self.module()
        torch.manual_seed(121)
        h = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool)
        maps = (torch.rand(1, 3, 2, 2, dtype=torch.double) + .5).requires_grad_()
        z = torch.randn(1, 6, 3, dtype=torch.double, requires_grad=True)
        safe_maps = torch.where(h[None, :, :, None], maps, 0)
        lifted = torch.zeros(6, 4, dtype=torch.double)
        for v in range(3):
            for e in range(2):
                lifted[v*2:(v+1)*2, e*2:(e+1)*2] = torch.diag(safe_maps[0, v, e])
        norm = torch.diag(lifted.square().sum(1).clamp_min(1e-6).rsqrt())
        edge_inv = torch.diag(h.sum(0).repeat_interleave(2).double().reciprocal())
        pre_normalized = norm @ z[0]
        q = norm @ lifted @ edge_inv @ lifted.T
        block_diagonal = torch.zeros_like(q)
        for v in range(3):
            block_diagonal[v*2:(v+1)*2, v*2:(v+1)*2] = q[v*2:(v+1)*2, v*2:(v+1)*2]
        expected = (torch.eye(6, dtype=torch.double) + q - 2*block_diagonal) @ pre_normalized
        actual = (m.sheaf_propagation(maps, h) @ z)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual_grad = torch.autograd.grad(actual.square().sum(), (maps, z), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), (maps, z))
        for a, e in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-10)

    def test_sagnn_leave_self_out_and_static_dynamic_discrepancy(self):
        m = self.module()
        torch.manual_seed(22)
        op = m.StaticDynamicDiscrepancy(8, 2).double()
        x = torch.randn(2, 3, 8, dtype=torch.double)
        dynamic, static = op.embeddings(x)
        q = op.query(op.query_norm(x)).reshape(2, 3, 2, 4).transpose(1, 2)
        k = op.key(op.key_norm(x)).reshape(2, 3, 2, 4).transpose(1, 2)
        v = op.value(op.value_norm(x)).reshape(2, 3, 2, 4).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / 2
        logits = logits.masked_fill(torch.eye(3, dtype=torch.bool), -torch.inf)
        attended = (logits.softmax(-1) @ v).transpose(1, 2).reshape(2, 3, 8)
        projected = op.output(attended)
        expected_dynamic = op.dynamic_final_norm(op.dynamic_norm(projected + op.dynamic_ffn(projected)))
        expected_static = op.static_final_norm(op.static_norm(op.static_ffn(x)))
        torch.testing.assert_close(dynamic, expected_dynamic)
        torch.testing.assert_close(static, expected_static)
        torch.testing.assert_close(op(x), (dynamic-static).square().mean(1))

    def test_all_patterns_poison_batch_independence_and_backward(self):
        m = self.module()
        availability = torch.tensor([[bool(bits & (1 << k)) for k in range(3)] for bits in range(1, 8)])
        active = torch.cat((torch.ones(7, 1, dtype=torch.bool), ~availability), 1)
        active = torch.cat((active, torch.zeros(1, 4, dtype=torch.bool)))
        availability = torch.cat((availability, torch.zeros(1, 3, dtype=torch.bool)))
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(23)
                op = m.build_hypergraph(method, 16, 8, 4).double()
                local = torch.randn(8, 16, dtype=torch.double, requires_grad=True)
                evidence = torch.randn(8, 4, 32, dtype=torch.double, requires_grad=True)
                clean = torch.where(active[..., None], evidence, 0)
                poisoned = torch.where(active[..., None], evidence, torch.full_like(evidence, float('nan')))
                y = op(local, clean, active, availability)
                self.assertEqual(y.shape, (8, 128))
                self.assertEqual(op.output_dim, 128)
                torch.testing.assert_close(y, op(local, poisoned, active, availability), rtol=0, atol=0)
                self.assertTrue(torch.equal(y[-1], torch.zeros(128, dtype=torch.double)))
                isolated = op(local[:1], clean[:1], active[:1], availability[:1])
                torch.testing.assert_close(y[:1], isolated, rtol=1e-10, atol=1e-10)
                before = {name: p.detach().clone() for name, p in op.named_parameters()}
                optimizer = torch.optim.SGD(op.parameters(), lr=.01)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(evidence.grad).all())
                self.assertTrue((evidence.grad[~active] == 0).all())
                for name, p in op.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(torch.count_nonzero(p.grad).item(), 0, name)
                optimizer.step()
                self.assertTrue(any(not torch.equal(before[n], p) for n, p in op.named_parameters()))

    def test_fixed_core_counts_and_unknown_method(self):
        m = self.module()
        self.assertEqual(len(m.build_hypergraph('allset_transformer', 16, 8, 4).rounds), 2)
        ed = m.build_hypergraph('ed_hnn', 16, 8, 4)
        self.assertEqual(ed.iterations, 3)
        self.assertEqual(sum(isinstance(x, m.EquivariantDiffusion) for x in ed.modules()), 1)
        self.assertEqual(len(m.build_hypergraph('sheaf_hypergnn_diag', 16, 8, 4).layers), 2)
        with self.assertRaises(ValueError):
            m.build_hypergraph('not_a_method', 16, 8, 4)

    def test_zero_extreme_inputs_empty_batch_and_strict_reload(self):
        m = self.module()
        active = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        availability = ~active[:, 1:]
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(24)
                op = m.build_hypergraph(method, 16, 8, 4).double().eval()
                for scale in (0., 100.):
                    local = torch.full((2, 16), scale, dtype=torch.double)
                    evidence = torch.full((2, 4, 32), scale, dtype=torch.double)
                    out = op(local, evidence, active, availability)
                    self.assertTrue(torch.isfinite(out).all())
                snapshot = {k: v.clone() for k, v in op.state_dict().items()}
                rng = torch.random.get_rng_state().clone()
                again = op(local, evidence, active, availability)
                torch.testing.assert_close(out, again, rtol=0, atol=0)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                for k, v in op.state_dict().items():
                    self.assertTrue(torch.equal(v, snapshot[k]), k)
                clone = m.build_hypergraph(method, 16, 8, 4).double().eval()
                clone.load_state_dict(snapshot, strict=True)
                torch.testing.assert_close(out, clone(local, evidence, active, availability), rtol=0, atol=0)
                self.assertEqual(op(local[:0], evidence[:0], active[:0], availability[:0]).shape, (0, 128))
                dead = torch.zeros_like(active)
                poison_l = torch.full_like(local, float('nan'))
                poison_e = torch.full_like(evidence, float('inf'))
                self.assertTrue(torch.equal(op(poison_l, poison_e, dead, availability), torch.zeros(2, 128, dtype=torch.double)))

    def test_incidence_permutation_equivariance(self):
        m = self.module()
        torch.manual_seed(25)
        x = torch.randn(1, 9, 128, dtype=torch.double)
        roles = torch.tensor([0] + [1] * 8)
        heads = torch.tensor([-1] + list(range(8)))
        h = m.make_incidence(roles, heads, 8)
        permutation = torch.tensor([4, 2, 0, 8, 3, 6, 1, 7, 5])
        for method in METHODS:
            with self.subTest(method=method):
                op = m.build_hypergraph(method, 16, 8, 4).double()
                reference = op.process(x, h, roles)
                permuted = op.process(x[:, permutation], h[permutation], roles[permutation])
                torch.testing.assert_close(reference, permuted, rtol=1e-9, atol=1e-9)

    def test_smooth_core_input_gradchecks(self):
        m = self.module()
        torch.manual_seed(26)
        h = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool)
        x = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
        ed = m.EquivariantDiffusion(4).double()
        self.assertTrue(torch.autograd.gradcheck(lambda z: ed(z, z * .5, h), (x,), fast_mode=True))
        pma = m.SeedPooling(4, 2).double()
        self.assertTrue(torch.autograd.gradcheck(lambda z: pma(z, h.T), (x,), fast_mode=True))
        sagnn = m.StaticDynamicDiscrepancy(4, 2).double()
        self.assertTrue(torch.autograd.gradcheck(sagnn, (x,), fast_mode=True))
        maps = (torch.rand(1, 3, 2, 2, dtype=torch.double) + .2).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda z: m.sheaf_propagation(z, h), (maps,), fast_mode=True))


if __name__ == '__main__':
    unittest.main()
