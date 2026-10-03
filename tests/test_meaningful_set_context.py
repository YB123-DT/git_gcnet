"""Mechanism tests for three complete evidence-set readouts (no training runs)."""
import importlib
import importlib.util
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn


METHODS = ('perceiver_io', 'dgcnn_dynamic_edgeconv', 'graph_multiset_transformer')


class SetContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        name = 'gcnet_missing_m3.meaningful_blocks_set_context'
        self.assertIsNotNone(importlib.util.find_spec(name), 'set/context core implementation is missing')
        return importlib.import_module(name)

    def fixture(self, dtype=torch.float32):
        torch.manual_seed(410)
        availability = torch.tensor([(a, t, v) for a in (0, 1) for t in (0, 1)
                                     for v in (0, 1) if a or t or v], dtype=torch.bool)
        active = torch.cat((torch.ones(7, 1, dtype=torch.bool), ~availability), 1)
        return (torch.randn(7, 7, dtype=dtype), torch.randn(7, 4, 6, dtype=dtype),
                active, availability)

    def test_factory_shapes_masks_gradients_and_updates(self):
        module = self.module()
        for method in METHODS:
            with self.subTest(method=method):
                net = module.build_set_context(method, 7, 2, 3)
                self.assertEqual(net.output_dim, 128)
                self.assertFalse(any(isinstance(m, (nn.Dropout, nn.modules.batchnorm._BatchNorm))
                                     for m in net.modules()))
                local, evidence, active, availability = self.fixture()
                evidence.requires_grad_()
                local.requires_grad_()
                optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
                before = {n: p.detach().clone() for n, p in net.named_parameters()}
                result = net(local, evidence, active, availability)
                self.assertEqual(result.shape, (7, 128))
                self.assertTrue(torch.isfinite(result).all())
                (result.square().mean() + result[:, 0].mean()).backward()
                for name, param in net.named_parameters():
                    self.assertIsNotNone(param.grad, (method, name))
                    self.assertTrue(torch.isfinite(param.grad).all(), (method, name))
                self.assertGreater(float(local.grad.abs().sum()), 0)
                self.assertGreater(float(evidence.grad[active].abs().sum()), 0)
                self.assertEqual(int(evidence.grad[~active].count_nonzero()), 0)
                optimizer.step()
                self.assertGreater(sum(not torch.equal(before[n], p) for n, p in net.named_parameters()), 5)
                prefixes = {
                    'perceiver_io': ('tokenizer.', 'latents', 'encode.', 'process.0.', 'process.1.',
                                     'process.2.', 'query.', 'query_bias', 'decode.'),
                    'dgcnn_dynamic_edgeconv': ('tokenizer.', 'stages.0.', 'stages.1.', 'stages.2.',
                                              'stages.3.', 'multiscale.', 'pool_projection.'),
                    'graph_multiset_transformer': ('tokenizer.', 'encoder.0.', 'encoder.1.',
                                                  'graph_seeds', 'final_seed', 'pool_graph.',
                                                  'interseed.', 'pool_final.', 'final.'),
                }[method]
                for prefix in prefixes:
                    self.assertTrue(any(n.startswith(prefix) and not torch.equal(before[n], p)
                                        for n, p in net.named_parameters()), (method, prefix))

    def test_poison_inactive_and_all_inactive_rows(self):
        module = self.module()
        for method in METHODS:
            with self.subTest(method=method):
                net = module.build_set_context(method, 7, 2, 3).eval()
                local, evidence, active, availability = self.fixture()
                safe = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
                poisoned = evidence.clone()
                poisoned[~active] = float('nan')
                with torch.no_grad():
                    expected = net(local, safe, active, availability)
                    self.assertTrue(torch.equal(expected, net(local, poisoned, active, availability)))
                    poisoned[~active] = float('inf')
                    self.assertTrue(torch.equal(expected, net(local, poisoned, active, availability)))
                    none = net(torch.full_like(local, float('nan')),
                               torch.full_like(evidence, float('nan')),
                               torch.zeros_like(active), availability)
                self.assertEqual(int(none.count_nonzero()), 0)
                self.assertEqual(net(local[:0], evidence[:0], active[:0], availability[:0]).shape, (0, 128))

    def test_independence_rng_and_strict_state(self):
        module = self.module()
        for method in METHODS:
            with self.subTest(method=method):
                net = module.build_set_context(method, 7, 2, 3).eval()
                args = self.fixture()
                rng = torch.get_rng_state().clone()
                state = {n: t.clone() for n, t in net.state_dict().items()}
                with torch.no_grad():
                    together = net(*args)
                    separate = torch.cat([net(*(x[i:i + 1] for x in args)) for i in range(7)])
                    torch.testing.assert_close(together, separate, atol=2e-6, rtol=1e-5)
                    self.assertTrue(torch.equal(together, net(*args)))
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(torch.equal(t, net.state_dict()[n]) for n, t in state.items()))
                rebuilt = module.build_set_context(method, 7, 2, 3).eval()
                rebuilt.load_state_dict(state, strict=True)
                with torch.no_grad():
                    self.assertTrue(torch.equal(together, rebuilt(*args)))
        with self.assertRaises(ValueError):
            module.build_set_context('set_transformer_full', 7, 2, 3)

    def test_perceiver_attention_scale_and_query_residual(self):
        module = self.module()
        attention = module._PerceiverAttention(8, 2).double()
        with torch.no_grad():
            for layer in (attention.query, attention.key, attention.value, attention.output):
                layer.weight.copy_(torch.eye(8, dtype=torch.float64))
                layer.bias.zero_()
        torch.manual_seed(92)
        q, kv = torch.randn(2, 3, 8, dtype=torch.float64), torch.randn(2, 5, 8, dtype=torch.float64)
        qh = q.reshape(2, 3, 2, 4).transpose(1, 2)
        kh = kv.reshape(2, 5, 2, 4).transpose(1, 2)
        expected = ((qh @ kh.transpose(-1, -2) / 2).softmax(-1) @ kh)
        expected = expected.transpose(1, 2).reshape(2, 3, 8)
        torch.testing.assert_close(attention(q, kv), expected, atol=1e-12, rtol=1e-12)
        for residual in (True, False):
            block = module._PerceiverCrossAttention(8, 2, query_residual=residual).double()
            with torch.no_grad():
                for parameter in block.parameters():
                    parameter.zero_()
            expected = q if residual else torch.zeros_like(q)
            self.assertTrue(torch.equal(block(q, kv), expected))

    def test_perceiver_full_chain_and_local_query(self):
        module = self.module()
        net = module.build_set_context('perceiver_io', 7, 2, 3)
        self.assertEqual(len(net.process), 3)
        self.assertEqual(tuple(net.latents.shape), (8, 128))
        self.assertFalse(net.decode.query_residual)
        counts = {'encode': 0, 'process': 0, 'decode': 0}
        hooks = []
        def count(name):
            def hook(*unused):
                counts[name] += 1
            return hook
        hooks.append(net.encode.register_forward_hook(count('encode')))
        hooks.extend(layer.register_forward_hook(count('process')) for layer in net.process)
        hooks.append(net.decode.register_forward_hook(count('decode')))
        args = self.fixture()
        result = net(*(x[-1:] for x in args))
        for handle in hooks:
            handle.remove()
        self.assertEqual(counts, {'encode': 1, 'process': 3, 'decode': 1})
        changed = list(x[-1:].clone() for x in args)
        changed[0].add_(.5)
        self.assertFalse(torch.equal(result, net(*changed)))

    def test_dgcnn_four_dynamic_graphs_and_ties(self):
        module = self.module()
        identical = torch.zeros(1, 5, 3)
        ids = torch.tensor([8, 2, 9, 0, 4])
        indices = module._stable_knn(identical, 4, ids)
        expected_ids = torch.tensor([0, 2, 4, 8]).expand(1, 5, 4)
        self.assertTrue(torch.equal(ids[indices], expected_ids))
        points = torch.tensor([[[0.], [1.], [5.]]])
        first = module._stable_knn(points, 2, torch.arange(3))
        points[:, 2] = .1
        second = module._stable_knn(points, 2, torch.arange(3))
        self.assertFalse(torch.equal(first, second))
        permutation = torch.tensor([2, 4, 0, 3, 1])
        permuted_indices = module._stable_knn(identical[:, permutation], 4, ids[permutation])
        permuted_neighbor_ids = ids[permutation][permuted_indices]
        self.assertTrue(torch.equal(permuted_neighbor_ids[:, torch.argsort(permutation)], expected_ids))
        net = module.build_set_context('dgcnn_dynamic_edgeconv', 7, 2, 3)
        self.assertEqual(len(net.stages), 4)
        shapes = []
        original = module._stable_knn
        def capture(x, *args, **kwargs):
            shapes.append(tuple(x.shape))
            return original(x, *args, **kwargs)
        with patch.object(module, '_stable_knn', side_effect=capture):
            net(*(x[-1:] for x in self.fixture()))
        self.assertEqual([shape[-1] for shape in shapes], [128, 64, 64, 128])

    def test_edgeconv_center_difference_neighbor_max(self):
        module = self.module()
        block = module._EdgeConv(3, 4).double()
        x = torch.tensor([[[0., 1., 2.], [1., 2., 4.], [2., 1., 0.]]], dtype=torch.float64)
        # k=4 includes all three nodes; independently form the center-relative edges.
        edges = torch.stack([torch.stack([torch.cat((x[0, j] - x[0, i], x[0, i]))
                                          for j in range(3)]) for i in range(3)])[None]
        expected = torch.nn.functional.leaky_relu(block.norm(block.linear(edges)), .2).max(2).values
        torch.testing.assert_close(block(x, torch.arange(3)), expected, atol=1e-12, rtol=1e-12)

    def test_gmt_graph_kv_and_full_pooling(self):
        module = self.module()
        roles = torch.tensor([0, 1, 1, 2])
        heads = torch.tensor([-1, 0, 1, 0])
        graph = module._normalized_graph(roles, heads, torch.float64)
        adjacency = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 1.],
                                  [1., 0., 1., 0.], [1., 1., 0., 1.]], dtype=torch.float64)
        degree = adjacency.sum(-1).rsqrt()
        torch.testing.assert_close(graph, degree[:, None] * adjacency * degree[None, :])
        torch.manual_seed(42)
        block = module._GMTMAB(8, 8, 2, graph_keys=True).double()
        q, x = torch.randn(1, 2, 8, dtype=torch.float64), torch.randn(1, 4, 8, dtype=torch.float64)
        pq = block.query(q).reshape(1, 2, 2, 4).transpose(1, 2)
        pk = block.key(x, graph).reshape(1, 4, 2, 4).transpose(1, 2)
        pv = block.value(x, graph).reshape(1, 4, 2, 4).transpose(1, 2)
        expected = pq + (pq @ pk.transpose(-1, -2) / math.sqrt(8)).softmax(-1) @ pv
        expected = block.norm1(expected.transpose(1, 2).reshape(1, 2, 8))
        expected = block.norm2(expected + torch.relu(block.feedforward(expected)))
        torch.testing.assert_close(block(q, x, graph), expected, atol=1e-12, rtol=1e-12)
        self.assertFalse(torch.allclose(block(q, x, graph), block(q, x, torch.eye(4, dtype=x.dtype))))
        net = module.build_set_context('graph_multiset_transformer', 7, 2, 3)
        self.assertEqual(len(net.encoder), 2)
        self.assertEqual(tuple(net.graph_seeds.shape), (4, 256))
        self.assertEqual(tuple(net.final_seed.shape), (1, 128))
        seen = []
        hooks = [getattr(net, name).register_forward_hook(
            lambda module, inputs, output, name=name: seen.append((name, tuple(output.shape))))
            for name in ('pool_graph', 'interseed', 'pool_final')]
        net(*(x[-1:] for x in self.fixture()))
        for hook in hooks:
            hook.remove()
        self.assertEqual(seen, [('pool_graph', (1, 4, 256)), ('interseed', (1, 4, 128)),
                                ('pool_final', (1, 1, 128))])

    def test_actual_eight_heads_poison_backward(self):
        module = self.module()
        _, _, active, availability = self.fixture()
        for method in METHODS:
            with self.subTest(method=method):
                net = module.build_set_context(method, 256, 8, 64)
                local = torch.randn(7, 256, requires_grad=True)
                evidence = torch.randn(7, 4, 512)
                evidence[~active] = float('nan')
                evidence.requires_grad_()
                output = net(local, evidence, active, availability)
                safe = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
                self.assertTrue(torch.equal(output, net(local, safe, active, availability)))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(local.grad).all())
                self.assertTrue(torch.isfinite(evidence.grad).all())
                self.assertEqual(int(evidence.grad[~active].count_nonzero()), 0)
                for name, param in net.named_parameters():
                    self.assertIsNotNone(param.grad, (method, name))
                    self.assertTrue(torch.isfinite(param.grad).all(), (method, name))

    def test_smooth_primitive_gradchecks_and_gcn_bias_order(self):
        module = self.module()
        torch.manual_seed(12)
        q = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
        x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        attention = module._PerceiverAttention(4, 2).double()
        self.assertTrue(torch.autograd.gradcheck(attention, (q, x), fast_mode=True))
        graph = module._normalized_graph(torch.tensor([0, 1, 1]), torch.tensor([-1, 0, 1]),
                                         torch.float64)
        block = module._GMTMAB(4, 4, 2, graph_keys=True).double()
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: block(a, b, graph), (q, x),
                                                fast_mode=True))
        gcn = module._GraphLinear(4, 3).double()
        expected = graph @ x @ gcn.linear.weight.T + gcn.linear.bias
        torch.testing.assert_close(gcn(x, graph), expected, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
