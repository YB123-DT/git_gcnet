"""Independent equation and masking checks for the four graph transfers."""
import importlib
import math
import unittest

import torch
from torch import nn


torch.set_num_threads(1)
METHODS = ('rrn_evidence', 'egt_evidence', 'residual_gated_graph_evidence', 'pna_evidence')


class MeaningfulGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(314)
        try:
            self.g = importlib.import_module('gcnet_missing_m3.meaningful_blocks_graph')
        except ModuleNotFoundError as exc:
            self.fail(f'Graph implementation is not available yet: {exc}')

    @staticmethod
    def batch(heads=2, value_dim=3):
        availability = torch.tensor([[bool(k & (1 << b)) for b in range(3)] for k in range(1, 8)])
        active = torch.cat([torch.ones(7, 1, dtype=torch.bool), ~availability], 1)
        return torch.randn(7, 6), torch.randn(7, 4, heads * value_dim), active, availability

    def test_topology_has_true_head_links_and_correct_degrees(self):
        adjacency, relations = self.g.graph_structure(8)
        self.assertEqual(tuple(adjacency.shape), (33, 33))
        self.assertEqual(tuple(relations.shape), (33, 33, 14))
        self.assertFalse(adjacency.diagonal().any())
        self.assertTrue(torch.equal(adjacency, adjacency.T))
        self.assertTrue(adjacency[1, 9])  # Same true head across Base / Gap-A.
        self.assertFalse(adjacency[1, 10])  # Different source AND head.
        self.assertTrue(adjacency[1, 2])  # Within Base heads.
        self.assertTrue(adjacency[0, 32])
        for groups in (1, 2, 3):
            sub = adjacency[:1 + 8 * groups, :1 + 8 * groups]
            self.assertEqual(int(sub[0].sum()), 8 * groups)
            self.assertTrue((sub[1:].sum(-1) == 7 + groups).all())

    def test_all_seven_masks_poison_batch_independence_and_gradients(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = self.g.build_graph(method, 6, 2, 3)
                self.assertEqual(model.output_dim, 128)
                local, evidence, active, availability = self.batch()
                local.requires_grad_()
                evidence.requires_grad_()
                out = model(local, evidence, active, availability)
                self.assertEqual(tuple(out.shape), (7, 128))
                self.assertTrue(torch.isfinite(out).all())
                poisoned = evidence.detach().clone()
                poisoned[~active] = float('nan')
                poisoned_out = model(local.detach(), poisoned, active, availability)
                torch.testing.assert_close(out, poisoned_out, rtol=0, atol=0)
                single = model(local[:1], evidence[:1], active[:1], availability[:1])
                torch.testing.assert_close(out[:1], single, rtol=2e-5, atol=3e-6)
                other = evidence.detach().clone()
                other[1:] *= 1e4
                torch.testing.assert_close(out[:1], model(local, other, active, availability)[:1], rtol=0, atol=0)
                weight = torch.linspace(-1, 1, 128)
                (out * weight).square().mean().backward()
                self.assertTrue(torch.isfinite(local.grad).all())
                self.assertTrue(torch.isfinite(evidence.grad).all())
                self.assertEqual(float(evidence.grad[~active].abs().sum()), 0)
                core_grads = [p.grad for p in model.core.parameters() if p.requires_grad]
                self.assertTrue(core_grads)
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in core_grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in core_grads), 0)
                parameter = next(p for p in model.core.parameters() if p.grad.abs().sum() > 0)
                previous = parameter.detach().clone()
                torch.optim.SGD(model.parameters(), lr=0.01).step()
                self.assertFalse(torch.equal(previous, parameter))

    def test_no_active_history_and_empty_batch_are_safe(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = self.g.build_graph(method, 6, 2, 3)
                out = model(torch.full((2, 6), float('nan')),
                            torch.full((2, 4, 6), float('nan')),
                            torch.zeros(2, 4, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool))
                self.assertTrue(torch.equal(out, torch.zeros_like(out)))
                empty = model(torch.empty(0, 6), torch.empty(0, 4, 6),
                              torch.empty(0, 4, dtype=torch.bool), torch.empty(0, 3, dtype=torch.bool))
                self.assertEqual(tuple(empty.shape), (0, 128))

    def test_complete_core_depths_and_no_cross_batch_normalization(self):
        models = {m: self.g.build_graph(m, 6, 2, 3) for m in METHODS}
        self.assertEqual(models['rrn_evidence'].core.steps, 5)
        self.assertEqual(len(models['egt_evidence'].core.layers), 3)
        self.assertEqual(len(models['residual_gated_graph_evidence'].core.cells), 3)
        self.assertEqual(len(models['pna_evidence'].core.layers), 4)
        for model in models.values():
            self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()))
        for layer in models['pna_evidence'].core.layers:
            self.assertEqual(len(layer.messages), 4)
        # Every persistent edge update must feed a subsequent node update.
        egt = models['egt_evidence'].core.layers
        self.assertTrue(egt[0].edge_update and egt[1].edge_update)
        self.assertFalse(egt[2].edge_update)

    def test_rrn_shared_iterations_match_explicit_equation(self):
        core = self.g.RRNCore(dim=4, steps=2)
        x = torch.randn(1, 3, 4)
        adjacency = torch.tensor([[[False, True, True], [True, False, True], [True, True, False]]])
        relations = torch.randn(1, 3, 3, 14)
        mask = torch.ones(1, 3, dtype=torch.bool)
        h = x.clone()
        hidden, cell = torch.zeros_like(x), torch.zeros_like(x)
        edge = core.edge_embedding(relations)
        for _ in range(2):
            incoming = []
            for i in range(3):
                incoming.append(sum(core.message(torch.cat([h[:, j], h[:, i], edge[:, i, j]], -1))
                                    for j in range(3) if i != j))
            update = core.post(torch.cat([torch.stack(incoming, 1), x], -1))
            hidden, cell = core.cell(update.reshape(-1, 4), (hidden.reshape(-1, 4), cell.reshape(-1, 4)))
            h, hidden, cell = hidden.reshape(1, 3, 4), hidden.reshape(1, 3, 4), cell.reshape(1, 3, 4)
        torch.testing.assert_close(core(x, mask, adjacency, relations), h)

    def test_egt_attention_and_edge_update_match_equations(self):
        layer = self.g.EGTLayer(dim=4, edge_dim=3, heads=2, edge_update=True)
        h, e = torch.randn(1, 3, 4), torch.randn(1, 3, 3, 3)
        mask = torch.tensor([[True, True, False]])
        hn, en = layer.node_norm(h), layer.edge_norm(e)
        q, k, v = layer.qkv(hn).reshape(1, 3, 3, 2, 2).unbind(2)
        scores = torch.einsum('bihd,bjhd->bijh', q, k) / math.sqrt(2)
        scores = scores.clamp(-5, 5) + layer.edge_bias(en)
        pair = mask[:, :, None] & mask[:, None, :]
        gates = torch.where(pair[..., None], layer.edge_gate(en).sigmoid(), 0.)
        attention = scores.masked_fill(~mask[:, None, :, None], torch.finfo(scores.dtype).min).softmax(2) * gates
        value = torch.einsum('bijh,bjhd->bihd', attention, v)
        value *= torch.log1p(gates.sum(2))[..., None]
        hp = h + layer.node_output(value.reshape(1, 3, 4))
        expected_h = hp + layer.node_ffn(layer.node_ffn_norm(hp))
        ep = e + layer.edge_output(scores)
        expected_e = ep + layer.edge_ffn(layer.edge_ffn_norm(ep))
        actual_h, actual_e = layer(h, e, mask)
        torch.testing.assert_close(actual_h, torch.where(mask[..., None], expected_h, 0.))
        torch.testing.assert_close(actual_e, torch.where(pair[..., None], expected_e, 0.))

    def test_gated_cell_is_two_full_graph_convolutions(self):
        cell = self.g.GatedGraphCell(dim=4)
        h = torch.randn(1, 3, 4)
        adjacency = torch.tensor([[[False, True, False], [True, False, True], [False, True, False]]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        def oracle(conv, x):
            values = []
            for i in range(3):
                value = conv.self_map(x[:, i]) + conv.node_bias
                for j in range(3):
                    if adjacency[0, i, j]:
                        gate = (conv.gate_target(x[:, i]) + conv.gate_source(x[:, j]) + conv.edge_bias).sigmoid()
                        value = value + gate * conv.neighbor_map(x[:, j])
                values.append(value)
            return conv.norm(torch.stack(values, 1))
        expected = (oracle(cell.conv2, oracle(cell.conv1, h).relu()) + cell.residual(h)).relu()
        torch.testing.assert_close(cell(h, mask, adjacency), expected)

    def test_pna_uses_all_statistics_scalers_and_correct_neighbor_axis(self):
        messages = torch.tensor([[[[[100.]], [[1.]], [[3.]]],
                                  [[[7.]], [[200.]], [[99.]]],
                                  [[[12.]], [[13.]], [[300.]]]]])
        adjacency = torch.tensor([[[False, True, True], [True, False, False], [False, False, False]]])
        delta = 2.
        got = self.g.pna_aggregate(messages, adjacency, delta)
        self.assertEqual(tuple(got.shape), (1, 3, 1, 12))
        a = torch.tensor([2., math.sqrt(1 + 1e-5), 1., 3.])
        scale = math.log(3) / delta
        torch.testing.assert_close(got[0, 0, 0], torch.cat([a, a * scale, a / scale]))
        self.assertTrue(torch.equal(got[0, 2], torch.zeros_like(got[0, 2])))
        self.assertAlmostEqual(self.g.pna_degree_reference(8), 2.3785469096888603, places=12)

    def test_factory_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            self.g.build_graph('graph_by_name_only', 6, 2, 3)

    def test_real_head_shapes_pack_active_nodes_and_state_roundtrip(self):
        availability = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=torch.bool)
        active = torch.cat([torch.ones(3, 1, dtype=torch.bool), ~availability], -1)
        local, evidence = torch.randn(3, 256), torch.randn(3, 4, 512)
        for method in METHODS:
            with self.subTest(method=method):
                model = self.g.build_graph(method, 256, 8, 64)
                calls = []
                hook = model.core.register_forward_pre_hook(lambda module, args: calls.append((args[0].shape[1], bool(args[1].all()))))
                rng = torch.random.get_rng_state().clone()
                out = model(local, evidence, active, availability)
                hook.remove()
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertEqual(sorted(calls), [(9, True), (17, True), (25, True)])
                self.assertTrue(torch.isfinite(out).all())
                copy = self.g.build_graph(method, 256, 8, 64)
                copy.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(out, copy(local, evidence, active, availability), rtol=0, atol=0)
                # Availability remains authoritative even if active is overpermissive.
                poisoned = evidence.clone()
                poisoned[~active] = float('inf')
                torch.testing.assert_close(out, model(local, poisoned, torch.ones_like(active), availability), rtol=0, atol=0)
                (out * torch.linspace(-1, 1, 128)).square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.core.parameters()))

    def test_full_pna_towers_match_independent_layer_equation(self):
        layer = self.g.PNALayer(dim=8, towers=2, delta=2.)
        x, relations = torch.randn(1, 3, 8), torch.randn(1, 3, 3, 14)
        mask = torch.ones(1, 3, dtype=torch.bool)
        adjacency = torch.tensor([[[False, True, True], [True, False, False], [False, True, False]]])
        edges = layer.edge_embedding(relations)
        expected_nodes = []
        for i in range(3):
            neighbors = [j for j in range(3) if adjacency[0, i, j]]
            towers = []
            for t in range(2):
                own = x[:, i, t * 4:(t + 1) * 4]
                messages = torch.stack([layer.messages[t](torch.cat([
                    own, x[:, j, t * 4:(t + 1) * 4], edges[:, i, j]], -1)) for j in neighbors], 1)
                mean = messages.mean(1)
                std = ((messages.square().mean(1) - mean.square()).clamp_min(0) + 1e-5).sqrt()
                stats = torch.cat([mean, std, messages.amin(1), messages.amax(1)], -1)
                scale = math.log1p(len(neighbors)) / 2.
                towers.append(layer.updates[t](torch.cat([own, stats, stats * scale, stats / scale], -1)))
            expected_nodes.append(layer.norm(layer.mixing(torch.cat(towers, -1))).relu())
        expected = torch.stack(expected_nodes, 1)
        torch.testing.assert_close(layer(x, mask, adjacency, relations), expected, rtol=1e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
