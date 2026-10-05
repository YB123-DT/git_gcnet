"""Focused CPU contracts for the separately registered Nested GNN variant."""
import unittest

import torch
from torch import nn

from gcnet_missing_m3.meaningful_input_new40 import build_new40
from gcnet_missing_m3.meaningful_new40_structure import NestedGNN, topology


VARIANT = 'nested_gnn_rootaware_evidence'


class NestedRootAwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_separate_variant_preserves_forty_catalog(self):
        from gcnet_missing_m3 import meaningful_new40_registry as registry
        self.assertEqual(len(registry.NEW40_METHODS), 40)
        self.assertNotIn(VARIANT, registry.NEW40_METHODS)
        self.assertIn(VARIANT, getattr(registry, 'NEW40_VARIANTS', ()))

    def test_pool_preserves_root_and_excludes_it_from_neighbor_mean(self):
        model = NestedGNN(root_aware=True)
        states, inputs = [], []
        handles = [norm.register_forward_hook(
            lambda module, args, output: states.append(output.detach().clone()))
            for norm in model.norms]
        handles.append(model.root_pool.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].detach().clone())))
        columns = torch.tensor([0, 1, 2, 3])
        x = torch.randn(2, 4, 64)
        model(x, columns, 2)
        for handle in handles:
            handle.remove()
        a = topology(columns, 2, x.dtype)
        self.assertEqual(len(inputs), 12)
        for root in range(4):
            neighbors = (a[root].bool() | (torch.arange(4) == root)).nonzero(as_tuple=True)[0]
            marker = neighbors == root
            for depth in range(3):
                h = states[root * 3 + depth]
                expected = torch.cat((h[:, marker].squeeze(1), h[:, ~marker].mean(1)), -1)
                torch.testing.assert_close(inputs[root * 3 + depth], expected, rtol=0, atol=0)

    def test_singleton_neighbor_half_is_exact_zero(self):
        model = NestedGNN(root_aware=True)
        inputs = []
        handle = model.root_pool.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].detach()))
        output = model(torch.randn(2, 1, 64), torch.tensor([0]), 2)
        handle.remove()
        self.assertEqual(len(inputs), 3)
        for value in inputs:
            self.assertEqual(value[:, 64:].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(output).all())

    def test_only_one_shared_linear_adds_8256_parameters(self):
        original, variant = NestedGNN(), NestedGNN(root_aware=True)
        extra = set(dict(variant.named_parameters())) - set(dict(original.named_parameters()))
        self.assertEqual(extra, {'root_pool.weight', 'root_pool.bias'})
        self.assertIsInstance(variant.root_pool, nn.Linear)
        self.assertEqual((variant.root_pool.in_features, variant.root_pool.out_features), (128, 64))
        self.assertEqual(sum(p.numel() for p in variant.parameters()) -
                         sum(p.numel() for p in original.parameters()), 8256)

    def test_variant_keeps_original_and_downstream_initialization_rng(self):
        torch.manual_seed(66)
        original = build_new40('nested_gnn_rooted_evidence', 16, 2, 4)
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(66)
        variant = build_new40(VARIANT, 16, 2, 4)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        variant_state = variant.state_dict()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(variant_state[name], value, atol=0, rtol=0)

    def test_original_forward_is_exact_historical_mean_pooling(self):
        model = NestedGNN()
        x, columns = torch.randn(2, 4, 64), torch.tensor([0, 1, 2, 3])
        a = topology(columns, 2, x.dtype)
        roots = []
        for root in range(x.shape[1]):
            neighbors = (a[root].bool() | (torch.arange(len(columns)) == root)).nonzero(as_tuple=True)[0]
            induced, marker = a[neighbors][:, neighbors], (neighbors == root).long()
            h = x[:, neighbors] + model.root_embedding(marker) + model.distance(1 - marker)
            states = []
            for layer, norm, eps in zip(model.layers, model.norms, model.epsilon):
                h = norm(layer((1 + eps) * h + induced @ h))
                states.append(h.mean(1))
            roots.append(model.pool(torch.cat(states, -1)))
        rooted = torch.stack(roots, 1)
        expected = model.readout(torch.cat((x, rooted, rooted.mean(1, keepdim=True).expand_as(x)), -1))
        torch.testing.assert_close(model(x, columns, 2), expected, atol=0, rtol=0)

    def test_raw_factory_core_has_finite_nonzero_projector_gradients(self):
        model = build_new40(VARIANT, 16, 2, 4).core
        x = torch.randn(2, 5, 64, requires_grad=True)
        value = model(x, torch.tensor([0, 1, 2, 3, 4]), 2)
        self.assertEqual(value.shape, x.shape)
        self.assertTrue(torch.isfinite(value).all())
        value.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.root_pool.weight.grad.abs().sum(), 0)

    def test_public_adapter_identity_mask_safety_and_training_updates(self):
        from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
        torch.manual_seed(66)
        adapter = MeaningfulInputAdapter(256, 1024, 1600, VARIANT, 8, 64)
        local, base = torch.randn(3, 7, 256), torch.randn(3, 7, 1024)
        gap = torch.randn(3, 7, 3, 1024)
        availability = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                     [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).expand(3, -1, -1)
        umask = torch.ones(7, 3)
        umask[:, -1] = 0
        valid = umask.T.bool()
        local = torch.where(valid[..., None], local, 0.)
        base = torch.where(valid[..., None], base, 0.)
        gap = torch.where((valid[..., None] & ~availability.bool())[..., None], gap, 0.)
        expected = (local, base, gap)
        actual = adapter(local, base, gap, availability, umask)
        for value, original in zip(actual, expected):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
        dirty_local = local.masked_fill(~valid[..., None], float('nan'))
        dirty_base = base.masked_fill(~valid[..., None], float('inf'))
        dirty_gap = gap.masked_fill(availability.bool()[..., None], float('nan'))
        dirty_gap[~valid] = float('inf')
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)
        initial_projection = adapter.core.core.root_pool.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            clean = adapter(local, base, gap, availability, umask)
            dirty = adapter(dirty_local, dirty_base, dirty_gap, availability, umask)
            for a, b in zip(clean, dirty):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            sum(value.square().mean() for value in clean).backward()
            for parameter in adapter.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertFalse(torch.equal(initial_projection, adapter.core.core.root_pool.weight))
        values = adapter(local, base, gap, availability, umask)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(values, expected)))
        for original, value in zip(expected, values):
            self.assertTrue(torch.equal(value[0], original[0]))
            self.assertEqual(value[~valid].count_nonzero(), 0)
            self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(values[2][availability.bool()].count_nonzero(), 0)
        torch.testing.assert_close(values[1][..., 512:], base[..., 512:], rtol=0, atol=0)
        torch.testing.assert_close(values[2][..., 512:], gap[..., 512:], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
