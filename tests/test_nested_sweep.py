"""Numerical contracts for the bounded Nested comparison sweep."""
import importlib.util
import unittest

import torch

from gcnet_missing_m3.meaningful_input_new40 import build_new40
from gcnet_missing_m3.meaningful_new40_structure import NestedGNN


METHODS = (
    'nested_ab_plain_gin', 'nested_ab_no_markers', 'nested_ab_no_head_edges',
    'nested_ab_last_layer', 'nested_dim32', 'nested_dim128', 'nested_depth1',
    'nested_depth2', 'nested_groups1', 'nested_groups4',
)


class NestedSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sweep_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('gcnet_missing_m3.nested_sweep'))

    def test_registry_is_bounded_comparison_catalog(self):
        from gcnet_missing_m3.meaningful_new40_registry import NEW40_METHODS, NEW40_VARIANTS
        self.assertEqual(len(NEW40_METHODS), 40)
        self.assertTrue(set(METHODS).issubset(NEW40_VARIANTS))

    def test_default_core_matches_original_initialization_and_forward(self):
        from gcnet_missing_m3.nested_sweep import NestedSweepGNN
        torch.manual_seed(66)
        original = NestedGNN()
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(66)
        current = NestedSweepGNN()
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertEqual(set(original.state_dict()), set(current.state_dict()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(current.state_dict()[name], value, atol=0, rtol=0)
        current.load_state_dict(original.state_dict())
        for columns in (torch.tensor([0]), torch.tensor([0, 1, 2, 9, 17])):
            x = torch.randn(2, len(columns), 64)
            torch.testing.assert_close(current(x, columns, 8), original(x, columns, 8), atol=0, rtol=0)

    def test_all_methods_shapes_identity_masks_and_trainability(self):
        active = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
        local, evidence = torch.randn(3, 16), torch.randn(3, 4, 512)
        evidence = torch.where(active[..., None], evidence, 0.)
        dirty = evidence.masked_fill(~active[..., None], float('nan'))
        dirty_local = local.masked_fill(~active.any(-1)[:, None], float('nan'))
        for method in METHODS:
            with self.subTest(method=method):
                model = build_new40(method, 16, 8, 64)
                clean = model(local, evidence, active, None)
                torch.testing.assert_close(clean[0], local, atol=0, rtol=0)
                torch.testing.assert_close(clean[1], evidence, atol=0, rtol=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                before = model.core.layers[0][0].weight.detach().clone()
                for _ in range(3):
                    optimizer.zero_grad()
                    clean = model(local, evidence, active, None)
                    unsafe = model(dirty_local, dirty, active, None)
                    # Direct TokenAdapter preserves inactive Local; caller masks it.
                    torch.testing.assert_close(clean[0][:2], unsafe[0][:2], atol=0, rtol=0)
                    torch.testing.assert_close(clean[1], unsafe[1], atol=0, rtol=0)
                    self.assertEqual(clean[1].shape, (3, 4, 512))
                    self.assertTrue(all(torch.isfinite(value).all() for value in clean))
                    sum(value.square().mean() for value in clean).backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                    optimizer.step()
                self.assertFalse(torch.equal(before, model.core.layers[0][0].weight))

    def test_grouping_concatenates_all_actual_heads_before_projection(self):
        for groups in (1, 4):
            model = build_new40(f'nested_groups{groups}', 16, 8, 64)
            values = torch.arange(512.).reshape(1, 1, 512).expand(1, 4, -1)
            inputs = []
            hooks = [layer.register_forward_pre_hook(
                lambda module, args: inputs.append(args[0].detach().clone()))
                for layer in model.tokenizer.memory]
            tokens, mask = model.tokenizer(torch.zeros(1, 16), values, torch.ones(1, 4, dtype=torch.bool))
            for hook in hooks:
                hook.remove()
            self.assertEqual(tokens.shape, (1, 1 + 4 * groups, 64))
            self.assertEqual(mask.shape, (1, 1 + 4 * groups))
            torch.testing.assert_close(torch.cat(inputs, -1), values, atol=0, rtol=0)
            self.assertEqual(model.value_dim, 512 // groups)
            self.assertEqual(model.num_heads, groups)

    def test_each_ablation_changes_only_its_named_mechanism(self):
        from gcnet_missing_m3.nested_sweep import NESTED_SWEEP, NestedSweepGNN
        self.assertEqual(tuple(NESTED_SWEEP), METHODS)
        plain = build_new40(METHODS[0]).core
        calls = []
        hook = plain.norms[0].register_forward_hook(lambda *args: calls.append(1))
        plain(torch.randn(1, 5, 64), torch.tensor([0, 1, 2, 9, 17]), 8)
        hook.remove()
        self.assertEqual(len(calls), 1)
        no_markers = build_new40(METHODS[1]).core
        self.assertFalse(any('root_embedding' in name or 'distance' in name
                             for name, _ in no_markers.named_parameters()))
        no_edges = build_new40(METHODS[2]).core
        columns = torch.tensor([0, 1, 2, 9])
        adj = no_edges.adjacency(columns, 8, torch.float32)
        self.assertEqual(adj[1, 3], 0)
        self.assertEqual(adj[1, 2], 1)
        self.assertEqual(adj[0, 3], 1)
        self.assertEqual(build_new40(METHODS[3]).core.pool[0].in_features, 64)
        self.assertEqual(NestedSweepGNN().pool[0].in_features, 192)
        for depth in (1, 2):
            self.assertEqual(len(build_new40(f'nested_depth{depth}').core.layers), depth)
        for dim in (32, 128):
            model = build_new40(f'nested_dim{dim}')
            self.assertEqual(model.local_decoder.in_features, dim)

    def test_public_caller_preserves_first_utterance_and_upper_read_half(self):
        from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
        local, base = torch.randn(3, 2, 256), torch.randn(3, 2, 1024)
        gap = torch.randn(3, 2, 3, 1024)
        availability = torch.tensor([[[1, 0, 1], [0, 1, 0]]]).expand(3, -1, -1)
        umask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        valid = umask.T.bool()
        local = torch.where(valid[..., None], local, 0.)
        base = torch.where(valid[..., None], base, 0.)
        gap = torch.where((valid[..., None] & ~availability.bool())[..., None], gap, 0.)
        for method in METHODS:
            with self.subTest(method=method):
                model = MeaningfulInputAdapter(256, 1024, 1600, method, 8, 64)
                initial = model(local, base, gap, availability, umask)
                for actual, expected in zip(initial, (local, base, gap)):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                # Move zero bridges to exercise the activated caller path.
                with torch.no_grad():
                    model.core.local_decoder.weight.fill_(0.01)
                    for decoder in model.core.memory_decoders:
                        decoder.weight.fill_(0.01)
                changed = model(local, base, gap, availability, umask)
                for actual, expected in zip(changed, (local, base, gap)):
                    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
                    self.assertEqual(actual[~valid].count_nonzero(), 0)
                torch.testing.assert_close(changed[1][..., 512:], base[..., 512:], atol=0, rtol=0)
                torch.testing.assert_close(changed[2][..., 512:], gap[..., 512:], atol=0, rtol=0)
                self.assertEqual(changed[2][availability.bool()].count_nonzero(), 0)

    def test_invalid_grouping_rejects_head_splitting(self):
        with self.assertRaisesRegex(ValueError, 'divide'):
            build_new40('nested_groups4', 16, 6, 64)


if __name__ == '__main__':
    unittest.main()
