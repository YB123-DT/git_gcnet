"""Contracts for Local8 tokenization on the original Nested mean-pool core."""
import unittest

import torch

from gcnet_missing_m3.meaningful_input_new40 import build_new40


METHOD = 'nested_local8_evidence'


class NestedLocal8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_registered_separately_from_bounded_sweep(self):
        from gcnet_missing_m3.meaningful_new40_registry import NEW40_VARIANTS, NESTED_SWEEP
        self.assertIn(METHOD, NEW40_VARIANTS)
        self.assertNotIn(METHOD, NESTED_SWEEP)

    def test_common_parameters_and_factory_rng_match_original(self):
        old_method = 'nested_gnn_rooted_evidence'
        torch.manual_seed(66)
        old = build_new40(old_method)
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(66)
        new = build_new40(METHOD)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        for prefix in ('core', 'tokenizer.memory', 'tokenizer.role_embedding',
                       'tokenizer.head_embedding', 'tokenizer.norm', 'memory_decoders'):
            old_values = dict(old.named_parameters())
            new_values = dict(new.named_parameters())
            for name, value in old_values.items():
                if name.startswith(prefix + '.'):
                    torch.testing.assert_close(new_values[name], value, atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in new.parameters()), 159683)
        self.assertFalse(any('legacy' in name for name, _ in new.named_parameters()))

    def test_contiguous_local_groups_and_actual_memory_heads(self):
        model = build_new40(METHOD)
        local = torch.arange(256.).reshape(1, 256)
        evidence = torch.arange(2048.).reshape(1, 4, 512)
        seen_local, seen_memory = [], []
        hooks = [layer.register_forward_pre_hook(
            lambda module, args: seen_local.append(args[0].detach().clone()))
            for layer in model.tokenizer.local]
        hooks += [layer.register_forward_pre_hook(
            lambda module, args: seen_memory.append(args[0].detach().clone()))
            for layer in model.tokenizer.memory]
        tokens, mask = model.tokenizer(local, evidence, torch.ones(1, 4, dtype=torch.bool))
        for hook in hooks:
            hook.remove()
        self.assertEqual(tokens.shape, (1, 40, 64))
        self.assertTrue(mask.all())
        self.assertEqual(len(seen_local), 8)
        self.assertTrue(all(value.shape == (1, 32) for value in seen_local))
        torch.testing.assert_close(torch.cat(seen_local, -1), local, atol=0, rtol=0)
        torch.testing.assert_close(torch.stack(seen_memory, 2), evidence.reshape(1, 4, 8, 64), atol=0, rtol=0)
        self.assertEqual(model.tokenizer.role_ids.tolist(), sum(([role] * 8 for role in range(5)), []))
        self.assertEqual(model.tokenizer.head_ids.tolist(), list(range(8)) * 5)
        self.assertTrue(all((layer.in_features, layer.out_features) == (64, 32)
                            for layer in model.local_decoders))

    def test_all_availability_patterns_pack_only_real_nodes(self):
        from gcnet_missing_m3.meaningful_blocks_common import active_groups
        model = build_new40(METHOD)
        availability = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                                     [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)
        active = torch.cat((torch.ones(7, 1, dtype=torch.bool), ~availability), 1)
        tokens, mask = model.tokenizer(torch.randn(7, 256), torch.randn(7, 4, 512), active)
        sizes = {}
        for rows, columns, packed in active_groups(tokens, mask):
            for row in rows.tolist():
                sizes[row] = packed.shape[1]
            self.assertEqual(columns[:8].tolist(), list(range(8)))
        self.assertEqual([sizes[row] for row in range(7)], [16, 24, 24, 24, 32, 32, 32])
        columns = torch.arange(40)
        adjacency = model.core.adjacency(columns, 8, torch.float32)
        self.assertFalse(adjacency.diag().bool().any())
        self.assertEqual(adjacency[3, 39], 1)  # every Local group connects globally
        self.assertEqual(adjacency[8, 15], 1)  # same role
        self.assertEqual(adjacency[8, 16], 1)  # same actual head
        self.assertEqual(adjacency[8, 17], 0)

    def test_zero_identity_nan_masks_and_three_step_learning(self):
        model = build_new40(METHOD)
        active = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]], dtype=torch.bool)
        local = torch.randn(3, 256)
        evidence = torch.randn(3, 4, 512).masked_fill(~active[..., None], 0.)
        dirty = evidence.masked_fill(~active[..., None], float('nan'))
        clean = model(local, evidence, active, None)
        torch.testing.assert_close(clean[0], local, atol=0, rtol=0)
        torch.testing.assert_close(clean[1], evidence, atol=0, rtol=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.core.layers[0][0].weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            clean = model(local, evidence, active, None)
            unsafe = model(local.masked_fill(~active.any(-1)[:, None], float('nan')), dirty, active, None)
            torch.testing.assert_close(clean[0][:2], unsafe[0][:2], atol=0, rtol=0)
            torch.testing.assert_close(clean[1], unsafe[1], atol=0, rtol=0)
            self.assertTrue(all(torch.isfinite(value).all() for value in clean))
            sum(value.square().mean() for value in clean).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
        self.assertFalse(torch.equal(before, model.core.layers[0][0].weight))
        self.assertTrue(all(not torch.equal(layer.weight, torch.zeros_like(layer.weight))
                            for layer in model.local_decoders))

    def test_public_first_utterance_padding_and_upper_512_preserved(self):
        from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
        model = MeaningfulInputAdapter(256, 1024, 1600, METHOD, 8, 64)
        local, base, gap = torch.randn(3, 2, 256), torch.randn(3, 2, 1024), torch.randn(3, 2, 3, 1024)
        availability = torch.tensor([[[1, 0, 1], [0, 1, 0]]]).expand(3, -1, -1)
        umask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        valid = umask.T.bool()
        local = local.masked_fill(~valid[..., None], 0.)
        base = base.masked_fill(~valid[..., None], 0.)
        gap = gap.masked_fill((~valid[..., None] | availability.bool())[..., None], 0.)
        initial = model(local, base, gap, availability, umask)
        for actual, expected in zip(initial, (local, base, gap)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with torch.no_grad():
            for layer in (*model.core.local_decoders, *model.core.memory_decoders):
                layer.weight.fill_(0.01)
        changed = model(local, base, gap, availability, umask)
        for actual, expected in zip(changed, (local, base, gap)):
            torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
            self.assertEqual(actual[~valid].count_nonzero(), 0)
        torch.testing.assert_close(changed[1][..., 512:], base[..., 512:], atol=0, rtol=0)
        torch.testing.assert_close(changed[2][..., 512:], gap[..., 512:], atol=0, rtol=0)
        self.assertEqual(changed[2][availability.bool()].count_nonzero(), 0)
        self.assertTrue((changed[0][1] - local[1]).abs().sum() > 0)


if __name__ == '__main__':
    unittest.main()
