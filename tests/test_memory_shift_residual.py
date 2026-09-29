import unittest

import torch

from gcnet_missing_m3 import osram


class MemoryShiftResidualTests(unittest.TestCase):
    def module(self):
        self.assertTrue(hasattr(osram, 'MemoryShiftFilter'))
        return osram.MemoryShiftFilter(4, 3, 5, relation_dim=7)

    def inputs(self):
        return (torch.randn(3, 2, 4), torch.randn(3, 2, 3),
                torch.randn(3, 2, 3, 3),
                torch.tensor([[[1., 0, 1], [0, 1, 0]]] * 3),
                torch.tensor([[1., 1, 1], [1, 1, 0]]), torch.randn(3, 2, 5))

    def test_equation_scalar_filters_and_active_count_denominator(self):
        f = self.module()
        local, base, gap, avail, mask, anchor = self.inputs()
        with torch.no_grad():
            f.residual_projector.weight.fill_(.2)
            f.residual_projector.bias.fill_(.3)
            for p in f.filter.parameters(): p.zero_()
        out = f(local, base, gap, avail, mask, anchor)
        valid = mask.T.bool()
        active = torch.cat((valid[..., None], valid[..., None] & ~avail.bool()), -1)
        evidence = torch.cat((base.unsqueeze(2), gap), 2)
        q = f.local_relation(local).unsqueeze(2).expand(-1, -1, 4, -1)
        types = f.evidence_type.weight.view(1, 1, 4, -1).expand_as(q)
        k = f.memory_relation(evidence) + types
        shift = k - q
        gate = torch.sigmoid(f.filter(torch.cat((q, k, shift, shift.abs(), q*k, types), -1)))
        self.assertEqual(gate.shape, (3, 2, 4, 1))
        filtered = (gate * shift * active[..., None]).sum(2) / active.sum(-1).clamp_min(1)[..., None]
        expected = f.residual_projector(filtered)
        torch.testing.assert_close(out[valid], expected[valid])
        self.assertEqual(out[~valid].count_nonzero().item(), 0)
        diag = f.last_diagnostics
        self.assertEqual(diag['valid_count'], 5)
        self.assertEqual(diag['active_counts'], dict(base=5, gap_audio=2, gap_text=3, gap_visual=2))
        self.assertEqual(set(diag['filter_mean'].values()), {.5})
        self.assertAlmostEqual(diag['filtered_shift_norm'], filtered[valid].norm(dim=-1).mean().item())
        self.assertAlmostEqual(diag['residual_anchor_norm_ratio'],
            (out[valid].norm(dim=-1) / anchor[valid].norm(dim=-1).clamp_min(1e-8)).mean().item())

    def test_mask_nan_padding_and_bias_type_leakage(self):
        f = self.module()
        local, base, gap, avail, mask, anchor = self.inputs()
        with torch.no_grad():
            f.residual_projector.weight.fill_(.1)
            f.residual_projector.bias.fill_(.5)
            f.memory_relation.bias.fill_(10.)
            f.evidence_type.weight.fill_(10.)
        expected = f(local, base, gap, avail, mask, anchor)
        valid = mask.T.bool(); inactive = avail.bool() | ~valid[..., None]
        gap[inactive] = float('nan')
        for x in (local, base, avail, anchor): x[~valid] = float('nan')
        for x in (local, base, gap): x.requires_grad_()
        actual = f(local, base, gap, avail, mask, anchor)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertEqual(gap.grad[inactive].count_nonzero().item(), 0)
        for x in (local, base, gap): self.assertTrue(torch.isfinite(x.grad).all())
        mask.zero_()
        out = f(local, base, gap, avail, mask, anchor)
        self.assertEqual(out.count_nonzero().item(), 0)
        self.assertEqual(f.last_diagnostics['valid_count'], 0)
        self.assertEqual(set(f.last_diagnostics['filter_mean'].values()), {None})

    def test_zero_initialization_then_filter_learns_second_step(self):
        f = self.module()
        args = self.inputs()
        out = f(*args)
        self.assertEqual(out.count_nonzero().item(), 0)
        out.sum().backward()
        self.assertGreater(f.residual_projector.weight.grad.abs().sum().item(), 0)
        self.assertEqual(f.filter[-1].weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            for p in f.parameters(): p.add_(p.grad, alpha=-.01)
        f.zero_grad()
        f(*args).square().sum().backward()
        for module in (f.filter, f.local_relation, f.memory_relation, f.evidence_type):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_shapes_and_binary_availability(self):
        f = self.module()
        args = list(self.inputs())
        args[3][0, 0, 0] = .2
        with self.assertRaisesRegex(ValueError, 'binary'): f(*args)
        args = list(self.inputs()); args[2] = args[2][..., :2]
        with self.assertRaisesRegex(ValueError, 'shapes'): f(*args)

    def test_exact_flat_initial_train_eval_and_upstream_writes(self):
        kwargs = dict(latent_dim=8, output_dim=10, num_heads=2,
                      key_dim=3, value_dim=4, dropout=.3, bidirectional=False)
        torch.manual_seed(81); flat = osram.OSRAMBackbone(**kwargs)
        rng = torch.get_rng_state()
        torch.manual_seed(81)
        new = osram.OSRAMBackbone(**kwargs, osram_readout_fusion='memory-shift-residual')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in flat.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        for module in (new.emotion_adapter, new.local_skip, new.emotion_norm):
            self.assertTrue(all(p.requires_grad for p in module.parameters()))
        node = torch.randn(4, 2, 8)
        latents = {name: torch.randn_like(node) for name in osram.MODALITIES}
        avail = torch.tensor([[[1., 0, 1], [0, 1, 0]]] * 4)
        mask = torch.tensor([[1., 1, 1, 1], [1, 1, 0, 0]])
        avail[~mask.T.bool()] = 0
        args = (node, latents, avail, torch.zeros(2, 4, dtype=torch.long), mask)
        for training in (True, False):
            flat.train(training); new.train(training)
            old_writes, new_writes = [], []
            torch.manual_seed(17)
            old_hidden, old_contexts = flat(*args, post_write_observer=lambda *x: old_writes.append(x))
            after = torch.get_rng_state()
            torch.manual_seed(17)
            hidden, contexts = new(*args, post_write_observer=lambda *x: new_writes.append(x))
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            self.assertTrue(torch.equal(hidden, old_hidden))
            for key in contexts: self.assertTrue(torch.equal(contexts[key], old_contexts[key]))
            self.assertEqual(len(old_writes), len(new_writes))
            for before, after in zip(old_writes, new_writes):
                self.assertEqual(before[0], after[0])
                for a, b in zip(before[1:], after[1:]): self.assertTrue(torch.equal(a, b))
        with torch.no_grad(): new.memory_shift_filter.residual_projector.weight.fill_(.1)
        old_writes, new_writes = [], []
        old_hidden, old_contexts = flat(*args, post_write_observer=lambda *x: old_writes.append(x))
        hidden, contexts = new(*args, post_write_observer=lambda *x: new_writes.append(x))
        self.assertFalse(torch.equal(hidden, old_hidden))
        self.assertEqual(hidden[~mask.T.bool()].count_nonzero().item(), 0)
        for key in contexts: self.assertTrue(torch.equal(contexts[key], old_contexts[key]))
        for before, after in zip(old_writes, new_writes):
            for a, b in zip(before[1:], after[1:]): self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
