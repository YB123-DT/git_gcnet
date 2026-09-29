import unittest

import torch

from gcnet_missing_m3 import osram


class PostGRNTests(unittest.TestCase):
    def module(self):
        self.assertTrue(hasattr(osram, 'PostGRN'), 'optional post-Flat GRN is missing')
        return osram.PostGRN(4, 3, 7, dropout=0.)

    def inputs(self):
        return (torch.randn(3, 2, 7), torch.randn(3, 2, 4),
                torch.randn(3, 2, 3), torch.randn(3, 2, 3, 3),
                torch.tensor([[[1., 0, 1], [0, 1, 0]]] * 3),
                torch.tensor([[1., 1, 1], [1, 1, 0]]))

    def test_identity_equation_diagnostics_and_no_post_norm(self):
        m = self.module(); args = self.inputs(); valid = args[-1].T.bool()
        out = m(*args)
        self.assertTrue(torch.equal(out[valid], args[0][valid]))
        self.assertEqual(out[~valid].count_nonzero().item(), 0)
        self.assertEqual(m.linear_x.out_features, 128)
        self.assertEqual(m.linear_gate.out_features, 7)
        self.assertEqual(m.condition_norm.normalized_shape, (19,))
        self.assertEqual(m.linear_gate.bias.count_nonzero().item(), 0)
        with torch.no_grad(): m.linear_value.bias.fill_(2.)
        x, local, base, gap, avail, mask = args
        c = torch.cat((local, base, torch.where(~avail.bool()[..., None], gap, 0.).flatten(2), avail), -1)
        z = m.dropout(m.linear2(torch.nn.functional.elu(m.linear_x(x) + m.linear_c(m.condition_norm(c)))))
        gate = m.linear_gate(z).sigmoid(); residual = gate * m.linear_value(z)
        actual = m(*args)
        torch.testing.assert_close(actual[valid], (x + residual)[valid], rtol=0, atol=0)
        d = m.last_diagnostics
        self.assertEqual(d['valid_count'], 5)
        self.assertAlmostEqual(d['gate_mean'], gate[valid].mean().item())
        self.assertAlmostEqual(d['gate_saturation_fraction'], ((gate[valid] <= .05) | (gate[valid] >= .95)).float().mean().item())
        self.assertAlmostEqual(d['gated_residual_flat_norm_ratio'], (residual[valid].norm(dim=-1) / x[valid].norm(dim=-1).clamp_min(1e-8)).mean().item())

    def test_nan_inactive_gap_padding_and_finite_gradients(self):
        m = self.module(); args = list(self.inputs()); valid = args[-1].T.bool()
        with torch.no_grad(): m.linear_value.weight.fill_(.1)
        expected = m(*args)
        inactive = args[4].bool() | ~valid[..., None]
        args[3][inactive] = float('nan')
        for x in args[:3] + [args[4]]: x[~valid] = float('nan')
        for x in args[:4]: x.requires_grad_()
        actual = m(*args)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        for x in args[:4]: self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(args[3].grad[inactive].count_nonzero().item(), 0)
        args[-1].zero_()
        self.assertEqual(m(*args).count_nonzero().item(), 0)
        self.assertEqual(m.last_diagnostics['valid_count'], 0)

    def backbone(self, **kwargs):
        return osram.OSRAMBackbone(latent_dim=4, output_dim=7, num_heads=1,
                                   key_dim=2, value_dim=2, dropout=.2, bidirectional=False, **kwargs)

    def backbone_inputs(self):
        availability = torch.tensor([[[1., 0, 1], [0, 1, 0]]] * 3)
        availability[-1, -1] = 0
        return (torch.randn(3, 2, 4), {n: torch.randn(3, 2, 4) for n in osram.MODALITIES},
                availability, torch.zeros(2, 3, dtype=torch.long),
                torch.tensor([[1., 1, 1], [1, 1, 0]]))

    def test_flat_initialization_rng_and_train_eval_identity(self):
        torch.manual_seed(42); old = self.backbone(); rng = torch.get_rng_state()
        torch.manual_seed(42); off = self.backbone(osram_post_grn=False)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(old.state_dict()), set(off.state_dict()))
        torch.manual_seed(42); new = self.backbone(osram_post_grn=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in old.state_dict().items(): self.assertTrue(torch.equal(v, new.state_dict()[k]))
        self.assertTrue(all(p.requires_grad for p in new.parameters()))
        args = self.backbone_inputs()
        for train in (False, True):
            old.train(train); off.train(train); new.train(train)
            torch.manual_seed(9); expected, contexts = old(*args)
            torch.manual_seed(9); disabled, _ = off(*args)
            torch.manual_seed(9); actual, new_contexts = new(*args)
            self.assertTrue(torch.equal(expected, disabled))
            self.assertTrue(torch.equal(expected, actual))
            for k in contexts: self.assertTrue(torch.equal(contexts[k], new_contexts[k]))
        self.assertIn('post_grn', new.last_diagnostics)
        with self.assertRaisesRegex(ValueError, 'flat'):
            self.backbone(osram_post_grn=True, osram_readout_fusion='memory-shift-residual')

    def test_condition_respects_emotion_ablation(self):
        for ablation in ('local-only', 'local-base', 'local-gap'):
            m = self.backbone(osram_post_grn=True, osram_emotion_ablation=ablation)
            captured = []
            handle = m.post_grn.register_forward_pre_hook(lambda module, args: captured.append(args))
            m(*self.backbone_inputs()); handle.remove()
            if ablation in ('local-only', 'local-gap'): self.assertEqual(captured[0][2].count_nonzero().item(), 0)
            if ablation in ('local-only', 'local-base'): self.assertEqual(captured[0][3].count_nonzero().item(), 0)

    def test_joint_updates_finite_over_multiple_steps(self):
        m = self.backbone(osram_post_grn=True); args = self.backbone_inputs()
        before = {k: v.detach().clone() for k, v in m.named_parameters()}
        opt = torch.optim.Adam(m.parameters(), lr=.01)
        for _ in range(4):
            opt.zero_grad(); out, _ = m(*args)
            (out - torch.randn_like(out)).square().mean().backward()
            for p in m.parameters():
                if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all())
            opt.step()
        for key in ('post_grn.linear_value.weight', 'post_grn.linear_x.weight', 'post_grn.linear_gate.weight',
                    'emotion_adapter.4.weight', 'local_skip.weight', 'query_projector.weight'):
            self.assertFalse(torch.equal(before[key], dict(m.named_parameters())[key]), key)


if __name__ == '__main__':
    unittest.main()
