import unittest
import torch
from gcnet_missing_m3 import osram
from tests.test_osram_post_grn import PostGRNTests


class HierarchicalEvidenceGateTests(unittest.TestCase):
    def test_identity_masks_and_identity_backbone(self):
        self.assertTrue(hasattr(osram, 'HierarchicalEvidenceGate'))
        helper = PostGRNTests()
        _, local, base, gap, avail, mask = helper.inputs()
        gate = osram.HierarchicalEvidenceGate(4, 3)
        active = torch.cat((mask.T.bool()[..., None], mask.T.bool()[..., None] & ~avail.bool()), -1)
        expected = torch.where(active[..., None], torch.cat((base[..., None, :], gap), -2), 0.)
        torch.testing.assert_close(gate(local, base, gap, avail, mask), expected, rtol=0, atol=0)
        gap[~active[..., 1:]] = float('nan')
        torch.testing.assert_close(gate(local, base, gap, avail, mask), expected, rtol=0, atol=0)
        for training in (False, True):
            torch.manual_seed(42); old = helper.backbone(); rng = torch.get_rng_state()
            torch.manual_seed(42); off = helper.backbone(osram_hierarchical_evidence_gate=False)
            self.assertEqual(set(old.state_dict()), set(off.state_dict()))
            for key, value in old.state_dict().items(): self.assertTrue(torch.equal(value, off.state_dict()[key]))
            torch.manual_seed(42); new = helper.backbone(osram_hierarchical_evidence_gate=True)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            old.train(training); new.train(training)
            torch.manual_seed(5); a, ca = old(*helper.backbone_inputs()); after = torch.get_rng_state()
            torch.manual_seed(5); b, cb = new(*helper.backbone_inputs())
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key in ca: torch.testing.assert_close(ca[key], cb[key], rtol=0, atol=0)

    def test_audio_only_softmax_and_filtered_condition(self):
        self.assertTrue(hasattr(osram, 'HierarchicalEvidenceGate'))
        gate = osram.HierarchicalEvidenceGate(4, 3)
        with torch.no_grad():
            gate.evidence_input.weight.zero_(); gate.evidence_input.bias.zero_()
            gate.type_embedding.weight.zero_(); gate.type_embedding.weight[2, 0] = torch.log(torch.tensor(3.))
            gate.evidence_input.weight[0, 256] = 1.; gate.evidence_output.weight[0, 0] = 1.
        local = torch.zeros(1, 1, 4); base = torch.ones(1, 1, 3); gap = torch.ones(1, 1, 3, 3)
        out = gate(local, base, gap, torch.tensor([[[1., 0., 0.]]]), torch.ones(1, 1))
        torch.testing.assert_close(out[0, 0, :, 0], torch.tensor([.6, 0., 1.8, .6]))
        observed = []
        hook = gate.filtered_relation.register_forward_pre_hook(lambda m, a: observed.append(a[0].detach().clone()))
        with torch.no_grad(): gate.feature_output.bias.fill_(1.)
        gate(local, base, gap, torch.zeros(1, 1, 3), torch.ones(1, 1)); hook.remove()
        torch.testing.assert_close(observed[0], torch.ones_like(observed[0]) * (2 * torch.sigmoid(torch.tensor(1.))))

    def test_config_and_finite_joint_updates(self):
        self.assertTrue(hasattr(osram, 'HierarchicalEvidenceGate'))
        from dataclasses import replace
        from gcnet_missing_m3.train_gcnet import TrainConfig
        config = TrainConfig(backbone_type='osram', training_objective='emotion-only', osram_bidirectional=False,
                             osram_hierarchical_evidence_gate=True)
        for key, value in [('osram_local_evidence_gate', True), ('osram_post_grn', True), ('train_rate_mode', 'conversation-mixed')]:
            with self.assertRaises(ValueError): replace(config, **{key: value})
        helper = PostGRNTests(); model = helper.backbone(osram_hierarchical_evidence_gate=True)
        before = {k: v.clone() for k, v in model.named_parameters()}
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        for _ in range(4):
            opt.zero_grad(); out, _ = model(*helper.backbone_inputs()); (out - torch.randn_like(out)).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            opt.step()
        for name in ('hierarchical_evidence_gate.feature_output.weight', 'hierarchical_evidence_gate.evidence_output.weight', 'emotion_adapter.0.weight', 'local_skip.weight'):
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)

    def test_all_active_only_base_padding_and_gradients(self):
        gate = osram.HierarchicalEvidenceGate(4, 3)
        local = torch.randn(3, 1, 4, requires_grad=True)
        base = torch.randn(3, 1, 3, requires_grad=True)
        gap = torch.randn(3, 1, 3, 3, requires_grad=True)
        avail = torch.tensor([[[1., 1., 1.]], [[0., 0., 0.]], [[float('nan')] * 3]])
        mask = torch.tensor([[1., 1., 0.]])
        with torch.no_grad():
            local[2] = float('nan'); base[2] = float('nan'); gap[0] = float('nan'); gap[2] = float('nan')
        out = gate(local, base, gap, avail, mask)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out[0, 0, 0], base[0, 0], rtol=0, atol=0)
        torch.testing.assert_close(out[1, 0, 1:], gap[1, 0], rtol=0, atol=0)
        self.assertEqual(out[2].count_nonzero(), 0)
        out.square().sum().backward()
        for value in (local, base, gap): self.assertTrue(torch.isfinite(value.grad).all())
        self.assertEqual(gap.grad[0].count_nonzero(), 0)
        self.assertEqual(gap.grad[2].count_nonzero(), 0)
        out = gate(local, base, gap, avail, torch.zeros_like(mask))
        self.assertEqual(out.count_nonzero(), 0)
        self.assertEqual(gate.last_diagnostics['valid_count'], 0)

    def test_ablation_inputs_and_train_eval_diagnostics_without_regularizer(self):
        helper = PostGRNTests()
        for mode in ('local-only', 'local-base', 'local-gap'):
            model = helper.backbone(osram_hierarchical_evidence_gate=True, osram_emotion_ablation=mode)
            captured = []
            hook = model.hierarchical_evidence_gate.register_forward_pre_hook(lambda m, a: captured.append(a))
            model(*helper.backbone_inputs()); hook.remove()
            if mode in ('local-only', 'local-gap'): self.assertEqual(captured[0][1].count_nonzero(), 0)
            if mode in ('local-only', 'local-base'): self.assertEqual(captured[0][2].count_nonzero(), 0)
        from gcnet_missing_m3 import train_gcnet as tr
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        c = tr.TrainConfig(dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
            latent_dim=8, osram_output_dim=19, osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
            osram_bidirectional=False, osram_hierarchical_evidence_gate=True)
        model = _build_model(c, (1, 1, 1))
        batch = [torch.randn(3, 2, 1) for _ in range(6)]
        batch += [torch.zeros(2, 3), torch.ones(2, 3), torch.tensor([[1., -1., 1.], [-1., 1., -1.]]), ['a', 'b']]
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        result = tr.train_epoch(model, [batch, batch], opt, c, tr._schedules(c, 'train'), 0, (1, 1, 1), torch.device('cpu'))
        self.assertAlmostEqual(result['loss'], result['classification_loss'])
        self.assertEqual(result['hierarchical_evidence_gate']['valid_count'], 12)
        self.assertNotIn('local_evidence_gate_regularization', result)
        evaluated, _ = tr.evaluate_rate(model, [batch], tr._schedules(c, 'test')[0], 'CMUMOSI', (1, 1, 1), torch.device('cpu'), False)
        self.assertEqual(evaluated['hierarchical_evidence_gate']['valid_count'], 6)
        with self.assertRaises(ValueError): osram.OSRAMBackbone(latent_dim=4, osram_hierarchical_evidence_gate=True, bidirectional=True)


if __name__ == '__main__': unittest.main()
