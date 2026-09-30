import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from gcnet_missing_m3.osram import HierarchicalEvidenceGate
from tests.test_osram_post_grn import PostGRNTests


class FeatureOnlyGateTests(unittest.TestCase):
    def test_feature_only_bypasses_competition_and_preserves_initialization(self):
        torch.manual_seed(12)
        full = HierarchicalEvidenceGate(4, 3)
        torch.manual_seed(12)
        gate = HierarchicalEvidenceGate(4, 3, feature_only=True)
        for key, value in full.state_dict().items():
            self.assertTrue(torch.equal(value, gate.state_dict()[key]), key)
        _, local, base, gap, avail, mask = PostGRNTests().inputs()
        active = torch.cat((mask.T.bool()[..., None], mask.T.bool()[..., None] & ~avail.bool()), -1)
        evidence = torch.where(active[..., None], torch.cat((base[..., None, :], gap), -2), 0.)
        with patch('torch.softmax', side_effect=AssertionError('competition must not execute')):
            torch.testing.assert_close(gate(local, base, gap, avail, mask), evidence, rtol=0, atol=0)
            with torch.no_grad():
                gate.feature_output.bias.fill_(1.)
                gate.evidence_output.weight.fill_(float('nan'))
                gate.filtered_relation.weight.fill_(float('nan'))
            gap[~active[..., 1:]] = float('nan')
            out = gate(local, base, gap, avail, mask)
        torch.testing.assert_close(out, evidence * (2 * torch.sigmoid(torch.tensor(1.))))
        out.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in gate.parameters() if p.grad is not None))
        for module in (gate.filtered_relation, gate.evidence_input, gate.evidence_output):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        for name in ('base', 'gap_a', 'gap_t', 'gap_v'):
            if gate.last_diagnostics[name]['active_count']:
                self.assertEqual(gate.last_diagnostics[name]['reweight_mean'], 1.)

    def test_guards_identity_joint_updates_and_padding(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig
        with self.assertRaises(ValueError): TrainConfig(osram_hierarchical_feature_only=True)
        config = TrainConfig(backbone_type='osram', training_objective='emotion-only',
            osram_bidirectional=False, osram_hierarchical_evidence_gate=True,
            osram_hierarchical_feature_only=True)
        with self.assertRaises(ValueError): replace(config, train_rate_mode='conversation-mixed')
        helper = PostGRNTests()
        with self.assertRaises(ValueError): helper.backbone(osram_hierarchical_feature_only=True)
        torch.manual_seed(3); old = helper.backbone()
        torch.manual_seed(3); model = helper.backbone(osram_hierarchical_evidence_gate=True, osram_hierarchical_feature_only=True)
        for training in (False, True):
            old.train(training); model.train(training)
            torch.manual_seed(4); expected, contexts = old(*helper.backbone_inputs())
            torch.manual_seed(4); actual, gated_contexts = model(*helper.backbone_inputs())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key in contexts: torch.testing.assert_close(contexts[key], gated_contexts[key], rtol=0, atol=0)
        before = {k: v.clone() for k, v in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for _ in range(4):
            optimizer.zero_grad()
            out, _ = model(*helper.backbone_inputs())
            (out - torch.randn_like(out)).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
        for name in ('hierarchical_evidence_gate.feature_output.weight', 'emotion_adapter.0.weight', 'local_skip.weight'):
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)

    def test_builder_train_epoch_no_additional_loss(self):
        from gcnet_missing_m3 import train_gcnet as tr
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        config = tr.TrainConfig(dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
            latent_dim=8, osram_output_dim=19, osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
            osram_bidirectional=False, osram_hierarchical_evidence_gate=True,
            osram_hierarchical_feature_only=True)
        model = _build_model(config, (1, 1, 1))
        self.assertTrue(model.osram.hierarchical_evidence_gate.feature_only)
        batch = [torch.randn(3, 2, 1) for _ in range(6)]
        batch += [torch.zeros(2, 3), torch.ones(2, 3), torch.tensor([[1., -1., 1.], [-1., 1., -1.]]), ['a', 'b']]
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        result = tr.train_epoch(model, [batch, batch], optimizer, config,
            tr._schedules(config, 'train'), 0, (1, 1, 1), torch.device('cpu'))
        self.assertAlmostEqual(result['loss'], result['classification_loss'])
        self.assertNotIn('local_evidence_gate_regularization', result)
        self.assertEqual(result['hierarchical_evidence_gate']['valid_count'], 12)
        for name in ('base', 'gap_a', 'gap_t', 'gap_v'):
            if result['hierarchical_evidence_gate'][name]['active_count']:
                self.assertEqual(result['hierarchical_evidence_gate'][name]['reweight_mean'], 1.)


if __name__ == '__main__': unittest.main()
