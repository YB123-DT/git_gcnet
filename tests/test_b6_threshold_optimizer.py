"""Threshold-only B6 initialization and auditable optimizer-rate isolation."""
import inspect
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
from gcnet_missing_m3.meaningful_input_structured import _LearnedRobustPCA
from tests.test_b6_structure_config import LegacyPCA, METHOD


class SmallStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.osram = nn.Module()
        self.osram.other = nn.Linear(2, 2)
        self.osram.meaningful_block = MeaningfulInputAdapter(8, 1024, 12, METHOD, 8, 64)
        self.observed_set = nn.Linear(2, 2)
        self.smax_fc = nn.Linear(2, 1)


class B6ThresholdOptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(66)

    def config(self, **kwargs):
        from gcnet_missing_m3.train_gcnet import TrainConfig
        defaults = dict(osram_meaningful_block=METHOD, backbone_type='osram',
            training_objective='emotion-only', disable_unused_aux_modules=True,
            osram_bidirectional=False, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64)
        defaults.update(kwargs)
        return TrainConfig(**defaults)

    def test_threshold_config_routes_and_cli(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
        from gcnet_missing_m3.model import MissingM3GraphModel
        from gcnet_missing_m3.osram import OSRAMBackbone
        for cls in (TrainConfig, MissingM3GraphModel, OSRAMBackbone, MeaningfulInputAdapter):
            self.assertEqual(inspect.signature(cls).parameters['osram_rpca_threshold_init'].default, .1)
        args = build_parser().parse_args(['--audio-feature', 'a', '--text-feature', 't',
            '--video-feature', 'v', '--output-dir', 'unused', '--osram-rpca-threshold-init', '.3',
            '--osram-rpca-lr-multiplier', '10'])
        self.assertEqual((args.osram_rpca_threshold_init, args.osram_rpca_lr_multiplier), (.3, 10.))
        from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
        config = self.config(dataset='MELD', device='cpu', backbone_type='osram',
            training_objective='emotion-only', disable_unused_aux_modules=True,
            osram_bidirectional=False, osram_num_heads=8, osram_key_dim=64,
            osram_value_dim=64, osram_output_dim=12, latent_dim=8,
            osram_rpca_threshold_init=.3)
        model = _build_model(config, (3, 4, 5), torch.device('cpu'))
        expected = torch.tensor([.3 * .8 ** t for t in range(1, 15)])
        torch.testing.assert_close(F.softplus(model.osram.meaningful_block.core.threshold_logits), expected)

    def test_default_threshold_preserves_legacy_checkpoint_forward_and_gradients(self):
        old = LegacyPCA()
        new = _LearnedRobustPCA(threshold_init=.1)
        for name, tensor in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[name], tensor, atol=0, rtol=0)
        new.load_state_dict(old.state_dict(), strict=True)
        y1 = torch.randn(2, 12, 4, requires_grad=True)
        y2 = y1.detach().clone().requires_grad_()
        f1, f2 = old.decompose(y1), new._decompose(y2)
        torch.testing.assert_close(f1, f2, atol=0, rtol=0)
        f1.sum().backward(); f2.sum().backward()
        torch.testing.assert_close(y1.grad, y2.grad, atol=0, rtol=0)
        for name in ('threshold_logits', 'step_logits'):
            torch.testing.assert_close(getattr(old, name).grad, getattr(new, name).grad, atol=0, rtol=0)

    def test_threshold_changes_only_threshold_parameters_and_masks_are_safe(self):
        legacy = _LearnedRobustPCA()
        changed = _LearnedRobustPCA(threshold_init=.5)
        self.assertFalse(torch.equal(legacy.threshold_logits, changed.threshold_logits))
        torch.testing.assert_close(legacy.step_logits, changed.step_logits, atol=0, rtol=0)
        torch.testing.assert_close(legacy.gamma, changed.gamma, atol=0, rtol=0)
        with torch.no_grad(): changed.gamma.fill_(.3)
        active = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0]]).bool()
        evidence = torch.randn(2, 4, 8, requires_grad=True)
        dirty = evidence.masked_fill(~active[..., None], float('nan'))
        _, output = changed(torch.randn(2, 8), dirty, active, torch.zeros(2, 3))
        self.assertEqual(output[~active].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertTrue(torch.isfinite(evidence.grad).all())
        for parameter in changed.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_svd_initializer_is_independent_of_learned_threshold_initialization(self):
        original_svd = torch.linalg.svd
        inputs = []
        def record(matrix, **kwargs):
            inputs.append(matrix.detach().clone())
            return original_svd(matrix, **kwargs)
        y = torch.randn(2, 12, 4)
        with patch('torch.linalg.svd', side_effect=record):
            _LearnedRobustPCA(threshold_init=.01)._decompose(y)
            _LearnedRobustPCA(threshold_init=1.)._decompose(y)
        torch.testing.assert_close(inputs[0], inputs[1], atol=0, rtol=0)

    def test_optimizer_only_actual_core_parameters_receive_multiplier(self):
        from gcnet_missing_m3.train_gcnet import _optimizer_parameter_groups
        model = SmallStudent()
        config = self.config(learning_rate=.001, backbone_lr_multiplier=.5,
            projector_lr_multiplier=2., classifier_lr_multiplier=3.)
        normal, normal_record = _optimizer_parameter_groups(model, config)
        changed, changed_record = _optimizer_parameter_groups(model, replace(config, osram_rpca_lr_multiplier=10.))
        self.assertEqual(list(normal_record), ['backbone', 'projector', 'classifier'])
        self.assertEqual([g['lr'] for g in normal], [.0005, .002, .003])
        expected_core = {id(p) for p in model.osram.meaningful_block.core.parameters()}
        self.assertEqual(len(expected_core), 3)
        actual_ids = [id(p) for g in changed for p in g['params']]
        self.assertEqual(len(actual_ids), len(set(actual_ids)))
        self.assertEqual(set(actual_ids), {id(p) for p in model.parameters()})
        before = {id(p): g['lr'] for g in normal for p in g['params']}
        after = {id(p): g['lr'] for g in changed for p in g['params']}
        for key in before:
            self.assertAlmostEqual(after[key], before[key] * (10 if key in expected_core else 1))
        self.assertEqual(changed_record['rpca']['parameter_count'], 29)
        self.assertEqual(changed_record['rpca']['learning_rate'], .005)
        again, again_record = _optimizer_parameter_groups(model, replace(config, osram_rpca_lr_multiplier=1.))
        self.assertEqual(again_record, normal_record)
        self.assertEqual([[id(p) for p in g['params']] for g in again],
                         [[id(p) for p in g['params']] for g in normal])

    def test_config_rejects_invalid_values_and_incompatible_mode(self):
        config = self.config()
        for name in ('osram_rpca_threshold_init', 'osram_rpca_lr_multiplier'):
            for value in (True, False, 0, -1, float('nan'), float('inf'), '0.1'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(config, **{name: value})
        for value in (True, 0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): _LearnedRobustPCA(threshold_init=value)
        with self.assertRaises(ValueError):
            replace(config, osram_meaningful_block='none', osram_rpca_lr_multiplier=2.)
        with self.assertRaises(ValueError):
            replace(config, pretrained_learning_rate=.0001, osram_rpca_lr_multiplier=2.)

    def test_differential_group_optimizer_state_roundtrip(self):
        from gcnet_missing_m3.train_gcnet import _optimizer_parameter_groups
        config = self.config(osram_rpca_lr_multiplier=10.)
        model = SmallStudent()
        groups, _ = _optimizer_parameter_groups(model, config)
        optimizer = torch.optim.Adam(groups)
        sum(p.square().sum() for p in model.parameters()).backward(); optimizer.step()
        resumed = SmallStudent()
        resumed.load_state_dict(model.state_dict(), strict=True)
        new_groups, _ = _optimizer_parameter_groups(resumed, config)
        restored = torch.optim.Adam(new_groups)
        restored.load_state_dict(optimizer.state_dict())
        self.assertEqual([g['lr'] for g in restored.param_groups], [g['lr'] for g in optimizer.param_groups])
        self.assertEqual(len(restored.state), len(optimizer.state))


if __name__ == '__main__':
    unittest.main()
