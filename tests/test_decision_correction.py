"""Contracts for evidence-centered task-space corrections."""
import importlib.util
import inspect
import subprocess
import types
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from gcnet_missing_m3.osram import OSRAMBackbone, MODALITIES
from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model


def config(**kwargs):
    values = dict(dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
                  disable_unused_aux_modules=True, osram_bidirectional=False, latent_dim=8,
                  osram_output_dim=10, osram_num_heads=2, osram_key_dim=3,
                  osram_value_dim=4, dropout=.3, osram_decision_correction=True)
    values.update(kwargs)
    return TrainConfig(**values)


def inputs():
    x = torch.randn(5, 2, 3)
    u = torch.tensor([[1., 1, 1, 1, 1], [1., 1, 1, 0, 0]])
    a = torch.tensor([[[1., 0, 1], [0, 1, 0]]] * 5)
    a[~u.T.bool()] = 0
    return [x], a, torch.zeros(2, 5, dtype=torch.long), u, [5, 3]


class DecisionHeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def head(self, classes=1):
        self.assertIsNotNone(importlib.util.find_spec('gcnet_missing_m3.decision_correction'),
                             'the centered decision head must be implemented')
        from gcnet_missing_m3.decision_correction import EvidenceCenteredDecisionHead
        return EvidenceCenteredDecisionHead(8, 8, classes)

    def evidence(self):
        local, base = torch.randn(5, 2, 8), torch.randn(5, 2, 8)
        gap = torch.randn(5, 2, 3, 8)
        u = torch.tensor([[0., 1, 0, 1, 1], [0., 0, 1, 1, 0]])
        a = torch.tensor([[[1., 0, 1], [0, 1, 0]]] * 5)
        a[~u.T.bool()] = 0
        return local, base, gap, a, u

    def test_exact_centering_after_nonzero_updates_train_and_eval(self):
        for classes in (1, 6):
            h = self.head(classes)
            args = self.evidence()
            optimizer = torch.optim.Adam(h.parameters(), lr=.01)
            for _ in range(3):
                optimizer.zero_grad()
                outputs = h(*args)
                valid = args[-1].T.bool()
                loss = sum(o[valid].square().mean() if classes == 1 else
                           torch.nn.functional.cross_entropy(o[valid], torch.arange(int(valid.sum())) % classes)
                           for k, o in outputs.items() if k in ('local', 'base', 'full')) / 3
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in h.parameters() if p.grad is not None))
                optimizer.step()
            self.assertGreater(h.base_delta[-1].weight.abs().sum(), 0)
            self.assertGreater(h.gap_delta[-1].weight.abs().sum(), 0)
            self.assertIsNone(h.base_delta[-1].bias)
            self.assertIsNone(h.gap_delta[-1].bias)
            for training in (True, False):
                h.train(training)
                local, base, gap, a, u = args
                zero = h(local, torch.zeros_like(base), torch.zeros_like(gap), a, u)
                self.assertTrue(torch.equal(zero['local'], zero['base']))
                self.assertTrue(torch.equal(zero['base'], zero['full']))
                all_observed = h(local, base, gap, torch.where(u.T.bool()[..., None], 1., 0.).expand_as(a), u)
                self.assertTrue(torch.equal(all_observed['base'], all_observed['full']))
                out = h(*args)
                first = u.T.bool() & (u.T.long().cumsum(0) == 1)
                for key in ('delta_base', 'delta_gap'):
                    self.assertEqual(out[key][first].count_nonzero(), 0)
                for value in out.values():
                    self.assertEqual(value[~u.T.bool()].count_nonzero(), 0)

    def test_nan_inactive_slots_are_removed_before_networks(self):
        h = self.head()
        with torch.no_grad():
            h.base_delta[-1].weight.normal_()
            h.gap_delta[-1].weight.normal_()
        args = self.evidence()
        expected = h(*args)
        local, base, gap, a, u = (x.clone() for x in args)
        local[~u.T.bool()] = float('nan')
        base[~u.T.bool()] = float('nan')
        gap[a.bool() | ~u.T.bool()[..., None]] = float('nan')
        actual = h(local, base, gap, a, u)
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)
        sum(value.square().sum() for value in actual.values()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in h.parameters()))

    def test_both_centered_evaluations_are_differentiable(self):
        h = self.head()
        with torch.no_grad():
            h.base_delta[-1].weight.normal_(std=.1)
            h.gap_delta[-1].weight.normal_(std=.1)
        captured = []
        handles = [m.register_forward_hook(lambda m, i, o: (o.retain_grad(), captured.append(o)) and None)
                   for m in (h.base_delta, h.gap_delta)]
        out = h(*self.evidence())
        out['full'].square().sum().backward()
        self.assertEqual(len(captured), 4)
        self.assertTrue(all(t.grad is not None and t.grad.abs().sum() > 0 for t in captured))
        for handle in handles:
            handle.remove()

    def test_mlp_shape_initialization_and_conditioners(self):
        h = self.head(6)
        self.assertEqual(h.local_head[1].in_features, 8)
        self.assertEqual(h.base_delta[1].in_features, 16)
        self.assertEqual(h.gap_delta[1].in_features, 43)
        for module in (h.local_head, h.base_delta, h.gap_delta):
            self.assertEqual(tuple(type(layer) for layer in module),
                             (torch.nn.LayerNorm, torch.nn.Linear, torch.nn.GELU, torch.nn.Linear))
            self.assertEqual(module[1].out_features, 128)
        seen = {'base': [], 'gap': []}
        handles = [module.register_forward_pre_hook(lambda m, args, key=key: seen[key].append(args[0]))
                   for key, module in (('base', h.base_delta), ('gap', h.gap_delta))]
        out = h(*self.evidence())
        self.assertTrue(torch.equal(out['local'], out['base']))
        self.assertTrue(torch.equal(out['base'], out['full']))
        self.assertTrue(torch.equal(seen['base'][0][..., :8], seen['base'][1][..., :8]))
        self.assertTrue(torch.equal(seen['gap'][0][..., :16], seen['gap'][1][..., :16]))
        self.assertTrue(torch.equal(seen['gap'][0][..., -3:], seen['gap'][1][..., -3:]))
        for handle in handles:
            handle.remove()


class DecisionIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_switch_and_config_guards(self):
        self.assertIn('osram_decision_correction', inspect.signature(MissingM3GraphModel).parameters)
        self.assertIn('osram_decision_correction', inspect.signature(OSRAMBackbone).parameters)
        cfg = config()
        for changes in (dict(osram_relation_block=True), dict(osram_gap_increment_filter=True),
                        dict(osram_post_grn=True), dict(paired_history_views=True),
                        dict(osram_readout_fusion='local-gated'), dict(readout_type='availability-affine'),
                        dict(osram_bidirectional=True), dict(training_objective='joint'),
                        dict(emotion_loss_mode='pattern-balanced'), dict(initial_backbone_checkpoint='old.pt')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes)
        args = build_parser().parse_args(['--audio-feature', 'a', '--text-feature', 't',
                                         '--video-feature', 'v', '--output-dir', 'unused',
                                         '--osram-decision-correction'])
        self.assertTrue(args.osram_decision_correction)

    def test_one_scan_no_flat_and_joint_learning_checkpoint(self):
        cfg = config()
        net = _build_model(cfg, (1, 1, 1))
        args = inputs()
        old = ('osram.emotion_adapter.', 'osram.local_skip.', 'osram.emotion_norm.', 'smax_fc.')
        for name, parameter in net.named_parameters():
            self.assertEqual(parameter.requires_grad, not name.startswith(old), name)
        optimizer = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=.01)
        before = {n: p.detach().clone() for n, p in net.named_parameters()}
        for _ in range(3):
            optimizer.zero_grad()
            with patch.object(net.osram, '_scan', wraps=net.osram._scan) as scan, \
                 patch.object(net.observed_set, 'forward', wraps=net.observed_set.forward) as encoder, \
                 patch.object(net.osram.local_path, 'forward', wraps=net.osram.local_path.forward) as local, \
                 patch.object(net.osram.emotion_adapter, 'forward', side_effect=AssertionError('Flat adapter called')), \
                 patch.object(net.osram.local_skip, 'forward', side_effect=AssertionError('Flat skip called')), \
                 patch.object(net.osram.emotion_norm, 'forward', side_effect=AssertionError('Flat norm called')), \
                 patch.object(net.smax_fc, 'forward', side_effect=AssertionError('Flat classifier called')):
                logits = net(*args)[0]
            self.assertEqual((scan.call_count, encoder.call_count, local.call_count), (1, 1, 1))
            out = net.last_decision_outputs
            self.assertIs(logits, out['full'])
            self.assertEqual(net.osram.last_decision_evidence['base'].shape[-1], 8)
            sum(out[key][args[3].T.bool()].square().mean() for key in ('local', 'base', 'full')).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
            optimizer.step()
        for prefix in ('decision_head.base_delta.', 'decision_head.gap_delta.', 'observed_set.', 'osram.local_path.', 'osram.key_projectors.'):
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in net.named_parameters() if n.startswith(prefix)), prefix)
        restored = _build_model(cfg, (1, 1, 1))
        restored.load_state_dict(net.state_dict(), strict=True)
        net.eval(); restored.eval()
        self.assertTrue(torch.equal(net(*args)[0], restored(*args)[0]))
        self.assertIsNotNone(net.last_decision_outputs)

    def test_enable_preserves_initialization_memory_and_optimizer_membership(self):
        from gcnet_missing_m3.train_gcnet import _optimizer_parameter_groups
        cfg = config()
        off = _build_model(replace(cfg, osram_decision_correction=False), (1, 1, 1))
        rng = torch.get_rng_state()
        on = _build_model(cfg, (1, 1, 1))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in off.state_dict().items():
            self.assertTrue(torch.equal(value, on.state_dict()[name]), name)
        args = inputs()
        for training in (True, False):
            off.train(training); on.train(training)
            torch.manual_seed(17); off(*args)
            torch.manual_seed(17); on(*args)
            for name in ('base', 'gap', 'local'):
                self.assertTrue(torch.equal(off.last_osram_context[name], on.last_osram_context[name]), name)
        groups, provenance = _optimizer_parameter_groups(on, cfg)
        members = {id(p) for group in groups for p in group['params']}
        self.assertEqual(members, {id(p) for p in on.parameters() if p.requires_grad})
        self.assertTrue({id(p) for p in on.decision_head.parameters()} <= members)
        self.assertIn('classifier', provenance)
        self.assertEqual(on.inactive_flat_parameter_count,
                         sum(p.numel() for p in on.parameters() if not p.requires_grad))

    def test_scan_nan_poison_padding_and_inactive_gap(self):
        net = _build_model(config(), (1, 1, 1)).eval()
        with torch.no_grad():
            net.decision_head.base_delta[-1].weight.normal_(std=.1)
            net.decision_head.gap_delta[-1].weight.normal_(std=.1)
        args = inputs()
        expected = net(*args)[0]
        original = net.osram._scan

        def poisoned(*scan_args, **kwargs):
            base, gap, diagnostics = original(*scan_args, **kwargs)
            base, gap = base.clone(), gap.clone()
            base[~args[3].T.bool()] = float('nan')
            gap[args[1].bool() | ~args[3].T.bool()[..., None]] = float('nan')
            return base, gap, diagnostics

        with patch.object(net.osram, '_scan', side_effect=poisoned):
            actual = net(*args)[0]
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual[~args[3].T.bool()].count_nonzero(), 0)

    def test_default_off_backbone_matches_prechange_source(self):
        source = subprocess.check_output(['git', 'show', '29a9d7c:gcnet_missing_m3/osram.py'], text=True)
        legacy = types.ModuleType('legacy_decision_osram')
        exec(compile(source, 'legacy_decision_osram.py', 'exec'), legacy.__dict__)
        kw = dict(latent_dim=8, output_dim=10, num_heads=2, key_dim=3, value_dim=4,
                  dropout=.4, bidirectional=False)
        torch.manual_seed(45); old = legacy.OSRAMBackbone(**kw); rng = torch.get_rng_state()
        torch.manual_seed(45); new = OSRAMBackbone(**kw)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
        with torch.no_grad():
            old.emotion_adapter[-1].weight.normal_(std=.1)
        new.load_state_dict(old.state_dict(), strict=True)
        _, a, q, u, lengths = inputs()
        node = torch.randn(5, 2, 8)
        args = (node, {name: torch.randn_like(node) for name in MODALITIES}, a, q, u, lengths)
        for training in (True, False):
            old.train(training); new.train(training)
            torch.manual_seed(72); expected = old(*args)[0]; rng = torch.get_rng_state()
            torch.manual_seed(72); actual = new(*args)[0]
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_default_off_matches_prechange_model_weights_outputs_and_rng(self):
        source = subprocess.check_output(['git', 'show', '29a9d7c:gcnet_missing_m3/model.py'], text=True)
        legacy = types.ModuleType('gcnet_missing_m3.legacy_decision_model')
        legacy.__package__ = 'gcnet_missing_m3'
        exec(compile(source, 'legacy_decision_model.py', 'exec'), legacy.__dict__)
        from tests.test_completion_memory_write import model_kwargs
        kw = model_kwargs()
        kw.update(completion_path='none', dropout=.3, projector_dropout=.2)
        torch.manual_seed(91); old = legacy.MissingM3GraphModel(**kw)
        rng = torch.get_rng_state()
        torch.manual_seed(91); new = MissingM3GraphModel(**kw)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
        from tests.test_completion_memory_write import inputs as relation_inputs
        x, a, q, u, lengths = relation_inputs()
        args = ([x], a, q, u, lengths)
        for training in (True, False):
            old.train(training); new.train(training)
            torch.manual_seed(19); expected = old(*args)[0]; rng = torch.get_rng_state()
            torch.manual_seed(19); actual = new(*args)[0]
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
