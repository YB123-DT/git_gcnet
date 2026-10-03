"""Optional external readouts preserve the original Flat backbone contract."""
from dataclasses import replace
import inspect
import subprocess
import types
import unittest
from unittest.mock import patch

import torch

from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
from tests.test_completion_memory_write import inputs, model_kwargs


def config(method='film', **changes):
    values = dict(dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
                  disable_unused_aux_modules=True, osram_bidirectional=False, latent_dim=8,
                  osram_output_dim=10, osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
                  dropout=.3, projector_dropout=.2, osram_readout_candidate=method)
    values.update(changes)
    return tr.TrainConfig(**values)


class ReadoutCandidateIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def model(self, method='film'):
        return _build_model(config(method), (3, 4, 5))

    def args(self):
        x, a, q, u, lengths = inputs()
        return [x], a, q, u, lengths

    def test_flag_config_and_cli_guards(self):
        self.assertIn('osram_readout_candidate', inspect.signature(tr.TrainConfig).parameters)
        self.assertIn('osram_readout_candidate', inspect.signature(MissingM3GraphModel).parameters)
        self.assertIn('osram_readout_candidate', inspect.signature(OSRAMBackbone).parameters)
        cfg = config()
        for changes in (dict(osram_readout_candidate='unknown'), dict(osram_readout_fusion='local-gated'),
                        dict(osram_bidirectional=True), dict(osram_forward_slot_reuse=True),
                        dict(osram_relation_block=True), dict(osram_relation_dual_readout=True),
                        dict(osram_gap_increment_filter=True), dict(osram_decision_correction=True),
                        dict(osram_post_grn=True), dict(osram_local_skip_gate=True),
                        dict(osram_memory_only_adapter=True), dict(osram_history_input_gate=True),
                        dict(osram_local_evidence_gate=True), dict(osram_hierarchical_evidence_gate=True),
                        dict(osram_hierarchical_feature_only=True), dict(osram_history_query_adapter=True),
                        dict(paired_history_views=True), dict(classification_completion=True),
                        dict(training_objective='joint'), dict(emotion_loss_mode='pattern-balanced'),
                        dict(train_rate_mode='conversation-mixed'), dict(readout_type='availability-affine'),
                        dict(initial_backbone_checkpoint='old.pt'), dict(joint_pretrain_checkpoint='old.pt')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes)
        parsed = tr.build_parser().parse_args(['--audio-feature', 'a', '--text-feature', 't',
                                               '--video-feature', 'v', '--output-dir', 'unused',
                                               '--osram-readout-candidate', 'film'])
        self.assertEqual(parsed.osram_readout_candidate, 'film')
        self.assertEqual(tr.TrainConfig().osram_readout_candidate, 'none')

    def test_default_off_has_no_new_state_and_strict_loads_old_state(self):
        old_osram = types.ModuleType('legacy_readout_candidate_osram')
        source = subprocess.check_output(['git', 'show', '5eb6061:gcnet_missing_m3/osram.py'], text=True)
        exec(compile(source, 'legacy_readout_candidate_osram.py', 'exec'), old_osram.__dict__)
        old_model = types.ModuleType('gcnet_missing_m3.legacy_readout_candidate_model')
        old_model.__package__ = 'gcnet_missing_m3'
        source = subprocess.check_output(['git', 'show', '5eb6061:gcnet_missing_m3/model.py'], text=True)
        exec(compile(source, 'legacy_readout_candidate_model.py', 'exec'), old_model.__dict__)
        old_model.OSRAMBackbone = old_osram.OSRAMBackbone
        kw = model_kwargs()
        kw.update(completion_path='none', dropout=.3, projector_dropout=.2)
        torch.manual_seed(19)
        old = old_model.MissingM3GraphModel(**kw)
        rng = torch.get_rng_state()
        torch.manual_seed(19)
        off = MissingM3GraphModel(**kw, osram_readout_candidate='none')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(any('readout_candidate' in name for name in off.state_dict()))
        self.assertEqual(old.state_dict().keys(), off.state_dict().keys())
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, off.state_dict()[name]), name)
        off.load_state_dict(old.state_dict(), strict=True)
        args = self.args()
        for training in (False, True):
            old.train(training); off.train(training)
            torch.manual_seed(17); expected = old(*args)[0]; rng = torch.get_rng_state()
            torch.manual_seed(17); actual = off(*args)[0]
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA cfg84 Flat equivalence')
    def test_cuda_cfg84_zero_init_matches_original_flat_and_rng(self):
        cfg = config(latent_dim=256, osram_output_dim=1600, osram_num_heads=8,
                     osram_key_dim=64, osram_value_dim=64, dropout=.5, projector_dropout=.1)
        old = _build_model(replace(cfg, osram_readout_candidate='none'), (512, 1024, 1024)).cuda()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        enabled = _build_model(cfg, (512, 1024, 1024)).cuda()
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        with torch.no_grad():
            old.osram.emotion_adapter[-1].weight.normal_(std=.005)
        enabled.load_state_dict(old.state_dict(), strict=False)
        x = torch.randn(4, 2, 2560, device='cuda')
        a = torch.tensor([[[1., 0, 1], [0, 1, 0]]] * 4, device='cuda')
        u = torch.tensor([[1., 1, 1, 1], [1, 1, 0, 0]], device='cuda')
        a[~u.T.bool()] = 0
        q = torch.zeros(2, 4, dtype=torch.long, device='cuda')
        for training in (True, False):
            old.train(training); enabled.train(training)
            before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            expected = old([x], a, q, u, [4, 2])[0]
            after_cpu, after_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            torch.set_rng_state(before_cpu); torch.cuda.set_rng_state(before_cuda)
            actual = enabled([x], a, q, u, [4, 2])[0]
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(after_cpu, torch.get_rng_state()))
            self.assertTrue(torch.equal(after_cuda, torch.cuda.get_rng_state()))

    def test_enabled_init_rng_flat_identity_and_single_scan(self):
        old = self.model('none')
        rng = torch.get_rng_state()
        enabled = self.model()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, enabled.state_dict()[name]), name)
        with torch.no_grad():
            old.osram.emotion_adapter[-1].weight.normal_(std=.05)
        enabled.load_state_dict(old.state_dict(), strict=False)
        args = self.args()
        for training in (False, True):
            old.train(training); enabled.train(training)
            torch.manual_seed(11); expected = old(*args)[0]; rng = torch.get_rng_state()
            torch.manual_seed(11)
            with patch.object(enabled.osram, '_scan', wraps=enabled.osram._scan) as scan, \
                 patch.object(enabled.osram.emotion_adapter, 'forward', wraps=enabled.osram.emotion_adapter.forward) as adapter, \
                 patch.object(enabled.osram.local_skip, 'forward', wraps=enabled.osram.local_skip.forward) as skip, \
                 patch.object(enabled.smax_fc, 'forward', wraps=enabled.smax_fc.forward) as head:
                actual = enabled(*args)[0]
            self.assertEqual((scan.call_count, adapter.call_count, skip.call_count, head.call_count), (1, 1, 1, 1))
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for name in ('base', 'gap', 'local'):
                self.assertTrue(torch.equal(old.last_osram_context[name], enabled.last_osram_context[name]), name)
            self.assertIn('readout_candidate', enabled.osram.last_diagnostics)

    def test_finite_joint_updates_and_strict_checkpoint_rebuild(self):
        net = self.model()
        self.assertTrue(all(p.requires_grad for p in net.parameters()))
        groups, _ = tr._optimizer_parameter_groups(net, config())
        members = {id(p) for group in groups for p in group['params']}
        self.assertEqual(members, {id(p) for p in net.parameters()})
        optimizer = torch.optim.Adam(groups)
        before = {name: p.detach().clone() for name, p in net.named_parameters()}
        args = self.args()
        for _ in range(3):
            optimizer.zero_grad()
            logits = net(*args)[0]
            loss = tr._task_loss('CMUMOSI', logits, torch.randn_like(args[3]), args[3])
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
            optimizer.step()
        for prefix in ('osram.readout_candidate.', 'osram.emotion_adapter.', 'osram.local_skip.', 'smax_fc.', 'observed_set.'):
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in net.named_parameters() if n.startswith(prefix)), prefix)
        restored = self.model()
        restored.load_state_dict(net.state_dict(), strict=True)
        net.eval(); restored.eval()
        self.assertTrue(torch.equal(net(*args)[0], restored(*args)[0]))

    def test_training_keeps_single_original_task_loss(self):
        cfg = config()
        net = self.model()
        # Raw loader order: host A/T/V, guest A/T/V, qmask, umask, labels, IDs.
        batch = [torch.randn(6, 2, width) for width in (3, 4, 5, 3, 4, 5)]
        batch += [torch.zeros(2, 6), torch.ones(2, 6), torch.tensor([[1., -1.] * 3, [-1., 1.] * 3]), ['a', 'b']]
        groups, _ = tr._optimizer_parameter_groups(net, cfg)
        with patch.object(net.osram, '_scan', wraps=net.osram._scan) as scan, \
             patch.object(tr, '_task_loss', wraps=tr._task_loss) as task:
            metrics = tr.train_epoch(net, [batch], torch.optim.Adam(groups), cfg,
                                     tr._schedules(cfg, 'train'), 0, (3, 4, 5), torch.device('cpu'))
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(task.call_count, 1)
        self.assertEqual(metrics['loss'], metrics['classification_loss'])
        self.assertEqual(metrics['jepa_loss'], 0)
        self.assertEqual(metrics['model_forward_count'], 1)
        self.assertEqual(metrics['optimizer_steps'], 1)
        self.assertNotIn('decision_correction', metrics)

    def test_nan_inactive_gap_and_emotion_ablation_are_safe(self):
        net = self.model().eval()
        with torch.no_grad():
            net.osram.emotion_adapter[-1].weight.normal_(std=.05)
            net.osram.readout_candidate.output.weight.normal_(std=.05)
        args = self.args()
        expected = net(*args)[0]
        original = net.osram._scan

        def poisoned(*scan_args, **kwargs):
            base, gap, diagnostics = original(*scan_args, **kwargs)
            base, gap = base.clone(), gap.clone()
            base[~args[3].T.bool()] = float('nan')
            gap[args[1].bool() | ~args[3].T.bool()[..., None]] = float('nan')
            return base, gap, diagnostics

        with patch.object(net.osram, '_scan', side_effect=poisoned):
            self.assertTrue(torch.equal(expected, net(*args)[0]))
        captured = []
        handle = net.osram.readout_candidate.register_forward_pre_hook(lambda m, inputs: captured.append(inputs))
        net.osram.osram_emotion_ablation = 'local-only'
        net(*args)
        handle.remove()
        self.assertEqual(captured[0][1].count_nonzero(), 0)
        self.assertEqual(captured[0][2].count_nonzero(), 0)


if __name__ == '__main__':
    unittest.main()
