"""Identity, stochastic replay, and learning contracts for Gap modulation."""
import subprocess
import types
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from gcnet_missing_m3.osram import OSRAMBackbone, MODALITIES


class GapIncrementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.kw = dict(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
                       value_dim=4, dropout=.4, bidirectional=False)

    def inputs(self):
        x = torch.randn(4, 2, 8)
        a = torch.tensor([[[1., 0, 1], [0, 1, 0]]] * 4)
        u = torch.tensor([[1., 1, 1, 1], [1, 1, 0, 0]])
        a[~u.T.bool()] = 0
        return x, {m: torch.randn_like(x) for m in MODALITIES}, a, torch.zeros(2, 4, dtype=torch.long), u

    def test_switch_exists(self):
        import inspect
        self.assertIn('osram_gap_increment_filter', inspect.signature(OSRAMBackbone).parameters)

    def test_legacy_off_identity_rng_and_gradients(self):
        source = subprocess.check_output(['git', 'show', '6901379:gcnet_missing_m3/osram.py'], text=True)
        legacy = types.ModuleType('legacy_gap')
        exec(compile(source, 'legacy_gap.py', 'exec'), legacy.__dict__)
        torch.manual_seed(19)
        old = legacy.OSRAMBackbone(**self.kw)
        state = torch.get_rng_state()
        torch.manual_seed(19)
        off = OSRAMBackbone(**self.kw)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        off.load_state_dict(old.state_dict(), strict=True)
        torch.manual_seed(19)
        on = OSRAMBackbone(**self.kw, osram_gap_increment_filter=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        # Exercise nonzero adapter as well as its default zero initialization.
        with torch.no_grad():
            old.emotion_adapter[-1].weight.normal_(std=.05)
        off.load_state_dict(old.state_dict())
        on.load_state_dict(old.state_dict(), strict=False)
        args = self.inputs()
        for training in (False, True):
            results = []
            for net in (old, off, on):
                net.train(training)
                net.zero_grad()
                torch.manual_seed(33)
                h = net(*args)[0]
                h.square().sum().backward()
                results.append((h, torch.get_rng_state()))
            for h, rng in results[1:]:
                self.assertTrue(torch.equal(h, results[0][0]))
                self.assertTrue(torch.equal(rng, results[0][1]))
            for name, p in old.named_parameters():
                other = dict(on.named_parameters())[name]
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, other.grad, rtol=0, atol=0)

    def test_single_scan_shared_dropout_and_endpoints(self):
        net = OSRAMBackbone(**self.kw, osram_gap_increment_filter=True)
        with torch.no_grad(): net.emotion_adapter[-1].weight.normal_(std=.05)
        args = self.inputs()
        args[2].fill_(1)
        args[2][~args[-1].T.bool()] = 0
        captured = []
        handle = net.emotion_adapter.register_forward_hook(lambda m, i, o: captured.append(o))
        with patch.object(net, '_scan', wraps=net._scan) as scan, patch.object(net.local_path, 'forward', wraps=net.local_path.forward) as local, patch.object(net.local_skip, 'forward', wraps=net.local_skip.forward) as skip:
            net(*args)
            self.assertEqual((scan.call_count, local.call_count, skip.call_count), (1, 1, 1))
        handle.remove()
        self.assertEqual(len(captured), 2)
        self.assertTrue(torch.equal(*captured))
        self.assertEqual(net.last_diagnostics['gap_increment_filter']['delta_norm'], 0)
        net.eval()
        args = self.inputs()
        anchors, adapters = [], []
        hs = net.local_skip.register_forward_hook(lambda m, i, o: anchors.append(o))
        ha = net.emotion_adapter.register_forward_hook(lambda m, i, o: adapters.append(o))
        h = net(*args)[0]
        valid = args[-1].T.bool()[..., None]
        full = torch.where(valid, net.emotion_norm(anchors[0] + adapters[0]), 0.)
        self.assertTrue(torch.equal(h, full))
        with torch.no_grad(): net.gap_increment_filter.gate_mlp[-1].bias.fill_(-100)
        h = net(*args)[0]
        base = torch.where(valid, net.emotion_norm(anchors[-1] + adapters[-1]), 0.)
        torch.testing.assert_close(h, base, atol=2e-6, rtol=2e-6)
        hs.remove(); ha.remove()

    def test_inactive_gap_nan_and_padding(self):
        net = OSRAMBackbone(**self.kw, osram_gap_increment_filter=True).eval()
        with torch.no_grad():
            net.emotion_adapter[-1].weight.normal_(std=.05)
            net.gap_increment_filter.gate_mlp[-1].bias.fill_(.7)
        args = self.inputs()
        expected = net(*args)[0]
        original = net._scan
        def poisoned(*a, **kw):
            base, gap, diagnostics = original(*a, **kw)
            gap = gap.clone()
            gap[args[2].bool() | ~args[-1].T.bool()[..., None]] = float('nan')
            base = base.clone()
            base[~args[-1].T.bool()] = float('nan')
            return base, gap, diagnostics
        with patch.object(net, '_scan', side_effect=poisoned):
            actual = net(*args)[0]
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(actual[~args[-1].T.bool()].count_nonzero(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA cfg84 equivalence check')
    def test_cuda_cfg84_identity_and_rng(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        cfg = TrainConfig(dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
                          disable_unused_aux_modules=True, osram_bidirectional=False, osram_output_dim=1600,
                          osram_num_heads=8, osram_key_dim=64, osram_value_dim=64, osram_write_step=.6)
        torch.manual_seed(91); old = _build_model(cfg, (512, 1024, 1024)).cuda()
        torch.manual_seed(91); new = _build_model(replace(cfg, osram_gap_increment_filter=True), (512, 1024, 1024)).cuda()
        with torch.no_grad(): old.osram.emotion_adapter[-1].weight.normal_(std=.005)
        new.load_state_dict(old.state_dict(), strict=False)
        x = torch.randn(4, 2, 2560, device='cuda')
        _, _, a, q, u = self.inputs()
        a, q, u = a.cuda(), q.cuda(), u.cuda()
        for training in (True, False):
            old.train(training); new.train(training)
            cpu_before, before = torch.get_rng_state(), torch.cuda.get_rng_state()
            expected = old([x], a, q, u, [4, 2], predict_missing=False)[0]
            cpu_after, after = torch.get_rng_state(), torch.cuda.get_rng_state()
            torch.set_rng_state(cpu_before); torch.cuda.set_rng_state(before)
            actual = new([x], a, q, u, [4, 2], predict_missing=False)[0]
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(after, torch.cuda.get_rng_state()))
            self.assertTrue(torch.equal(cpu_after, torch.get_rng_state()))

    def test_config_cli_and_full_model_learning(self):
        from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        cfg = TrainConfig(backbone_type='osram', training_objective='emotion-only', osram_bidirectional=False,
                          latent_dim=8, osram_output_dim=10, osram_num_heads=2, osram_key_dim=3,
                          osram_value_dim=4, osram_gap_increment_filter=True, disable_unused_aux_modules=True)
        for kw in (dict(osram_relation_block=True), dict(osram_post_grn=True), dict(paired_history_views=True),
                   dict(osram_readout_fusion='local-gated'), dict(readout_type='target-specific'),
                   dict(train_rate_mode='conversation-mixed'), dict(training_objective='joint')):
            with self.assertRaises(ValueError): replace(cfg, **kw)
        parsed = build_parser().parse_args(['--audio-feature','a','--text-feature','t','--video-feature','v','--output-dir','unused','--osram-gap-increment-filter'])
        self.assertTrue(parsed.osram_gap_increment_filter)
        net = _build_model(cfg, (1, 1, 1))
        optimizer = torch.optim.Adam(net.parameters(), lr=.01)
        x = torch.randn(4, 2, 3)
        _, _, a, q, u = self.inputs()
        initial = net.osram.gap_increment_filter.gate_mlp[-1].weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            with patch.object(net.smax_fc, 'forward', wraps=net.smax_fc.forward) as task:
                out = net([x], a, q, u, [4, 2], predict_missing=False)[0]
                self.assertEqual(task.call_count, 1)
            out.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
            optimizer.step()
        self.assertFalse(torch.equal(initial, net.osram.gap_increment_filter.gate_mlp[-1].weight))
        restored = _build_model(cfg, (1, 1, 1))
        restored.load_state_dict(net.state_dict(), strict=True)
        net.eval(); restored.eval()
        self.assertTrue(torch.equal(net([x], a, q, u, [4, 2], predict_missing=False)[0], restored([x], a, q, u, [4, 2], predict_missing=False)[0]))

    def test_epoch_diagnostics_and_single_task_loss(self):
        from gcnet_missing_m3 import train_gcnet as tr
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        cfg = tr.TrainConfig(dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
                             disable_unused_aux_modules=True, latent_dim=8, osram_output_dim=10,
                             osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
                             osram_bidirectional=False, osram_gap_increment_filter=True)
        net = _build_model(cfg, (1, 1, 1))
        batch = [torch.randn(6, 2, 1) for _ in range(6)]
        batch += [torch.zeros(2, 6), torch.ones(2, 6), torch.tensor([[1., -1.] * 3, [-1., 1.] * 3]), ['a', 'b']]
        with patch.object(net.osram, '_scan', wraps=net.osram._scan) as scan:
            result = tr.train_epoch(net, [batch, batch], torch.optim.Adam(net.parameters(), lr=.001),
                                    cfg, tr._schedules(cfg, 'train'), 0, (1, 1, 1), torch.device('cpu'))
        self.assertEqual(scan.call_count, 2)
        self.assertEqual(result['model_forward_count'], 2)
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertEqual(result['jepa_loss'], 0)
        self.assertNotIn('relation_dual_readout', result)
        self.assertNotIn('paired_history', result)
        self.assertEqual(result['gap_increment_filter']['valid_tokens'], 24)
        d = result['gap_increment_filter']
        self.assertAlmostEqual(d['gate_saturation_fraction'], d['gate_low_fraction'] + d['gate_high_fraction'])
        self.assertTrue(all(p.requires_grad for p in net.osram.parameters()))
        metrics, _ = tr.evaluate_rate(net, [batch, batch], tr._schedules(cfg, 'test')[.5],
                                      cfg.dataset, (1, 1, 1), torch.device('cpu'), collect=False)
        self.assertEqual(metrics['gap_increment_filter']['valid_tokens'], 24)


if __name__ == '__main__':
    unittest.main()
