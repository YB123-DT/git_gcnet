import unittest
import torch
from gcnet_missing_m3 import osram
from tests import test_osram_post_grn as helpers


class HistoryInputGateTests(unittest.TestCase):
    def test_feature_and_initial_identity(self):
        self.assertTrue(hasattr(osram, 'HistoryInputGate'))
        helper = helpers.PostGRNTests()
        torch.manual_seed(42); old = helper.backbone(); rng = torch.get_rng_state()
        torch.manual_seed(42); off = helper.backbone(osram_history_input_gate=False)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(old.state_dict()), set(off.state_dict()))
        torch.manual_seed(42); new = helper.backbone(osram_history_input_gate=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in old.state_dict().items(): self.assertTrue(torch.equal(v, new.state_dict()[k]))
        args = helper.backbone_inputs()
        for train in (False, True):
            old.train(train); new.train(train)
            torch.manual_seed(5); a, ca = old(*args)
            torch.manual_seed(5); b, cb = new(*args)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for k in ca: torch.testing.assert_close(ca[k], cb[k], rtol=0, atol=0)
            a.square().sum().backward(); b.square().sum().backward()
            for k, p in old.named_parameters():
                if p.grad is not None: torch.testing.assert_close(p.grad, dict(new.named_parameters())[k].grad, rtol=1e-5, atol=1e-6)
            old.zero_grad(); new.zero_grad()

    def test_ablation_condition_and_weighted_statistics(self):
        from types import SimpleNamespace
        from gcnet_missing_m3 import train_gcnet as tr
        helper = helpers.PostGRNTests()
        for ablation in ('local-only','local-base','local-gap'):
            m = helper.backbone(osram_history_input_gate=True,osram_emotion_ablation=ablation)
            captured = []
            handle = m.history_input_gate.register_forward_pre_hook(lambda module,args: captured.append(args))
            m(*helper.backbone_inputs()); handle.remove()
            if ablation in ('local-only','local-gap'): self.assertEqual(captured[0][1].count_nonzero().item(),0)
            if ablation in ('local-only','local-base'): self.assertEqual(captured[0][2].count_nonzero().item(),0)
        totals = {}
        for n, value in ((2,.9),(6,1.1)):
            d = dict(valid_count=n,alpha_mean=value,alpha_second_moment=value**2,
                     alpha_below_one_fraction=float(value<1),alpha_above_one_fraction=float(value>1),
                     alpha_saturation_fraction=0.)
            tr._accumulate_history_input_gate(SimpleNamespace(osram=SimpleNamespace(last_diagnostics={'history_input_gate':d})),totals)
        metrics = tr._history_input_gate_metrics(totals)['history_input_gate']
        self.assertAlmostEqual(metrics['alpha_mean'],1.05)
        self.assertAlmostEqual(metrics['alpha_std'],(.0075)**.5)
        self.assertAlmostEqual(metrics['alpha_below_one_fraction'],.25)

    def test_mask_bounds_and_updates(self):
        self.assertTrue(hasattr(osram, 'HistoryInputGate'))
        m = osram.HistoryInputGate(4, 3)
        _, local, base, gap, availability, umask = helpers.PostGRNTests().inputs()
        valid = umask.T.bool(); inactive = availability.bool() | ~valid[..., None]
        with torch.no_grad(): m.output.weight.fill_(.1)
        args = (local, base, gap, availability, umask)
        expected = m(*args)
        gap[inactive] = float('nan')
        for x in (local, base, availability): x[~valid] = float('nan')
        actual = m(*args)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (3, 2, 1))
        self.assertTrue(((actual[valid] >= .8) & (actual[valid] <= 1.2)).all())
        self.assertEqual(actual[~valid].count_nonzero().item(), 0)
        opt = torch.optim.Adam(m.parameters(), lr=.01)
        for _ in range(4):
            opt.zero_grad(); m(*args).square().sum().backward()
            for p in m.parameters(): self.assertTrue(torch.isfinite(p.grad).all())
            opt.step()

    def test_config_builder_and_full_model(self):
        from dataclasses import replace
        from gcnet_missing_m3 import train_gcnet as tr
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        config = tr.TrainConfig(backbone_type='osram', training_objective='emotion-only',
            osram_bidirectional=False, osram_output_dim=37, latent_dim=8,
            osram_num_heads=2, osram_key_dim=3, osram_value_dim=4)
        args = tr.build_parser().parse_args(['--audio-feature','a','--text-feature','t',
            '--video-feature','v','--output-dir','unused','--osram-history-input-gate'])
        self.assertTrue(args.osram_history_input_gate)
        self.assertFalse(config.osram_history_input_gate)
        gated = replace(config, osram_history_input_gate=True)
        for key, value in [('osram_post_grn',True),('osram_history_query_adapter',True),
                           ('train_rate_mode','conversation-mixed'),('training_objective','joint'),
                           ('osram_readout_fusion','memory-shift-residual')]:
            with self.subTest(key=key), self.assertRaises(ValueError): replace(gated, **{key:value})
        old = _build_model(config,(3,4,5)); new = _build_model(gated,(3,4,5))
        x = torch.randn(3,2,12); a = torch.tensor([[[1.,0,1],[0,1,0]]]*3)
        u = torch.tensor([[1.,1,1],[1,1,0]]); a[~u.T.bool()] = 0
        q = torch.zeros(2,3,dtype=torch.long)
        for train in (False,True):
            old.train(train); new.train(train)
            torch.manual_seed(6); expected = old([x],a,q,u,[3,2],predict_missing=False)[0]
            torch.manual_seed(6); actual = new([x],a,q,u,[3,2],predict_missing=False)[0]
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        self.assertTrue(all(p.requires_grad for p in new.osram.parameters()))
        before = new.osram.history_input_gate.output.weight.detach().clone()
        opt = torch.optim.Adam(new.parameters(),lr=.01)
        for _ in range(3):
            opt.zero_grad(); new([x],a,q,u,[3,2],predict_missing=False)[0].square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None))
            opt.step()
        self.assertFalse(torch.equal(before,new.osram.history_input_gate.output.weight))
        totals = {}; tr._accumulate_history_input_gate(new,totals)
        self.assertEqual(tr._history_input_gate_metrics(totals)['history_input_gate']['valid_count'],5)


if __name__ == '__main__': unittest.main()
