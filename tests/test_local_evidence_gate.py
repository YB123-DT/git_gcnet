import unittest
import torch
from gcnet_missing_m3 import osram
from tests.test_osram_post_grn import PostGRNTests


class LocalEvidenceGateTests(unittest.TestCase):
    def test_identity_and_masked_regularizer(self):
        self.assertTrue(hasattr(osram, 'LocalConditionedEvidenceGate'))
        m = osram.LocalConditionedEvidenceGate(4, 3)
        _, local, base, gap, availability, umask = PostGRNTests().inputs()
        args = (local, base, gap, availability, umask)
        active = torch.cat((umask.T.bool()[..., None], umask.T.bool()[..., None] & ~availability.bool()), -1)
        gates = m(*args)
        torch.testing.assert_close(gates, active.to(gates.dtype))
        self.assertEqual(m.regularization.item(), 0.)
        with torch.no_grad(): m.output.bias.fill_(.5)
        expected = m(*args).detach().clone()
        gap[~active[..., 1:]] = float('nan')
        for value in (local, base, availability): value[~umask.T.bool()] = float('nan')
        torch.testing.assert_close(m(*args), expected)
        torch.testing.assert_close(m.regularization, ((expected[active]-1)**2).mean())
        opt = torch.optim.Adam(m.parameters(), lr=.01)
        for _ in range(3):
            opt.zero_grad(); gates = m(*args)
            (gates.square().mean()+.001*m.regularization).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            opt.step()

    def test_backbone_initial_equivalence(self):
        self.assertTrue(hasattr(osram, 'LocalConditionedEvidenceGate'))
        helper = PostGRNTests()
        torch.manual_seed(42); old = helper.backbone(); rng = torch.get_rng_state()
        torch.manual_seed(42); off = helper.backbone(osram_local_evidence_gate=False)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(old.state_dict()),set(off.state_dict()))
        for k,v in old.state_dict().items(): self.assertTrue(torch.equal(v,off.state_dict()[k]))
        torch.manual_seed(42); new = helper.backbone(osram_local_evidence_gate=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k,v in old.state_dict().items(): self.assertTrue(torch.equal(v,new.state_dict()[k]))
        for train in (False, True):
            old.train(train); new.train(train)
            torch.manual_seed(5); a, ca = old(*helper.backbone_inputs())
            torch.manual_seed(5); b, cb = new(*helper.backbone_inputs())
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            for k in ca: torch.testing.assert_close(ca[k],cb[k],rtol=0,atol=0)

    def test_config(self):
        from dataclasses import replace
        from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
        c = TrainConfig(backbone_type='osram',training_objective='emotion-only',osram_bidirectional=False)
        self.assertTrue(hasattr(c,'osram_local_evidence_gate'))
        self.assertFalse(c.osram_local_evidence_gate)
        self.assertEqual(c.osram_local_evidence_gate_reg_weight,.001)
        gated = replace(c,osram_local_evidence_gate=True)
        for k,v in [('osram_history_input_gate',True),('osram_post_grn',True),('train_rate_mode','conversation-mixed'),('training_objective','joint'),('text_core',True),('osram_bidirectional',True)]:
            with self.assertRaises(ValueError): replace(gated,**{k:v})
        args = build_parser().parse_args(['--audio-feature','a','--text-feature','t','--video-feature','v','--output-dir','x','--osram-local-evidence-gate'])
        self.assertTrue(args.osram_local_evidence_gate)

    def test_train_epoch_adds_regularization_once(self):
        from gcnet_missing_m3 import train_gcnet as tr
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        c = tr.TrainConfig(dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',
            latent_dim=8,osram_output_dim=19,osram_num_heads=2,osram_key_dim=3,osram_value_dim=4,
            osram_bidirectional=False,osram_local_evidence_gate=True)
        model = _build_model(c,(1,1,1))
        with torch.no_grad(): model.osram.local_evidence_gate.output.bias.fill_(.5)
        regs = []
        hook = model.osram.local_evidence_gate.register_forward_hook(lambda m,a,o: regs.append(float(m.regularization.detach())))
        batch = [torch.randn(3,2,1) for _ in range(6)]
        batch += [torch.zeros(2,3),torch.ones(2,3),torch.tensor([[1.,-1.,1.],[-1.,1.,-1.]]),['a','b']]
        opt = torch.optim.Adam(model.parameters(),lr=.001)
        result = tr.train_epoch(model,[batch,batch],opt,c,tr._schedules(c,'train'),0,(1,1,1),torch.device('cpu'))
        hook.remove()
        self.assertEqual(len(regs),2)
        self.assertAlmostEqual(result['loss']-result['classification_loss'],.001*sum(regs)/2,places=6)
        self.assertAlmostEqual(result['local_evidence_gate_regularization'],sum(regs)/2)
        self.assertEqual(result['optimizer_steps'],2)

    def test_ablation_and_inactive_gradients(self):
        helper = PostGRNTests()
        for mode in ('local-only','local-base','local-gap'):
            model = helper.backbone(osram_local_evidence_gate=True,osram_emotion_ablation=mode)
            captured = []
            hook = model.local_evidence_gate.register_forward_pre_hook(lambda m,a: captured.append(a))
            model(*helper.backbone_inputs()); hook.remove()
            if mode in ('local-only','local-gap'): self.assertEqual(captured[0][1].count_nonzero(),0)
            if mode in ('local-only','local-base'): self.assertEqual(captured[0][2].count_nonzero(),0)
        m = osram.LocalConditionedEvidenceGate(4,3)
        _,local,base,gap,availability,umask = helper.inputs()
        gap.requires_grad_()
        with torch.no_grad(): m.output.weight.normal_()
        gates = m(local,base,gap,availability,umask)
        gates.sum().backward()
        inactive = availability.bool() | ~umask.T.bool()[...,None]
        self.assertEqual(gap.grad[inactive].count_nonzero(),0)
        previous = m.regularization
        m(local,base,gap,availability,torch.zeros_like(umask))
        self.assertIsNot(previous,m.regularization)
        self.assertEqual(m.regularization.item(),0.)

    def test_type_specific_gates_and_diagnostic_weighting(self):
        from types import SimpleNamespace
        from gcnet_missing_m3 import train_gcnet as tr
        m = osram.LocalConditionedEvidenceGate(4,3)
        # Route only the first evidence-type coordinate through one hidden unit.
        with torch.no_grad():
            m.input.weight.zero_(); m.input.bias.zero_(); m.output.weight.zero_()
            m.type_embedding.weight.zero_()
            m.type_embedding.weight[:,0] = torch.arange(4.)
            m.input.weight[0,514] = 1.; m.output.weight[0,0] = 1.
        gates = m(torch.zeros(1,1,4),torch.zeros(1,1,3),torch.zeros(1,1,3,3),torch.zeros(1,1,3),torch.ones(1,1))
        self.assertTrue((gates[...,1:] > gates[...,:-1]).all())
        totals = {}
        for count, mean in ((2,.9),(6,1.1)):
            diagnostic = {name:dict(active_count=count,gate_mean=mean,gate_saturation_fraction=0.)
                          for name in ('base','gap_a','gap_t','gap_v')}
            tr._accumulate_local_evidence_gate(SimpleNamespace(osram=SimpleNamespace(last_diagnostics={'local_evidence_gate':diagnostic})),totals)
        for row in tr._local_evidence_gate_metrics(totals)['local_evidence_gate'].values():
            self.assertEqual(row['active_count'],8)
            self.assertAlmostEqual(row['gate_mean'],1.05)

if __name__ == '__main__': unittest.main()
