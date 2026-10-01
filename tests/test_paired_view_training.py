import unittest
import copy
from dataclasses import replace
import torch
from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model


class PairedTrainingTests(unittest.TestCase):
    def config(self, **kwargs):
        return tr.TrainConfig(dataset='CMUMOSI', backbone_type='osram',
            training_objective='emotion-only', latent_dim=8, osram_output_dim=19,
            osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
            osram_bidirectional=False, **kwargs)

    def test_default_off_and_guards(self):
        self.assertTrue(hasattr(self.config(), 'paired_history_views'))
        self.assertFalse(self.config().paired_history_views)
        cfg = self.config(paired_history_views=True)
        for changes in ({'osram_local_skip_gate': True}, {'train_rate_mode':'conversation-mixed'},
                        {'osram_bidirectional':True}, {'history_drop_prob':1.1},
                        {'history_contrast_temperature':0.}, {'history_contrast_weight':-1.}):
            with self.assertRaises(ValueError): replace(cfg, **changes)

    def test_projector_registration_preserves_rng(self):
        self.assertTrue(hasattr(tr, '_attach_history_projector'))
        model = _build_model(self.config(), (1,1,1))
        before = torch.get_rng_state().clone()
        tr._attach_history_projector(model, self.config(paired_history_views=True))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(model.history_contrast_projector(torch.randn(2,19)).shape,(2,128))
        optimizer = torch.optim.Adam(model.parameters())
        ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in ids for p in model.history_contrast_projector.parameters()))

    def test_two_forwards_training_and_default_off(self):
        self.assertTrue(hasattr(tr, '_attach_history_projector'))
        torch.manual_seed(3)
        batch = [torch.randn(20,2,1) for _ in range(6)]
        batch += [torch.zeros(2,20),torch.ones(2,20),torch.tensor([[1.,-1.]*10,[-1.,1.]*10]),['a','b']]
        cfg=self.config(paired_history_views=True,history_drop_prob=.2)
        model=_build_model(cfg,(1,1,1)); tr._attach_history_projector(model,cfg)
        initial_projector = model.history_contrast_projector[0].weight.detach().clone()
        calls=[]
        hook=model.register_forward_pre_hook(lambda m,args: calls.append(args[1].detach().clone()))
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        result=tr.train_epoch(model,[batch,batch],opt,cfg,tr._schedules(cfg,'train'),0,(1,1,1),torch.device('cpu'))
        hook.remove()
        self.assertEqual(len(calls),4)
        self.assertTrue((calls[1]<=calls[0]).all())
        self.assertTrue((calls[3]<=calls[2]).all())
        self.assertEqual(result['model_forward_count'],4)
        self.assertGreater(result['paired_history']['observed_dropped'],0)
        self.assertAlmostEqual(result['loss'],result['classification_loss']+.1*result['paired_history']['infonce_loss'],places=5)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertGreater(result['paired_history']['eligible_anchor_count'],0)
        self.assertFalse(torch.equal(initial_projector,model.history_contrast_projector[0].weight))

    def test_view1_rng_and_masks_match_legacy_and_memory_resets(self):
        torch.manual_seed(37)
        cfg=self.config()
        base=_build_model(cfg,(1,1,1))
        paired=copy.deepcopy(base)
        cfg2=replace(cfg,paired_history_views=True,history_drop_prob=.5,history_contrast_weight=0.)
        tr._attach_history_projector(paired,cfg2)
        batch=[torch.randn(6,2,1) for _ in range(6)]
        batch += [torch.zeros(2,6),torch.ones(2,6),torch.tensor([[1.,-1.]*3,[-1.,1.]*3]),['a','b']]
        captures=[]
        for model,config in [(base,cfg),(paired,cfg2)]:
            current=[]
            hook=model.register_forward_hook(lambda m,args,out: current.append((args[1].clone(),out[0].detach().clone())))
            torch.manual_seed(19)
            result=tr.train_epoch(model,[batch,batch],torch.optim.Adam(model.parameters(),lr=0.),config,
                tr._schedules(config,'train'),0,(1,1,1),torch.device('cpu'))
            rng=torch.get_rng_state().clone()
            hook.remove(); captures.append((current,rng,result))
        self.assertTrue(torch.equal(captures[0][1],captures[1][1]))
        for old,new in zip(captures[0][0],captures[1][0][::2]):
            for a,b in zip(old,new): torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertNotIn('paired_history',captures[0][2])
        self.assertTrue(all(p.grad is None for p in paired.history_contrast_projector.parameters()))
        # Calling a different trajectory between identical eval calls must not carry state.
        paired.eval()
        first=tr._prepare_view(batch,tr._schedules(cfg,'train')[0.],0,(1,1,1))
        other=tr._prepare_view(batch,tr._schedules(cfg,'train')[.7],0,(1,1,1))
        def forward(v):
            return paired([v['incomplete']],v['availability'],v['qmask'],v['umask'],v['lengths'],predict_missing=False)[0]
        with torch.no_grad():
            a=forward(first); forward(other); b=forward(first)
        torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
