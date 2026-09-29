from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel
from tests.test_completion_memory_write import model_kwargs, inputs


class PostGRNIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA cfg84 verification')
    def test_cuda_cfg84_identity_and_finite_joint_updates(self):
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        config=tr.TrainConfig(dataset='CMUMOSI',backbone_type='osram',
            training_objective='emotion-only',disable_unused_aux_modules=True,
            osram_bidirectional=False,osram_output_dim=1600,osram_num_heads=8,
            osram_key_dim=64,osram_value_dim=64,osram_write_step=.6)
        flat=_build_model(config,(512,1024,1024)).cuda()
        new=_build_model(replace(config,osram_post_grn=True),(512,1024,1024)).cuda()
        x=torch.randn(4,2,2560,device='cuda')
        a=torch.tensor([[[1.,0,1],[0,1,0]]]*4,device='cuda')
        u=torch.tensor([[1.,1,1,1],[1,1,0,0]],device='cuda');a[~u.T.bool()]=0
        q=torch.zeros(2,4,dtype=torch.long,device='cuda')
        for train in (False,True):
            flat.train(train);new.train(train);before=torch.cuda.get_rng_state()
            old=flat([x],a,q,u,[4,2],predict_missing=False)[0]
            torch.cuda.set_rng_state(before)
            actual=new([x],a,q,u,[4,2],predict_missing=False)[0]
            self.assertTrue(torch.equal(old,actual))
        optimizer=torch.optim.Adam(new.parameters(),lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            out=new([x],a,q,u,[4,2],predict_missing=False)[0]
            out[u.T.bool()].square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None))
            torch.nn.utils.clip_grad_norm_(new.parameters(),1.)
            optimizer.step()

    def config(self):
        return tr.TrainConfig(backbone_type='osram', osram_bidirectional=False,
                              training_objective='emotion-only', osram_post_grn=True)

    def test_cli_default_and_config_guards(self):
        self.assertTrue(hasattr(tr.TrainConfig(), 'osram_post_grn'))
        self.assertFalse(tr.TrainConfig().osram_post_grn)
        args=tr.build_parser().parse_args(['--audio-feature','a','--text-feature','t',
              '--video-feature','v','--output-dir','unused','--osram-post-grn'])
        self.assertTrue(args.osram_post_grn)
        config=self.config()
        self.assertEqual(config.osram_readout_fusion,'flat')
        for key,value in [('osram_readout_fusion','memory-shift-residual'),
                          ('training_objective','joint'),('train_rate_mode','conversation-mixed'),
                          ('osram_history_query_adapter',True),
                          ('completion_path','pre_osram_joint_dual_projector')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(config,**{key:value})

    def test_model_init_shared_state_and_no_freezes(self):
        kwargs=model_kwargs();kwargs['completion_path']='none'
        torch.manual_seed(71);flat=MissingM3GraphModel(**kwargs)
        rng=torch.get_rng_state()
        torch.manual_seed(71);new=MissingM3GraphModel(**kwargs,osram_post_grn=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in flat.state_dict().items():self.assertTrue(torch.equal(v,new.state_dict()[k]),k)
        self.assertTrue(all(p.requires_grad for p in new.osram.parameters()))
        x,a,q,u,lengths=inputs()
        flat.eval();new.eval()
        self.assertTrue(torch.equal(flat([x],a,q,u,lengths,predict_missing=False)[0],
                                    new([x],a,q,u,lengths,predict_missing=False)[0]))

    def test_model_builder_propagates_independent_flag(self):
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        model=_build_model(replace(self.config(), osram_output_dim=37,latent_dim=8,
                                   osram_num_heads=2,osram_key_dim=3,osram_value_dim=4), (3,4,5))
        self.assertTrue(model.osram.osram_post_grn)
        self.assertEqual(model.osram.output_dim,37)

    def test_diagnostics_weighted_over_valid_tokens(self):
        self.assertTrue(hasattr(tr,'_accumulate_post_grn'))
        totals={};model=SimpleNamespace(osram=SimpleNamespace(last_diagnostics={}))
        for n,g,s,r in [(2,.2,.5,2.),(6,.8,0.,4.)]:
            model.osram.last_diagnostics={'post_grn':dict(valid_count=n,gate_mean=g,
                gate_saturation_fraction=s,gated_residual_flat_norm_ratio=r)}
            tr._accumulate_post_grn(model,totals)
        actual=tr._post_grn_metrics(totals)['post_grn']
        self.assertEqual(actual['valid_count'],8)
        self.assertAlmostEqual(actual['gate_mean'],.65)
        self.assertAlmostEqual(actual['gate_saturation_fraction'],.125)
        self.assertAlmostEqual(actual['gated_residual_flat_norm_ratio'],3.5)
        totals={};tr._accumulate_post_grn(SimpleNamespace(),totals)
        self.assertEqual(tr._post_grn_metrics(totals),{})


if __name__=='__main__': unittest.main()
