import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser, _relation_dual_readout_loss
from tests.test_completion_memory_write import model_kwargs, inputs


class DualReadoutTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA cfg84 dual-readout check')
    def test_cuda_cfg84_same_initial_prediction_and_rng(self):
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        cfg = TrainConfig(dataset='CMUMOSI', backbone_type='osram', training_objective='emotion-only',
            disable_unused_aux_modules=True, osram_bidirectional=False, osram_relation_block=True,
            osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
            osram_write_step=.6)
        torch.manual_seed(91); single=_build_model(cfg,(512,1024,1024)).cuda()
        torch.manual_seed(91); dual=_build_model(replace(cfg,osram_relation_dual_readout=True),(512,1024,1024)).cuda()
        x=torch.randn(4,2,2560,device='cuda');a=torch.tensor([[[1.,0,1],[0,1,0]]]*4,device='cuda')
        u=torch.tensor([[1.,1,1,1],[1,1,0,0]],device='cuda');a[~u.T.bool()]=0
        q=torch.zeros(2,4,dtype=torch.long,device='cuda')
        for training in (True,False):
            single.train(training);dual.train(training)
            before=torch.cuda.get_rng_state();expected=single([x],a,q,u,[4,2],predict_missing=False)[0]
            after=torch.cuda.get_rng_state();torch.cuda.set_rng_state(before)
            actual=dual([x],a,q,u,[4,2],predict_missing=False)[0]
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(torch.equal(after,torch.cuda.get_rng_state()))
            if training:
                self.assertTrue(torch.equal(actual[u.T.bool()],dual.last_relation_base_logits[u.T.bool()]))

    def setUp(self):
        torch.set_num_threads(1)

    def model(self, dual=True):
        kw = model_kwargs()
        kw.update(completion_path='none', osram_relation_block=True,
                  osram_relation_dual_readout=dual)
        return MissingM3GraphModel(**kw)

    def test_config(self):
        cfg = TrainConfig(backbone_type='osram', training_objective='emotion-only',
                          osram_bidirectional=False, osram_relation_block=True,
                          osram_relation_dual_readout=True)
        for kw in (dict(osram_relation_block=False), dict(emotion_loss_mode='pattern-groupdro'),
                   dict(readout_type='availability-affine'), dict(paired_history_views=True),
                   dict(osram_ablation='local-only')):
            with self.assertRaises(ValueError): replace(cfg, **kw)
        kw=model_kwargs();kw.update(completion_path='none',osram_relation_block=True,
            osram_relation_dual_readout=True,osram_ablation='local-only')
        with self.assertRaises(ValueError): MissingM3GraphModel(**kw)
        args = build_parser().parse_args(['--audio-feature','a','--text-feature','t',
            '--video-feature','v','--output-dir','unused','--osram-relation-dual-readout'])
        self.assertTrue(args.osram_relation_dual_readout)

    def test_same_trajectory_and_shared_layers(self):
        m = self.model(); x,a,q,u,lengths = inputs()
        counts = {}
        handles = []
        for name, layer in [('encoder',m.observed_set),('skip',m.osram.local_skip),
                            ('adapter',m.osram.emotion_adapter),('norm',m.osram.emotion_norm),
                            ('head',m.smax_fc)]:
            counts[name] = 0
            def hook(_m,_i,_o,name=name): counts[name] += 1
            handles.append(layer.register_forward_hook(hook))
        with patch.object(m.osram, '_scan', wraps=m.osram._scan) as scan:
            full = m([x],a,q,u,lengths,predict_missing=False)[0]
            self.assertEqual(scan.call_count,1)
        for h in handles: h.remove()
        self.assertEqual(counts,dict(encoder=1,skip=1,adapter=1,norm=2,head=2))
        valid = u.T.bool()
        self.assertTrue(torch.equal(full[valid],m.last_relation_base_logits[valid]))
        self.assertEqual(m.last_relation_base_logits[~valid].count_nonzero(),0)
        self.assertEqual(m.osram.last_relation_base_hidden[~valid].count_nonzero(),0)
        m.eval(); m([x],a,q,u,lengths,predict_missing=False)
        self.assertIsNone(m.last_relation_base_logits)
        self.assertIsNone(m.osram.last_relation_base_hidden)

    def test_rng_state_parameters_and_zero_init_gradients(self):
        torch.manual_seed(81); single = self.model(False); rng = torch.get_rng_state()
        torch.manual_seed(81); dual = self.model(True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(single.state_dict().keys(),dual.state_dict().keys())
        for key,v in single.state_dict().items(): self.assertTrue(torch.equal(v,dual.state_dict()[key]))
        x,a,q,u,lengths = inputs(); valid=u.T.bool()
        p=single([x],a,q,u,lengths,predict_missing=False)[0]
        f=dual([x],a,q,u,lengths,predict_missing=False)[0]
        self.assertTrue(torch.equal(p,f))
        p[valid].square().mean().backward()
        (.5*f[valid].square().mean()+.5*dual.last_relation_base_logits[valid].square().mean()).backward()
        # Shared path gradient is identical at zero-init; relation gradient is
        # intentionally halved because only the full loss trains that branch.
        for (name,s),(_,d) in zip(single.named_parameters(),dual.named_parameters()):
            if s.grad is not None:
                factor=.5 if 'relation_block.' in name else 1.
                torch.testing.assert_close(d.grad,s.grad*factor)

    def test_base_gradients_include_memory_and_not_relation(self):
        m=self.model(); x,a,q,u,lengths=inputs()
        with torch.no_grad():
            m.osram.relation_block.relation_out.weight.normal_(std=.01)
            # Original adapter is also zero-init; emulate a trained checkpoint
            # so a base loss can already reach upstream memory projections.
            m.osram.emotion_adapter[-1].weight.normal_(std=.1)
        optimizer=torch.optim.Adam(m.parameters(),lr=.001)
        m([x],a,q,u,lengths,predict_missing=False)
        m.last_relation_base_logits[u.T.bool()].square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.osram.query_projector.parameters()))
        self.assertTrue(all(p.grad is None for p in m.osram.relation_block.parameters()))
        # Inference-time removal of just Memory from the adapter changes BASE,
        # proving base is the full Flat path, not a local-only supervision head.
        reference=m.last_relation_base_logits.detach().clone()
        def zero_memory(_module,args):
            changed=args[0].clone(); changed[...,m.osram.latent_dim:]=0
            return (changed,)
        handle=m.osram.emotion_adapter.register_forward_pre_hook(zero_memory)
        m([x],a,q,u,lengths,predict_missing=False);handle.remove()
        self.assertFalse(torch.equal(reference[u.T.bool()],m.last_relation_base_logits[u.T.bool()]))
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            full=m([x],a,q,u,lengths,predict_missing=False)[0]
            loss=.5*full[u.T.bool()].square().mean()+.5*m.last_relation_base_logits[u.T.bool()].square().mean()
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.osram.relation_block.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            optimizer.step()

    def test_task_helper_preserves_mask_and_original_loss(self):
        from gcnet_missing_m3.train_gcnet import _emotion_loss
        m=self.model(); x,a,q,u,lengths=inputs()
        full=m([x],a,q,u,lengths,predict_missing=False)[0]
        labels=torch.randn_like(u)
        cfg=TrainConfig(dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',
            osram_bidirectional=False,osram_relation_block=True,osram_relation_dual_readout=True)
        view=dict(labels=labels,umask=u,availability=a)
        loss,_=_emotion_loss(cfg.dataset,full,labels,u,a,cfg.emotion_loss_mode,cfg.mosi_task_mode,
                            cfg.task_regression_loss,cfg.task_smooth_l1_beta)
        combined,base=_relation_dual_readout_loss(m,cfg,view,loss)
        torch.testing.assert_close(combined,loss)
        torch.testing.assert_close(base,loss)


if __name__ == '__main__': unittest.main()
