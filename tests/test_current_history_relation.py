import unittest
import torch
from gcnet_missing_m3.osram import CurrentHistoryRelationBlock, OSRAMBackbone, MODALITIES


class RelationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA cfg84 equivalence check')
    def test_cuda_cfg84_initial_logits(self):
        from dataclasses import replace
        from gcnet_missing_m3.train_gcnet import TrainConfig
        from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
        cfg = TrainConfig(dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',disable_unused_aux_modules=True,osram_bidirectional=False,osram_output_dim=1600,osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,osram_write_step=.6)
        torch.manual_seed(91); old=_build_model(cfg,(512,1024,1024)).cuda()
        torch.manual_seed(91); new=_build_model(replace(cfg,osram_relation_block=True),(512,1024,1024)).cuda()
        x=torch.randn(4,2,2560,device='cuda');a=torch.tensor([[[1.,0,1],[0,1,0]]]*4,device='cuda')
        u=torch.tensor([[1.,1,1,1],[1,1,0,0]],device='cuda');a[~u.T.bool()]=0
        q=torch.zeros(2,4,dtype=torch.long,device='cuda')
        for training in (True,False):
            old.train(training);new.train(training)
            before=torch.cuda.get_rng_state();expected=old([x],a,q,u,[4,2],predict_missing=False)[0]
            after=torch.cuda.get_rng_state();torch.cuda.set_rng_state(before)
            actual=new([x],a,q,u,[4,2],predict_missing=False)[0]
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(torch.equal(after,torch.cuda.get_rng_state()))

    def test_ablation_inputs_and_memory_unchanged_after_update(self):
        kw=dict(latent_dim=8,output_dim=10,num_heads=2,key_dim=3,value_dim=4,dropout=0,bidirectional=False)
        torch.manual_seed(42);old=OSRAMBackbone(**kw)
        torch.manual_seed(42);new=OSRAMBackbone(**kw,osram_relation_block=True)
        with torch.no_grad(): new.relation_block.relation_out.weight.normal_()
        node=torch.randn(4,2,8);a=torch.tensor([[[1.,0,1],[0,1,0]]]*4)
        args=(node,{k:torch.randn_like(node) for k in MODALITIES},a,torch.zeros(2,4,dtype=torch.long),torch.ones(2,4))
        writes, new_writes=[],[]
        h,c=old(*args,post_write_observer=lambda *x:writes.append(x))
        hn,cn=new(*args,post_write_observer=lambda *x:new_writes.append(x))
        self.assertFalse(torch.equal(h,hn))
        for k in c:self.assertTrue(torch.equal(c[k],cn[k]))
        for before,after in zip(writes,new_writes):
            for b,n in zip(before[1:],after[1:]):self.assertTrue(torch.equal(b,n))
        received=[]
        hook=new.relation_block.register_forward_pre_hook(lambda _,x:received.append(x))
        new.osram_emotion_ablation='local-only';new(*args);hook.remove()
        self.assertEqual(received[0][1].count_nonzero(),0)
        self.assertEqual(received[0][2].count_nonzero(),0)

    def test_config_and_cli(self):
        from dataclasses import replace
        from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser
        cfg = TrainConfig(backbone_type='osram', training_objective='emotion-only', osram_bidirectional=False, osram_relation_block=True)
        self.assertEqual(cfg.osram_relation_mode, 'pairwise')
        for kw in (dict(osram_bidirectional=True),dict(osram_post_grn=True),dict(paired_history_views=True),dict(train_rate_mode='conversation-mixed'),dict(osram_relation_dim=0)):
            with self.assertRaises(ValueError): replace(cfg,**kw)
        args = build_parser().parse_args(['--audio-feature','a','--text-feature','t','--video-feature','v','--output-dir','unused','--osram-relation-block','--osram-relation-mode','control'])
        self.assertTrue(args.osram_relation_block)
        self.assertEqual(args.osram_relation_mode, 'control')

    def test_full_model_logits_and_learning(self):
        from gcnet_missing_m3.model import MissingM3GraphModel
        from tests.test_completion_memory_write import model_kwargs, inputs
        kw = model_kwargs(); kw.update(completion_path='none',dropout=.2,projector_dropout=.1)
        torch.manual_seed(44); old = MissingM3GraphModel(**kw); rng = torch.get_rng_state()
        torch.manual_seed(44); new = MissingM3GraphModel(**kw,osram_relation_block=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in old.state_dict().items(): self.assertTrue(torch.equal(v,new.state_dict()[k]),k)
        x,a,q,u,lengths = inputs()
        for training in (False,True):
            old.train(training);new.train(training)
            before = torch.get_rng_state(); expected = old([x],a,q,u,lengths,predict_missing=False)[0]
            after = torch.get_rng_state();torch.set_rng_state(before)
            actual = new([x],a,q,u,lengths,predict_missing=False)[0]
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
            self.assertTrue(torch.equal(expected,actual))
        optimizer = torch.optim.Adam(new.parameters(),lr=.001)
        initial = {k:p.detach().clone() for k,p in new.osram.named_parameters()}
        for _ in range(3):
            optimizer.zero_grad()
            new([x],a,q,u,lengths,predict_missing=False)[0].square().sum().backward()
            for p in new.parameters():
                if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()
        current = dict(new.osram.named_parameters())
        for prefix in ('relation_block.','local_skip.','emotion_adapter.','query_projector.'):
            self.assertTrue(any(not torch.equal(v,current[k]) for k,v in initial.items() if k.startswith(prefix)),prefix)

    def test_pairwise_equation_and_control_capacity(self):
        pair = CurrentHistoryRelationBlock(256,1024,1600,dropout=0)
        control = CurrentHistoryRelationBlock(256,1024,1600,dropout=0,mode='control')
        n = sum(p.numel() for p in pair.parameters()); nc = sum(p.numel() for p in control.parameters())
        self.assertLess(abs(n-nc)/n,.001)
        self.assertFalse(hasattr(control,'evidence_type'))
        local,base,gap = torch.randn(3,1,256),torch.randn(3,1,1024),torch.randn(3,1,3,1024)
        a = torch.tensor([[[1.,0,1]]]*3); u = torch.ones(1,3); anchor=torch.ones(3,1,1600)
        with torch.no_grad(): pair.relation_out.weight.normal_()
        actual = pair(local,base,gap,a,u,anchor)
        q = pair.local_relation(local)[...,None,:].expand(3,1,4,128)
        k = pair.memory_relation(torch.cat((base[...,None,:512],gap[...,:512]),2))
        ty = pair.evidence_type.weight.view(1,1,4,16).expand(3,1,4,16)
        active=torch.tensor([[[0,0,0,0]],[[1,0,1,0]],[[1,0,1,0]]],dtype=torch.bool)
        z=torch.cat((q,k,q*k,(q-k).abs(),ty),-1)
        z=torch.where(active[...,None],z,torch.zeros_like(z))
        r=pair.relation_mlp(z);r=torch.where(active[...,None],r,torch.zeros_like(r))
        expected=pair.relation_out(r.sum(2)/active.sum(-1).clamp_min(1)[...,None]);expected[0]=0
        torch.testing.assert_close(actual,expected)

    def test_mask_history_and_gradients(self):
        for mode in ('pairwise', 'control'):
            m = CurrentHistoryRelationBlock(4, 6, 8, relation_dim=5, relation_out_dim=3, dropout=0, mode=mode)
            local, base, gap = torch.randn(4,2,4), torch.randn(4,2,6), torch.randn(4,2,3,6)
            avail = torch.tensor([[[1.,0,1],[0,1,0]]] * 4)
            mask = torch.tensor([[1,1,1,1],[0,1,1,0]])
            anchor = torch.randn(4,2,8)
            args = (local,base,gap,avail,mask,anchor)
            self.assertEqual(m(*args).count_nonzero(), 0)
            with torch.no_grad():
                m.relation_out.weight.fill_(.1); m.relation_out.bias.fill_(.2)
            expected = m(*args)
            valid = mask.T.bool()
            nohistory = ~(valid & ((valid.long().cumsum(0)-valid.long())>0))
            self.assertEqual(expected[nohistory].count_nonzero(), 0)
            gap[avail.bool() | ~valid[...,None]] = float('nan')
            for x in (local,base,anchor): x[~valid] = float('nan')
            base[...,3:] = float('nan'); gap[...,3:] = float('nan')
            for x in (local,base,gap): x.requires_grad_()
            out = m(*args)
            torch.testing.assert_close(out,expected)
            out.square().sum().backward()
            for p in m.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            for x in (local,base,gap): self.assertTrue(torch.isfinite(x.grad).all())

    def test_original_initialization_and_memory(self):
        kw = dict(latent_dim=8,output_dim=10,num_heads=2,key_dim=3,value_dim=4,dropout=.3,bidirectional=False)
        torch.manual_seed(42); old = OSRAMBackbone(**kw); rng = torch.get_rng_state()
        torch.manual_seed(42); new = OSRAMBackbone(**kw,osram_relation_block=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in old.state_dict().items(): self.assertTrue(torch.equal(v,new.state_dict()[k]),k)
        old.load_state_dict(old.state_dict(),strict=True)
        self.assertFalse(any('relation_block' in k for k in old.state_dict()))
        node = torch.randn(4,2,8)
        args = (node,{k:torch.randn_like(node) for k in MODALITIES},torch.tensor([[[1.,0,1],[0,1,0]]] * 4),torch.zeros(2,4,dtype=torch.long),torch.ones(2,4))
        for training in (True,False):
            old.train(training);new.train(training)
            torch.manual_seed(17); h,c = old(*args);rng = torch.get_rng_state()
            torch.manual_seed(17); hn,cn = new(*args)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(h,hn))
            for k in c: self.assertTrue(torch.equal(c[k],cn[k]))
        self.assertTrue(all(p.requires_grad for p in new.parameters()))

if __name__ == '__main__': unittest.main()
