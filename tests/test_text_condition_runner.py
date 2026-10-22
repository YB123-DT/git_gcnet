import torch


def test_isolated_prefix_uses_raw_current_text_and_never_changes_past():
    from experiments.osram_text_condition_optimization_audit_20261022.run import prefix_view
    full = torch.arange(4*2*6).reshape(4,2,6).float()
    avail = torch.ones(4,2,3)
    avail[2,1,1] = 0
    original = dict(complete=full,incomplete=full,availability=avail,
        qmask=torch.zeros(2,4),umask=torch.ones(2,4),labels=torch.zeros(2,4),
        lengths=[4,4],conversation_ids=['a','b'])
    yes = prefix_view(original,1,2,(2,2,2),True)
    no = prefix_view(original,1,2,(2,2,2),False)
    assert yes['incomplete'].shape == (3,1,6)
    assert yes['lengths'] == [3] and yes['conversation_ids'] == ['b']
    assert torch.equal(yes['incomplete'][:2],no['incomplete'][:2])
    assert torch.equal(yes['incomplete'][2,0,2:4],full[2,1,2:4])
    assert not no['incomplete'][2,0,2:4].any()
    assert torch.equal(yes['incomplete'][2,0,:2],no['incomplete'][2,0,:2])
    assert original['availability'][2,1,1] == 0


def test_select_targets_is_label_free_valid_and_leaves_observed_nontext():
    from experiments.osram_text_condition_optimization_audit_20261022.run import choose_targets
    view=dict(availability=torch.tensor([[[1.,1.,1.],[0.,1.,0.]],
                                         [[1.,0.,0.],[1.,1.,0.]],
                                         [[0.,1.,0.],[0.,0.,0.]]]),
              umask=torch.tensor([[1.,1.,1.],[1.,1.,0.]]),
              labels=torch.tensor([[0.,1.,-1.],[1.,0.,0.]]),conversation_ids=['a','b'])
    first=choose_targets(view,'CMUMOSI','0.7')
    view['labels'].mul_(-100)
    assert first == choose_targets(view,'CMUMOSI','0.7') == [(0,1),(1,1)]


def test_real_tiny_flat_pair_shares_prewrite_state_and_preserves_past_outputs():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_text_condition_optimization_audit_20261022.run import prefix_view,target_loss
    cfg=TrainConfig(dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',
        disable_unused_aux_modules=True,latent_dim=8,osram_output_dim=10,osram_num_heads=2,
        osram_key_dim=3,osram_value_dim=4,osram_bidirectional=False,osram_write_step=.6)
    model=_build_model(cfg,(4,5,6),torch.device('cpu')).requires_grad_(True).eval()
    original=dict(complete=torch.randn(3,1,15),availability=torch.ones(3,1,3),
        qmask=torch.zeros(1,3),umask=torch.ones(1,3),labels=torch.ones(1,3),
        lengths=[3],conversation_ids=['a'])
    yes=prefix_view(original,0,2,(4,5,6),True)
    no=prefix_view(original,0,2,(4,5,6),False)
    lv,mv=target_loss(model,yes,True);lm,mm=target_loss(model,no,True)
    assert mv['calls']==mm['calls']==3
    torch.testing.assert_close(mv['memory'],mm['memory'],rtol=0,atol=0)
    torch.testing.assert_close(mv['past_prediction'],mm['past_prediction'],rtol=0,atol=0)
    assert torch.isfinite(lv) and torch.isfinite(lm)
    params=list(model.parameters())
    gradients=torch.autograd.grad(lv,params,allow_unused=True)
    assert any(g is not None and bool(g.abs().sum()>0) for g in gradients)
    assert all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
    assert 'forward' not in model.observed_set.__dict__
    from experiments.osram_text_condition_optimization_audit_20261022.run import isolated_current_encoding
    import pytest
    with pytest.raises(RuntimeError,match='forced'):
        with isolated_current_encoding(model):
            raise RuntimeError('forced')
    assert 'forward' not in model.observed_set.__dict__
    model.train()
    with pytest.raises(ValueError,match='eval mode'):
        target_loss(model,yes)


def test_test_strata_reuses_actual_skip_and_executes_one_scan(tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from experiments.osram_text_condition_optimization_audit_20261022 import run
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.osram=torch.nn.Module()
            self.osram.latent_dim=2
            self.osram.local_skip=torch.nn.Linear(2,3)
            self.osram.emotion_adapter=torch.nn.Linear(6,3)
            self.osram.emotion_norm=torch.nn.LayerNorm(3)
            self.smax_fc=torch.nn.Linear(3,1)
            self.scans=0
        def forward(self,features,availability,qmask,umask,lengths,**kwargs):
            self.scans+=1
            x=features[0]
            hidden=self.osram.emotion_norm(self.osram.local_skip(x[...,:2].contiguous())+
                                          self.osram.emotion_adapter(x))
            return self.smax_fc(hidden),None,None,None
    model=Tiny().eval()
    features=torch.randn(3,1,6)
    a=torch.ones(3,1,3)
    data=[None]*10
    data[6]=torch.zeros(1,3);data[7]=torch.ones(1,3)
    data[8]=torch.tensor([[1.,-1.,1.]]);data[9]=['a']
    view=dict(incomplete=features,availability=a,qmask=data[6],umask=data[7],lengths=[3])
    with torch.no_grad():
        prediction=model([features],a,data[6],data[7],[3])[0].reshape(-1).numpy()
    meta=metadata(data)
    np.savez(tmp_path/'saved.npz',**meta,availability=a.transpose(0,1).reshape(-1,3).numpy(),
             prediction=prediction)
    loader=type('Loader',(list,),{})([data])
    loader.sampler=SimpleNamespace(set_epoch=lambda epoch:None)
    trainer=SimpleNamespace(_move_batch=lambda raw,device:raw,
        _prepare_view_from_primary_masks=lambda *args:view)
    bundle=dict(trainer=trainer,test_loader=loader,dimensions=(2,2,2),reference=tmp_path,
                reference_summary={'best_by_rate':{'0.3':{'predictions':'saved.npz'}}})
    count=[]
    handle=model.osram.local_skip.register_forward_hook(lambda *args:count.append(1))
    monkeypatch.setattr(run,'RATES',('0.3',))
    result=run.test_strata(bundle,model,{'0.3':[dict(meta,availability=a)]},tmp_path)
    handle.remove()
    assert model.scans==2  # one saved-reference setup plus one actual replay
    assert len(count)==1  # never recompute skip from the concatenated slice
    assert result['pooled_test_slots']['overall']['N']==3
