from unittest.mock import patch
import torch
from gcnet_missing_m3 import osram


def inputs():
    torch.manual_seed(12)
    l=torch.randn(3,2,8);b=torch.randn(3,2,12);g=torch.randn(3,2,3,12)
    a=torch.tensor([[[1,1,1],[1,0,1]],[[0,1,0],[1,1,0]],[[0,0,0],[0,0,0]]]).float()
    u=torch.tensor([[1,1,0],[1,1,0]]).float()
    return l,b,g,a,u


def fusion():
    assert hasattr(osram,'LocalCrossAttentionFusion'), 'cross-attention not implemented'
    return osram.LocalCrossAttentionFusion(8,12,10,dropout=0).eval()


def test_shape_zero_init_active_softmax_and_padding():
    m=fusion();args=inputs();h=m(*args);valid=args[-1].T.bool()
    expected=m.emotion_norm(m.local_skip(args[0])).masked_fill(~valid[...,None],0)
    assert torch.equal(h,expected)
    assert m.attention_dim==512 and m.num_heads==4
    active=torch.cat([valid[...,None],valid[...,None]&~args[3].bool()],-1)
    assert torch.equal(m.last_active,active)
    weights=m.last_attention
    assert torch.count_nonzero(weights.masked_select(~active.unsqueeze(-2)))==0
    torch.testing.assert_close(weights.sum(-1)[valid],torch.ones(4,4))
    assert not h[~valid].count_nonzero()


def test_inactive_nan_ignored_active_content_and_local_change():
    m=fusion();torch.nn.init.normal_(m.context_out[-1].weight,std=.1)
    args=list(inputs());h=m(*args);weights=m.last_attention.clone()
    changed=list(args);changed[2]=args[2].clone()
    active=(args[-1].T.bool()[...,None]&~args[3].bool())
    changed[2][~active]=float('nan')
    assert torch.equal(h,m(*changed))
    changed=list(args);changed[2]=args[2].clone();changed[2][0,1,1]+=torch.randn(12)*3
    assert not torch.equal(h,m(*changed))
    changed=list(args);changed[0]=args[0]+torch.randn_like(args[0]);m(*changed)
    assert not torch.equal(weights,m.last_attention)
    empty=list(args);empty[-1]=torch.zeros_like(args[-1]);assert not m(*empty).count_nonzero()


def test_gradients_after_zero_init_first_update():
    m=fusion();args=inputs();optimizer=torch.optim.SGD(m.parameters(),lr=.01)
    for step in range(2):
        optimizer.zero_grad();h=m(*args);loss=(h*torch.arange(10.)).sum();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        if step:
            for name in ('query','key','value','context_out','evidence_type'):
                assert sum(float(p.grad.abs().sum()) for p in getattr(m,name).parameters())>0,name
        optimizer.step()


def test_shared_parameters_rng_context_and_memory_unchanged():
    fusion()
    kw=dict(latent_dim=8,output_dim=10,num_heads=2,key_dim=3,value_dim=3,
            dropout=0,n_speakers=1,bidirectional=False,write_step=.6)
    torch.manual_seed(31);flat=osram.OSRAMBackbone(**kw).eval();rng=torch.get_rng_state()
    torch.manual_seed(31);cross=osram.OSRAMBackbone(**kw,osram_readout_fusion='local-cross-attn').eval()
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in flat.state_dict().items():assert torch.equal(v,cross.state_dict()[k]),k
    local,_,_,a,u=inputs();z={k:torch.randn_like(local) for k in osram.MODALITIES}
    args=(local,z,a,torch.zeros(2,3).long(),u,[2,2]);outputs=[];traces=[]
    for model in (flat,cross):
        states=[];old=model.block_write
        def capture(*a,**kw):
            out=old(*a,**kw);states.append(out.detach().clone());return out
        with patch.object(model,'block_write',side_effect=capture):outputs.append(model(*args))
        traces.append(states)
    assert len(traces[0])==len(traces[1])
    assert all(torch.equal(a,b) for a,b in zip(*traces))
    for k in ('local','base','gap'):assert torch.equal(outputs[0][1][k],outputs[1][1][k])
    assert torch.equal(outputs[0][0],outputs[1][0])
