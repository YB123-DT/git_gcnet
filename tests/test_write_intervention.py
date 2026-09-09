import torch
from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3.write_intervention import protect_update, WriteIntervention


def test_protection_and_global_norm_same_state():
    delta=torch.tensor([[[[2.,3.],[4.,5.]]]],dtype=torch.float64)
    history=torch.tensor([[[[1.,0.,0.],[0.,0.,0.]]]],dtype=torch.float64)
    protected,glob,ratio=protect_update(delta,history,torch.tensor([[True,False,False]]),.001)
    assert torch.linalg.vector_norm(protected[...,0])<.005
    torch.testing.assert_close(protected[...,1],delta[...,1])
    torch.testing.assert_close(protected.norm(),glob.norm(),rtol=1e-8,atol=1e-8)
    torch.testing.assert_close(glob,delta*ratio[...,None,None])
    none,g,_=protect_update(delta,history,torch.zeros(1,3,dtype=torch.bool),.001)
    assert torch.equal(none,delta) and torch.equal(g,delta)


def test_forward_causal_complete_and_restore():
    torch.manual_seed(10)
    model=OSRAMBackbone(latent_dim=4,output_dim=4,num_heads=2,key_dim=2,value_dim=2,
                       dropout=0,bidirectional=False).eval()
    torch.nn.init.normal_(model.emotion_adapter[-1].weight)
    node=torch.randn(3,1,4);latents={n:torch.randn(3,1,4) for n in ('audio','text','visual')}
    q=torch.zeros(1,3,dtype=torch.long);u=torch.ones(1,3)
    full=torch.ones(3,1,3)
    original=model(node,latents,full,q,u,[3])[0]
    for mode in ('reference','protected','global'):
        with WriteIntervention(model,mode) as audit:
            out=model(node,latents,full,q,u,[3])[0]
            assert torch.equal(out,original)
            assert all(not r['has_protection'] for r in audit.records)
    assert 'block_write' not in model.__dict__


    a=torch.tensor([[[1.,0.,0.]],[[0.,1.,0.]],[[0.,0.,1.]]])
    for mode in ('protected','global'):
        with WriteIntervention(model,mode) as audit:
            out=model(node,latents,a,q,u,[3])[0]
            assert not any(r['has_protection'] for r in audit.records if r['time_index']==0)
            changed={k:v.clone() for k,v in latents.items()}
            for v in changed.values():v[2,:,0]+=10
            future=model(node,changed,a,q,u,[3])[0]
            assert torch.equal(out[:2],future[:2])
    assert 'block_write' not in model.__dict__


def test_zero_and_collinear_protected_updates_are_finite():
    delta=torch.randn(2,2,3,2,dtype=torch.float64)
    history=torch.ones(2,2,2,3,dtype=torch.float64)
    history=history/history.norm(dim=-2,keepdim=True)
    mask=torch.tensor([[True,True,False],[False,False,False]])
    protected,glob,ratio=protect_update(delta,history,mask)
    assert torch.isfinite(protected).all() and torch.isfinite(glob).all()
    assert torch.equal(protected[1],delta[1])
    torch.testing.assert_close(protected.norm(dim=(-2,-1)),glob.norm(dim=(-2,-1)),rtol=1e-7,atol=1e-8)
    p,g,r=protect_update(torch.zeros_like(delta),history,mask)
    assert torch.count_nonzero(p)==0 and torch.count_nonzero(g)==0 and torch.isfinite(r).all()
