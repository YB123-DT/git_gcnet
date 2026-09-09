import torch
import copy
import inspect
from gcnet_missing_m3.memory_retention import MemoryRetentionDiagnostics
from gcnet_missing_m3.osram import OSRAMBackbone


def test_instrumentation_exact_forward_backward_and_rng_parity():
    assert inspect.signature(OSRAMBackbone.forward).parameters['collect_memory_retention_diagnostics'].default is False
    torch.manual_seed(8)
    model = OSRAMBackbone(latent_dim=4,output_dim=4,num_heads=2,key_dim=2,
                          value_dim=2,n_speakers=1,dropout=.1,bidirectional=False)
    torch.nn.init.normal_(model.emotion_adapter[-1].weight)
    other = copy.deepcopy(model)
    node = torch.randn(3,1,4,requires_grad=True)
    latent = {m:torch.randn(3,1,4,requires_grad=True) for m in ('audio','text','visual')}
    a=torch.tensor([[[1.,0.,0.]],[[0.,1.,0.]],[[0.,0.,1.]]])
    q=torch.zeros(1,3,dtype=torch.long);u=torch.ones(1,3)
    rng=torch.get_rng_state()
    out,ctx=model(node,latent,a,q,u,[3])
    after=torch.get_rng_state();out.square().sum().backward()
    grads=[node.grad.clone()]+[x.grad.clone() for x in latent.values()]
    node.grad=None
    for x in latent.values(): x.grad=None
    records=[];collector=MemoryRetentionDiagnostics(records.append,dataset='toy',missing_rate=.5,sample_ids=['x'])
    torch.set_rng_state(rng)
    out2,ctx2=other(node,latent,a,q,u,[3],collect_memory_retention_diagnostics=True,memory_retention_diagnostics=collector)
    assert torch.equal(after,torch.get_rng_state())
    assert torch.equal(out,out2)
    for k in ctx: assert torch.equal(ctx[k],ctx2[k])
    out2.square().sum().backward()
    for x,y in zip(grads,[node.grad]+[x.grad for x in latent.values()]): assert torch.equal(x,y)
    for p,p2 in zip(model.parameters(),other.parameters()):
        assert (p.grad is None and p2.grad is None) or torch.equal(p.grad,p2.grad)
    assert model.state_dict().keys()==other.state_dict().keys()
    for k,v in model.state_dict().items(): assert torch.equal(v,other.state_dict()[k])
    assert records


def test_disabled_flag_ignores_collector():
    model=OSRAMBackbone(latent_dim=2,output_dim=2,num_heads=1,key_dim=2,value_dim=2,bidirectional=False)
    records=[]
    d=MemoryRetentionDiagnostics(records.append,dataset='toy',missing_rate=.5,sample_ids=['a'])
    node=torch.ones(1,1,2)
    model(node,{m:node for m in ('audio','text','visual')},torch.tensor([[[1.,0.,0.]]]),
          torch.zeros(1,1,dtype=torch.long),torch.ones(1,1),[1],memory_retention_diagnostics=d)
    assert not records and d.last_key is None


def test_manual_retention_and_probe_update():
    records = []
    d = MemoryRetentionDiagnostics(records.append, dataset='toy', missing_rate=.5,
                                   sample_ids=['conversation0'])
    keys = torch.tensor([[[[1., 0., 0.], [0., 1., 0.]]]], requires_grad=True)
    values = torch.tensor([[[[2., 3., 0.], [0., 0., 0.]]]], requires_grad=True)
    m = torch.tensor([[[[2., 0.], [0., 0.]]]], requires_grad=True)
    valid = torch.tensor([True])
    d.observe(0, m, m, m, keys, values, torch.tensor([[1., 0., 0.]]), valid)
    assert len(records) == 2 and all(r['status'] == 'NO_HISTORY' for r in records)
    records.clear()
    post = torch.zeros_like(m)
    d.observe(1, m, m*.5, post, keys, values, torch.tensor([[0., 1., 0.]]), valid)
    r = next(x for x in records if x['status'] == 'retention')
    assert r['history_distance'] == 1
    assert r['err_pre'] == 0
    assert abs(r['err_decay']-.5) < 1e-6
    assert abs(r['err_post']-1) < 1e-6
    assert abs(r['write_damage']-.5) < 1e-6
    assert r['max_key_overlap'] == 0
    assert not d.last_key.requires_grad and not d.last_value.requires_grad
    assert d.last_seen_time.tolist() == [[0, 1, -1]]
    records.clear()
    # The missing Audio key equals its old probe but must be excluded from overlap.
    # The observed Text key is orthogonal; a beneficial write has negative damage.
    d.observe(2, post, post, m, keys, values, torch.tensor([[0.,1.,0.]]), valid)
    r=next(x for x in records if x.get('target_modality')=='audio')
    assert r['history_distance']==2
    assert r['write_damage'] < -.99
    assert r['max_key_overlap']==0
    d.observe(3, m, m, m, keys, values, torch.zeros(1, 3), torch.tensor([False]))
    assert d.last_seen_time.tolist() == [[0, 2, -1]]


def test_single_key_block_write_effective_coefficient():
    model = OSRAMBackbone(latent_dim=2, output_dim=2, num_heads=1,key_dim=2,value_dim=2)
    m = torch.tensor([[[[.2, .3], [.4, .5]]]], dtype=torch.float64)
    k = torch.tensor([[[[1., 0., 0.], [0., 0., 0.]]]], dtype=torch.float64)
    v = torch.tensor([[[[2., 0., 0.], [3., 0., 0.]]]], dtype=torch.float64)
    out = model.block_write(m,k,v,torch.tensor([[1.,0.,0.]]), beta=torch.full((1,3),.5))
    coefficient = .5/(.5+.001)
    expected = m + coefficient * (v[...,0:1]-m@k[...,0:1])@k[...,0:1].transpose(-1,-2)
    torch.testing.assert_close(out,expected,rtol=1e-12,atol=1e-12)
    assert abs(coefficient-.998003992)<1e-9
