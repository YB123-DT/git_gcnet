from pathlib import Path
import subprocess

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from test_osram import _inputs


def make(**kwargs):
    return OSRAMBackbone(latent_dim=8,output_dim=10,num_heads=2,key_dim=3,value_dim=4,
        n_speakers=2,dropout=kwargs.pop('dropout',0.),bidirectional=False,write_step=.6,**kwargs)


def test_disabled_matches_pre_source_role_checkpoint_output_and_rng():
    source=subprocess.check_output(['git','show','27b8c1f:gcnet_missing_m3/osram.py'],
        cwd=Path(__file__).resolve().parents[1],text=True)
    ns={'__name__':'gcnet_missing_m3._before_source_role','__package__':'gcnet_missing_m3'}
    exec(compile(source,'old_osram.py','exec'),ns)
    torch.manual_seed(315)
    old=ns['OSRAMBackbone'](latent_dim=8,output_dim=10,num_heads=2,key_dim=3,value_dim=4,
        n_speakers=2,dropout=.2,bidirectional=False,write_step=.6)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(315);new=make(dropout=.2,osram_source_role=False)
    assert torch.equal(rng,torch.get_rng_state())
    new.load_state_dict(old.state_dict(),strict=True)
    torch.nn.init.normal_(old.emotion_adapter[-1].weight,std=.1)
    new.load_state_dict(old.state_dict(),strict=True)
    args=_inputs();rng=torch.get_rng_state().clone()
    a,ac=old(*args);a[...,0].sum().backward();after=torch.get_rng_state().clone()
    torch.set_rng_state(rng);b,bc=new(*args);b[...,0].sum().backward()
    assert torch.equal(a,b) and torch.equal(after,torch.get_rng_state())
    for k in ac:assert torch.equal(ac[k],bc[k])
    for name,p in old.named_parameters():
        q=dict(new.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:assert torch.equal(p.grad,q.grad),name


def test_zero_init_keeps_initialization_logits_rng_and_contexts_exact():
    torch.manual_seed(32);base=make(dropout=.2);rng=torch.get_rng_state().clone()
    torch.manual_seed(32);new=make(dropout=.2,osram_source_role=True)
    assert torch.equal(rng,torch.get_rng_state())
    for name,p in base.state_dict().items():assert torch.equal(new.state_dict()[name],p),name
    torch.nn.init.normal_(base.emotion_adapter[-1].weight,std=.2)
    new.load_state_dict(base.state_dict(),strict=False)
    args=_inputs();rng=torch.get_rng_state().clone();a,ac=base(*args);after=torch.get_rng_state().clone()
    torch.set_rng_state(rng);b,bc=new(*args)
    assert torch.equal(a,b) and torch.equal(after,torch.get_rng_state())
    for name in ac:assert torch.equal(ac[name],bc[name])
    trace=new.last_source_trace
    assert trace['source_base'].requires_grad and trace['source_gap'].requires_grad
    torch.testing.assert_close(trace['source_base'].sum(2),bc['base'][...,:8],rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(trace['source_gap'].sum(3),bc['gap'][...,:8],rtol=3e-5,atol=3e-6)
    assert not new.source_role.last_gamma[0].any()


def test_module_trains_without_modifying_memory_queries_local_or_backward_half():
    m=make(osram_source_role=True).eval();args=_inputs()
    torch.nn.init.normal_(m.emotion_adapter[-1].weight,std=.1)
    optimizer=torch.optim.Adam(m.parameters(),lr=.01)
    initial={n:p.detach().clone() for n,p in m.source_role.named_parameters()}
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True);h,_=m(*args)
        (h[...,0]-.3).square().mean().backward()
        gradients=[p.grad for p in m.source_role.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        optimizer.step()
    assert any(not torch.equal(initial[n],p) for n,p in m.source_role.named_parameters())
    on,context=m(*args)
    assert not on[-1].any()
    m.osram_source_role=False;off,old=m(*args);m.osram_source_role=True
    for k in context:assert torch.equal(context[k],old[k]),k
    assert not torch.equal(on,off)
    changed=list(args);changed[0]=args[0].clone();changed[0][2]+=5
    changed[1]={k:v.clone() for k,v in args[1].items()}
    for v in changed[1].values():v[2]-=7
    future,_=m(*changed);assert torch.equal(on[:2],future[:2])
    clone=make(osram_source_role=True).eval();clone.load_state_dict(m.state_dict(),strict=True)
    assert torch.equal(on,clone(*args)[0])


@pytest.mark.parametrize('attribute',['osram_ablation','osram_emotion_ablation'])
def test_runtime_ablation_never_reintroduces_disabled_evidence(attribute):
    m=make(osram_source_role=True).eval()
    torch.nn.init.constant_(m.source_role.net[-1].bias,1.)
    captures=[]
    hook=m.emotion_adapter.register_forward_pre_hook(lambda module,args:captures.append(args[0].detach()))
    try:
        for ablation in ('local-only','local-base','local-gap'):
            setattr(m,attribute,ablation)
            # osram_ablation does not offer local-gap; emotion ablation does.
            if attribute=='osram_ablation' and ablation=='local-gap':continue
            m(*_inputs());x=captures[-1]
            if ablation in ('local-only','local-gap'):assert not x[...,8:24].any()
            if ablation in ('local-only','local-base'):assert not x[...,24:].any()
    finally:hook.remove()


def test_flag_rejects_tasr_overlap_and_has_trainconfig_builder_route():
    from dataclasses import replace
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    with pytest.raises(ValueError,match='source.role|TASR'):make(osram_source_role=True,osram_tasr=True)
    cfg=replace(TrainConfig(),dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',
        disable_unused_aux_modules=True,latent_dim=8,osram_output_dim=10,osram_num_heads=2,
        osram_key_dim=3,osram_value_dim=4,osram_source_role=True,osram_bidirectional=False)
    m=_build_model(cfg,(4,5,6),torch.device('cpu'))
    assert m.osram.osram_source_role and hasattr(m.osram,'source_role')
