"""V3 is a read-input correction, not the old global hidden residual."""
import os
from pathlib import Path
import subprocess

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from test_osram import _inputs


def make(**kwargs):
    return OSRAMBackbone(latent_dim=8,output_dim=10,num_heads=2,key_dim=3,
        value_dim=4,n_speakers=2,dropout=.2,bidirectional=False,write_step=.6,**kwargs)


def test_disabled_v3_strict_legacy_weights_training_output_gradients_and_rng():
    path=os.environ.get('OSRAM_V3_LEGACY_SOURCE')
    source=Path(path).read_text() if path else subprocess.check_output(
        ['git','show','7d9e9c4:gcnet_missing_m3/osram.py'],text=True)
    ns={'__name__':'gcnet_missing_m3._v3_legacy','__package__':'gcnet_missing_m3'}
    exec(compile(source,'before_v3_osram.py','exec'),ns)
    torch.manual_seed(66)
    old=ns['OSRAMBackbone'](latent_dim=8,output_dim=10,num_heads=2,key_dim=3,
        value_dim=4,n_speakers=2,dropout=.2,bidirectional=False,write_step=.6)
    initial=torch.get_rng_state().clone()
    torch.manual_seed(66); new=make(osram_source_role_v3=False)
    assert torch.equal(initial,torch.get_rng_state())
    new.load_state_dict(old.state_dict(),strict=True)
    state=torch.get_rng_state().clone()
    a,ac=old(*_inputs()); a[...,0].sum().backward(); after=torch.get_rng_state().clone()
    torch.set_rng_state(state)
    b,bc=new(*_inputs()); b[...,0].sum().backward()
    assert torch.equal(a,b) and torch.equal(after,torch.get_rng_state())
    for name in ac: assert torch.equal(ac[name],bc[name])
    for name,p in old.named_parameters():
        q=dict(new.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None: assert torch.equal(p.grad,q.grad),name


def test_zero_init_v3_single_scan_and_exact_original_nonzero_flat_predictions():
    torch.manual_seed(66); baseline=make(); initial=torch.get_rng_state().clone()
    torch.manual_seed(66); model=make(osram_source_role_v3=True)
    assert torch.equal(initial,torch.get_rng_state())
    for name,v in baseline.state_dict().items(): assert torch.equal(v,model.state_dict()[name])
    with torch.no_grad(): baseline.emotion_adapter[-1].weight.normal_(std=.03)
    state=model.state_dict(); state.update(baseline.state_dict()); model.load_state_dict(state,strict=True)
    before=torch.get_rng_state().clone(); expected,ec=baseline(*_inputs()); after=torch.get_rng_state().clone()
    scans=[]; scan=model._scan
    def counted(*args,**kwargs):
        scans.append(1); return scan(*args,**kwargs)
    model._scan=counted
    torch.set_rng_state(before); actual,ac=model(*_inputs())
    assert len(scans)==1
    assert torch.equal(expected,actual) and torch.equal(after,torch.get_rng_state())
    for name in ec: assert torch.equal(ec[name],ac[name])
    assert model.last_source_trace['source_base'].requires_grad
    assert model.last_source_trace['source_gap'].requires_grad
    assert not hasattr(model,'source_relation')
    assert not hasattr(model.source_role_v3,'output_projection')


def test_v3_default_off_cli_factory_and_no_stacking():
    from dataclasses import replace
    from gcnet_missing_m3.train_gcnet import TrainConfig,build_parser
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    cfg=replace(TrainConfig(),dataset='CMUMOSI',backbone_type='osram',
        training_objective='emotion-only',disable_unused_aux_modules=True,
        latent_dim=8,osram_output_dim=10,osram_num_heads=2,osram_key_dim=3,
        osram_value_dim=4,osram_bidirectional=False,osram_source_role_v3=True)
    assert not TrainConfig().osram_source_role_v3
    model=_build_model(cfg,(4,5,6),torch.device('cpu'))
    assert model.osram.osram_source_role_v3 and hasattr(model.osram,'source_role_v3')
    required=['--audio-feature','a','--text-feature','t','--video-feature','v','--output-dir','/tmp/not-used-v3']
    parser=build_parser()
    assert not parser.parse_args(required).osram_source_role_v3
    assert parser.parse_args(required+['--osram-source-role-v3']).osram_source_role_v3
    for extra in [{'osram_source_relation':True},{'osram_source_role':True},
                  {'osram_tasr':True},{'osram_meaningful_block':'nested_gnn_rooted_evidence'}]:
        with pytest.raises(ValueError): make(osram_source_role_v3=True,**extra)


def test_causal_real_source_gradients_and_unchanged_padding():
    model=make(osram_source_role_v3=True).eval()
    with torch.no_grad(): model.emotion_adapter[-1].weight.normal_(std=.03)
    for layer in model.source_role_v3.decoders:
        with torch.no_grad(): layer.weight.normal_(std=.01)
    output,_=model(*_inputs())
    trace=model.last_source_trace
    for name in ('source_base','source_gap'): trace[name].retain_grad()
    output[...,0].sum().backward()
    for name in ('source_base','source_gap'):
        g=trace[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
    args=list(_inputs()); args[0]=args[0].clone(); args[0][2]+=5
    args[1]={name:v.clone() for name,v in args[1].items()}
    for v in args[1].values(): v[2]+=7
    actual,_=model(*args)
    torch.testing.assert_close(actual[:2],output[:2],rtol=0,atol=0)
    assert not actual[3].any()


def test_role_corrections_enter_flat_forward_slices_not_local_skip_or_raw_contexts():
    model=make(osram_source_role_v3=True).eval()
    for role,layer in enumerate(model.source_role_v3.decoders):
        with torch.no_grad(): layer.bias.fill_(float(role+1))
    captured=[]
    h=model.emotion_adapter.register_forward_pre_hook(lambda _,v:captured.append(v[0].detach().clone()))
    args=_inputs(); _,contexts=model(*args); h.remove()
    features=captured[-1]; f=model.num_heads*model.value_dim; valid=args[4].T.bool()
    availability=args[2]; has_history=model.last_source_trace['trajectory'][...,3].bool().any(-1)
    enabled=valid & has_history & (~availability.bool()).any(-1)
    base=features[...,8:8+2*f]; gap=features[...,8+2*f:].reshape(*features.shape[:2],3,2*f)
    torch.testing.assert_close(features[...,:8],contexts['local'],rtol=0,atol=0)
    torch.testing.assert_close(base[...,:f]-contexts['base'][...,:f],
                              enabled[...,None].expand_as(base[...,:f]).float(),rtol=0,atol=1e-6)
    assert not base[...,f:].any() and not gap[...,f:].any()
    for role in range(3):
        active=enabled & ~availability[...,role].bool()
        expected=active[...,None].expand_as(gap[...,role,:f]).float()*(role+2)
        torch.testing.assert_close(gap[...,role,:f]-contexts['gap'][...,role,:f],expected,rtol=0,atol=1e-6)
