"""Second read uses the real pre-write state without changing CHM writes."""
import subprocess

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from test_source_role_v3_connections_integration import _inputs


def make(enabled=False, mode='free', dropout=.2):
    return OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
        key_dim=3, value_dim=4, n_speakers=2, dropout=dropout,
        bidirectional=False, write_step=.6,
        osram_conditional_reread=enabled, osram_reread_mode=mode)


def test_disabled_matches_prior_code_forward_backward_rng_and_checkpoint():
    source=subprocess.check_output(['git','show','515d785:gcnet_missing_m3/osram.py'],text=True)
    ns={'__name__':'gcnet_missing_m3._pre_reread','__package__':'gcnet_missing_m3'}
    exec(compile(source,'pre_reread.py','exec'),ns)
    torch.manual_seed(66)
    old=ns['OSRAMBackbone'](latent_dim=8,output_dim=10,num_heads=2,key_dim=3,
        value_dim=4,n_speakers=2,dropout=.2,bidirectional=False,write_step=.6)
    init=torch.get_rng_state().clone()
    torch.manual_seed(66);new=make()
    assert torch.equal(init,torch.get_rng_state())
    new.load_state_dict(old.state_dict(),strict=True)
    args=_inputs();rng=torch.get_rng_state().clone()
    y,c=old(*args);y[...,0].sum().backward();after=torch.get_rng_state().clone()
    torch.set_rng_state(rng);z,d=new(*args);z[...,0].sum().backward()
    assert torch.equal(y,z) and torch.equal(after,torch.get_rng_state())
    for name in c:assert torch.equal(c[name],d[name]),name
    for name,p in old.named_parameters():
        g=dict(new.named_parameters())[name].grad
        assert (p.grad is None)==(g is None),name
        if g is not None:assert torch.equal(g,p.grad),name


@pytest.mark.parametrize('mode',['free','span'])
def test_zero_init_exact_single_scan_single_local_and_unchanged_write(mode):
    torch.manual_seed(66);flat=make();init=torch.get_rng_state().clone()
    torch.manual_seed(66);model=make(True,mode)
    assert torch.equal(init,torch.get_rng_state())
    for name,value in flat.state_dict().items():assert torch.equal(value,model.state_dict()[name])
    args=_inputs();rng=torch.get_rng_state().clone();y,c=flat(*args)
    torch.set_rng_state(rng);calls=[];scan_calls=[];states=[]
    handle=model.local_path.register_forward_hook(lambda *a:calls.append(1))
    scan=model._scan
    def count(*a,**kw):scan_calls.append(1);return scan(*a,**kw)
    model._scan=count
    z,d=model(*args,post_write_observer=lambda t,m,v:states.append(m))
    handle.remove()
    assert torch.equal(y,z) and calls==[1] and scan_calls==[1]
    for name in c:assert torch.equal(c[name],d[name]),name
    expected=[];flat(*args,post_write_observer=lambda t,m,v:expected.append(m))
    assert all(torch.equal(a,b) for a,b in zip(states,expected))


@pytest.mark.parametrize('mode',['free','span'])
def test_nonzero_second_read_keeps_original_gap_local_write_and_is_causal(mode):
    torch.manual_seed(66);model=make(True,mode,dropout=0).eval()
    with torch.no_grad():
        model.conditional_reread.output_projection.weight.normal_(std=.2)
        model.emotion_adapter[-1].weight.normal_(std=.1)
    args=_inputs();states=[];seen=[]
    hook=model.conditional_reread.register_forward_pre_hook(lambda _,a:seen.append(a))
    y,c=model(*args,post_write_observer=lambda t,m,v:states.append(m));hook.remove()
    assert len(seen)==3
    # The current target reads the alpha-decayed state AFTER previous write.
    alpha=torch.sigmoid(model.alpha_logits).view(1,2,1,1)
    torch.testing.assert_close(seen[1][3],states[0]*alpha)
    assert not seen[0][3].any()
    assert torch.equal(seen[1][0],c['local'][1])
    assert torch.equal(seen[1][5],torch.cat((torch.ones(2,1,dtype=torch.bool),~args[2][1].bool()),-1))
    future=list(args);future[0]=args[0].clone();future[0][2].add_(100)
    future[1]={k:v.clone() for k,v in args[1].items()}
    for v in future[1].values():v[2].add_(100)
    y2,_=model(*future)
    assert torch.equal(y[:2],y2[:2])
    model.osram_conditional_reread=False
    z,d=model(*args)
    for name in ('local','base','gap'):assert torch.equal(c[name],d[name])
    assert not torch.equal(y[1:3],z[1:3])
    assert not y[3].any()


def test_config_parser_and_model_factory_forward_new_options():
    from dataclasses import replace
    from gcnet_missing_m3.train_gcnet import TrainConfig,build_parser
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    args=build_parser().parse_args(['--audio-feature','a','--text-feature','t',
        '--video-feature','v','--output-dir','/tmp/unused-reread',
        '--osram-conditional-reread','--osram-reread-mode','span'])
    assert args.osram_conditional_reread and args.osram_reread_mode=='span'
    cfg=replace(TrainConfig(),dataset='CMUMOSI',backbone_type='osram',
        training_objective='emotion-only',disable_unused_aux_modules=True,
        latent_dim=8,osram_output_dim=10,osram_num_heads=2,osram_key_dim=3,
        osram_value_dim=4,osram_bidirectional=False,osram_conditional_reread=True,
        osram_reread_mode='span')
    model=_build_model(cfg,(4,5,6),torch.device('cpu'))
    assert model.osram.conditional_reread.mode=='span'


@pytest.mark.parametrize('mode',['free','span'])
def test_zero_init_preserves_nonzero_trained_flat_exit_and_dropout_rng(mode):
    torch.manual_seed(66);flat=make()
    with torch.no_grad():flat.emotion_adapter[-1].weight.normal_(std=.1)
    torch.manual_seed(66);model=make(True,mode)
    model.load_state_dict({**model.state_dict(),**flat.state_dict()},strict=True)
    args=_inputs();rng=torch.get_rng_state().clone()
    a,_=flat(*args);after=torch.get_rng_state().clone()
    torch.set_rng_state(rng);b,_=model(*args)
    assert torch.equal(a,b) and torch.equal(after,torch.get_rng_state())


def test_second_read_rejects_counterfactual_readout_stacking():
    with pytest.raises(ValueError,match='Second read'):
        OSRAMBackbone(bidirectional=False,osram_conditional_reread=True,
                      osram_counterfactual_readout='marginal')
