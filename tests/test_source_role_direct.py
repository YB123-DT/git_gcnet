from dataclasses import asdict,replace
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from gcnet_missing_m3.source_role import CoupledSourceRole
from test_osram import _inputs
from test_source_role_integration import make


def inputs():
    torch.manual_seed(73)
    return [torch.randn(3,2,8),torch.randn(3,2,3,8),torch.randn(3,2,3,3,8),
        torch.ones(3,2,3,4),torch.zeros(3,2,3),torch.ones(2,3)]


def test_direct_zero_scores_are_unit_coefficients_and_source_sum_not_zero():
    module=CoupledSourceRole(8,8,mode='direct');a=inputs()
    out,gamma=module(*a)
    assert torch.equal(gamma,torch.ones_like(gamma))
    reads=torch.cat((a[1].unsqueeze(2),a[2]),2)
    assert torch.equal(out,reads.sum(3))
    assert sum(p.numel() for p in module.parameters())==sum(p.numel() for p in CoupledSourceRole(8,8).parameters())
    assert 'weighted_read_norm' in module.last_diagnostics and 'delta_norm' not in module.last_diagnostics


@pytest.mark.parametrize('bias',[-8.,-1.,1.,8.])
def test_direct_coefficients_follow_twice_sigmoid_without_normalization(bias):
    m=CoupledSourceRole(8,8,mode='direct');torch.nn.init.constant_(m.net[-1].bias,bias)
    out,g=m(*inputs());expected=2*torch.sigmoid(torch.tensor(bias))
    torch.testing.assert_close(g,torch.full_like(g,expected),rtol=0,atol=0)
    assert (g>=0).all() and (g<=2).all()
    out.square().sum().backward()
    assert m.net[-1].weight.grad is not None and torch.isfinite(m.net[-1].weight.grad).all()


def test_direct_safe_masks_remove_poisoned_inactive_sources_and_padding():
    m=CoupledSourceRole(8,8,mode='direct');a=inputs()
    a[3][0]=0;a[3][:,:,1,3]=0;a[4][:,:,0]=1;a[5][1,2]=0
    clean=m(*a)
    a[1][:,:,1]=float('nan');a[2][:,:,:,1]=float('nan');a[2][:,:,0]=float('nan')
    a[0][0]=float('nan');a[1][0]=float('nan');a[2][0]=float('nan')
    for k in range(5):a[k][2,1]=float('nan')
    out,gamma=m(*a)
    for before,after in zip(clean,(out,gamma)):torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert not gamma[...,1].any() and not out[0].any() and not out[2,1].any()
    out.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_direct_readout_replaces_forward_evidence_without_original_addback():
    m=make(osram_source_role=True,osram_source_role_mode='direct').eval()
    torch.nn.init.normal_(m.emotion_adapter[-1].weight,std=.1)
    torch.nn.init.constant_(m.source_role.net[-1].bias,-1.)
    outputs=[];adapter=[]
    hooks=[m.source_role.register_forward_hook(lambda _,x,y:outputs.append(y[0].detach())),
           m.emotion_adapter.register_forward_pre_hook(lambda _,x:adapter.append(x[0].detach()))]
    try:
        hidden,context=m(*_inputs())
    finally:
        for h in hooks:h.remove()
    weighted=outputs[-1];x=adapter[-1];fd=8;ld=8
    evidence=x[...,ld:].reshape(4,2,4,2*fd)
    assert torch.equal(evidence[...,:fd],weighted)
    assert not evidence[...,fd:].any() and not hidden[-1].any()
    assert torch.equal(x[...,:ld],context['local'])
    m.osram_source_role=False;_,raw=m(*_inputs())
    for k in raw:assert torch.equal(raw[k],context[k])


def test_direct_zero_init_is_tolerance_equal_to_flat_and_reloadable():
    torch.manual_seed(37);base=make().eval();rng=torch.get_rng_state().clone()
    torch.manual_seed(37);direct=make(osram_source_role=True,osram_source_role_mode='direct').eval()
    assert torch.equal(rng,torch.get_rng_state())
    torch.nn.init.normal_(base.emotion_adapter[-1].weight,std=.1)
    state=direct.state_dict();state.update(base.state_dict());direct.load_state_dict(state,strict=True)
    old,_=base(*_inputs());new,_=direct(*_inputs())
    torch.testing.assert_close(new,old,atol=1e-5,rtol=1e-5)
    clone=make(osram_source_role=True,osram_source_role_mode='direct').eval()
    clone.load_state_dict(direct.state_dict(),strict=True)
    assert torch.equal(new,clone(*_inputs())[0])


def test_residual_default_keeps_previous_module_exact():
    source=subprocess.check_output(['git','show','f6bc52e:gcnet_missing_m3/source_role.py'],
        cwd=Path(__file__).resolve().parents[1],text=True)
    ns={};exec(compile(source,'previous_source_role.py','exec'),ns)
    torch.manual_seed(12);old=ns['CoupledSourceRole'](8,8)
    torch.manual_seed(12);new=CoupledSourceRole(8,8)
    assert list(old.state_dict())==list(new.state_dict())
    with torch.no_grad():old.net[-1].weight.normal_(std=.05)
    new.load_state_dict(old.state_dict(),strict=True)
    for before,after in zip(old(*inputs()),new(*inputs())):assert torch.equal(before,after)


def test_worker_direct_config_and_reference_guard_whitelist_only_mode():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_coupled_source_role_20261020.worker import source_role_config,validate_reference
    cfg=replace(TrainConfig(),dataset='CMUMOSI',backbone_type='osram',training_objective='emotion-only',
        osram_bidirectional=False,
        missing_mask_protocol='gcnet-random-mask-v1',checkpoint_selection='test-oracle-per-rate',checkpoint_selection_metric='weighted_f1')
    new=source_role_config(cfg,source_role_mode='direct')
    expected=asdict(cfg);expected.update(osram_source_role=True,osram_source_role_mode='direct')
    assert asdict(new)==expected
    current=dict(config=expected,split={},seed_protocol={},mask_hashes={})
    old=dict(status='complete',epoch=150,initial_backbone_sha256='init',identity=copy.deepcopy(current))
    old['identity']['config']=json.loads(json.dumps(asdict(cfg)))
    assert validate_reference(current,old,'init',150,source_role_mode='direct')['only_source_role_settings_changed']
    with pytest.raises(ValueError):validate_reference(current,old,'init',150)


def test_zero_init_check_direct_uses_declared_tolerance_not_fake_bitwise_parity():
    from experiments.osram_coupled_source_role_20261020.worker import check_zero_init_arrays
    old=dict(prediction=np.array([[.5]],dtype=np.float32),labels=np.array([1.]))
    new=dict(old,prediction=old['prediction']+np.float32(2e-7))
    proof=check_zero_init_arrays(old,new,'direct')
    assert not proof['zero_init_predictions_exact'] and proof['zero_init_predictions_within_tolerance']
    with pytest.raises(AssertionError):check_zero_init_arrays(old,new,'residual')


def test_direct_plan_is_exactly_one_mosi_seed66_job(tmp_path):
    from experiments.osram_mosi_source_role_direct_20261020.run import plan_jobs
    jobs=plan_jobs(tmp_path,'sha',['GPU-healthy'])
    assert len(jobs)==1 and jobs[0]['dataset']=='CMUMOSI' and jobs[0]['seed']==66
    assert jobs[0]['command'][2]=='experiments.osram_mosi_source_role_direct_20261020.worker'
