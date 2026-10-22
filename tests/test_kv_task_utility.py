"""Frozen joint-write forks must match an actual omit-write scan."""
import importlib.util

import pytest
import torch
from torch.nn import functional as F

from gcnet_missing_m3.osram import OSRAMBackbone, MODALITIES
from experiments.osram_meld_write_retention_20261014.run import WriteCapture
from experiments.osram_meld_write_retention_20261014.task_replay import original_read


def core():
    name = 'experiments.osram_kv_task_utility_20261022.core'
    assert importlib.util.find_spec(name) is not None, 'joint-write utility core is not implemented'
    return __import__(name, fromlist=['kv_quality', 'omit_write_forks'])


def inputs():
    torch.manual_seed(904)
    model = OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
        key_dim=3, value_dim=4, n_speakers=2, dropout=0.,
        bidirectional=False, write_step=.6).eval()
    availability = torch.tensor([[[1,1,1],[1,0,1]], [[0,1,1],[1,1,0]],
        [[1,0,0],[0,0,0]], [[0,1,0],[0,0,0]], [[1,1,0],[0,0,0]]]).float()
    availability = torch.cat((availability,availability[-1:].repeat(4,1,1)))
    valid = availability.bool().any(-1)
    keys = {m:F.normalize(torch.randn(9,2,2,3),dim=-1) for m in MODALITIES}
    values = {m:torch.randn(9,2,2,4) for m in MODALITIES}
    queries = F.normalize(torch.randn(9,2,4,2,3),dim=-1)
    return model,keys,values,queries,availability,valid


def test_quality_uses_actual_weighted_joint_columns_and_energy_normalization():
    api = core()
    model,keys,values,_,a,_ = inputs()
    k = torch.stack([keys[m][0] for m in MODALITIES],-1)
    v = torch.stack([values[m][0] for m in MODALITIES],-1)
    before = torch.randn(2,2,4,3)
    after = model.block_write(before,k,v,a[0])
    measured = api.kv_quality(model,before,after,k,v,a[0])
    scale = (a[0][:,None,:]*model.beta_logits.sigmoid().sqrt()[None]).unsqueeze(2)
    kb,vb = k*scale,v*scale
    energy = vb.square().sum((1,2,3))
    eb = (vb-before@kb).square().sum((1,2,3))/energy.clamp_min(1e-12)
    ea = (vb-after@kb).square().sum((1,2,3))/energy.clamp_min(1e-12)
    torch.testing.assert_close(measured['e_before'],eb)
    torch.testing.assert_close(measured['e_after'],ea)
    torch.testing.assert_close(measured['q_gain'],eb-ea)
    assert (measured['q_gain'] >= -1e-6).all()


@pytest.mark.parametrize('source',[0,1,3])
def test_forks_match_complete_omit_scan_and_keep_original_reads(source):
    api = core()
    model,keys,values,queries,a,valid = inputs()
    rng = torch.get_rng_state().clone()
    with torch.no_grad(), WriteCapture(model,valid.T) as captured:
        base,gap,_ = model._scan(keys,values,queries,a,valid,False)
    original = model.block_write
    calls = [0]
    def omit(memory,*args,**kwargs):
        t = calls[0];calls[0] += 1
        return memory if t == source else original(memory,*args,**kwargs)
    model.block_write = omit
    try:
        with torch.no_grad():
            ob,og,_ = model._scan(keys,values,queries,a,valid,False)
    finally:
        del model.block_write
    assert torch.equal(base[:source+1],ob[:source+1])
    assert torch.equal(gap[:source+1],og[:source+1])
    for distance,states,error in api.omit_write_forks(model,captured.snapshots,source,distances=(1,2,4,8)):
        t = source+distance
        k = torch.stack([keys[m][t] for m in MODALITIES],-1)
        kb,kg = original_read(model,states['keep'],queries[t],k,a[t],valid[t])
        rb,rg = original_read(model,states['omit'],queries[t],k,a[t],valid[t])
        torch.testing.assert_close(kb,base[t],rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(kg,gap[t],rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(rb,ob[t],rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(rg,og[t],rtol=2e-5,atol=2e-5)
        assert error < 2e-5
        if t >= 2:
            assert torch.equal(states['keep'][1],captured.snapshots[1][-1][1])
    assert torch.equal(torch.get_rng_state(),rng)
    assert 'block_write' not in model.__dict__


def test_zero_write_step_has_zero_quality_gain_and_no_counterfactual_effect():
    api = core()
    model,keys,values,q,a,valid = inputs();model.write_step = 0.
    with torch.no_grad(),WriteCapture(model,valid.T) as captured:
        model._scan(keys,values,q,a,valid,False)
    _,pre,k,v,av,post = captured.snapshots[0]
    assert torch.equal(api.kv_quality(model,pre,post,k,v,av)['q_gain'],torch.zeros(2))
    for _,states,_ in api.omit_write_forks(model,captured.snapshots,0,distances=(1,2,4)):
        assert torch.equal(states['keep'],states['omit'])


def test_invalid_distance_rejected():
    api = core();model,*_ = inputs()
    with pytest.raises(ValueError):
        list(api.omit_write_forks(model,[],0,distances=(0,)))


def test_mask_bank_hash_rejects_a_single_changed_bit():
    import hashlib
    from experiments.osram_kv_task_utility_20261022 import run
    assert hasattr(run,'validate_mask_bank'), 'exact original mask bank verification missing'
    bank = {'0.3':[dict(availability=torch.ones(2,1,3))]}
    expected = {'0.3':hashlib.sha256(bank['0.3'][0]['availability'].numpy().tobytes()).hexdigest()}
    assert run.validate_mask_bank(bank,expected)==expected
    bank['0.3'][0]['availability'][0,0,0] = 0
    with pytest.raises(ValueError,match='mask'):
        run.validate_mask_bank(bank,expected)
