import copy
import pytest

from experiments.osram_mosei_frozen_nested_lr_20261027 import scheduler as s
from experiments.osram_mosei_frozen_nested_lr_20261027.protocol import jobs_for


def test_recovery_cannot_expand_scope_or_change_source():
    state=dict(source_commit='pinned',smoke=jobs_for('smoke'),train=jobs_for('train'))
    s.validate_state(state,'pinned')
    changed=copy.deepcopy(state);changed['train'][0]['rate']='0.8'
    with pytest.raises(ValueError):s.validate_state(changed,'pinned')
    with pytest.raises(ValueError):s.validate_state(state,'other')
    changed=copy.deepcopy(state);changed['train'].pop()
    with pytest.raises(ValueError):s.validate_state(changed,'pinned')


def test_healthy_uuid_mapping_is_fixed_and_never_gpu4():
    assert set(s.GPUS)=={0,1,2}
    assert len(set(s.GPUS.values()))==3
    assert 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264' not in s.GPUS.values()


def test_average_requires_exact_complete_eight_rates():
    from experiments.osram_mosei_frozen_nested_lr_20261027.summarize import aggregates
    rows=[dict(config='lr1e4',rate=str(i/10),ACC=80+i,WF1=79+i,MSE=1.,
               Flat_ACC=79.,Flat_WF1=78.,Flat_MSE=1.1) for i in range(8)]
    result=aggregates(rows)
    assert result['lr1e4']['mean']['ACC']==83.5
    assert result['lr1e4']['high']['WF1']==85.
    assert 'lr1e3' not in result
    assert aggregates(rows[:-1])=={}
    assert aggregates(rows+[rows[0]])=={}


def test_flat_summary_never_inherits_nested_contributions_or_cost():
    from experiments.osram_mosei_frozen_nested_lr_20261027.summarize import flat_row
    nested=dict(ACC=81.,WF1=80.,MSE=.9,Flat_ACC=80.,Flat_WF1=79.,Flat_MSE=1.,
        added_parameters=159235,full_parameters=13500000,corrections=17,harms=9,
        delta_Flat_ACC=1.,selected_epoch=3,seconds=999.,peak_memory_mib=800.)
    actual=flat_row(nested,11,'baseline.npz')
    assert actual['ACC']==80. and actual['WF1']==79. and actual['MSE']==1.
    assert actual['added_parameters']==actual['corrections']==actual['harms']==actual['delta_Flat_ACC']==0
    assert actual['selected_epoch']==11 and actual['predictions']=='baseline.npz'
    assert actual['seconds'] is None and actual['peak_memory_mib'] is None
    assert actual['full_parameters']==13500000-159235
