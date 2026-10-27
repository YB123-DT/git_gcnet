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
