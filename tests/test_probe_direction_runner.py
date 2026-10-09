"""Stored interventions, not a newly selected deletion experiment."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'experiments/osram_probe_direction_20261009/run.py'


def runner():
    assert PATH.exists(), 'stored-plan replay is not implemented'
    spec=importlib.util.spec_from_file_location('direction_runner',PATH)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source():
    return dict(conversation_id='x',utterance_index=3,family='recent_vs_earlier_T',
        rate=.7,split='test',pattern=5,y=.2,pred_real=.1,pred_delete=.3,pred_control=.4,
        position_delete=2,position_control=0,modality_delete='T',modality_control='T',
        lag_delete=1,lag_control=3,deleted_bits=1,control_deleted_bits=1)


def test_exact_plan_preserves_both_arms_and_positions():
    m=runner()
    r=source()
    result=m.stored_cases([r],.7)
    assert [(x['arm'],x['position'],x['lag']) for x in result]==[('delete',2,1),('control',0,3)]
    assert r==source()
    assert [x['pred_expected'] for x in result]==[.3,.4]


def test_plan_rejects_current_deletion_duplicates_and_rate_mismatch():
    m=runner()
    with pytest.raises(ValueError,match='duplicate'):
        m.stored_cases([source(),source()],.7)
    r=source(); r['position_delete']=3
    with pytest.raises(ValueError,match='past'):
        m.stored_cases([r],.7)
    with pytest.raises(ValueError,match='rate'):
        m.stored_cases([source()],.6)


def test_plan_never_uses_labels_or_scores_to_select():
    m=runner()
    r=source(); r.update(y=-100,pred_real=-20,pred_delete=80,pred_control=30)
    a,b=m.stored_cases([source()],.7),m.stored_cases([r],.7)
    for x,y in zip(a,b):
        assert (x['position'],x['lag'],x['arm'])==(y['position'],y['lag'],y['arm'])


def test_pair_cache_checks_inactive_slots():
    m=runner()
    a=np.array([[1,0,1]])
    r=np.zeros((1,4,512))
    m.check_memory_slots(r,a)
    r[0,2,0]=3
    m.check_memory_slots(r,a)
    r[0,1,0]=2
    with pytest.raises(ValueError,match='inactive'):
        m.check_memory_slots(r,a)
