import importlib
from copy import deepcopy

import pytest


def runner():
    try:
        return importlib.import_module('experiments.osram_meld_related_history_diagnostic_20261018.run')
    except ModuleNotFoundError:
        pytest.fail('Frozen B1/B2 diagnostic runner missing')


def test_limited_inference_plan_and_fixed_reference():
    r = runner()
    assert r.KINDS == ('B1','B2')
    assert r.RATES == tuple(i/10 for i in range(8))
    assert r.COMMON_RATE == .1 and r.MATCHED_EPOCH == 13
    assert r.MATCHED_SOURCE_RATE == .5
    assert 4 not in r.GPU_UUIDS


def test_epoch13_requires_full_same_trajectory_identity():
    r = runner()
    branch = dict(kind='B2',input_dim=2564)
    config = dict(seed=66,epochs=50,missing_mask_protocol='gcnet-random-mask-v1')
    identity = dict(branch=branch,config_sha256='fixed',numerical_commit='same')
    summary = dict(status='complete',identity=identity,config=config,branch=branch)
    state = dict(identity=identity,next_epoch=50,history=[dict(epoch=i) for i in range(1,51)])
    checkpoint = dict(epoch=13,config=config,related_history=branch)
    r.assert_trajectory(summary,state,checkpoint,13)
    bad = deepcopy(state); bad['identity']['config_sha256']='other'
    with pytest.raises(ValueError):r.assert_trajectory(summary,bad,checkpoint,13)
    bad = deepcopy(checkpoint); bad['epoch']=10
    with pytest.raises(ValueError):r.assert_trajectory(summary,state,bad,13)
    bad = deepcopy(checkpoint); bad['config']['seed']=67
    with pytest.raises(ValueError):r.assert_trajectory(summary,state,bad,13)
