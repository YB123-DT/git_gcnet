import importlib
from copy import deepcopy
import pytest


def runner():
    try:
        return importlib.import_module('experiments.osram_meld_small_lr_3seed_20261018.run')
    except ModuleNotFoundError:
        pytest.fail('The finite two-seed extension is not implemented')


def test_only_seed_changes_from_winning_config():
    r=runner();original=r.screen.make_config(r.screen.source_config(),'O6')
    before=deepcopy(original)
    for seed in (67,68):
        assert r.make_config(seed)==dict(original,seed=seed)
    assert original==before
    assert original['learning_rate']==1e-5 and original['weight_decay']==2e-4
    assert original['missing_mask_protocol']=='gcnet-random-mask-v1'


@pytest.mark.parametrize('seed',[65,66,69,True,'67'])
def test_no_retraining_completed_seed_or_expanding_seeds(seed):
    with pytest.raises(ValueError):runner().make_config(seed)


def test_three_seed_aggregate_requires_complete_matching_config():
    r=runner();base=r.screen.make_config(r.screen.source_config(),'O6')
    summaries={str(s):dict(status='complete',config=dict(base,seed=s),testbest=dict(
        mean=dict(ACC=60+s-66,UA=35+s-66,WF1=57+s-66),
        high=dict(ACC=55+s-66,UA=30+s-66,WF1=50+s-66))) for s in (66,67,68)}
    result=r.aggregate_seeds(summaries)
    assert result['mean']['ACC']==61
    assert result['mean']['UA']==36
    assert result['std']['ACC']==1
    wrong=deepcopy(summaries);wrong['68']['config']['learning_rate']=.001
    with pytest.raises(ValueError):r.aggregate_seeds(wrong)
    wrong=deepcopy(summaries);wrong['67']['status']='running'
    with pytest.raises(ValueError):r.aggregate_seeds(wrong)
    with pytest.raises(ValueError):r.aggregate_seeds({'66':summaries['66']})
