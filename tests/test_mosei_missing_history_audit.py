import numpy as np
import pytest


def test_support_is_strict_past_per_modality_and_conversation():
    from experiments.osram_mosei_missing_history_audit_20261025.evaluate import history_support
    av=np.array([[1,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,0],[1,0,0]])
    ids=np.array(['a','a','a','a','b','b'])
    times=np.array([0,1,2,3,0,1])
    seen,counts=history_support(av,ids,times)
    np.testing.assert_array_equal(seen,[[0,0,0],[1,0,0],[1,0,1],[1,1,1],[0,0,0],[0,1,0]])
    np.testing.assert_array_equal(counts[3],[1,1,1])
    changed=av.copy();changed[3]=[0,1,1]
    np.testing.assert_array_equal(history_support(changed,ids,times)[0][:4],seen[:4])


def test_never_target_is_not_no_history_and_cohorts_partition_missing():
    from experiments.osram_mosei_missing_history_audit_20261025.evaluate import make_cohorts
    av=np.array([[1,0,0],[0,0,1],[0,1,0],[1,0,0]])
    cohorts=make_cohorts(av,np.array(['a']*4),np.arange(4))
    np.testing.assert_array_equal(cohorts['T:never'],[1,1,0,0])
    np.testing.assert_array_equal(cohorts['T:never_with_history'],[0,1,0,0])
    np.testing.assert_array_equal(cohorts['T:seen'],[0,0,0,1])
    for m,idx in zip('ATV',range(3)):
        np.testing.assert_array_equal(cohorts[m+':never']|cohorts[m+':seen'],av[:,idx]==0)
        assert not (cohorts[m+':never']&cohorts[m+':seen']).any()


@pytest.mark.parametrize('av,times',[(np.array([[0,0,0]]),[0]),(np.array([[1,.2,0]]),[0]),(np.array([[1,0,0]]),[1])])
def test_invalid_masks_and_missing_history_are_rejected(av,times):
    from experiments.osram_mosei_missing_history_audit_20261025.evaluate import history_support
    with pytest.raises(ValueError):history_support(av,np.array(['a']),np.array(times))


def test_readouts_zero_evidence_not_corrections_and_preserve_local():
    from experiments.osram_mosei_missing_history_audit_20261025.evaluate import readout_cases, select_readout
    local=np.ones((2,1,4));base=np.full((2,1,1024),2.);gap=np.full((2,1,3,1024),3.)
    cases=readout_cases()
    assert len(cases)==13
    for name in cases:
        l,b,g=select_readout(local,base,gap,name)
        assert l is local
        keep_b,keep_g=cases[name]
        np.testing.assert_array_equal(b,base if keep_b else np.zeros_like(base))
        for i in range(3):np.testing.assert_array_equal(g[:,:,i],gap[:,:,i] if i in keep_g else np.zeros_like(gap[:,:,i]))
    np.testing.assert_array_equal(base,2.);np.testing.assert_array_equal(gap,3.)


def test_empty_cohort_is_na_and_neutrals_do_not_affect_polarity_counts():
    from experiments.osram_mosei_missing_history_audit_20261025.evaluate import measure
    y=np.array([1.,-1.,0.]); full=np.array([-1.,-1.,5.]); candidate=np.array([1.,1.,0.])
    got=measure(y,candidate,full,full,np.array([1,1,1],bool))
    assert (got['N_all'],got['N_binary'],got['ACC'])==(3,2,50.)
    assert (got['corrections_vs_full'],got['harms_vs_full'])==(1,1)
    assert measure(y,candidate,full,full,np.zeros(3,bool))['WF1'] is None
