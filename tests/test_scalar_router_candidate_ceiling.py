import numpy as np
import pytest

from experiments.osram_scalar_router_candidate_ceiling_20261020.analysis import (
    analyze_cohort, validate_predictions,
)


def test_candidate_capacity_router_errors_and_mse_use_same_rows():
    labels=np.array([1.,-1.,1.,-1.,0.])
    candidates=np.array([[-.5,.5,-.2],[-.5,.5,-.2],[.3,.4,.5],[.2,.3,.4],[-.4,0.,.4]])
    weights=np.array([[0.,1.,0.],[0.,1.,0.],[1.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    prediction=(weights*candidates).sum(1)
    row,groups=analyze_cohort(labels,candidates,prediction,np.ones(5,dtype=bool))
    assert row['N_regression']==5 and row['N_polarity']==4
    assert row['unanimous']==2 and row['cross_threshold']==2
    assert row['full_errors']==2 and row['candidate_correctable']==1
    assert row['corrections']==1 and row['harms']==1
    assert row['missed_corrections']==0
    assert row['oracle_ACC']==75. and row['router_ACC']==50.
    assert row['convex_mse_floor'] <= row['discrete_mse_floor'] <= row['full_MSE']
    assert groups['corrections']['N']==1 and groups['harms']['N']==1
    assert groups['corrections']['delta_MSE']<0 and groups['harms']['delta_MSE']>0


def test_unanimous_candidates_cannot_rescue_polarity_but_can_improve_mse():
    labels=np.array([1.,-1.])
    candidates=np.array([[-.8,-.2,-.1],[.9,.1,.2]])
    row,_=analyze_cohort(labels,candidates,candidates[:,1],np.ones(2,dtype=bool))
    assert row['unanimous']==2 and row['candidate_correctable']==0
    assert row['corrections']==row['harms']==0
    assert row['router_MSE']<row['full_MSE']
    assert row['oracle_ACC']==0.


def test_empty_eligible_cohort_is_na_not_zero_performance():
    row,groups=analyze_cohort(np.array([1.]),np.ones((1,3)),np.ones(1),np.zeros(1,dtype=bool))
    assert row['N_regression']==0 and row['N_polarity']==0
    assert row['router_ACC'] is None and row['unanimous_pct'] is None
    assert groups['corrections']['N']==0


def example_arrays():
    return dict(labels=np.array([1.,-1.,0.]),candidates=np.ones((3,3,1)),
        prediction=np.ones((3,1)),weights=np.array([[1.,0.,0.],[.2,.4,.4],[1.,0.,0.]]),
        availability=np.array([[0.,1.,0.],[1.,0.,0.],[1.,1.,1.]]),
        eligible=np.array([False,True,False]),conversation_ids=np.array(['a','a','b']),
        utterance_indices=np.array([0,1,0]),speakers=np.zeros(3))


def test_prediction_validation_checks_strict_past_eligibility_and_convexity():
    a=example_arrays()
    assert validate_predictions(a)['convexity_max_violation']==0.
    a['eligible'][0]=True
    with pytest.raises(ValueError,match='eligibility'):validate_predictions(a)
    a=example_arrays();a['prediction'][1]=2.
    with pytest.raises(ValueError,match='mixture'):validate_predictions(a)
    a=example_arrays();a['candidates'][0,1,0]=2.
    with pytest.raises(ValueError,match='inactive'):validate_predictions(a)
