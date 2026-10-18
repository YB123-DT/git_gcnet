"""Fixed1e-5 LR weight-decay follow-up keeps same-seed controls."""
import importlib
from copy import deepcopy
import json

import pytest


def run():
    try:return importlib.import_module('experiments.osram_meld_fixed_lr_wd_20261018.run')
    except ModuleNotFoundError as e:
        if e.name and 'osram_meld_fixed_lr_wd_20261018' in e.name:
            pytest.fail('Fixed-LR weight-decay follow-up is not implemented')
        raise


def test_exact_eight_jobs_change_only_weight_decay_and_reuse_matching_seed():
    r=run(); specs=r.job_specs()
    assert len(specs)==8 and len({s['id'] for s in specs})==8
    assert {(s['seed'],s['weight_decay']) for s in specs}=={
        (seed,wd) for seed in (67,68) for wd in (0.,1e-5,1e-4,5e-4)}
    for spec in specs:
        cfg=r.make_config(spec['seed'],spec['weight_decay'])
        assert cfg==dict(r.control_config(spec['seed']),weight_decay=spec['weight_decay'])
        assert cfg['learning_rate']==1e-5 and cfg['epochs']==50 and cfg['batch_size']==32
        assert cfg['missing_mask_protocol']=='gcnet-random-mask-v1'


@pytest.mark.parametrize('seed,wd',[(66,0.),(69,1e-5),(True,0.),(67,2e-4),(68,-1e-5),(68,float('nan'))])
def test_unapproved_seed_or_decay_rejected(seed,wd):
    with pytest.raises(ValueError):run().make_config(seed,wd)


def control_files(tmp_path,seed=67):
    r=run();summary=dict(status='complete',config=r.control_config(seed),parameters=13290795,
        identity=dict(numerical_commit=r.screen.original.HISTORICAL,
            dataset_sha256=r.screen.original.DATA_SHA),initial_state_sha256='same-init',
        testbest={'mean':dict(ACC=60.,UA=36.,WF1=57.),'high':dict(ACC=55.,UA=30.,WF1=50.)})
    trace={phase:{str(i):({'sha256':'h','batches':5,'N':100} if phase=='train' else {
        str(rate/10):dict(sha256='h',batches=5,N=100) for rate in range(8)})
        for i in range(1,51)} for phase in ('train','selection')}
    for name,data in [('SUMMARY.json',summary),('MASK_TRACE.json',trace),
        ('INITIAL_STATE.json',dict(sha256='same-init',parameters=13290795))]:
        (tmp_path/name).write_text(json.dumps(data))
    return summary,trace


def test_control_requires_complete_same_seed_original_numeric_and_exact_trace(tmp_path):
    summary,trace=control_files(tmp_path)
    assert run().validate_control(tmp_path,67)==summary
    trace['selection']['50'].pop('0.7')
    (tmp_path/'MASK_TRACE.json').write_text(json.dumps(trace))
    with pytest.raises(ValueError):run().validate_control(tmp_path,67)


def test_other_seed_control_and_bad_initial_hash_are_rejected(tmp_path):
    control_files(tmp_path,67)
    with pytest.raises(ValueError):run().validate_control(tmp_path,68)
    (tmp_path/'INITIAL_STATE.json').write_text(json.dumps(dict(sha256='wrong',parameters=13290795)))
    with pytest.raises(ValueError):run().validate_control(tmp_path,67)


def test_paired_seed_aggregate_is_two_seeds_not_three():
    r=run();rows=[]
    for seed in (67,68):
        rows.append(dict(status='complete',config=r.make_config(seed,0.),
            testbest=dict(mean=dict(ACC=seed,UA=seed,WF1=seed),high=dict(ACC=seed,UA=seed,WF1=seed))))
    result=r.aggregate_pair(rows,0.)
    assert result['seeds']==[67,68] and result['N_seeds']==2
    assert result['mean']['ACC']==67.5
    with pytest.raises(ValueError):r.aggregate_pair(rows[:1],0.)
