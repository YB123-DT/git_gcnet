import importlib
import types
from copy import deepcopy

import pytest


def runner():
    try:
        return importlib.import_module('experiments.osram_meld_wd_seed66_20261018.run')
    except ModuleNotFoundError:
        pytest.fail('Missing finite seed66 WD extension')


def test_four_jobs_change_only_wd_from_completed_o6():
    r = runner()
    original = r.screen.make_config(r.screen.source_config(), 'O6')
    assert len(r.JOBS) == 4 and set(r.JOBS.values()) == {0., 1e-5, 1e-4, 5e-4}
    for job, wd in r.JOBS.items():
        assert r.make_config(job) == dict(original, weight_decay=wd)
        assert r.make_config(job)['seed'] == 66
        assert r.make_config(job)['epochs'] == 50
    with pytest.raises(ValueError):
        r.make_config('S67_WD0')


def rows_for(wd):
    r = runner()
    config = r.screen.make_config(r.screen.source_config(), 'O6')
    return [dict(status='complete',config=dict(config, seed=seed, weight_decay=wd),
        testbest={scope:{metric:float(seed) for metric in ('ACC','UA','WF1')}
                  for scope in ('mean','high')}) for seed in (66,67,68)]


def test_three_seed_mean_requires_exact_complete_matching_triplet():
    r = runner()
    rows = rows_for(1e-4)
    result = r.aggregate_three(rows, 1e-4)
    assert result['N_seeds'] == 3 and result['seeds'] == [66,67,68]
    assert result['mean']['ACC'] == 67 and result['mean_sample_std']['UA'] == 1
    for invalid in (rows[:2], [rows[0],rows[0],rows[2]]):
        with pytest.raises(ValueError):
            r.aggregate_three(invalid, 1e-4)
    changed = deepcopy(rows); changed[0]['status'] = 'running'
    with pytest.raises(ValueError):
        r.aggregate_three(changed, 1e-4)
    changed = deepcopy(rows); changed[1]['config']['missing_mask_protocol'] = 'conversation-mask-v2'
    with pytest.raises(ValueError):
        r.aggregate_three(changed, 1e-4)


def test_gpu2_only_does_not_share_b1b2_gpu0():
    assert runner().GPU_UUIDS == {2:'GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'}


def test_worker_still_validates_o6_after_limiting_jobs(monkeypatch, tmp_path):
    r = runner()
    config = r.make_config('S66_WD0')
    control_config = dict(config, weight_decay=2e-4)
    control = tmp_path / 'control'
    r.write_json(control / 'SUMMARY.json', dict(status='complete',config=control_config,
        parameters=13290795,initial_state_sha256='same-init',identity=dict(
            numerical_commit=r.screen.original.HISTORICAL,dataset_sha256=r.screen.original.DATA_SHA)))
    r.write_json(control / 'INITIAL_STATE.json', dict(sha256='same-init',parameters=13290795))
    item = dict(sha256='same-mask',N=100,batches=2)
    trace = {phase:{str(epoch):(dict(item) if phase=='train' else
        {str(i/10):dict(item) for i in range(8)}) for epoch in range(1,51)}
        for phase in ('train','selection')}
    r.write_json(control / 'MASK_TRACE.json', trace)
    config_path = tmp_path / 'config.json'; r.write_json(config_path,config)
    monkeypatch.setattr(r,'CONTROL',control)
    for name in ('JOB_SPECS','GPU_UUIDS','validate_control','check_snapshot'):
        monkeypatch.setattr(r.screen,name,getattr(r.screen,name))
    monkeypatch.setattr(r.screen,'worker',lambda args:r.screen.validate_control(args.control))
    r.worker(types.SimpleNamespace(job='S66_WD0',config=config_path,driver_commit='fixed-source'))
