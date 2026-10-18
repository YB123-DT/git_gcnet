import importlib
import json
from pathlib import Path

import pytest


def module():
    return importlib.import_module('experiments.osram_paired_meld_iemocap6_20261018.spec')


def test_exact_six_seed66_jobs_preserve_accepted_protocol(tmp_path):
    m = module()
    jobs = m.build_jobs(tmp_path)
    assert len(jobs) == 6 and len({j['id'] for j in jobs}) == 6
    assert {(j['dataset'],j['fold']) for j in jobs} == {('MELD',1)} | {('IEMOCAPSix',i) for i in range(1,6)}
    for job in jobs:
        cfg = job['config']
        assert cfg['seed'] == 66 and cfg['paired_history_views']
        assert cfg['history_task_view2_weight'] == .5 and cfg['history_contrast_weight'] == 0
        assert cfg['history_drop_prob'] == .2
        assert cfg['missing_mask_protocol'] == 'gcnet-random-mask-v1'
        assert cfg['osram_readout_fusion'] == 'flat' and cfg['osram_output_dim'] == 1600
        assert cfg['osram_meaningful_block'] == 'none'
        assert cfg['learning_rate'] == (1e-5 if job['dataset']=='MELD' else .001)
        assert cfg['epochs'] == (50 if job['dataset']=='MELD' else 100)
        assert cfg['weight_decay'] == 1e-5 and cfg['batch_size'] == 32
        assert Path(job['output']).is_relative_to(tmp_path)
        assert Path(job['baseline_output']).is_absolute()


def test_configuration_change_is_only_the_approved_paired_flags():
    m = module()
    base = dict(dataset='MELD',seed=66,fold=1,paired_history_views=False,
                history_contrast_weight=.1,history_drop_prob=.2,history_task_view2_weight=.5)
    original = dict(base)
    result = m.paired_config(base)
    assert base == original
    assert {k for k in result if result[k]!=base[k]} == {'paired_history_views','history_contrast_weight'}


def test_complete_fivefold_mean_is_unweighted_and_no_partial_mean(tmp_path):
    m = module()
    jobs = m.build_jobs(tmp_path)
    metrics=lambda n: dict(ACC=n,UA=n+1,WF1=n+2,CE=n/100)
    summaries={j['id']:dict(status='complete',dataset=j['dataset'],seed=66,fold=j['fold'],
        testbest=dict(rates={str(i/10):metrics(j['fold']+i) for i in range(8)}),
        persistent=dict(patterns={p:dict(metrics=metrics(j['fold'])) for p in m.PATTERNS})) for j in jobs}
    result=m.aggregate_jobs(jobs,summaries)
    assert result['status']=='complete'
    assert result['datasets']['IEMOCAPSix']['random']['rates']['0.0']['ACC']==3
    assert result['datasets']['IEMOCAPSix']['persistent']['macro_six']['ACC']==3
    summaries.pop(jobs[-1]['id'])
    partial=m.aggregate_jobs(jobs,summaries)
    assert partial['status']!='complete'
    assert 'IEMOCAPSix' not in partial['datasets']
