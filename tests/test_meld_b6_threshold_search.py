"""Threshold/init and B6-only LR HPO at fixed best-ACC rank2/14."""
import importlib
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    return importlib.import_module('experiments.osram_meld_b6_threshold_hpo_20261015.run')


def test_make_config_only_changes_two_authorized_settings():
    m=module()
    previous=json.loads((ROOT/'experiments/osram_meld_b6_structure_hpo_20261015/SUMMARY.json').read_text())['trials'][3]['config']
    c=m.make_config(.03,10.)
    assert all(c[k]==v for k,v in previous.items())
    assert c['osram_rpca_threshold_init']==.03 and c['osram_rpca_lr_multiplier']==10.
    assert (c['osram_rpca_rank'],c['osram_rpca_steps'])==(2,14)
    for a,b in ((0.,1.),(.001,1.),(2.,1.),(.1,0.),(.1,31.),(True,1.),(.1,float('nan'))):
        with pytest.raises(ValueError):m.make_config(a,b)


def test_initial_five_and_explicit_six_process_admission(tmp_path):
    m=module()
    assert len(m.INITIAL_CONFIGS)==5 and m.INITIAL_CONFIGS[0]=={'threshold_init':.1,'b6_lr_multiplier':1.}
    assert len({tuple(sorted(v.items())) for v in m.INITIAL_CONFIGS})==5
    for v in m.INITIAL_CONFIGS:m.make_config(**v)
    from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler
    with pytest.raises(ValueError):Scheduler(tmp_path/'old',gpus=(0,),per_gpu=6)
    s=m.ThresholdScheduler(tmp_path/'new',gpus=(0,),per_gpu=6)
    assert s.capacity=={0:6} and s.global_per_gpu_limit==6
    with pytest.raises(ValueError):m.ThresholdScheduler(tmp_path/'bad',gpus=(4,),per_gpu=6)


def test_float_search_callbacks_use_real_optuna_and_resume(tmp_path):
    optuna=pytest.importorskip('optuna')
    m=module()
    from experiments.osram_meld_b6_structure_hpo_20261015.run import run_search
    class CPUResults:
        batches=[]
        configs=[]
        def run_jobs(self,jobs,return_on_first_completion=False):
            self.batches.append(len(jobs));rows=[]
            for job in jobs:
                self.configs.append(job['config'])
                metrics=dict(ACC=60.,UA=35.,WF1=56.,CE=1.2)
                full=dict(mean=metrics,high=metrics,rates={str(i/10):metrics for i in range(8)})
                summary=dict(status='complete',testbest=full,paired=dict(on=full,off=full,changes={},mask_digest='fixed'),
                             parameters=123,block_parameters=29,training_seconds=1.,peakmemorybytes=1)
                rows.append(dict(job,status='complete',summary=summary))
            return rows
    fake=CPUResults();flat=dict(on=dict(mean=dict(ACC=60.,UA=35.)),mask_digest='fixed')
    kwargs=dict(config_factory=m.make_config,initial_configs=m.INITIAL_CONFIGS,
                search_space=m.SEARCH_SPACE,study_name='threshold_test',sampler=m.make_sampler(),
                metadata=m.REPORT_METADATA,budget=20,max_pending=18)
    # Crash after submitting all5starts must not bypass their feedback barrier.
    study=optuna.create_study(study_name='threshold_test',
        storage='sqlite:///'+str(tmp_path/'STRUCTURE.sqlite3'),direction='maximize',sampler=m.make_sampler())
    for c in m.INITIAL_CONFIGS:study.enqueue_trial(c)
    for _ in range(5):
        t=study.ask()
        for name,spec in m.SEARCH_SPACE.items():t.suggest_float(name,**spec)
    report=run_search(tmp_path,flat,fake,**kwargs)
    assert report['status']=='complete' and len(report['trials'])==20
    assert fake.batches[0]==5 and max(fake.batches)<=18
    assert all(c['osram_rpca_rank']==2 and c['osram_rpca_steps']==14 for c in fake.configs)
    assert all(set(t['params'])=={'threshold_init','b6_lr_multiplier'} for t in report['trials'])
    assert len({tuple(sorted(t['params'].items())) for t in report['trials']})==20
    csv=(tmp_path/'RESULT.csv').read_text().splitlines()[0]
    assert 'threshold_init' in csv and 'b6_lr_multiplier' in csv
    count=len(fake.batches)
    run_search(tmp_path,flat,fake,**kwargs)
    assert len(fake.batches)==count


def test_cli_uses_only_physical_gpu2():
    m=module()
    assert m.parse_args([]).gpus==[2]
    for gpus in ('0','0,2,3','4'):
        with pytest.raises(SystemExit):m.parse_args(['--gpus',gpus])
