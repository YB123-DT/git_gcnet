"""Bounded B6 structural HPO, fixed trial16 training parameters."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

PACKAGE = 'experiments.osram_meld_b6_structure_hpo_20261015.run'
ROOT = Path(__file__).resolve().parents[1]


def module():
    return importlib.import_module(PACKAGE)


def test_only_two_module_knobs_change_trial16():
    source = json.loads((ROOT / 'experiments/osram_meld_block2_topacc_3seed_20261015/MANIFEST.json').read_text())
    trial16 = next(c['summary']['config'] for c in source['configurations'] if c['id']=='B6_trial16')
    config = module().make_config(2, 24)
    assert {k for k in trial16 if trial16[k] != config[k]} == set()
    assert config['osram_rpca_rank']==2 and config['osram_rpca_steps']==24
    assert config['seed']==66 and config['epochs']==50
    for rank, steps in [(0,14),(4,14),(1,3),(1,29),(1,13),(True,14),(1,14.)]:
        with pytest.raises(ValueError):
            module().make_config(rank, steps)


def test_five_distinct_warmstarts_include_exact_old_structure():
    configs=module().INITIAL_CONFIGS
    assert len(configs)==5
    assert len({tuple(sorted(c.items())) for c in configs})==5
    assert configs[0]=={'rank':1,'steps':14}
    for c in configs:
        module().make_config(**c)


def test_global_gpu_limit_counts_external_processes():
    assert module().available_slots(4,[1,2,3],[4,5])==0
    assert module().available_slots(4,[1,2],[2,3])==1
    assert module().available_slots(4,[],[])==4


def test_scheduler_can_return_after_first_completion_without_relaunch(tmp_path):
    from experiments.osram_meld_block2_auto_20261014.scheduler import Scheduler
    scheduler=Scheduler(tmp_path/'queue',gpus=(0,),per_gpu=2,
                        smoke_costs={'B6':{'peak_memory_mib':1,'disk_bytes':0}})
    scheduler.poll_seconds=.01
    scheduler.disk_margin_bytes=0
    scheduler.gpu_snapshot=lambda:{0:{'uuid':'GPU-test','free_mib':32000}}
    def command(job, config, uuid):
        script=('import pathlib,json,time;time.sleep('+str(.05 if job['id']=='fast' else .8)+');'
                'p=pathlib.Path('+repr(job['output'])+');p.mkdir(parents=True,exist_ok=True);'
                '(p/"SUMMARY.json").write_text(json.dumps({"status":"complete"}))')
        return [sys.executable,'-c',script]
    scheduler.command=command
    jobs=[dict(id=n,method='B6',config={},output=str(tmp_path/n)) for n in ('fast','slow')]
    try:
        rows=scheduler.run_jobs(jobs,return_on_first_completion=True)
        assert rows[0]['status']=='complete' and rows[1]['status']=='running'
        assert scheduler.state['status']=='running'
        done=scheduler.run_jobs(jobs)
        assert all(r['status']=='complete' and r['attempts']==1 for r in done)
    finally:
        for child in scheduler.children.values():
            child.wait(timeout=3)


def test_completed_export_does_not_mix_testbest_with_paired():
    metrics={'ACC':60.,'UA':35.,'WF1':56.,'CE':1.2}
    full={'mean':metrics,'high':metrics,'rates':{str(i/10):metrics for i in range(8)}}
    summary={'status':'complete','testbest':full,
             'paired':{'on':full,'off':full,'changes':{},'mask_digest':'same'},
             'parameters':123,'block_parameters':29,'training_seconds':5.,'peakmemorybytes':1}
    trial={'number':0,'state':'COMPLETE','params':{'rank':1,'steps':14},'value':0.,
           'user_attrs':{'summary':summary}}
    report=module().build_report([trial])
    assert report['best_testbest_ACC']==0 and report['best_paired_objective']==0
    assert report['trials'][0]['testbest']['mean']['UA']==35.
    assert report['trials'][0]['paired']['on']['mean']['UA']==35.


def test_real_optuna_unique_async_search_and_idempotent_resume(tmp_path):
    optuna=pytest.importorskip('optuna')
    m=module()
    class CPUResults:
        def __init__(self):
            self.jobs={};self.batches=[]
        def run_jobs(self,jobs,return_on_first_completion=False):
            self.batches.append(len(jobs))
            for job in jobs:
                self.jobs[job['id']]=job
            rows=[]
            for job in jobs:
                r=job['config']['osram_rpca_rank'];s=job['config']['osram_rpca_steps']
                metrics={'ACC':60.,'UA':35.,'WF1':56.,'CE':1.2}
                full={'mean':metrics,'high':metrics,'rates':{str(i/10):metrics for i in range(8)}}
                paired_metrics=dict(metrics,ACC=60.+r/10,UA=35.+s/10)
                paired_full={'mean':paired_metrics,'high':paired_metrics,
                             'rates':{str(i/10):paired_metrics for i in range(8)}}
                rows.append(dict(job,status='complete',summary={'status':'complete',
                    'paired':{'on':paired_full,'mask_digest':'fixed'},
                    'testbest':full,'parameters':123,
                    'block_parameters':2*s+1,'training_seconds':1.,'peakmemorybytes':1}))
            return rows
    results=CPUResults()
    flat={'on':{'mean':{'ACC':60.,'UA':35.}},'mask_digest':'fixed'}
    # Real SQLite crash window: incomplete proposal must be filled, not duplicated.
    study=optuna.create_study(study_name='meld_B6_trial16_structure_seed66',
        storage='sqlite:///'+str(tmp_path/'STRUCTURE.sqlite3'),direction='maximize',
        sampler=m.unique_tpe_sampler())
    for cfg in m.INITIAL_CONFIGS:
        study.enqueue_trial(cfg)
    study.ask().suggest_categorical('rank',[1,2,3])
    report=m.run_search(tmp_path,flat,results,budget=20,max_pending=12)
    assert report['status']=='complete' and len(report['trials'])==20
    pairs=[(j['config']['osram_rpca_rank'],j['config']['osram_rpca_steps']) for j in results.jobs.values()]
    assert len(pairs)==len(set(pairs))==20
    assert results.batches[0]==5 and max(results.batches)<=12
    count=len(results.batches)
    m.run_search(tmp_path,flat,results,budget=20,max_pending=12)
    assert len(results.batches)==count


def test_provenance_gpu_list_survives_json_roundtrip():
    assert module().gpu_provenance((0,2,3))==json.loads(json.dumps(module().gpu_provenance((0,2,3))))
