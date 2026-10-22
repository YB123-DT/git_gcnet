from dataclasses import asdict,replace
import importlib.util

import pytest

from test_mosi_memory_grid_worker import reference_config


def api():
    name='experiments.osram_mosi_head48_step04_20261022.worker'
    assert importlib.util.find_spec(name) is not None,'single joint head48/step0.4 entry missing'
    return __import__(name,fromlist=['main'])


def test_only_the_requested_two_factors_change_for_seed66():
    worker=api();old=reference_config();cfg=worker.candidate_config(old,worker.MODE)
    assert {k for k,v in asdict(cfg).items() if asdict(old)[k]!=v}=={
        'osram_key_dim','osram_value_dim','osram_write_step'}
    assert cfg.osram_key_dim==cfg.osram_value_dim==48 and cfg.osram_write_step==.4
    assert cfg.seed==66 and cfg.epochs==150 and cfg.learning_rate==3e-4


def test_single_job_scope_rejects_extra_seeds_or_modes():
    worker=api()
    for seed in (65,67,68):
        with pytest.raises(ValueError):worker.candidate_config(replace(reference_config(),seed=seed),worker.MODE)
    with pytest.raises(ValueError):worker.candidate_config(reference_config(),'head_48')


def test_original_grid_loop_reused_and_requested_scope_is_explicit(monkeypatch):
    worker=api();seen={}
    def capture(argv,**kwargs):seen.update(kwargs)
    monkeypatch.setattr(worker.base,'main',capture)
    worker.main(['--unused'])
    assert seen['allowed_seeds']==(66,)
    assert seen['allowed_modes']==(worker.MODE,)
    assert seen['config_factory'] is worker.candidate_config
    assert seen['source_catalog_factory'] is worker.source_catalog


def test_report_is_completed_only_and_rejects_changed_evaluation_masks(tmp_path,monkeypatch):
    import json
    name='experiments.osram_mosi_head48_step04_20261022.report'
    assert importlib.util.find_spec(name) is not None,'completed-only four-control report missing'
    report=__import__(name,fromlist=['summarize'])
    from test_mosi_memory_grid_queue import summary
    from experiments.osram_mosi_memory_grid_20261022.config import candidate_config
    controls={mode:tmp_path/mode for mode in ('baseline','step_0p4','head_48')}
    monkeypatch.setattr(report,'CONTROLS',controls)
    for mode,path in controls.items():
        path.mkdir();s=summary();s['identity']['config']=asdict(candidate_config(reference_config(),mode))
        s['identity']['source_hashes']={p:p for p in report.CORE}
        s['identity']['initial_model_sha256']='head48init' if mode=='head_48' else mode
        (path/'SUMMARY.json').write_text(json.dumps(s))
    out=tmp_path/'new';out.mkdir();s=summary(status='running',epoch=5)
    s['identity']['source_hashes']={p:p for p in report.CORE}
    s['parameters']['total']=11245857
    s['identity']['config']=asdict(api().candidate_config(reference_config(),api().MODE))
    s['identity']['initial_model_sha256']='head48init'
    (out/'SUMMARY.json').write_text(json.dumps(s))
    assert report.summarize(out)['candidate_complete'] is False
    s['status']='complete';s['epoch']=150
    (out/'SUMMARY.json').write_text(json.dumps(s))
    r=report.summarize(out)
    assert r['candidate_complete'] and len(r['aggregate'])==4 and len(r['per_rate'])==32
    s['identity']['mask_hashes']={'test':'different'}
    (out/'SUMMARY.json').write_text(json.dumps(s))
    with pytest.raises(ValueError):report.summarize(out)
