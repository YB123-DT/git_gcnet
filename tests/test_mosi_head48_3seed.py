from argparse import Namespace
from dataclasses import asdict, replace
import importlib.util
import inspect
from pathlib import Path

import pytest

from test_mosi_memory_grid_worker import reference_config


def api(name):
    key = 'experiments.osram_mosi_head48_3seed_20261022.'+name
    assert importlib.util.find_spec(key) is not None, 'head48 supplement '+name+' missing'
    return __import__(key,fromlist=['main'])


@pytest.mark.parametrize('seed',[67,68])
def test_only_key_value_width_changes_for_the_same_seed(seed):
    worker = api('worker')
    original = replace(reference_config(),seed=seed)
    candidate = worker.candidate_config(original,'head_48')
    assert {k for k,v in asdict(candidate).items() if asdict(original)[k]!=v}=={'osram_key_dim','osram_value_dim'}
    assert candidate.seed==seed and candidate.osram_key_dim==candidate.osram_value_dim==48
    assert candidate.osram_write_step==.6 and candidate.osram_num_heads==8
    assert worker.reference_for_seed(seed).as_posix().endswith(f'/seed_{seed}/flat')


def test_no_repeat66_or_additional_parameter_search():
    worker = api('worker')
    for seed in (65,66,69):
        with pytest.raises(ValueError): worker.reference_for_seed(seed)
        with pytest.raises(ValueError): worker.candidate_config(replace(reference_config(),seed=seed),'head_48')
    for mode in ('baseline','head_32','step_0p4'):
        with pytest.raises(ValueError): worker.candidate_config(replace(reference_config(),seed=67),mode)


def test_worker_reuses_original_loop_with_explicit_seed_scoped_hooks(monkeypatch):
    worker = api('worker');seen={}
    def captured(argv,**kwargs): seen.update(kwargs);return 'ok'
    monkeypatch.setattr(worker.base,'main',captured)
    assert worker.main(['--unused'])=='ok'
    assert seen['allowed_seeds']==(67,68)
    assert tuple(seen['allowed_modes'])==('head_48',)
    assert seen['config_factory'] is worker.candidate_config
    assert seen['reference_factory'] is worker.reference_for_seed
    assert seen['source_catalog_factory'] is worker.source_catalog


def test_original_worker_defaults_still_only_allow_original_grid_seed66():
    from experiments.osram_mosi_memory_grid_20261022 import worker
    params=inspect.signature(worker.main).parameters
    assert 'allowed_seeds' in params,'original worker lacks explicit supplemental seed hooks'
    assert params['allowed_seeds'].default==(66,)
    assert params['config_factory'].default is worker.candidate_config


def test_queue_has_only_two_fresh_head48_train_jobs_and_original_protocol():
    run = api('run')
    jobs=run.plan_jobs(Path('/test'),'sha',['GPU-valid'])
    assert len(jobs)==2 and {j['seed'] for j in jobs}=={67,68}
    assert all(j['mode']=='head_48' and j['phase']=='train' and j['dataset']=='CMUMOSI' for j in jobs)
    for job in jobs:
        command=job['command']
        assert command[command.index('-m')+1]=='experiments.osram_mosi_head48_3seed_20261022.worker'
        assert job['output']==f"/test/runs/seed_{job['seed']}"
        assert job['backbone_lr']==3e-4 and job['final_backbone_lr']==3e-4/10
        assert job['mask_protocol']=='gcnet-random-mask-v1'
    for devices in ([],['bad'],['GPU-a','GPU-b'],[run.BAD_UUID]):
        with pytest.raises(ValueError): run.plan_jobs(Path('/test'),'sha',devices)
