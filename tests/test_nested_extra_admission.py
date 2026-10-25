import importlib

import pytest


def test_only_unstarted_mosei_rates_are_extra_admitted():
    module = importlib.import_module('experiments.osram_frozen_nested_cross_20261025.admit_mosei')
    jobs = [dict(dataset='CMUMOSEI',seed=66,fold=1,rate=str(r/10),status='pending' if r>=5 else 'running')
        for r in range(8)]
    jobs.append(dict(dataset='MELD',seed=66,fold=1,rate='0.7',status='pending'))
    assert [j['rate'] for j in module.extra_targets(jobs)] == ['0.5','0.6','0.7']
    jobs[5]['status'] = 'running'
    assert [j['rate'] for j in module.extra_targets(jobs)] == ['0.6','0.7']


def test_controller_guard_rejects_another_task_or_pid_command():
    module = importlib.import_module('experiments.osram_frozen_nested_cross_20261025.admit_mosei')
    state = dict(status='train',source_commit='source',expected_train_jobs=56)
    cmd = ['python','-u','-m',module.CONTROLLER,'--root','/allowed','--source-commit','source']
    module.validate_controller(state,cmd,'/allowed')
    with pytest.raises(ValueError):module.validate_controller(state,cmd,'/other')
    with pytest.raises(ValueError):module.validate_controller(state,['python','other.py'],'/allowed')


def test_lock_timeout_cannot_overwrite_queue_or_start_replacement(tmp_path,monkeypatch):
    module = importlib.import_module('experiments.osram_frozen_nested_cross_20261025.admit_mosei')
    (tmp_path/'QUEUE.json').write_text('original')
    calls=[]
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:calls.append(a))
    with pytest.raises(RuntimeError,match='lock'):
        module.resume_controller(tmp_path,{},[],{}, {},None,False)
    assert not calls and (tmp_path/'QUEUE.json').read_text() == 'original'
