import importlib
from types import SimpleNamespace

import pytest

PACKAGE = 'experiments.osram_frozen_nested_gcnet_20261025'


def test_locked_scope_and_exact_existing_learning_rate():
    run = importlib.import_module(PACKAGE+'.run')
    assert run.validate_scope('0.7') == '0.7'
    assert [run.learning_rate(e) for e in (0, 99, 100, 149)] == [3e-4, 3e-4, 3e-4/10, 3e-4/10]
    for rate in (-1, .8, True, 'unknown'):
        with pytest.raises(ValueError): run.validate_scope(rate)
    with pytest.raises(ValueError): run.validate_scope('.3', seed=67)


def test_only_eight_frozen_original_nested_jobs():
    queue = importlib.import_module(PACKAGE+'.scheduler')
    assert len(queue.jobs_for('smoke')) == 1
    jobs = queue.jobs_for('train')
    assert len(jobs) == len({j['id'] for j in jobs}) == 8
    assert {j['rate'] for j in jobs} == {str(i/10) for i in range(8)}
    assert all(j['seed'] == 66 and j['mode'] == 'nested' and j['stage'] == 'stage2' for j in jobs)
    with pytest.raises(ValueError): queue.jobs_for('other')


def test_launch_binds_uuid_and_runs_only_new_entrypoint(tmp_path, monkeypatch):
    queue = importlib.import_module(PACKAGE+'.scheduler')
    launched = []
    def popen(command, **kwargs):
        launched.append((command, kwargs))
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(queue.subprocess, 'Popen', popen)
    args = SimpleNamespace(root=tmp_path, source_commit='locked', resume=False)
    job = queue.jobs_for('smoke')[0]
    queue.launch(args, job, 'GPU-test', 'smoke')
    command, kwargs = launched[0]
    assert PACKAGE+'.run' in command and '--smoke' in command
    assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-test'
    assert '--mode' not in command and '--rate' in command
    assert kwargs['start_new_session'] is True


def test_incomplete_jobs_cannot_be_reported_as_complete(tmp_path):
    module = importlib.import_module(PACKAGE+'.summarize')
    result = module.summarize(tmp_path, 'locked', [dict(id='pending', status='pending')])
    assert result['status'] == 'incomplete' and not result['aggregate']
    assert result['failures'] and result['expected_jobs'] == 8


def test_preflight_keeps_recovery_metadata_and_cumulative_cost():
    run = importlib.import_module(PACKAGE+'.run')
    previous = dict(baseline_artifact={'sha256':'accepted'}, seconds=14., peak_memory_mib=900., peak_rss_gib=2.)
    summary = run.starting_summary(previous, '.3', 'locked')
    assert summary['baseline_artifact'] == previous['baseline_artifact']
    assert run.costs(previous, 5., 800., 3.) == dict(seconds=19., peak_memory_mib=900., peak_rss_gib=3.)
    assert previous['seconds'] == 14. and summary['status'] == 'preflight'


def test_interrupted_smoke_can_recover_full_state(tmp_path, monkeypatch):
    import torch
    queue = importlib.import_module(PACKAGE+'.scheduler')
    state = dict(block={}, optimizer={}, rng={}, identity=dict(source_commit='locked',
        method='nested_gnn_rooted_evidence', seed=66, evaluation_rate='0.7', smoke=True),
        epoch=1, history=[{'epoch':1}], best={'epoch':1})
    torch.save(state, tmp_path/'last.pt')
    job = dict(id='nested_0.7', status='failed', rate='0.7', output=str(tmp_path))
    queue.reconcile([job], 'smoke', 'locked')
    assert job['status'] == 'pending'
