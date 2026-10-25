"""Four finite configurations, separate stages and explicit recovery."""
import importlib
import json
from types import SimpleNamespace

import pytest

PACKAGE = 'experiments.osram_recent_noarea_serial_20261025'


def test_finite_stage_by_wiring_queue():
    module = importlib.import_module(PACKAGE+'.scheduler')
    smoke, train = module.jobs_for('smoke'), module.jobs_for('train')
    assert len(smoke) == 4
    assert {(r['stage'], r['mode']) for r in smoke} == {
        (stage, mode) for stage in ('stage1', 'stage2') for mode in ('parallel', 'serial')}
    assert len(train) == len({r['id'] for r in train}) == 18
    assert all(r['seed'] == 66 and r['dataset'] == 'CMUMOSI' for r in train)
    assert len([r for r in train if r['stage'] == 'stage1']) == 2
    assert {(r['mode'], r['rate']) for r in train if r['stage'] == 'stage2'} == {
        (m, f'{i/10:.1f}') for m in ('parallel', 'serial') for i in range(8)}


def test_healthy_map_and_measured_capacity():
    module = importlib.import_module(PACKAGE+'.scheduler')
    assert module.parse_gpu_map('0=GPU-a,1=GPU-b') == {0:'GPU-a', 1:'GPU-b'}
    for invalid in ('4=GPU-bad', '0=GPU-a,2=GPU-a', '3=GPU-c'):
        with pytest.raises(ValueError): module.parse_gpu_map(invalid)
    job = dict(stage='stage1', mode='serial')
    snapshot = dict(disk_free_gib=100, ram_available_gib=100,
                    gpus={'GPU-a':dict(free_mib=10000)})
    peaks = {'stage1:serial':dict(gpu_mib=4000, rss_gib=5)}
    assert module.fits(snapshot, 'GPU-a', job, [], peaks, 6)
    assert not module.fits(snapshot, 'GPU-a', job, [job], peaks, 6)


def test_detached_commands_select_the_requested_stage(tmp_path, monkeypatch):
    module = importlib.import_module(PACKAGE+'.scheduler')
    captured = []
    def popen(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    args = SimpleNamespace(root=tmp_path, source_commit='source', resume=False)
    for stage in ('stage1', 'stage2'):
        job = next(r for r in module.jobs_for('smoke') if r['stage'] == stage)
        module.launch(args, job, 'GPU-a', 'smoke')
        cmd, options = captured[-1]
        assert PACKAGE+'.'+stage in cmd and '--smoke' in cmd
        assert ('--rate' in cmd) == (stage == 'stage2')
        assert options['start_new_session'] is True
        assert options['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-a'


def test_incomplete_stage_results_cannot_form_means(tmp_path):
    module = importlib.import_module(PACKAGE+'.summarize')
    result = module.summarize(tmp_path, 'source', [dict(id='pending', status='pending')])
    assert result['status'] == 'incomplete'
    assert not result['aggregate']
    assert json.loads((tmp_path/'RESULT.json').read_text())['expected_jobs'] == 18


def test_resume_resets_unlaunched_blocked_jobs_without_output_path(tmp_path, monkeypatch):
    module = importlib.import_module(PACKAGE+'.scheduler')
    failed = dict(id='failed', stage='stage2', mode='serial', rate='0.7', status='failed',
        output=str(tmp_path), source_commit='same', pid=-1)
    untouched = dict(id='untouched', stage='stage2', mode='parallel', rate='0.1', status='blocked')
    monkeypatch.setattr(module, 'process_matches', lambda job: False)
    monkeypatch.setattr(module, 'recoverable', lambda job: job.get('id') == 'failed')
    module.reconcile([failed, untouched], 'train')
    assert failed['status'] == untouched['status'] == 'pending'
    assert 'output' not in untouched
