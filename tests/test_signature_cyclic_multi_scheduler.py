"""Finite cyclic queue admission and complete-only summaries."""
import json
from pathlib import Path

import pytest


def test_finite_scope_and_no_duplicate_jobs():
    from experiments.osram_signature_cyclic_multi_20261025.scheduler import jobs_for
    jobs = jobs_for('train')
    assert len(jobs) == len({j['id'] for j in jobs}) == 48
    assert {(j['dataset'], j['seed']) for j in jobs} == {
        ('CMUMOSI', 66), ('CMUMOSI', 67), ('CMUMOSEI', 66),
        ('CMUMOSEI', 67), ('CMUMOSEI', 68), ('MELD', 66)}
    assert len(jobs_for('smoke')) == 3


def test_bad_gpu_and_duplicate_uuid_rejected():
    from experiments.osram_signature_cyclic_multi_20261025.scheduler import parse_gpu_map
    assert parse_gpu_map('0=GPU-a,2=GPU-b') == {0: 'GPU-a', 2: 'GPU-b'}
    for value in ('4=GPU-a', '0=GPU-a,1=GPU-a', '0=GPU-a,0=GPU-b', '0=0'):
        with pytest.raises(ValueError):
            parse_gpu_map(value)


def test_admission_accounts_for_not_yet_resident_peers():
    from experiments.osram_signature_cyclic_multi_20261025.scheduler import fits
    snapshot = {'ram_available_gib': 80, 'disk_free_gib': 100,
                'gpus': {'GPU-a': {'free_mib': 14000}}}
    job = {'dataset': 'CMUMOSI'}
    peers = [{'dataset': 'CMUMOSI'}] * 3
    peaks = {'CMUMOSI': {'gpu_mib': 2000, 'rss_gib': 2}}
    assert fits(snapshot, 'GPU-a', job, peers, peaks, 6)
    assert not fits(snapshot, 'GPU-a', job, peers * 2, peaks, 6)
    snapshot['gpus']['GPU-a']['free_mib'] = 11000
    assert not fits(snapshot, 'GPU-a', job, peers, peaks, 6)
    snapshot['gpus']['GPU-a']['free_mib'] = 14000
    snapshot['ram_available_gib'] = 20
    assert not fits(snapshot, 'GPU-a', job, peers, peaks, 6)


def test_incomplete_group_not_aggregated(tmp_path):
    from experiments.osram_signature_cyclic_multi_20261025.summarize import summarize
    result = summarize(tmp_path, 'abc', [{'dataset': 'MELD', 'seed': 66,
        'rate': '0.0', 'status': 'running', 'output': str(tmp_path/'run')}])
    assert result['aggregate'] == []
    assert result['status'] == 'incomplete'


def test_surviving_worker_is_not_duplicated(monkeypatch):
    from experiments.osram_signature_cyclic_multi_20261025 import scheduler
    job = {'id': 'CMUMOSI_66_0.0', 'status': 'running', 'pid': 123,
           'output': '/tmp/unrelated-placeholder'}
    monkeypatch.setattr(scheduler, 'process_matches', lambda job: True)
    scheduler.reconcile([job], 'train')
    assert job['status'] == 'running' and job['pid'] == 123


def test_partial_run_requires_complete_state_to_resume(tmp_path, monkeypatch):
    from experiments.osram_signature_cyclic_multi_20261025 import scheduler
    job = {'status': 'running', 'pid': 123, 'output': str(tmp_path)}
    monkeypatch.setattr(scheduler, 'process_matches', lambda job: False)
    scheduler.reconcile([job], 'train')
    assert job['status'] == 'failed'
    (tmp_path/'last.pt').write_bytes(b'full-state')
    scheduler.reconcile([job], 'train')
    assert job['status'] == 'pending'


def test_summary_rejects_wrong_selected_metric_and_preserves_same_epoch(tmp_path):
    from experiments.osram_signature_cyclic_multi_20261025.summarize import selected_row
    output = tmp_path/'run'
    output.mkdir()
    checkpoint = tmp_path/'parent.pt'
    checkpoint.write_bytes(b'parent')
    import hashlib
    summary = {'status': 'complete', 'epoch': 50, 'parent_id': hashlib.sha256(b'parent').hexdigest(),
        'checks': {key: True for key in ('parent_unchanged', 'parent_grad_none', 'checkpoint_replay_exact',
                   'cyclic_train_rates_verified', 'only_requested_test_rate')},
        'baseline': {'ACC': 60., 'UA': 35., 'WF1': 57., 'CE': 1.3},
        'best': {'epoch': 4, 'checkpoint': 'best.pt', 'predictions': 'prediction.npz',
            'checkpoint_sha256': hashlib.sha256(b'best').hexdigest(),
            'prediction_sha256': hashlib.sha256(b'pred').hexdigest(),
            'metrics': {'ACC': 62., 'UA': 36., 'WF1': 59., 'CE': 1.2}},
        'identity': {'dataset': 'MELD', 'seed': 66, 'evaluation_rate': '0.0',
            'cyclic_training': True, 'test_mask_sha256': 'm',
            'mask_protocol': 'gcnet-random-mask-v1', 'initial_block_sha256': 'init',
            'parent': {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(b'parent').hexdigest()}}}
    (output/'best.pt').write_bytes(b'best')
    (output/'prediction.npz').write_bytes(b'pred')
    history = [{'epoch': n, 'optimizer_steps': 2, 'training_rates': [((n-1+i)%8)/10 for i in range(2)],
                'mask_trace_sha256': str(n), 'gradients': {'nonzero': 12},
                'test': {'ACC': 62. if n == 4 else 61.,
                'UA': 36. if n == 4 else 40., 'WF1': 59., 'CE': 1.2}}
               for n in range(1, 51)]
    (output/'SUMMARY.json').write_text(json.dumps(summary))
    (output/'history.json').write_text(json.dumps(history))
    row = selected_row({'dataset': 'MELD', 'seed': 66, 'rate': '0.0', 'output': str(output)})
    assert row['UA'] == 36. and row['selected_epoch'] == 4
    summary['best']['epoch'] = 3
    (output/'SUMMARY.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        selected_row({'dataset': 'MELD', 'seed': 66, 'rate': '0.0', 'output': str(output)})
