"""Bounded queue and same-parent NoArea/Signature comparison contracts."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE = 'experiments.osram_signature_noarea_iemocap68_20261025'


def test_queue_contains_only_seed68_fivefold_eight_rates():
    scheduler = importlib.import_module(PACKAGE+'.scheduler')
    jobs = scheduler.jobs_for('train')
    assert len(jobs) == len({j['id'] for j in jobs}) == 40
    assert {(j['seed'], j['fold'], j['rate']) for j in jobs} == {
        (68, f, f'{r/10:.1f}') for f in range(1, 6) for r in range(8)}
    assert [j['fold'] for j in jobs[:5]] == [5, 1, 2, 3, 4]
    assert [(j['seed'], j['fold'], j['rate']) for j in scheduler.jobs_for('smoke')] == [(68, 5, '0.7')]


def test_gpu_map_and_capacity_are_bounded():
    scheduler = importlib.import_module(PACKAGE+'.scheduler')
    for value in ('3=GPU-ok', '4=GPU-bad', '0=GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'):
        with pytest.raises(ValueError):
            scheduler.parse_gpu_map(value)
    assert scheduler.parse_gpu_map('0=GPU-a,1=GPU-b,2=GPU-c') == {0:'GPU-a', 1:'GPU-b', 2:'GPU-c'}
    snapshot = dict(disk_free_gib=100, ram_available_gib=100, gpus={'GPU-a':dict(free_mib=12000)})
    job = dict(dataset='IEMOCAPSix')
    peaks = {'IEMOCAPSix':dict(gpu_mib=4000, rss_gib=5)}
    assert scheduler.fits(snapshot, 'GPU-a', job, [], peaks, 6)
    assert not scheduler.fits(snapshot, 'GPU-a', job, [job, job], peaks, 6)
    assert not scheduler.fits(snapshot, 'GPU-a', job, [job], peaks, 1)


def test_resume_adopts_exact_live_worker_without_duplicate(tmp_path, monkeypatch):
    scheduler = importlib.import_module(PACKAGE+'.scheduler')
    job = dict(id='live', status='running', output=str(tmp_path), pid=42)
    monkeypatch.setattr(scheduler, 'process_matches', lambda row: True)
    scheduler.reconcile([job], 'train')
    assert job['status'] == 'running'


def test_foreign_live_pid_is_never_adopted(tmp_path):
    import os
    scheduler = importlib.import_module(PACKAGE+'.scheduler')
    assert not scheduler.process_matches(dict(pid=os.getpid(), command=['foreign-worker'],
        output=str(tmp_path), source_commit='source', worker_cwd=str(tmp_path)))


def test_recovery_requires_full_state_and_current_ablation(tmp_path, monkeypatch):
    import torch
    scheduler = importlib.import_module(PACKAGE+'.scheduler')
    monkeypatch.setattr(scheduler, 'process_matches', lambda row: False)
    job = dict(id='interrupted', status='running', output=str(tmp_path), pid=42,
               source_commit='source', fold=5, rate='0.7')
    identity = dict(source_commit='source', ablation='degree2_no_area', seed=68,
                    fold=5, evaluation_rate='0.7')
    torch.save(dict(block={}, identity=identity), tmp_path/'last.pt')
    scheduler.reconcile([job], 'train')
    assert job['status'] == 'failed'
    torch.save(dict(block={}, identity=identity, optimizer={}, rng={}, epoch=1,
                    history=[{}], best={}), tmp_path/'last.pt')
    job['status'] = 'running'
    scheduler.reconcile([job], 'train')
    assert job['status'] == 'pending'


def test_launch_detaches_worker_session_and_records_source(tmp_path, monkeypatch):
    scheduler = importlib.import_module(PACKAGE+'.scheduler')
    captured = {}
    def popen(command, **options):
        captured.update(command=command, **options)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(scheduler.subprocess, 'Popen', popen)
    args = SimpleNamespace(root=tmp_path, resume=False, source_commit='immutable-source')
    job = scheduler.jobs_for('train')[0]
    scheduler.launch(args, job, 'GPU-a', 'train')
    assert captured['start_new_session'] is True
    assert captured['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-a'
    assert job['source_commit'] == 'immutable-source'
    assert scheduler.MODULE in captured['command']


@pytest.fixture
def paired_job(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    original = importlib.import_module('experiments.osram_signature_iemocap_cyclic_20261025.summarize')
    summary = importlib.import_module(PACKAGE+'.summarize')
    normal_root = tmp_path/'original'
    monkeypatch.setattr(summary, 'NORMAL_ROOT', normal_root)
    job = dict(id='IEMOCAPSix_68_fold5_0.7', dataset='IEMOCAPSix', seed=68, fold=5, rate='0.7', status='complete')
    labels, prediction = np.arange(6), np.eye(6)
    availability = np.ones((6, 3), dtype=np.bool_)
    metrics, _ = original.prediction_metrics(labels, prediction)
    parent = tmp_path/'parent.pt'
    parent.write_bytes(b'parent')
    paths = []
    for output, source, ablation in ((tmp_path/'new', 'new-source', 'degree2_no_area'),
                                     (normal_root/job['id'], summary.NORMAL_SOURCE, None)):
        output.mkdir(parents=True)
        (output/'best.pt').write_bytes(b'branch')
        np.savez(output/'prediction.npz', labels=labels, prediction=prediction, base_prediction=prediction, availability=availability)
        identity = dict(stage2_epochs=100, dataset='IEMOCAPSix', seed=68, fold=5,
            evaluation_rate='0.7', source_commit=source, cyclic_training=True,
            mask_protocol='gcnet-random-mask-v1', initial_block_sha256='same-initial',
            parent=dict(checkpoint=str(parent), checkpoint_sha256=original.digest(parent)),
            test_mask_sha256=hashlib.sha256(availability.tobytes()).hexdigest())
        if ablation:
            identity['ablation'] = ablation
        best = dict(epoch=1, metrics=metrics, checkpoint='best.pt', checkpoint_sha256=original.digest(output/'best.pt'),
                    predictions='prediction.npz', prediction_sha256=original.digest(output/'prediction.npz'))
        record = dict(status='complete', epoch=100, identity=identity, parent_id=identity['parent']['checkpoint_sha256'],
            best=best, baseline=metrics, total_parameters=281856,
            checks={k:True for k in ('parent_unchanged', 'parent_grad_none', 'checkpoint_replay_exact',
                                    'cyclic_train_rates_verified', 'only_requested_test_rate')})
        history = [dict(epoch=i+1, training_rates=[(i%8)/10], optimizer_steps=1,
            mask_trace_sha256=f'trace{i}', gradients=dict(nonzero=1), test=metrics) for i in range(100)]
        (output/'SUMMARY.json').write_text(json.dumps(record))
        (output/'history.json').write_text(json.dumps(history))
        paths.append(output)
    job['output'] = str(paths[0])
    return summary, job, paths


def test_paired_result_validates_both_sources_masks_and_selected_metrics(paired_job):
    module, job, _ = paired_job
    row = module.paired_row(job, 'new-source')
    assert row['ACC'] == row['normal_ACC'] == 100.
    assert row['delta_signature_ACC'] == row['delta_ACC'] == 0.


@pytest.mark.parametrize('field', ['initial_block_sha256', 'test_mask_sha256', 'total_parameters', 'training_trace'])
def test_pair_mismatches_rejected(paired_job, field):
    module, job, paths = paired_job
    path = paths[1]/'SUMMARY.json'
    record = json.loads(path.read_text())
    if field == 'total_parameters':
        record[field] += 1
    elif field == 'training_trace':
        path = paths[1]/'history.json'
        record = json.loads(path.read_text())
        record[0]['mask_trace_sha256'] = 'different-mask-trace'
    else:
        record['identity'][field] = 'mismatch'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        module.paired_row(job, 'new-source')


def test_only_all40_valid_rows_receive_fivefold_summary(tmp_path, monkeypatch):
    module = importlib.import_module(PACKAGE+'.summarize')
    scheduler = importlib.import_module(PACKAGE+'.scheduler')
    def row(job, source):
        result = dict(dataset='IEMOCAPSix', seed=68, fold=job['fold'], rate=job['rate'],
                      initial_block_sha256='initial', training_trace_sha256='trace', selected_epoch=1)
        for key in module.KEYS:
            result.update({key:62., 'parent_'+key:61., 'delta_'+key:1.,
                           'normal_'+key:63., 'delta_signature_'+key:-1.})
        return result
    monkeypatch.setattr(module, 'paired_row', row)
    jobs = scheduler.jobs_for('train')
    for job in jobs:
        job['status'] = 'complete'
    complete = module.summarize(tmp_path, 'source', jobs)
    assert complete['status'] == 'complete'
    mean = next(r for r in complete['aggregate'] if r['scope'] == 'fivefold_eight_rate_mean')
    assert mean['ACC'] == 62. and mean['normal_ACC'] == 63. and mean['delta_signature_ACC'] == -1.
    assert not any(r['scope'].startswith('three_seed') for r in complete['aggregate'])
    partial = module.summarize(tmp_path, 'source', jobs[:-1])
    assert partial['status'] == 'incomplete'
    assert not any(r['scope'].startswith('fivefold') for r in partial['aggregate'])
