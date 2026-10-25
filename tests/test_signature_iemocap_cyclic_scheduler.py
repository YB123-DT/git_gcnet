import importlib
import json
from pathlib import Path

import pytest


MODULE = 'experiments.osram_signature_iemocap_cyclic_20261025'


def test_controller_accepts_twelve_workers_and_records_effective_limit(tmp_path, monkeypatch):
    scheduler = importlib.import_module(MODULE + '.scheduler')
    captured = {}
    monkeypatch.setattr(scheduler, 'guard_gpu', lambda *args: None)

    def capture(args, state, phase, peaks=None):
        captured.update(limit=args.per_gpu, state_limit=state['max_per_gpu'])
        raise RuntimeError('admission captured before launching any worker')

    monkeypatch.setattr(scheduler, 'run_phase', capture)
    result = scheduler.main(['--root', str(tmp_path/'queue'), '--source-commit', 'source',
                             '--gpu-map', '0=GPU-ok', '--per-gpu', '12'])
    assert result == 1
    assert captured == dict(limit=12, state_limit=12)


@pytest.mark.parametrize('limit', ['0', '13'])
def test_controller_rejects_out_of_bounds_capacity(tmp_path, limit):
    scheduler = importlib.import_module(MODULE + '.scheduler')
    with pytest.raises(SystemExit):
        scheduler.main(['--root', str(tmp_path/'queue'), '--source-commit', 'source',
                        '--gpu-map', '0=GPU-ok', '--per-gpu', limit])


def test_finite_queue_covers_every_seed_fold_rate_once():
    scheduler = importlib.import_module(MODULE + '.scheduler')
    jobs = scheduler.jobs_for('train')
    assert len(jobs) == 120
    assert len({j['id'] for j in jobs}) == 120
    assert {(j['seed'], j['fold'], j['rate']) for j in jobs} == {
        (s, f, f'{r / 10:.1f}') for s in (66, 67, 68)
        for f in range(1, 6) for r in range(8)}
    assert [(j['seed'], j['fold']) for j in jobs[:15]] == [
        (s, f) for f in (5, 1, 2, 3, 4) for s in (66, 67, 68)]
    smoke = scheduler.jobs_for('smoke')
    assert [(j['seed'], j['fold'], j['rate']) for j in smoke] == [(66, 5, '0.7')]


def test_capacity_uses_measured_peak_and_keeps_reserve():
    scheduler = importlib.import_module(MODULE + '.scheduler')
    snapshot = dict(disk_free_gib=100, ram_available_gib=100,
                    gpus={'GPU-ok': {'free_mib': 12000}})
    peaks = {'IEMOCAPSix': dict(gpu_mib=4000, rss_gib=5)}
    job = dict(dataset='IEMOCAPSix')
    assert scheduler.fits(snapshot, 'GPU-ok', job, [], peaks, 6)
    assert not scheduler.fits(snapshot, 'GPU-ok', job, [job, job], peaks, 6)
    assert not scheduler.fits(snapshot, 'GPU-ok', job, [job], peaks, 1)


def test_pooling_uses_confusion_not_mean_ua():
    summary = importlib.import_module(MODULE + '.summarize')
    # Unequal supports deliberately distinguish pooled and per-fold averages.
    metrics = summary.confusion_metrics([[8, 2], [1, 1]])
    assert metrics['ACC'] == pytest.approx(75.)
    assert metrics['UA'] == pytest.approx(65.)
    assert metrics['N'] == 12


def test_incomplete_folds_are_never_presented_as_fivefold(tmp_path):
    summary = importlib.import_module(MODULE + '.summarize')
    rows = [dict(dataset='IEMOCAPSix', seed=66, fold=5, rate=f'{r / 10:.1f}',
                 ACC=60., UA=50., WF1=55., CE=1.,
                 parent_ACC=59., parent_UA=49., parent_WF1=54., parent_CE=1.1,
                 delta_ACC=1., delta_UA=1., delta_WF1=1., delta_CE=-.1,
                 training_trace_sha256='trace', initial_block_sha256='initial')
            for r in range(8)]
    aggregate, pooled = summary.aggregate_rows(rows)
    assert any(r['scope'] == 'fold_eight_rate_mean' for r in aggregate)
    assert not any(r['scope'].startswith('fivefold') for r in aggregate)
    assert not pooled


def test_bad_gpu_is_rejected():
    scheduler = importlib.import_module(MODULE + '.scheduler')
    with pytest.raises(ValueError):
        scheduler.parse_gpu_map('4=GPU-bad')
    with pytest.raises(ValueError):
        scheduler.parse_gpu_map('0=GPU-46fb379f-cc90-dc82-9b5e-5d011f552264')


def test_resume_preserves_live_worker_without_launching(tmp_path, monkeypatch):
    scheduler = importlib.import_module(MODULE + '.scheduler')
    job = dict(id='live', status='running', output=str(tmp_path), pid=42)
    monkeypatch.setattr(scheduler, 'process_matches', lambda row: True)
    scheduler.reconcile([job], 'train')
    assert job['status'] == 'running'


def test_resume_marks_only_full_checkpoint_recoverable(tmp_path, monkeypatch):
    scheduler = importlib.import_module(MODULE + '.scheduler')
    monkeypatch.setattr(scheduler, 'process_matches', lambda row: False)
    job = dict(id='interrupted', status='running', output=str(tmp_path), pid=42)
    scheduler.reconcile([job], 'train')
    assert job['status'] == 'failed'
    (tmp_path / 'last.pt').touch()
    job['status'] = 'running'
    scheduler.reconcile([job], 'train')
    assert job['status'] == 'pending'


@pytest.fixture
def completed_job(tmp_path):
    import hashlib
    import numpy as np
    summary_module = importlib.import_module(MODULE + '.summarize')
    labels = np.arange(6, dtype=np.int64)
    prediction = np.eye(6, dtype=np.float64)
    availability = np.ones((6, 3), dtype=np.bool_)
    np.savez(tmp_path/'predictions_epoch_1.npz', labels=labels, prediction=prediction,
             base_prediction=prediction, availability=availability)
    (tmp_path/'parent.pt').write_bytes(b'parent')
    (tmp_path/'best_epoch_1.pt').write_bytes(b'branch')
    metrics, _ = summary_module.prediction_metrics(labels, prediction)
    identity = dict(stage2_epochs=2, dataset='IEMOCAPSix', seed=66, fold=5,
        evaluation_rate='0.7', source_commit='test-source', cyclic_training=True,
        mask_protocol='gcnet-random-mask-v1',
        parent=dict(checkpoint=str(tmp_path/'parent.pt'), checkpoint_sha256=summary_module.digest(tmp_path/'parent.pt')),
        test_mask_sha256=hashlib.sha256(availability.tobytes()).hexdigest(),
        initial_block_sha256='initial')
    best = dict(epoch=1, metrics=metrics, checkpoint='best_epoch_1.pt',
        checkpoint_sha256=summary_module.digest(tmp_path/'best_epoch_1.pt'),
        predictions='predictions_epoch_1.npz',
        prediction_sha256=summary_module.digest(tmp_path/'predictions_epoch_1.npz'))
    summary = dict(status='complete', epoch=2, identity=identity,
        parent_id=identity['parent']['checkpoint_sha256'], best=best, baseline=metrics,
        checks={k:True for k in ('parent_unchanged', 'parent_grad_none',
            'checkpoint_replay_exact', 'cyclic_train_rates_verified', 'only_requested_test_rate')})
    history = [dict(epoch=i+1, training_rates=[i/10], optimizer_steps=1,
                    mask_trace_sha256=f'trace{i}', gradients=dict(nonzero=1), test=metrics)
               for i in range(2)]
    (tmp_path/'SUMMARY.json').write_text(json.dumps(summary))
    (tmp_path/'history.json').write_text(json.dumps(history))
    return dict(dataset='IEMOCAPSix', seed=66, fold=5, rate='0.7', output=str(tmp_path), status='complete'), summary


def test_selected_metrics_validate_exact_epoch_mask_and_predictions(completed_job):
    summary_module = importlib.import_module(MODULE + '.summarize')
    job, _ = completed_job
    row = summary_module.selected_row(job, 'test-source')
    assert row['selected_epoch'] == 1
    assert row['ACC'] == 100.
    assert row['UA'] == 100.
    assert row['delta_CE'] == 0.


@pytest.mark.parametrize('change', ['epoch', 'mask', 'source', 'parent'])
def test_selected_metrics_reject_identity_and_reselection_changes(completed_job, change):
    summary_module = importlib.import_module(MODULE + '.summarize')
    job, summary = completed_job
    if change == 'epoch':
        summary['best']['epoch'] = 2  # Same-ACC tie must use earliest epoch.
    elif change == 'mask':
        summary['identity']['test_mask_sha256'] = 'wrong'
    elif change == 'source':
        summary['identity']['source_commit'] = 'wrong'
    else:
        summary['identity']['parent']['checkpoint_sha256'] = 'wrong'
    (Path(job['output'])/'SUMMARY.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        summary_module.selected_row(job, 'test-source')


def test_full_fold_metrics_keep_pooled_and_arithmetic_separate():
    summary_module = importlib.import_module(MODULE + '.summarize')
    rows = []
    for fold in range(1, 6):
        logits = [[1., 0., 0., 0., 0., 0.]]*(fold*6)
        labels = list(range(6))*fold
        metrics, confusion = summary_module.prediction_metrics(labels, logits)
        for r in range(8):
            row = dict(dataset='IEMOCAPSix', seed=66, fold=fold, rate=f'{r / 10:.1f}',
                       N=metrics['N'], confusion=confusion, parent_confusion=confusion)
            for key in ('ACC', 'UA', 'WF1', 'CE'):
                row.update({key:metrics[key], 'parent_'+key:metrics[key], 'delta_'+key:0.})
            rows.append(row)
    aggregate, pooled = summary_module.aggregate_rows(rows)
    assert len(pooled) == 8
    assert all(r['N'] == 90 for r in pooled)
    assert sum(r['scope'] == 'fivefold_arithmetic_mean' for r in aggregate) == 8
    assert sum(r['scope'] == 'fivefold_eight_rate_mean' for r in aggregate) == 1
    assert not any(r['scope'] == 'three_seed_fivefold_mean' for r in aggregate)


def test_partial_summary_is_machine_readable_without_inventing_mean(completed_job, tmp_path):
    summary_module = importlib.import_module(MODULE + '.summarize')
    job, _ = completed_job
    report = summary_module.summarize(tmp_path, 'test-source', [job])
    assert report['status'] == 'incomplete'
    assert len(report['per_rate']) == 1
    assert not report['aggregate']
    assert json.loads((tmp_path/'RESULT.json').read_text())['status'] == 'incomplete'
    assert (tmp_path/'RESULT.md').exists()
    assert (tmp_path/'RESULT.csv').exists()
