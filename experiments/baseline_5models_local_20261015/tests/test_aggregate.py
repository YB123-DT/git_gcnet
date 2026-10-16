import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aggregate_results as agg
from common_data import fixed_masks


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def local_records(monkeypatch):
    records = [dict(id='d' + str(f), utterance_ids=['u0', 'u1'],
                    labels=np.array([0, 1]), session=f, split='test') for f in range(1, 6)]
    monkeypatch.setattr(agg, 'load_dataset', lambda _: {'records': records, 'metadata': {}})
    return records


def task(tmp_path, dataset='MELD', seed=66, fold=0, model='ComP', version='causal'):
    config = dict(dataset=dataset, seed=seed, fold=fold, model=model, model_version=version,
                  missing_rate=.3, eval_mask_seed=20261015, n_classes=2, stage_epoch=1, epochs=3,
                  checkpoint_selection='test_best_acc', evaluation_protocol='test_oracle')
    ident = '{}_{}_{}_{}_{}'.format(dataset, model, version, seed, fold)
    return dict(id=ident, config=config, phase='formal', run_dir=str(tmp_path/ident), causality='Causal')


def save_fixture(task, records, correct=True, mask_tamper=False, result_type='formal'):
    """Synthetic persisted artifact contract; no actual training is represented."""
    run = Path(task['run_dir']);run.mkdir(parents=True)
    c = task['config']
    selected = records if c['dataset'] == 'MELD' else [r for r in records if r['session'] == c['fold']]
    labels = np.concatenate([r['labels'] for r in selected])
    predictions = labels.copy() if correct else 1-labels
    masks = np.concatenate([fixed_masks(c['dataset'], r['id'], r['utterance_ids'], c['missing_rate']) for r in selected])
    if mask_tamper:
        masks[0] = 1 - masks[0]
    sample_ids = np.array([r['id']+'::'+u for r in selected for u in r['utterance_ids']])
    np.savez(run/'predictions.npz', labels=labels, predictions=predictions, masks=masks, sample_ids=sample_ids)
    (run/'best.pt').write_bytes(b'fixture-checkpoint-placeholder' * 10)
    metrics=agg.compute_metrics(labels,predictions,c['n_classes'])
    manifest=dict(config=dict(c,device='cuda:0',seq_len=None),result_type=result_type,
                  expected_n_test=len(labels),expected_sample_ids=sample_ids.tolist(),
                  mask_key_records=[dict(id=r['id'],utterance_ids=r['utterance_ids']) for r in selected])
    from runtime import digest
    for filename, value in [('config.json', c), ('manifest.json', manifest),
                            ('status.json', dict(state='COMPLETED',exit_code=0,epochs=3,result_type=result_type)),
                            ('checkpoint_replay.json',dict(verified=True,epoch=2)),
                            ('best.json',dict(epoch=2,checkpoint_sha256=digest(run/'best.pt'),predictions_sha256=digest(run/'predictions.npz'))),
                            ('metrics.json',dict(**metrics,best_epoch=2,n_test=len(labels),checkpoint_selection='test_best_acc',evaluation_protocol='test_oracle'))]:
        (run/filename).write_text(json.dumps(value))


def run_registry(tmp_path, tasks):
    registry = tmp_path/'registry.json';registry.write_text(json.dumps({'tasks':tasks}))
    result = agg.aggregate(registry, tmp_path/'output')
    return result, tmp_path/'output'


def test_completed_text_is_not_a_completed_experiment(tmp_path):
    t = task(tmp_path)
    p = Path(t['run_dir']);p.mkdir();(p/'status.json').write_text('{"state":"COMPLETED","exit_code":0}')
    report, out = run_registry(tmp_path, [t])
    assert report['verified_formal_folds'] == 0
    assert rows(out/'progress.csv')[0]['status'] == 'INVALID_COMPLETED'
    assert rows(out/'results_summary.csv')[0]['count'] == '0'
    assert rows(out/'results_summary.csv')[0]['acc_mean'] == ''


def test_four_iemocap_folds_do_not_make_one_seed(tmp_path, local_records, monkeypatch):
    tasks = [task(tmp_path,dataset='IEMOCAPFour',fold=f) for f in range(1,5)]
    for t in tasks:save_fixture(t, local_records)
    report, out = run_registry(tmp_path,tasks)
    assert report['verified_formal_folds'] == 4
    assert report['complete_seed_results'] == 0
    seed = rows(out/'results_seed.csv')[0]
    assert seed['fold_count'] == '4' and seed['missing_folds'] == '5'
    assert seed['acc'] == ''


def test_two_seeds_have_sample_variance_but_no_three_seed_claim(tmp_path, local_records, monkeypatch):
    tasks = [task(tmp_path,seed=s) for s in (66,67)]
    for t in tasks:save_fixture(t,local_records,correct=t['config']['seed']==66)
    report,out = run_registry(tmp_path,tasks)
    summary = rows(out/'results_summary.csv')[0]
    assert report['complete_3seed_results'] == 0
    assert summary['count'] == '2' and summary['missing_seeds'] == '68'
    assert float(summary['acc_mean']) == 50
    assert float(summary['acc_variance']) == 5000
    assert float(summary['acc_std']) == pytest.approx(np.sqrt(5000))
    assert '50.00 ±' not in (out/'TABLE_A.md').read_text()


def test_fixed_mask_tamper_and_smoke_are_rejected(tmp_path, local_records, monkeypatch):
    monkeypatch.setattr(agg, 'valid_result', lambda _: True)
    tasks=[task(tmp_path,seed=s) for s in (66,67)]
    save_fixture(tasks[0],local_records,mask_tamper=True)
    save_fixture(tasks[1],local_records,result_type='smoke')
    report,out=run_registry(tmp_path,tasks)
    assert report['verified_formal_folds'] == 0
    assert {r['rejection_reason'] for r in rows(out/'progress.csv')} == {'FIXED_TEST_MASK_MISMATCH','NON_FORMAL_RESULT_TYPE'}


def test_five_folds_pool_predictions_then_exact_three_seeds(tmp_path, local_records, monkeypatch):
    tasks=[task(tmp_path,dataset='IEMOCAPFour',seed=s,fold=f) for s in (66,67,68) for f in range(1,6)]
    for t in tasks:save_fixture(t,local_records,correct=t['config']['fold']!=5)
    report,out=run_registry(tmp_path,tasks)
    assert report['complete_seed_results'] == 3
    assert report['complete_3seed_results'] == 1
    summary=rows(out/'results_summary.csv')[0]
    assert summary['status']=='COMPLETE_3SEED' and float(summary['acc_mean'])==80
    assert float(summary['acc_std'])==0
    assert {r['n_test'] for r in rows(out/'results_seed.csv')}=={'10'}
    assert 'test-selection bias' in (out/'TABLE_A.md').read_text()


def test_pairing_requires_both_exact_test_samples_and_masks(tmp_path, local_records, monkeypatch):
    tasks=[task(tmp_path,version=v) for v in ('causal','original')]
    for t in tasks:save_fixture(t,local_records,correct=t['config']['model_version']=='causal')
    _,out=run_registry(tmp_path,tasks)
    comparison=rows(out/'comparison.csv')[0]
    assert comparison['status']=='PAIRED' and float(comparison['acc_delta'])==100
    # A wrong sample id must not become a paired comparison even with verifier
    # approval: the independent dataset-grounded check rejects the result.
    run=Path(tasks[1]['run_dir'])
    with np.load(run/'predictions.npz') as z: values={k:z[k] for k in z.files}
    values['sample_ids'][0]='bad';np.savez(run/'predictions.npz',**values)
    _,out=run_registry(tmp_path,tasks)
    assert rows(out/'comparison.csv')[0]['status']=='MISSING_ORIGINAL'


def test_different_eval_mask_seeds_cannot_be_mixed(tmp_path, local_records):
    tasks=[task(tmp_path,seed=s) for s in (66,67,68)]
    tasks[-1]['config']['eval_mask_seed']=999
    for t in tasks:
        save_fixture(t,local_records)
    # The fixture deliberately writes the default mask seed; the last result is
    # invalid independently, and the planned cross-seed protocol also differs.
    _,out=run_registry(tmp_path,tasks)
    summary=rows(out/'results_summary.csv')[0]
    assert summary['status']=='INCOMPATIBLE_CONFIGS' and summary['acc_mean']==''


def test_unstarted_table_c_reports_verified_ancestry_without_execution_claim(tmp_path):
    t=task(tmp_path)
    provenance=tmp_path/'adapters/comp/SOURCE.json';provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps(dict(upstream_repository='https://example.invalid/source',
                                          upstream_commit='a'*40,source_commit='b'*40)))
    audit=tmp_path/'audit/data_audit.json';audit.parent.mkdir()
    audit.write_text(json.dumps({'datasets':{'MELD':{'metadata':{'selected_features_labels_sha256':'c'*64}}}}))
    _,out=run_registry(tmp_path,[t])
    row=rows(out/'TABLE_C.csv')[0]
    assert row['source_commit']=='a'*40 and row['adapter_commit']=='b'*40
    assert row['feature_version']=='c'*64
    assert row['configuration_scope']=='planned_unstarted'
    assert row['code_snapshot_sha256']==''
    assert row['status']=='MISSING'


def test_sdr_versions_remain_separate_and_only_released_enters_main_table(tmp_path, local_records):
    tasks = [task(tmp_path, model='SDR-GNN', version=v, seed=s)
             for v in ('released_compatible', 'implementation_fixed') for s in (66, 67, 68)]
    for t in tasks:
        t['causality'] = 'Non-causal'
        save_fixture(t, local_records, correct=t['config']['model_version'] == 'implementation_fixed')
    _, out = run_registry(tmp_path, tasks)
    summaries = rows(out/'results_summary.csv')
    assert len(summaries) == 2 and all(r['count'] == '3' for r in summaries)
    assert {r['model_version']: float(r['acc_mean']) for r in summaries} == {
        'released_compatible': 0., 'implementation_fixed': 100.}
    assert 'released-code compatible' in (out/'TABLE_A.md').read_text()
    assert 'implementation-fixed' not in (out/'TABLE_A.md').read_text()
    assert 'implementation-fixed' in (out/'TABLE_D.md').read_text()
    assert 'released-code compatible' in (out/'TABLE_D.md').read_text()
    paired = rows(out/'sdr_version_comparison.csv')
    assert len(paired) == 3 and all(r['status'] == 'PAIRED' for r in paired)
    assert all(float(r['acc_delta']) == 100. for r in paired)


def test_fixed_scores_cannot_fill_missing_released_sdr_result(tmp_path, local_records):
    released = task(tmp_path, model='SDR-GNN', version='released_compatible')
    fixed = task(tmp_path, model='SDR-GNN', version='implementation_fixed')
    save_fixture(fixed, local_records)
    _, out = run_registry(tmp_path, [released, fixed])
    summary = {r['model_version']: r for r in rows(out/'results_summary.csv')}
    assert summary['released_compatible']['count'] == '0'
    assert summary['released_compatible']['acc_mean'] == ''
    assert summary['implementation_fixed']['count'] == '1'
    assert rows(out/'sdr_version_comparison.csv')[0]['status'] == 'MISSING_RELEASED_COMPATIBLE'
