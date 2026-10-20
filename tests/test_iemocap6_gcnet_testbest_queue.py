from copy import deepcopy

import pytest


def test_rerun_has_thirty_new_gcnet_testbest_jobs_and_no_reused_validation_fold(tmp_path):
    from experiments.osram_iemocap6_gcnet_testbest_20261020.run import plan_jobs
    jobs = plan_jobs(tmp_path, 'source', ['GPU-a', 'GPU-b', 'GPU-c'])
    assert len(jobs) == 30
    assert {(j['fold'], j['seed'], j['mode']) for j in jobs} == {
        (f, s, m) for f in range(1, 6) for s in (66, 67, 68) for m in ('flat', 'joint')}
    assert len({j['output'] for j in jobs}) == 30
    assert all(j['checkpoint_selection'] == 'test-oracle-per-rate' for j in jobs)
    assert all(j['mask_protocol'] == 'gcnet-random-mask-v1' for j in jobs)
    assert all(j['command'][2] == 'experiments.osram_iemocap6_gcnet_testbest_20261020.worker'
               and j['dataset'] == 'IEMOCAPSix' and not j.get('reused') for j in jobs)


def valid_policy():
    return dict(identity=dict(config=dict(checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='accuracy', missing_mask_protocol='gcnet-random-mask-v1',
        epochs=100), split=dict(adapter_holdout=False, train_indices=[1, 2, 3, 4],
            original_train_indices=[1, 2, 3, 4], test_indices=[5])),
        selected_epoch_by_rate={str(i / 10): i + 1 for i in range(8)})


def test_report_rejects_validation_selection_holdout_missing_epoch_and_wrong_metric():
    from experiments.osram_iemocap6_gcnet_testbest_20261020.report import validate_policy
    good = valid_policy()
    assert validate_policy(good) == good['selected_epoch_by_rate']
    for field, value in [('checkpoint_selection', 'validation'),
                         ('checkpoint_selection_metric', 'weighted_f1'),
                         ('missing_mask_protocol', 'conversation-mask-v2')]:
        bad = deepcopy(good)
        bad['identity']['config'][field] = value
        with pytest.raises(ValueError):
            validate_policy(bad)
    bad = deepcopy(good)
    bad['identity']['split']['train_indices'] = [1, 2, 3]
    with pytest.raises(ValueError, match='training'):
        validate_policy(bad)
    bad = deepcopy(good)
    bad['selected_epoch_by_rate']['0.7'] = 101
    with pytest.raises(ValueError, match='epoch'):
        validate_policy(bad)
    bad = deepcopy(good)
    del bad['selected_epoch_by_rate']['0.7']
    with pytest.raises(ValueError, match='eight'):
        validate_policy(bad)


def test_report_requires_all_thirty_completed_new_runs(tmp_path):
    from experiments.osram_iemocap6_gcnet_testbest_20261020.report import write_report
    with pytest.raises(ValueError, match='30'):
        write_report(tmp_path, dict(jobs=[]))


def test_saved_metric_must_match_selected_epoch_not_only_its_predictions(tmp_path):
    import hashlib
    import json
    import numpy as np
    from experiments.osram_text_path_router_joint_20261019.run import summarize_arrays
    from experiments.osram_iemocap6_gcnet_testbest_20261020.report import load_run
    summary = valid_policy()
    summary.update(status='complete', phase='train', dataset='IEMOCAPSix', epoch=100,
        fold=1, seed=66, mode='flat', source_commit='sha', evaluation={})
    rates = tuple(summary['selected_epoch_by_rate'])
    summary['selected_epoch_by_rate'] = {r: 1 for r in rates}
    logits = np.eye(6, dtype=np.float32) * 2
    log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    arrays = dict(labels=np.arange(6), candidates=np.repeat(logits[:, None], 3, axis=1),
        log_probs_router=log_probs, weights=np.tile([1., 0., 0.], (6, 1)),
        eligible=np.zeros(6, dtype=bool), availability=np.tile([1., 0., 0.], (6, 1)),
        conversation_ids=np.array(['conversation'] * 6), utterance_indices=np.arange(6), speakers=np.zeros(6))
    metrics = summarize_arrays(arrays, 6)
    (tmp_path/'evaluation').mkdir()
    for rate in rates:
        path = tmp_path/'evaluation'/('random_'+rate.replace('.', 'p')+'.npz')
        np.savez_compressed(path, **arrays)
        summary['evaluation'][rate] = dict(deepcopy(metrics),
            output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), selected_epoch=1)
    summary['aggregate'] = dict(router={s: deepcopy(metrics['router']) for s in ('mean', 'high')})
    history = [dict(epoch=i + 1, test={r: deepcopy(metrics) for r in rates}) for i in range(100)]
    (tmp_path/'history.json').write_text(json.dumps(history))
    (tmp_path/'SUMMARY.json').write_text(json.dumps(summary))
    job = dict(output=str(tmp_path), fold=1, seed=66, mode='flat', source_commit='sha')
    assert load_run(job)[0]['selected_epoch_by_rate'] == {r: 1 for r in rates}
    # Scores can be internally self-consistent but belong to a different epoch.
    history[0]['test']['0.0']['router']['UA'] -= 1
    (tmp_path/'history.json').write_text(json.dumps(history))
    with pytest.raises(ValueError, match='selected epoch'):
        load_run(job)
