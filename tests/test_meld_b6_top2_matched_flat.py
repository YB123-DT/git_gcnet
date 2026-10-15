"""Seven approved new runs, not another hyperparameter search."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'experiments/osram_meld_b6_top2_matched_flat_3seed_20261015/run.py'
    spec = importlib.util.spec_from_file_location('b6_top2_matched', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sources():
    from experiments.osram_meld_b6_threshold_hpo_20261015.run import make_config
    result = {}
    for name, values in [('B6_default', (.1, 1.)), ('B6_tuned', (.33599569865951234, 2.8200380294401013))]:
        result[name] = dict(source_summary=name, source_summary_sha256='fixture',
                            summary=dict(status='complete', method='B6', config=make_config(*values)))
    return result


def test_exact_seven_jobs_and_shared_training_hyperparameters(tmp_path):
    m = module()
    before = sources()
    copied = deepcopy(before)
    jobs = m.make_jobs(before, tmp_path)
    assert before == copied
    assert len(jobs) == 7 and len({j['id'] for j in jobs}) == 7
    assert {(j['group'], j['config']['seed']) for j in jobs} == {
        ('Flat_matched', 66), ('Flat_matched', 67), ('Flat_matched', 68),
        ('B6_default', 67), ('B6_default', 68), ('B6_tuned', 67), ('B6_tuned', 68)}
    configs = [j['config'] for j in jobs]
    for key in ('learning_rate', 'weight_decay', 'dropout', 'batch_size', 'epochs',
                'missing_mask_protocol', 'checkpoint_selection_metric', 'train_missing_rates'):
        assert all(c[key] == configs[0][key] for c in configs)
    for job in jobs:
        c = job['config']
        assert c['epochs'] == 50 and c['batch_size'] == 32
        assert c['missing_mask_protocol'] == 'gcnet-random-mask-v1'
        assert c['checkpoint_selection'] == 'test-oracle-per-rate'
        assert c['checkpoint_selection_metric'] == 'accuracy'
        assert not c['osram_disable_local_skip'] and c['training_objective'] == 'emotion-only'
        if job['group'] == 'Flat_matched':
            assert job['method'] == 'Flat' and c['osram_meaningful_block'] == 'none'
            assert c['osram_rpca_lr_multiplier'] == 1.


def test_rejects_nonmatching_protocol_or_other_chosen_variant(tmp_path):
    m = module()
    for key, value in [('learning_rate', .001), ('osram_rpca_rank', 3),
                       ('missing_mask_protocol', 'conversation-mask-v2'), ('seed', 67)]:
        s = sources()
        s['B6_tuned']['summary']['config'][key] = value
        with pytest.raises(ValueError):
            m.make_jobs(s, tmp_path)


def test_keep_one_job_on_gpu3_and5_and_use_explicit_seed_guards(tmp_path):
    m = module()
    from experiments.osram_meld_b6_threshold_hpo_20261015 import run as frozen
    s = m.make_scheduler(frozen, tmp_path / 'queue', tmp_path / 'source')
    assert s.gpus == (2, 3, 5) and s.capacity == {2: 5, 3: 1, 5: 1}
    assert s.global_per_gpu_limit == 6
    jobs = m.make_jobs(sources(), tmp_path)
    for j in jobs:
        command = s.command(j, tmp_path / 'config.json', 'GPU-example')
        if j['config']['seed'] == 66:
            assert '--replication-seed' not in command
        else:
            assert command[-2:] == ['--replication-seed', str(j['config']['seed'])]
        assert '--testbest-only' not in command  # exact testbest + common-mask replay


def test_summary_reuses_two_seed66_results_and_never_ranks_partial_triplets(tmp_path):
    m = module()
    ss = sources()
    metrics = dict(ACC=61., UA=36., WF1=57., CE=1.2)
    def summary():
        return dict(status='complete', testbest={'mean': metrics, 'high': metrics,
                    'rates': {str(i / 10): metrics for i in range(8)}})
    for item in ss.values():
        item['summary'].update(summary())
    jobs = m.make_jobs(ss, tmp_path)
    records = [dict(j, status='complete', summary=summary()) for j in jobs]
    report = m.summarize(ss, records, tmp_path)
    assert report['status'] == 'complete'
    assert len(report['groups']) == 3
    assert all(set(g['seeds']) == {'66', '67', '68'} for g in report['groups'].values())
    assert all(g['three_seed']['mean']['ACC']['mean'] == 61. for g in report['groups'].values())
    records[-1].update(status='failed', failures=[{'reason': 'test failure'}])
    report = m.summarize(ss, records, tmp_path)
    assert report['status'] == 'incomplete_or_failed'
    assert 'three_seed' not in report['groups']['B6_tuned']
