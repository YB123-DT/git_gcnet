"""A selection-only rerun must preserve scope, masks and oracle labelling."""
from pathlib import Path

import pytest


def test_plan_is_six_gcnet_testbest_jobs_not_old_validation_workers(tmp_path):
    from experiments.osram_mosi_gcnet_testbest_20261020.run import plan_jobs
    jobs = plan_jobs(tmp_path, 'sha', ['GPU-a', 'GPU-b', 'GPU-c'])
    assert len(jobs) == 6
    assert {(j['seed'], j['mode']) for j in jobs} == {
        (s, m) for s in (66, 67, 68) for m in ('flat', 'joint')}
    assert all(j['dataset'] == 'CMUMOSI' for j in jobs)
    assert all(j['mask_protocol'] == 'gcnet-random-mask-v1' for j in jobs)
    assert all(j['command'][2] == 'experiments.osram_mosi_gcnet_testbest_20261020.worker' for j in jobs)
    assert all(j['checkpoint_selection'] == 'test-oracle-per-rate' for j in jobs)
    assert len({j['output'] for j in jobs}) == 6
    assert all(str(tmp_path) in j['output'] for j in jobs)


def test_oracle_report_requires_eight_actual_epochs_and_explicit_test_selection():
    from experiments.osram_mosi_gcnet_testbest_20261020.report import validate_policy
    epochs = {str(i / 10): i + 1 for i in range(8)}
    summary = dict(identity=dict(config=dict(checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1', missing_mask_protocol='gcnet-random-mask-v1')),
        selected_epoch_by_rate=epochs)
    assert validate_policy(summary) == epochs
    summary['identity']['config']['checkpoint_selection'] = 'validation'
    with pytest.raises(ValueError, match='Test-best'):
        validate_policy(summary)
    summary['identity']['config']['checkpoint_selection'] = 'test-oracle-per-rate'
    summary['selected_epoch_by_rate'] = {'0.0': 2}
    with pytest.raises(ValueError, match='eight'):
        validate_policy(summary)
    summary['selected_epoch_by_rate'] = dict(epochs, **{'0.7': 151})
    with pytest.raises(ValueError, match='epoch'):
        validate_policy(summary)


def test_incomplete_runs_cannot_be_ranked(tmp_path):
    from experiments.osram_mosi_gcnet_testbest_20261020.report import write_report
    with pytest.raises(ValueError, match='six'):
        write_report(tmp_path, dict(jobs=[]))
