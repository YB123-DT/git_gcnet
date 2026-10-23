import importlib

import pytest


def test_no_partial_jobs_can_be_ranked(tmp_path):
    report = importlib.import_module('experiments.osram_nested_cmerc_tuning_20261023.report')
    with pytest.raises(ValueError):
        report.write_report(tmp_path, {'jobs': [], 'source_commit': 'sha'})


def test_protection_includes_wf1_and_acc_not_just_a_single_winner():
    report = importlib.import_module('experiments.osram_nested_cmerc_tuning_20261023.report')
    baseline = {'ACC': 80., 'WF1': 80., 'UA': 70.}
    candidate = {'ACC': 80.2, 'WF1': 79.5, 'UA': 71.}
    measured = report.metric_deltas(candidate, baseline)
    assert measured == pytest.approx({'ACC': .2, 'WF1': -.5, 'UA': 1.})
