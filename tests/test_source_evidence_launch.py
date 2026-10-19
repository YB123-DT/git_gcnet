import importlib
from pathlib import Path

import pytest


def test_finite_audit_plan_never_trains_or_changes_anchors(tmp_path):
    try:
        api = importlib.import_module('experiments.osram_12_source_evidence_audit_20261019.launch')
    except ModuleNotFoundError:
        pytest.fail('Missing finite frozen audit launcher')
    jobs = api.build_jobs(tmp_path, 'abc123', 'GPU-healthy')
    assert len(jobs) == 2
    assert {job['dataset'] for job in jobs} == {'MELD', 'IEMOCAPSix'}
    for job in jobs:
        assert job['command'][0] == '/data2/yb/conda_envs/cmerc-repro/bin/python'
        assert 'experiments.osram_12_source_evidence_audit_20261019.run' in job['command']
        assert '--source-commit' in job['command']
        assert not {'--epochs','--learning-rate','--train'} & set(job['command'])
        assert Path(job['output']).parent == tmp_path
