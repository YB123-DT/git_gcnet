from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_mosei_capacity_cmerc import cfg


def test_only_write_step_changes_from_the_paired_flat():
    from experiments.osram_mosei_write_step_20261024.worker import STEPS, candidate_config
    assert STEPS == (.2, .4, .8, 1.)
    for step in STEPS:
        original = cfg()
        assert asdict(candidate_config(original, step)) == dict(asdict(original), osram_write_step=step)
    for change in (dict(learning_rate=.001), dict(seed=67), dict(osram_adapter_hidden_dim=256),
                   dict(osram_meaningful_block='nested_dim32')):
        with pytest.raises(ValueError):
            candidate_config(replace(cfg(), **change), .4)
    with pytest.raises(ValueError):
        candidate_config(cfg(), .6)


def test_plan_is_four_fresh_flat_runs_on_healthy_gpu7():
    from experiments.osram_mosei_write_step_20261024.run import GPU, plan_jobs
    jobs = plan_jobs(Path('/new'), 'sha')
    assert len(jobs) == 4
    assert {j['step'] for j in jobs} == {.2, .4, .8, 1.}
    assert len({j['output'] for j in jobs}) == 4
    for job in jobs:
        assert job['gpu_uuid'] == GPU == 'GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'
        assert job['epochs'] == 60 and job['method'] == 'none'
        assert 'cmerc-repro' in job['command'][0]
        assert '--resume' not in job['command']


def test_report_allows_only_the_named_step_difference():
    from experiments.osram_mosei_write_step_20261024.report import normalize_config
    old = asdict(cfg())
    assert normalize_config(dict(old, osram_write_step=.4), old) == old
    with pytest.raises(ValueError):
        normalize_config(dict(old, osram_write_step=.4, learning_rate=.001), old)
