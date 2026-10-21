from pathlib import Path
from types import SimpleNamespace

import pytest


def test_exact_finite_plan_no_mosei_duplicate():
    from experiments.osram_v3_classification_20261021.run import plan_jobs
    jobs = plan_jobs(Path('/tmp/v3_queue'))
    assert len(jobs) == 11
    assert {j['seed'] for j in jobs} == {66}
    assert [j['fold'] for j in jobs if j['dataset'] == 'IEMOCAPFour'] == [1, 2, 3, 4, 5]
    assert [j['fold'] for j in jobs if j['dataset'] == 'IEMOCAPSix'] == [1, 2, 3, 4, 5]
    assert sum(j['dataset'] == 'MELD' for j in jobs) == 1
    assert not any(j['dataset'] == 'CMUMOSEI' for j in jobs)
    assert len({j['output'] for j in jobs}) == 11


def test_command_preserves_seed_dataset_fold_and_environment():
    from experiments.osram_v3_classification_20261021.run import plan_jobs, worker_command, PYTHON
    args = SimpleNamespace(source_commit='reviewed-sha')
    job = plan_jobs(Path('/tmp/v3_queue'))[0]
    command = worker_command(args, job, 'GPU-healthy', 'train')
    assert command[0] == PYTHON
    for key, value in (('--seed', '66'), ('--fold', str(job['fold'])),
                       ('--dataset', job['dataset']), ('--source-commit', 'reviewed-sha')):
        assert command[command.index(key) + 1] == value
    assert command[command.index('--output') + 1] == job['output']


@pytest.mark.parametrize('free,active,unmeasured,expected', [
    (32000, 0, 0, 2), (32000, 2, 0, 0), (5000, 0, 0, 0), (16000, 1, 1, 0),
])
def test_resource_admission_reserves_loading_children(free, active, unmeasured, expected):
    from experiments.osram_v3_classification_20261021.run import capacity
    assert capacity(free, active, 6000, unmeasured, limit=2) == expected


def test_latest_user_gpu_caps():
    from experiments.osram_v3_classification_20261021.run import GPU_LIMITS, capacity
    assert sorted(GPU_LIMITS.values()) == [4, 8]
    assert capacity(32000, 0, 1500, limit=8) == 8


def test_gpu4_cannot_be_scheduled():
    from experiments.osram_v3_classification_20261021.run import validate_gpus
    with pytest.raises(ValueError):
        validate_gpus(['GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'])
