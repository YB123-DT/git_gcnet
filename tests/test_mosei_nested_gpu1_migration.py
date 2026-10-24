import copy
from pathlib import Path

import pytest


def original(tmp_path):
    from experiments.osram_mosei_nested_local_20261024.run import plan_jobs, GPU
    jobs = plan_jobs(tmp_path / 'old', 'source-sha', GPU)
    for i, job in enumerate(jobs):
        job.update(status='running' if i < 3 else 'failed', pid=100+i)
    return dict(source_commit='source-sha', jobs=jobs)


def test_only_four_existing_variants_migrate_without_hyperparameter_changes(tmp_path):
    from experiments.osram_mosei_nested_local_20261024.migrate_gpu1 import plan_migration, GPU1
    old = original(tmp_path)
    saved = copy.deepcopy(old)
    jobs = plan_migration(old, tmp_path / 'new')
    assert old == saved
    assert len(jobs) == 4
    for i, (before, after) in enumerate(zip(old['jobs'], jobs)):
        assert after['gpu_uuid'] == GPU1
        assert after['previous_output'] == before['output']
        assert after['resume'] == (i < 3)
        command = before['command'].copy()
        command[command.index('--gpu-uuid')+1] = GPU1
        command[command.index('--output')+1] = after['output']
        if i < 3:
            command += ['--resume']
        assert after['command'] == command
        for key in ('variant', 'method', 'seed', 'fold', 'epochs', 'learning_rate',
                    'nested_learning_rate', 'source_commit', 'mask_protocol'):
            assert after[key] == before[key]
        assert Path(after['output']).is_relative_to(tmp_path / 'new')


def test_migration_rejects_scope_expansion_or_unknown_failure(tmp_path):
    from experiments.osram_mosei_nested_local_20261024.migrate_gpu1 import plan_migration
    state = original(tmp_path)
    state['jobs'].append(copy.deepcopy(state['jobs'][0]))
    with pytest.raises(ValueError):
        plan_migration(state, tmp_path / 'new')
    state = original(tmp_path)
    state['jobs'][0]['status'] = 'failed'
    with pytest.raises(ValueError):
        plan_migration(state, tmp_path / 'new')


def test_pid_guard_requires_same_command_and_source(tmp_path):
    from experiments.osram_mosei_nested_local_20261024.migrate_gpu1 import validate_process
    root = tmp_path / 'proc'; process = root / '123'; process.mkdir(parents=True)
    source = tmp_path / 'source'; source.mkdir()
    (process / 'cwd').symlink_to(source)
    command = ['python', '-u', '-m', 'experiment.worker', '--output', '/expected']
    (process / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in command)+b'\0')
    validate_process(123, command, source, proc_root=root)
    with pytest.raises(RuntimeError):
        validate_process(123, command[:-1]+['/other'], source, proc_root=root)
    with pytest.raises(RuntimeError):
        validate_process(123, command, tmp_path / 'other', proc_root=root)
