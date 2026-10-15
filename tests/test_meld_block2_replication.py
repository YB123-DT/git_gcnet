"""Finite seed67/68 replication, without changing the seed66 screen protocol."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE = 'experiments.osram_meld_block2_auto_20261014'


def test_default_worker_guard_remains_seed66_only():
    methods = importlib.import_module(PACKAGE + '.methods')
    guard = getattr(methods, 'validate_training_config', None)
    assert callable(guard), 'explicit replication guard is missing'
    cfg = SimpleNamespace(epochs=50, seed=66, preserve_selection_artifacts=True)
    guard(cfg)
    cfg.seed = 67
    with pytest.raises(ValueError):
        guard(cfg)


@pytest.mark.parametrize('seed', [67, 68])
def test_replication_guard_matches_explicit_seed(seed):
    methods = importlib.import_module(PACKAGE + '.methods')
    guard = getattr(methods, 'validate_training_config', None)
    assert callable(guard), 'explicit replication guard is missing'
    cfg = SimpleNamespace(epochs=50, seed=seed, preserve_selection_artifacts=True)
    guard(cfg, replication_seed=seed)
    with pytest.raises(ValueError):
        guard(cfg, replication_seed=135-seed)
    cfg.epochs = 100
    with pytest.raises(ValueError):
        guard(cfg, replication_seed=seed)


def test_three_fixed_configs_make_only_six_new_jobs(tmp_path):
    spec = importlib.util.find_spec(PACKAGE + '.replicate')
    assert spec is not None, 'replication launcher is missing'
    module = importlib.import_module(PACKAGE + '.replicate')
    base = json.loads((Path(__file__).resolve().parents[1] / 'experiments' /
                      'osram_meld_block2_auto_20261014/BASE_CONFIG.json').read_text())
    config = importlib.import_module(PACKAGE + '.methods').build_config(base, 'B6', 'unused')
    source = dict(status='complete', method='B6', config=config,
                  provenance={'git_sha': 'source66', 'dataset_sha256': 'dataset'},
                  testbest={'mean': {'ACC': 60.9}})
    manifests = [{'id': f'B6_trial{i}', 'source_summary': f'/source/{i}/SUMMARY.json',
                  'summary': source} for i in (16, 13, 12)]
    jobs = module.make_jobs(manifests, tmp_path)
    assert len(jobs) == 6 and len({j['output'] for j in jobs}) == 6
    assert {j['config']['seed'] for j in jobs} == {67, 68}
    for job in jobs:
        assert {k for k in config if config[k] != job['config'][k]} == {'seed'}
    assert config['seed'] == 66
    scheduler = module.ReplicationScheduler(tmp_path / 'queue')
    command = scheduler.command(jobs[0], 'config.json', 'GPU-test')
    assert command[-2:] == ['--replication-seed', str(jobs[0]['config']['seed'])]


def test_incomplete_sources_cannot_be_replicated(tmp_path):
    spec = importlib.util.find_spec(PACKAGE + '.replicate')
    assert spec is not None, 'replication launcher is missing'
    module = importlib.import_module(PACKAGE + '.replicate')
    with pytest.raises(ValueError):
        module.make_jobs([{'id': 'B6_trial16', 'summary': {'status': 'running'}}], tmp_path)
