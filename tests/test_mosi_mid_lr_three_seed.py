import importlib
from pathlib import Path

import pytest


def test_plan_only_adds_seeds67_and68_at_3e4(tmp_path):
    from experiments.osram_mosi_mid_lr_3seed_20261021.run import plan_jobs
    jobs = plan_jobs(tmp_path, 'sha', ['GPU-good'])
    assert [j['seed'] for j in jobs] == [67, 68]
    assert len({j['output'] for j in jobs}) == 2
    for j in jobs:
        assert j['backbone_lr'] == 3e-4 and j['final_backbone_lr'] == 3e-4 / 10
        assert j['dataset'] == 'CMUMOSI' and j['mode'] == 'flat'
        assert j['command'][2] == 'experiments.osram_mosi_mid_lr_3seed_20261021.worker'
    with pytest.raises(ValueError):
        plan_jobs(tmp_path, 'sha', ['GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'])


def test_worker_selects_same_seed_reference_not_seed66(monkeypatch):
    module = importlib.import_module('experiments.osram_mosi_mid_lr_3seed_20261021.worker')
    captured = {}
    def capture(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
    monkeypatch.setattr(module.common, 'main', capture)
    module.main(['--seed', '67', '--mode', 'flat'])
    assert captured['allowed_seeds'] == (67, 68)
    assert captured['allowed_modes'] == ('flat',)
    assert captured['backbone_lr'] == 3e-4
    assert captured['final_backbone_lr'] == 3e-4 / 10
    assert captured['reference_seed_parent'].name == 'MOSI'
    assert captured['argv'] == ['--seed', '67', '--mode', 'flat']


def test_shared_reference_resolution_preserves_original_default():
    from experiments.osram_mosi_router_lr1e5_20261020.worker import reference_path, REFERENCE
    assert reference_path(66, 'flat') == REFERENCE/'flat'
    for seed in (67, 68):
        assert reference_path(seed, 'flat', reference_seed_parent=REFERENCE.parent) == REFERENCE.parent/f'seed_{seed}'/'flat'
    with pytest.raises(ValueError):
        reference_path(68, 'flat')


def test_report_reuses_completed66_and_refuses_partial_new_jobs(tmp_path):
    from experiments.osram_mosi_mid_lr_3seed_20261021.report import write_report, SEED66
    assert SEED66 == Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003')
    with pytest.raises(ValueError, match='completed'):
        write_report(tmp_path, {'jobs': []})
