import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('lr', [5e-4, 3e-4])
def test_explicit_final_lr_changes_at_epoch101_only(lr):
    from experiments.osram_mosi_router_lr1e5_20261020.worker import set_epoch_lrs
    optimizer = SimpleNamespace(param_groups=[dict(name='backbone', lr=1.)])
    for epoch in (0, 99, 100, 149):
        set_epoch_lrs(optimizer, epoch, backbone_lr=lr, final_backbone_lr=lr / 10)
        assert optimizer.param_groups[0]['lr'] == (lr if epoch < 100 else lr / 10)
    for invalid in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            set_epoch_lrs(optimizer, 0, backbone_lr=lr, final_backbone_lr=invalid)


def test_plan_is_exactly_two_fresh_seed66_flat_jobs(tmp_path):
    module = importlib.import_module('experiments.osram_mosi_mid_lr_20261021.run')
    jobs = module.plan_jobs(tmp_path, 'sha', ['GPU-good'])
    assert len(jobs) == 2
    assert {j['backbone_lr'] for j in jobs} == {5e-4, 3e-4}
    assert len({j['output'] for j in jobs}) == 2
    for j in jobs:
        assert (j['dataset'], j['mode'], j['seed'], j['phase']) == ('CMUMOSI', 'flat', 66, 'train')
        assert j['command'][2] == 'experiments.osram_mosi_mid_lr_20261021.worker'
        assert float(j['command'][j['command'].index('--backbone-lr') + 1]) == j['backbone_lr']
        assert '--resume' not in j['command']
    with pytest.raises(ValueError):
        module.plan_jobs(tmp_path, 'sha', ['GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'])


@pytest.mark.parametrize('lr', [5e-4, 3e-4])
def test_worker_routes_only_approved_lr_and_flat_mode(monkeypatch, lr):
    module = importlib.import_module('experiments.osram_mosi_mid_lr_20261021.worker')
    captured = {}
    def capture(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
    monkeypatch.setattr(module.common, 'main', capture)
    module.main(['--backbone-lr', str(lr), '--mode', 'flat'])
    assert captured['backbone_lr'] == lr
    assert captured['final_backbone_lr'] == lr / 10
    assert captured['allowed_modes'] == ('flat',)
    assert captured['argv'] == ['--mode', 'flat']
    with pytest.raises(SystemExit):
        module.main(['--backbone-lr', '0.001'])


def test_report_refuses_unfinished_runs(tmp_path):
    from experiments.osram_mosi_mid_lr_20261021.report import write_report
    with pytest.raises(ValueError, match='two completed'):
        write_report(tmp_path, {'jobs': []})
