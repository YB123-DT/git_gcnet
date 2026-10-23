from argparse import Namespace
from dataclasses import asdict, replace
import json

import pytest

from test_v3_connection_screen import baseline_config


def test_exact_four_reread_jobs_and_immutable_command(tmp_path):
    from experiments.osram_conditional_reread_20261023.run import approved_jobs, worker_command
    jobs = approved_jobs()
    assert {(j['dataset'], j['variant'], j['seed']) for j in jobs} == {
        (dataset, mode, 66) for dataset in ('CMUMOSI', 'CMUMOSEI') for mode in ('free', 'span')}
    args = Namespace(output=tmp_path, source_commit='immutable')
    for job in jobs:
        command = worker_command(args, job, 'train', 'GPU-healthy')
        assert 'experiments.osram_conditional_reread_20261023.worker' in command
        assert command[command.index('--output') + 1] == str(tmp_path / 'runs' / job['id'])
        assert worker_command(args, job, 'train', 'GPU-healthy', resume=True)[-1] == '--resume'
    with pytest.raises(ValueError):
        worker_command(args, dict(jobs[0], seed=67), 'train', 'GPU-healthy')


@pytest.mark.parametrize('dataset', ['CMUMOSI', 'CMUMOSEI'])
def test_only_reread_switch_and_mode_change(dataset):
    from experiments.osram_conditional_reread_20261023.worker import candidate_config, validate_reference_config
    cfg = baseline_config(dataset)
    for mode in ('free', 'span'):
        candidate = candidate_config(cfg, mode)
        assert {name for name, value in asdict(cfg).items() if asdict(candidate)[name] != value} == (
            {'osram_conditional_reread'} if mode == 'free' else {'osram_conditional_reread', 'osram_reread_mode'})
        assert not candidate.osram_source_role_v3 and not candidate.osram_tasr
        prior = json.loads(json.dumps(asdict(cfg)))
        from experiments.osram_conditional_reread_20261023.worker import REVIEWED_SCHEMA_DEFAULTS
        for name in REVIEWED_SCHEMA_DEFAULTS:
            prior.pop(name, None)
        validate_reference_config(candidate, prior)
        with pytest.raises(ValueError):
            validate_reference_config(candidate, dict(prior, batch_size=16))
    with pytest.raises(ValueError):
        candidate_config(replace(cfg, osram_source_role_v3=True), 'free')
    with pytest.raises(ValueError):
        candidate_config(replace(cfg, osram_reread_rank_rtol=.1), 'span')


def test_schedule_and_four_actual_updates():
    from experiments.osram_conditional_reread_20261023.worker import settings, set_learning_rates, smoke_limits
    assert smoke_limits('CMUMOSI') == (2, 2)
    assert smoke_limits('CMUMOSEI') == (1, 4)
    assert settings('CMUMOSEI')['epochs'] == 60
    optimizer = Namespace(param_groups=[dict(name='backbone', lr=0.)])
    for epoch, expected in ((99, 3e-4), (100, 3e-5), (149, 3e-5)):
        set_learning_rates(optimizer, 'CMUMOSI', epoch)
        assert optimizer.param_groups[0]['lr'] == expected


def test_real_query_and_output_gradient_contract():
    import torch
    from experiments.osram_conditional_reread_20261023.worker import ParameterGradientCapture, validate_gradients
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    module.expected_gradient_groups = lambda: ('0.', '1.')
    capture = ParameterGradientCapture(module)
    module(torch.ones(3, 2)).sum().backward()
    validate_gradients(module, capture.observed)
    capture.remove()
    assert not capture.handles
    with pytest.raises(RuntimeError):
        validate_gradients(module, {'1.weight': True})


def test_pair_effects_do_not_cross_datasets_or_rank_partial_runs(tmp_path):
    from experiments.osram_conditional_reread_20261023.report import paired_differences, summarize
    rows = [dict(dataset=dataset, model=model, scope='mean', ACC=value, WF1=value, MSE=1.)
            for dataset, model, value in [('CMUMOSI', 'free', 80.), ('CMUMOSI', 'span', 79.),
                                          ('CMUMOSEI', 'span', 83.)]]
    effects = paired_differences(rows)
    assert len(effects) == 1 and effects[0]['dataset'] == 'CMUMOSI'
    assert effects[0]['delta_WF1'] == 1.
    report = summarize(tmp_path)
    assert len(report['states']) == 4
    assert not report['aggregate'] and not report['pair_differences']


def test_health_and_concurrency_gate(monkeypatch):
    from experiments.osram_conditional_reread_20261023.run import gpu_free, select_gpu
    resources = {'GPU-healthy': dict(free_mib=20000, peers=[(2000, 3000)] * 3)}
    assert select_gpu(1000, resources, per_gpu=3) is None
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k: '4, GPU-renamed, 40000\n')
    with pytest.raises(ValueError):
        gpu_free('GPU-renamed')
