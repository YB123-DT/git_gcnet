from argparse import Namespace
from dataclasses import asdict

import pytest

from test_v3_connection_screen import baseline_config


def test_exact_six_mosi_paired_jobs(tmp_path):
    from experiments.osram_v3_paired_interfaces_20261022.run import approved_jobs, worker_command
    jobs = approved_jobs()
    assert len(jobs) == 6
    assert {job['dataset'] for job in jobs} == {'CMUMOSI'}
    assert {job['seed'] for job in jobs} == {66}
    assert len({job['variant'] for job in jobs}) == 6
    args = Namespace(output=tmp_path, source_commit='immutable')
    for job in jobs:
        command = worker_command(args, job, 'train', 'GPU-healthy')
        assert 'experiments.osram_v3_paired_interfaces_20261022.worker' in command
    with pytest.raises(ValueError):
        worker_command(args, dict(jobs[0], dataset='CMUMOSEI'), 'train', 'GPU-healthy')


def test_each_pair_changes_only_variant():
    from experiments.osram_v3_paired_interfaces_20261022.worker import PAIRS, candidate_config
    cfg = baseline_config()
    for pair in PAIRS:
        a, b = [asdict(candidate_config(cfg, variant)) for variant in pair]
        assert {key for key in a if a[key] != b[key]} == {'osram_source_role_v3_variant'}
    with pytest.raises(ValueError):
        candidate_config(baseline_config('CMUMOSEI'), PAIRS[0][0])


def test_structurally_zero_qk_does_not_require_all_parameter_gradients():
    from experiments.osram_v3_paired_interfaces_20261022.worker import validate_gradients
    module = Namespace(expected_gradient_groups=lambda: ('projection.', 'decoders.'))
    observed = {'projection.weight': True, 'decoders.0.weight': True,
        'source_attention.in_proj_weight.QK': False,
        'input.source_base': True, 'input.source_gap': True}
    validate_gradients(module, observed)
    with pytest.raises(RuntimeError):
        validate_gradients(module, dict(observed, **{'input.source_gap': False}))


def test_pair_report_differences_are_b_minus_a():
    from experiments.osram_v3_paired_interfaces_20261022.report import paired_differences
    rows = [dict(dataset='CMUMOSI', model=model, scope='mean', ACC=acc, WF1=wf1, MSE=mse)
        for model, acc, wf1, mse in [('topology-full', 80., 79., 1.),
                                   ('topology-diagonal', 81., 79.5, .9)]]
    delta = paired_differences(rows)
    assert len(delta) == 1
    assert delta[0]['delta_ACC'] == 1.
    assert delta[0]['delta_WF1'] == .5
    assert delta[0]['delta_MSE'] == pytest.approx(-.1)


def test_incomplete_results_are_excluded(tmp_path):
    from experiments.osram_v3_paired_interfaces_20261022.report import summarize
    report = summarize(tmp_path)
    assert not report['aggregate'] and not report['pair_differences']
    assert len(report['states']) == 6


def test_fixed_schedule_and_resource_limit(monkeypatch):
    from experiments.osram_v3_paired_interfaces_20261022.worker import settings, set_learning_rates
    from experiments.osram_v3_paired_interfaces_20261022.run import gpu_free, select_gpu
    assert settings('CMUMOSI') == dict(epochs=150, first_lr=3e-4, final_lr=3e-5)
    optimizer = Namespace(param_groups=[dict(name='backbone', lr=0.)])
    for epoch, expected in [(0, 3e-4), (99, 3e-4), (100, 3e-5), (149, 3e-5)]:
        set_learning_rates(optimizer, 'CMUMOSI', epoch)
        assert optimizer.param_groups[0]['lr'] == expected
    resources = {'GPU-healthy': dict(free_mib=20000, peers=[(2000, 3000)] * 3)}
    assert select_gpu(1000, resources, per_gpu=3) is None
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k: '4, GPU-renamed, 40000\n')
    with pytest.raises(ValueError):
        gpu_free('GPU-renamed')
