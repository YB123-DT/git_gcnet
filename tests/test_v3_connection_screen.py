from argparse import Namespace
from dataclasses import asdict, replace
import json

import pytest


def baseline_config(dataset='CMUMOSI'):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    return TrainConfig(dataset=dataset, fold=1, seed=66, backbone_type='osram',
        osram_bidirectional=False, osram_write_step=.6,
        epochs=150 if dataset == 'CMUMOSI' else 60,
        learning_rate=3e-4 if dataset == 'CMUMOSI' else 1e-3,
        missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1', evaluate_test=True,
        training_objective='emotion-only', osram_output_dim=1600,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64)


def test_exact_six_jobs_only_seed66_and_three_connections(tmp_path):
    from experiments.osram_v3_connection_screen_20261022.run import approved_jobs, worker_command
    jobs = approved_jobs()
    assert len(jobs) == 6
    assert {(j['dataset'], j['variant'], j['seed']) for j in jobs} == {
        (dataset, variant, 66) for dataset in ('CMUMOSI', 'CMUMOSEI')
        for variant in ('head-preserving', 'source-slots', 'flat-hidden')}
    assert len({j['id'] for j in jobs}) == 6
    args = Namespace(output=tmp_path, source_commit='fixed')
    for job in jobs:
        command = worker_command(args, job, 'train', 'GPU-healthy')
        for option, expected in (('--seed', '66'), ('--dataset', job['dataset']),
                                 ('--variant', job['variant']), ('--gpu-uuid', 'GPU-healthy')):
            assert command[command.index(option) + 1] == expected
        assert command[command.index('--output') + 1] == str(tmp_path / 'runs' / job['id'])
        assert '--resume' not in command
        assert worker_command(args, job, 'train', 'GPU-healthy', resume=True)[-1] == '--resume'
    with pytest.raises(ValueError):
        worker_command(args, dict(jobs[0], seed=67), 'train', 'GPU-healthy')
    with pytest.raises(ValueError):
        worker_command(args, dict(jobs[0], variant='standard'), 'train', 'GPU-healthy')


@pytest.mark.parametrize('dataset', ['CMUMOSI', 'CMUMOSEI'])
def test_only_enabled_flag_and_variant_change(dataset):
    from experiments.osram_v3_connection_screen_20261022.worker import candidate_config, VARIANTS
    cfg = baseline_config(dataset)
    for variant in VARIANTS:
        candidate = candidate_config(cfg, variant)
        assert {name for name, value in asdict(cfg).items()
                if asdict(candidate)[name] != value} == {
                    'osram_source_role_v3', 'osram_source_role_v3_variant'}
        assert candidate.osram_source_role_v3_mode == 'residual'
        assert candidate.osram_source_role_v3_dim == 64
        assert candidate.osram_source_role_v3_pair_hidden == 128
        assert candidate.osram_adapter_hidden_dim == 0


def test_protocol_changes_are_rejected():
    from experiments.osram_v3_connection_screen_20261022.worker import candidate_config
    cfg = baseline_config()
    forbidden = dict(seed=67, epochs=100, learning_rate=1e-3, weight_decay=0.,
        batch_size=16, optimizer='adamw', lr_schedule='cosine',
        missing_mask_protocol='conversation-mask-v2', training_objective='joint',
        osram_output_dim=512, osram_bidirectional=True, osram_write_step=1.,
        osram_source_role_v3=True, osram_source_role=True, osram_source_relation=True,
        osram_tasr=True, osram_adapter_hidden_dim=512, train_rate_mode='fixed',
        fixed_missing_rate=.3, train_missing_rates=(.1, .2),
        checkpoint_selection='validation', checkpoint_selection_metric='accuracy',
        evaluate_test=False, osram_source_role_v3_dim=128,
        osram_source_role_v3_pair_hidden=256)
    for name, value in forbidden.items():
        with pytest.raises(ValueError):
            candidate_config(replace(cfg, **{name: value}), 'head-preserving')
    with pytest.raises(ValueError):
        candidate_config(cfg, 'standard')


def test_reference_schema_migrations_are_explicit_and_bounded():
    from experiments.osram_v3_connection_screen_20261022.worker import (
        candidate_config, validate_reference_config, REVIEWED_SCHEMA_DEFAULTS)
    baseline = baseline_config()
    prior = json.loads(json.dumps(asdict(baseline)))
    for name in REVIEWED_SCHEMA_DEFAULTS:
        prior.pop(name, None)
    cfg = candidate_config(baseline, 'source-slots')
    validate_reference_config(cfg, prior)
    altered = dict(prior, learning_rate=1e-3)
    with pytest.raises(ValueError):
        validate_reference_config(cfg, altered)
    # An unrelated missing field must not silently acquire a dataclass default.
    prior.pop('gradient_clip_norm')
    with pytest.raises(ValueError):
        validate_reference_config(cfg, prior)


def test_immutable_learning_rates_and_four_update_smokes():
    from experiments.osram_v3_connection_screen_20261022.worker import settings, set_learning_rates, smoke_limits
    assert settings('CMUMOSI') == dict(epochs=150, first_lr=3e-4, final_lr=3e-5)
    assert settings('CMUMOSEI') == dict(epochs=60, first_lr=1e-3, final_lr=1e-3)
    assert smoke_limits('CMUMOSI') == (2, 2)
    assert smoke_limits('CMUMOSEI') == (1, 4)
    optimizer = Namespace(param_groups=[dict(name='backbone', lr=0.)])
    for dataset, epoch, expected in [('CMUMOSI', 0, 3e-4), ('CMUMOSI', 99, 3e-4),
            ('CMUMOSI', 100, 3e-5), ('CMUMOSI', 149, 3e-5), ('CMUMOSEI', 59, 1e-3)]:
        set_learning_rates(optimizer, dataset, epoch)
        assert optimizer.param_groups[0]['lr'] == expected
    with pytest.raises(ValueError):
        set_learning_rates(optimizer, 'CMUMOSEI', 60)


def test_bad_physical_gpu_and_uuid_rejected(monkeypatch):
    from experiments.osram_v3_connection_screen_20261022 import run
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k: '4, GPU-renamed, 40000\n')
    with pytest.raises(ValueError):
        run.gpu_free('GPU-renamed')
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k:
        '1, GPU-46fb379f-cc90-dc82-9b5e-5d011f552264, 40000\n')
    with pytest.raises(ValueError):
        run.gpu_free('GPU-46fb379f-cc90-dc82-9b5e-5d011f552264')


def test_admission_reserves_peer_margin_without_double_counting():
    from experiments.osram_v3_connection_screen_20261022.run import admission_required, select_gpu
    assert admission_required(1000, []) == 3500
    assert admission_required(1000, [(3000, 4500)]) == 3500
    assert admission_required(1000, [(3000, 1000)]) == 7000
    resources = {'GPU-a': dict(free_mib=12000, peers=[(2000, 3000)]),
                 'GPU-b': dict(free_mib=12000, peers=[])}
    assert select_gpu(1000, resources, per_gpu=6) == 'GPU-b'
    assert select_gpu(10000, resources, per_gpu=6) is None
    assert select_gpu(1000, resources, per_gpu=1) == 'GPU-b'
    with pytest.raises(ValueError):
        select_gpu(1000, resources, per_gpu=7)


def test_gradient_gate_uses_actual_variant_groups_and_source_inputs():
    from experiments.osram_v3_connection_screen_20261022.worker import validate_gradients
    module = Namespace(expected_gradient_groups=lambda: ('projection.', 'hidden_projection.'))
    observed = {'projection.weight': True, 'hidden_projection.weight': True,
                'input.source_base': True, 'input.source_gap': True}
    validate_gradients(module, observed)
    for key in observed:
        with pytest.raises(RuntimeError):
            validate_gradients(module, dict(observed, **{key: False}))


def test_incomplete_runs_and_smokes_are_not_performance(tmp_path):
    from experiments.osram_v3_connection_screen_20261022.run import approved_jobs
    from experiments.osram_v3_connection_screen_20261022.report import summarize
    job = approved_jobs()[0]
    output = tmp_path / 'runs' / job['id']
    output.mkdir(parents=True)
    (output / 'SUMMARY.json').write_text(json.dumps(dict(status='running', epoch=1)))
    result = summarize(tmp_path)
    assert not result['per_rate'] and not result['aggregate']
    assert result['states'][job['id']]['status'] == 'running'


def test_final_epoch_resume_replaces_stale_summary_from_complete_checkpoint():
    from experiments.osram_v3_connection_screen_20261022.worker import sync_restored_summary
    summary = dict(status='running', epoch=149, best_by_rate={'0.0': {'epoch': 148}})
    best = {f'{index / 10:.1f}': {'epoch': 150} for index in range(8)}
    sync_restored_summary(summary, start=150, best=best)
    assert summary['epoch'] == 150
    assert summary['best_by_rate'] == best
    assert summary['status'] == 'running'  # Completion still needs artifact validation.
    # The resume loop is empty at the full budget; these values must survive it.
    assert not list(range(summary['epoch'], 150))
