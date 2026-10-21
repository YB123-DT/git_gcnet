from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from experiments.osram_source_role_v3_20261021.worker import (
    source_catalog, validate_reference_config, validate_gradients,
    settings, set_learning_rates, gradient_check_epoch,
)


def test_fixed_two_dataset_budgets():
    assert settings('CMUMOSI')['epochs'] == 150
    assert settings('CMUMOSEI')['epochs'] == 60
    optimizer = SimpleNamespace(param_groups=[dict(name='backbone', lr=0.)])
    set_learning_rates(optimizer, 'CMUMOSI', 99)
    assert optimizer.param_groups[0]['lr'] == 3e-4
    set_learning_rates(optimizer, 'CMUMOSI', 100)
    assert optimizer.param_groups[0]['lr'] == 3e-5
    with pytest.raises(ValueError):
        settings('MELD')


def test_mosi_zero_initialized_paths_get_four_real_updates_before_gradient_gate():
    assert gradient_check_epoch('CMUMOSI') == 1
    assert gradient_check_epoch('CMUMOSEI') == 0
    # Two MOSI batches per epoch; ordinary gradients accumulate across epochs.
    assert (gradient_check_epoch('CMUMOSI') + 1) * 2 == 4


def test_only_v3_flag_and_reviewed_disabled_schema_may_differ():
    @dataclass
    class Config:
        seed: int = 66
        osram_source_role: bool = False
        osram_source_role_mode: str = 'residual'
        osram_source_relation: bool = False
        osram_source_relation_layout: str = 'source-mean'
        osram_source_role_v3: bool = True
    cfg = Config()
    validate_reference_config(cfg, {'seed': 66})
    for wrong in (replace(cfg, seed=67), replace(cfg, osram_source_relation=True),
                  replace(cfg, osram_source_relation_layout='role-tokens')):
        with pytest.raises(ValueError):
            validate_reference_config(wrong, {'seed': 66})


def test_source_catalog_includes_v3_and_reused_numerical_helpers():
    paths = source_catalog(('gcnet_missing_m3/osram.py',))
    assert len(paths) == len(set(paths))
    assert 'gcnet_missing_m3/source_role_v3.py' in paths
    assert 'experiments/osram_source_relation_attention_20261021/worker.py' in paths
    assert 'experiments/osram_source_role_v3_20261021/run.py' in paths


def test_decoder_gradient_alone_is_not_success():
    with pytest.raises(RuntimeError):
        validate_gradients({'decoders.0.weight': True})
    required = ('memory_projection.weight', 'local_projection.weight', 'pair_mlp.1.weight',
        'source_embedding.weight', 'gap_role_embedding.weight',
        'attention_source_embedding.weight', 'attention_role_embedding.weight',
        'source_norm.weight', 'source_attention.in_proj_weight',
        'output_attention.in_proj_weight', 'role_queries.0.weight', 'decoders.0.weight',
        'input.source_base', 'input.source_gap')
    observed = dict.fromkeys(required, True)
    observed.update({f'{group}.{role}.weight': True
                     for group in ('role_queries', 'decoders') for role in range(4)})
    validate_gradients(observed)
    observed['decoders.3.weight'] = False
    with pytest.raises(RuntimeError):
        validate_gradients(observed)


def test_controller_only_schedules_two_v3_seed66_jobs(tmp_path):
    from experiments.osram_source_role_v3_20261021.run import worker_command
    args = SimpleNamespace(output=tmp_path, gpu_uuid='GPU-healthy', source_commit='fixed')
    command = worker_command(args, 'CMUMOSI', 'train')
    assert 'experiments.osram_source_role_v3_20261021.worker' in command
    assert command[command.index('--seed') + 1] == '66'
    assert '--layout' not in command
