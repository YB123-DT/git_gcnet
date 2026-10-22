from dataclasses import asdict, replace

import pytest


def baseline_config():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    return TrainConfig(dataset='CMUMOSI', fold=1, backbone_type='osram',
        osram_bidirectional=False, osram_write_step=.6, epochs=150,
        learning_rate=3e-4, missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1', evaluate_test=True,
        training_objective='emotion-only', osram_output_dim=1600,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64)


def test_six_capacity_arms_change_only_authorized_widths():
    from experiments.osram_mosi_v3_capacity_20261022.worker import MODES, candidate_config
    assert MODES == {'f512_plain': (512, 0, 0), 'f512_v128': (512, 128, 256),
        'f512_v256': (512, 256, 512), 'f1024_plain': (1024, 0, 0),
        'f1024_v128': (1024, 128, 256), 'f1024_v256': (1024, 256, 512)}
    cfg = baseline_config()
    allowed = {'osram_adapter_hidden_dim', 'osram_source_role_v3',
               'osram_source_role_v3_dim', 'osram_source_role_v3_pair_hidden'}
    for mode, (flat, width, pair) in MODES.items():
        candidate = candidate_config(cfg, mode)
        assert candidate.osram_adapter_hidden_dim == flat
        assert candidate.osram_source_role_v3 == bool(width)
        if width:
            assert (candidate.osram_source_role_v3_dim,
                    candidate.osram_source_role_v3_pair_hidden) == (width, pair)
        assert {k for k, v in asdict(cfg).items() if v != asdict(candidate)[k]} <= allowed


def test_capacity_rejects_unapproved_protocols_and_modes():
    from experiments.osram_mosi_v3_capacity_20261022.worker import candidate_config
    cfg = baseline_config()
    changes = dict(dataset='CMUMOSEI', seed=67, epochs=60, learning_rate=1e-3,
        weight_decay=0., batch_size=16, optimizer='adamw', lr_schedule='cosine',
        missing_mask_protocol='conversation-mask-v2', training_objective='joint',
        osram_output_dim=512, osram_bidirectional=True, osram_write_step=1.,
        osram_source_role_v3=True, osram_source_role=True, osram_source_relation=True,
        osram_tasr=True, osram_adapter_hidden_dim=512, train_rate_mode='fixed',
        fixed_missing_rate=.3, train_missing_rates=(.1, .2), task_regression_loss='smooth_l1',
        checkpoint_selection='validation', checkpoint_selection_metric='accuracy',
        evaluate_test=False)
    for key, value in changes.items():
        with pytest.raises(ValueError):
            candidate_config(replace(cfg, **{key: value}), 'f512_v128')
    with pytest.raises(ValueError, match='approved'):
        candidate_config(cfg, 'wide_v3')


def test_mosi_schedule_and_smoke_keep_original_update_protocol():
    from experiments.osram_mosi_v3_capacity_20261022.worker import (
        smoke_epochs, settings, set_learning_rates,
    )
    assert settings('CMUMOSI') == dict(epochs=150, first_lr=3e-4, final_lr=3e-5)
    assert smoke_epochs() == 2  # Two genuine batch32 updates per epoch, four total.
    optimizer = type('Optimizer', (), {'param_groups': [{'name': 'backbone', 'lr': 0.}]})()
    for epoch, expected in ((0, 3e-4), (99, 3e-4), (100, 3e-5), (149, 3e-5)):
        set_learning_rates(optimizer, 'CMUMOSI', epoch)
        assert optimizer.param_groups[0]['lr'] == expected
    with pytest.raises(ValueError):
        set_learning_rates(optimizer, 'CMUMOSI', 150)


def test_gradient_instrumentation_accumulates_then_removes_hooks():
    import torch
    from experiments.osram_mosi_v3_capacity_20261022.worker import GradientCapture
    module = torch.nn.Linear(2, 1)
    capture = GradientCapture(module)
    module(torch.ones(1, 2)).sum().backward()
    assert capture.observed['weight'] and capture.observed['bias']
    module.zero_grad(set_to_none=True)
    module(torch.zeros(1, 2)).sum().backward()
    assert capture.observed['weight']  # An epoch with zero weight gradient cannot erase proof.
    capture.remove()
    assert not module._forward_pre_hooks
    assert all(not parameter._backward_hooks for parameter in module.parameters())
