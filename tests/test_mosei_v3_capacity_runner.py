from dataclasses import asdict

import pytest


def test_capacity_config_changes_only_authorized_widths():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_mosei_v3_capacity_20261022.worker import candidate_config
    cfg = TrainConfig(dataset='CMUMOSEI', fold=1, backbone_type='osram', osram_bidirectional=False, osram_write_step=.6,
                      epochs=60, missing_mask_protocol='gcnet-random-mask-v1',
                      checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
                      evaluate_test=True, training_objective='emotion-only',
                      osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64)
    plain = candidate_config(cfg, 'small_flat')
    wide = candidate_config(cfg, 'wide_v3')
    assert plain.osram_adapter_hidden_dim == wide.osram_adapter_hidden_dim == 1024
    assert not plain.osram_source_role_v3 and wide.osram_source_role_v3
    assert (wide.osram_source_role_v3_dim, wide.osram_source_role_v3_pair_hidden) == (256, 512)
    allowed = {'osram_adapter_hidden_dim', 'osram_source_role_v3',
               'osram_source_role_v3_dim', 'osram_source_role_v3_pair_hidden'}
    for new in (plain, wide):
        assert {k for k, v in asdict(cfg).items() if v != asdict(new)[k]} <= allowed
        assert new.osram_output_dim == 1600 and new.learning_rate == .001


def test_capacity_config_rejects_old_missing_other_dataset_and_extra_modules():
    from dataclasses import replace
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_mosei_v3_capacity_20261022.worker import candidate_config
    cfg = TrainConfig(dataset='CMUMOSEI', fold=1, backbone_type='osram', osram_bidirectional=False, osram_write_step=.6,
                      osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
                      epochs=60, missing_mask_protocol='gcnet-random-mask-v1', training_objective='emotion-only')
    for changed in (replace(cfg, missing_mask_protocol='conversation-mask-v2'),
                    replace(cfg, dataset='CMUMOSI'), replace(cfg, learning_rate=1e-4),
                    replace(cfg, osram_source_role_v3=True)):
        with pytest.raises(ValueError):
            candidate_config(changed, 'wide_v3')


def test_queue_limits_to_two_modes_and_two_jobs():
    from argparse import Namespace
    from pathlib import Path
    from experiments.osram_mosei_v3_capacity_20261022.run import worker_command
    args = Namespace(output=Path('/test'), gpu_uuid='GPU-test', source_commit='abc')
    command = worker_command(args, 'wide_v3', 'train')
    assert command[command.index('--mode') + 1] == 'wide_v3'
    assert command[command.index('--seed') + 1] == '66'
    assert '/test/runs/wide_v3' in command
    with pytest.raises(ValueError):
        worker_command(args, 'nested', 'train')
