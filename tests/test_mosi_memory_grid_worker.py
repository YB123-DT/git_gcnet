from dataclasses import asdict

import pytest
import torch


def reference_config():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    return TrainConfig(dataset='CMUMOSI',fold=1,seed=66,epochs=150,learning_rate=3e-4,
        batch_size=32,weight_decay=1e-5,optimizer='adam',lr_schedule='constant',
        backbone_type='osram',training_objective='emotion-only',disable_unused_aux_modules=True,
        missing_mask_protocol='gcnet-random-mask-v1',checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1',evaluate_test=True,osram_output_dim=1600,
        osram_num_heads=8,osram_key_dim=64,osram_value_dim=64,osram_write_step=.6,
        osram_bidirectional=False,train_rate_mode='cyclic',task_regression_loss='mse')


def test_two_five_point_sweeps_share_exactly_one_fresh_control():
    from experiments.osram_mosi_memory_grid_20261022.config import MODES,FAMILIES
    assert len(MODES)==9
    assert set(FAMILIES)=={'write_step','head_dim'}
    assert all(len(v)==5 for v in FAMILIES.values())
    assert set(FAMILIES['write_step']) & set(FAMILIES['head_dim'])=={'baseline'}
    assert [MODES[m][0] for m in FAMILIES['write_step']]==[.2,.4,.6,.8,1.]
    assert [MODES[m][1] for m in FAMILIES['head_dim']]==[32,48,64,96,128]


def test_only_write_strength_or_both_head_dimensions_change():
    from experiments.osram_mosi_memory_grid_20261022.config import MODES,candidate_config
    before=reference_config()
    for mode,(step,dim) in MODES.items():
        cfg=candidate_config(before,mode)
        changed={k for k,v in asdict(before).items() if v!=asdict(cfg)[k]}
        expected=({'osram_write_step'} if step!=.6 else set()) | (
            {'osram_key_dim','osram_value_dim'} if dim!=64 else set())
        assert changed==expected
        assert cfg.osram_num_heads==8 and cfg.osram_output_dim==1600
        assert cfg.osram_adapter_hidden_dim==0 and not cfg.osram_source_role_v3
        assert cfg.missing_mask_protocol=='gcnet-random-mask-v1'


def test_config_rejects_unapproved_backbone_or_mode():
    from dataclasses import replace
    from experiments.osram_mosi_memory_grid_20261022.config import candidate_config
    for cfg in (replace(reference_config(),learning_rate=1e-3),
                replace(reference_config(),osram_source_role_v3=True),
                replace(reference_config(),missing_mask_protocol='conversation-mask-v2')):
        with pytest.raises(ValueError):
            candidate_config(cfg,'baseline')
    with pytest.raises(ValueError):
        candidate_config(reference_config(),'head_256')


def test_compatible_fresh_initialization_copies_shared_shapes_only():
    from experiments.osram_mosi_memory_grid_20261022.worker import align_fresh_initialization
    source=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.Linear(3,2))
    target=torch.nn.Sequential(torch.nn.Linear(5,3),torch.nn.Linear(3,2))
    changed=target[0].weight.detach().clone()
    proof=align_fresh_initialization(source,target)
    assert torch.equal(target[0].weight,changed)
    assert torch.equal(target[0].bias,source[0].bias)
    assert torch.equal(target[1].weight,source[1].weight)
    assert set(proof['shape_changed'])=={'0.weight'}
    assert proof['same_shape_parameters_exact']


def test_existing_flat_build_accepts_larger_memory_without_new_readout():
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_mosi_memory_grid_20261022.config import candidate_config
    cfg=candidate_config(reference_config(),'head_128')
    model=_build_model(cfg,(4,5,6),torch.device('cpu'))
    assert model.osram.context_dim==2048
    assert model.osram.emotion_adapter[1].in_features==256+4*2048
    assert model.osram.emotion_adapter[1].out_features==1600
    assert model.smax_fc.in_features==1600
