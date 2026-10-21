"""Lock a finite original-Nested rerun onto the current MOSEI protocol."""
from dataclasses import make_dataclass, asdict, replace
import importlib

import pytest


def worker():
    return importlib.import_module('experiments.osram_mosei_nested_gcnet_3seed_20261021.worker')


def config():
    defaults = dict(dataset='CMUMOSEI', seed=66, fold=1, epochs=60, batch_size=32,
        learning_rate=.001, weight_decay=1e-5, optimizer='adam', lr_schedule='constant',
        missing_mask_protocol='gcnet-random-mask-v1', training_objective='emotion-only',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
        evaluate_test=True, osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64,
        osram_value_dim=64, osram_write_step=.6, osram_bidirectional=False,
        osram_readout_fusion='flat', osram_meaningful_block='none',
        osram_source_role=False, osram_source_role_mode='residual', osram_tasr=False,
        osram_source_relation=False, osram_source_relation_layout='source-mean',
        osram_source_tracking=False, paired_history_views=False, completion_path='none',
        task_regression_loss='mse', mosi_task_mode='regression',
        train_rate_mode='cyclic', train_missing_rates=tuple(i/10 for i in range(8)))
    cls = make_dataclass('Config', [(k, type(v)) for k, v in defaults.items()])
    return cls(**defaults)


def test_original_nested_changes_only_seed_and_module_switch():
    w = worker(); original = config()
    for seed in (66, 67, 68):
        for mode in ('flat', 'nested'):
            cfg = w.candidate_config(original, seed, mode)
            expected = asdict(original)
            expected.update(seed=seed, osram_meaningful_block=(w.METHOD if mode=='nested' else 'none'))
            assert asdict(cfg) == expected
    assert w.METHOD == 'nested_gnn_rooted_evidence'


@pytest.mark.parametrize('changes', [dict(learning_rate=1e-4), dict(epochs=100),
    dict(missing_mask_protocol='conversation-mask-v2'), dict(osram_source_relation=True),
    dict(osram_source_role=True), dict(paired_history_views=True),
    dict(osram_meaningful_block='mosei_nested_dim32_depth2'), dict(osram_bidirectional=True),
    dict(checkpoint_selection='validation')])
def test_reject_mismatched_baselines_and_unrequested_methods(changes):
    with pytest.raises(ValueError): worker().candidate_config(replace(config(), **changes), 66, 'nested')


def test_reference_config_allows_only_reviewed_default_schema_and_nested_switch():
    w = worker(); cfg = w.candidate_config(config(), 67, 'nested')
    old = asdict(cfg); old['osram_meaningful_block'] = 'none'
    for key in ('osram_source_role', 'osram_source_role_mode', 'osram_source_relation',
                'osram_source_relation_layout'):
        old.pop(key)
    w.validate_flat_config(cfg, old)
    old['learning_rate'] = 1e-4
    with pytest.raises(ValueError): w.validate_flat_config(cfg, old)


def test_reference_mapping_does_not_mislabel_direct_or_low_lr_models_as_flat():
    w = worker()
    assert '/seed_66/flat' in str(w.FLAT66)
    for seed in (67, 68):
        assert w.mask_reference(seed).name == 'direct'
        assert f'seed_{seed}' in str(w.mask_reference(seed))
    with pytest.raises(ValueError): w.mask_reference(65)
    for seed, mode in [(65, 'nested'), (66, 'rootaware')]:
        with pytest.raises(ValueError): w.candidate_config(config(), seed, mode)


def test_gradient_validation_requires_graph_not_only_zero_initialized_decoders():
    w = worker()
    with pytest.raises(RuntimeError): w.validate_nested_gradients({'local_decoder.weight': True})
    valid = {name: True for name in ['core.layers.0.0.weight', 'core.pool.0.weight',
        'core.readout.0.weight', 'tokenizer.local.weight', 'local_decoder.weight',
        'memory_decoders.0.weight']}
    w.validate_nested_gradients(valid)


def test_argument_surface_is_bounded_to_original_three_seeds_and_two_modes():
    p = worker().argument_parser()
    args = p.parse_args(['--phase','train','--mode','nested','--seed','68',
        '--gpu-uuid','GPU-health','--source-commit','sha','--output','/tmp/unused'])
    assert args.seed == 68 and args.mode == 'nested'
    with pytest.raises(SystemExit):
        p.parse_args(['--phase','train','--mode','direct','--seed','68',
            '--gpu-uuid','GPU-health','--source-commit','sha','--output','/tmp/unused'])
