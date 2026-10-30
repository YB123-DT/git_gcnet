"""Predeclared, capacity-matched CAM task-path ablations (internal screen)."""
from __future__ import annotations

import math
from pathlib import Path

RATES = tuple(f'{i / 10:.1f}' for i in range(8))
VARIANTS = {
    'A0': dict(osram_emotion_ablation='local-only', osram_gap_read='residual'),
    'A1': dict(osram_emotion_ablation='local-base', osram_gap_read='residual'),
    'A2': dict(osram_emotion_ablation='full', osram_gap_read='raw'),
    'A3': dict(osram_emotion_ablation='full', osram_gap_read='residual'),
}
LABEL = 'INTERNAL TEST-ORACLE SCREEN; per-rate TEST W-F1; not a single checkpoint score'
PYTHON = Path('/data2/yb/conda_envs/cmerc-repro/bin/python')
PARENT = Path('/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/'
              'execution_v1/runs/CMUMOSEI/fold_1/seed_66/none')
PARENT_CONFIG_SHA256 = 'cbe01df5e047fa01c35d34496d9a2881a794f5cdf71e8dff71cc31a29a977469'
INITIAL_SHA256 = '9a6a03edc69f7bd45c04464be8105eac96ec01d9a0922bbd2eb496a69ce82f77'
MASKS = Path('/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/'
             'execution_v1/runs/CMUMOSEI/seed_66/flat/fixed_masks.pt')
INPUTS = Path('/data2/yb/remote_experiments/osram_nested_cross_dataset_20261006/INPUTS.json')
INPUTS_SHA256 = 'fd03c4a6ca12abe8a55bde88531b7103761c4ed263c9fe381e30809aabd2a46c'
FORBIDDEN_GPU = 'GPU-46fb379f-cc90-dc82-9b5e-5d011f552264'

# These are invariants, not a new set of defaults. The complete accepted config
# is read from its hash-verified original file, then exactly two task switches
# are allowed to differ between A0--A3.
REQUIRED_CONFIG = dict(dataset='CMUMOSEI', fold=1, seed=66, epochs=60,
    batch_size=32, learning_rate=1e-4, weight_decay=1e-5, optimizer='adam',
    lr_schedule='constant', dropout=.5, projector_dropout=.1, latent_dim=256,
    backbone_type='osram', training_objective='emotion-only', train_rate_mode='cyclic',
    checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
    evaluate_test=True, missing_mask_protocol='gcnet-random-mask-v1',
    evaluation_protocol='official', mosi_task_mode='regression', task_regression_loss='mse',
    osram_output_dim=1600, osram_adapter_hidden_dim=0, osram_num_heads=8,
    osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
    osram_bidirectional=False, osram_readout_fusion='flat', osram_ablation='full',
    osram_meaningful_block='none', osram_tasr=False, osram_source_role=False,
    osram_source_role_v3=False, osram_source_relation=False, osram_conditional_reread=False,
    osram_source_tracking=False, osram_disable_local_skip=False, osram_forward_slot_reuse=False,
    completion_path='none', classification_completion=False, paired_history_views=False,
    disable_unused_aux_modules=True, simple_regression_predictor=False,
    initial_backbone_checkpoint=None, beta_mode='embedded')


def variant_config(raw, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown ablation {variant}')
    for key, expected in REQUIRED_CONFIG.items():
        if key not in raw or raw[key] != expected:
            raise ValueError(f'Accepted baseline differs: {key}')
    if raw.get('osram_emotion_ablation') != 'full' or raw.get('osram_gap_read') != 'residual':
        raise ValueError('Input must be the unablated original Flat configuration')
    if raw.get('gap_residual_strength') != 1.0:
        raise ValueError('Accepted baseline differs: gap_residual_strength')
    if tuple(raw.get('train_missing_rates', ())) != tuple(float(r) for r in RATES):
        raise ValueError('Accepted baseline differs: train_missing_rates')
    return dict(raw, **VARIANTS[variant])


def is_improvement(metrics, old):
    if any(not math.isfinite(float(metrics[k])) for k in ('WF1', 'MSE')):
        raise ValueError('Selection metrics must be finite')
    return old is None or metrics['WF1'] > old['WF1']


def aggregate(evaluated):
    if set(evaluated) != set(RATES):
        raise ValueError('Eight per-rate results required')
    return {scope: {key: sum(evaluated[r][key] for r in rates) / len(rates)
                    for key in ('ACC', 'WF1', 'MSE')}
            for scope, rates in (('mean', RATES), ('high', RATES[5:]))}
