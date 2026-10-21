from dataclasses import dataclass, replace
import math

import pytest

from experiments.osram_mosi_fixed_rate_seed66_20261021.worker import (
    fixed_config, rate_key, strict_improvement, validate_reference_config,
)


@dataclass
class Config:
    seed: int = 66
    train_rate_mode: str = 'cyclic'
    fixed_missing_rate: float | None = None
    train_missing_rates: tuple = tuple(i / 10 for i in range(8))
    learning_rate: float = 3e-4
    epochs: int = 150
    batch_size: int = 32
    weight_decay: float = 1e-5
    training_objective: str = 'emotion-only'
    missing_mask_protocol: str = 'gcnet-random-mask-v1'
    osram_readout_fusion: str = 'flat'
    osram_meaningful_block: str = 'none'
    osram_source_role: bool = False
    osram_source_role_mode: str = 'residual'
    osram_source_relation: bool = False
    osram_source_relation_layout: str = 'source-mean'
    osram_source_role_v3: bool = False
    osram_tasr: bool = False
    paired_history_views: bool = False
    osram_relation_block: bool = False


@pytest.mark.parametrize('rate', [i / 10 for i in range(8)])
def test_only_fixed_rate_fields_change(rate):
    base = Config(); cfg = fixed_config(base, rate)
    assert cfg == replace(base, train_rate_mode='fixed', fixed_missing_rate=rate,
                          train_missing_rates=(rate,))
    assert base.train_rate_mode == 'cyclic'


@pytest.mark.parametrize('rate', [True, False, -0.1, 0.8, 0.11, math.nan, math.inf])
def test_illegal_rate_rejected(rate):
    with pytest.raises(ValueError):
        rate_key(rate)


def test_no_seed_mask_architecture_or_budget_expansion():
    for cfg in (replace(Config(), seed=67), replace(Config(), epochs=100),
                replace(Config(), missing_mask_protocol='conversation-mask-v2'),
                replace(Config(), osram_source_role_v3=True),
                replace(Config(), osram_meaningful_block='nested_gnn_rooted_evidence'),
                replace(Config(), learning_rate=1e-3)):
        with pytest.raises(ValueError):
            fixed_config(cfg, .3)


def test_reference_normalizes_only_reviewed_disabled_schema():
    from dataclasses import asdict
    old = asdict(Config())
    for field in ('osram_source_role', 'osram_source_role_mode', 'osram_source_relation',
                  'osram_source_relation_layout', 'osram_source_role_v3'):
        old.pop(field)
    validate_reference_config(fixed_config(Config(), .3), old, .3)
    with pytest.raises(ValueError):
        validate_reference_config(replace(fixed_config(Config(), .3), weight_decay=2e-4), old, .3)


def test_selection_is_same_rate_strict_wf1_not_mse_or_acc():
    assert strict_improvement(None, dict(ACC=60., WF1=60., MSE=2.))
    old = dict(metrics=dict(ACC=60., WF1=60., MSE=2.))
    assert not strict_improvement(old, dict(ACC=70., WF1=60., MSE=1.))
    assert not strict_improvement(old, dict(ACC=70., WF1=59., MSE=1.))
    assert strict_improvement(old, dict(ACC=50., WF1=61., MSE=3.))
    with pytest.raises(ValueError):
        strict_improvement(old, dict(ACC=60., WF1=math.nan, MSE=1.))
