"""Reproducibility contract for the seed66 history-recency screen."""

from dataclasses import asdict
import importlib
import json
from pathlib import Path


def runner():
    return importlib.import_module('experiments.osram_flat_history_recency_20261011.run')


def reference():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    path = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/config.json')
    return asdict(TrainConfig(**json.loads(path.read_text())))


def test_candidate_changes_only_history_recency():
    base = reference()
    candidate = runner().candidate_config(base)
    assert {key: (base[key], value) for key, value in candidate.items()
            if value != base[key]} == {'osram_history_recency': (False, True)}


def test_schedule_and_single_seed_are_fixed():
    assert runner().schedule() == ((100, .001), (150, .0001))
    assert runner().specs() == [dict(
        id='history_recency_seed66', dataset='CMUMOSI',
        model='history_recency', seed=66, fold=1)]


def test_candidate_rejects_non_cfg84_or_competing_method():
    base = reference()
    for field, value in (
        ('osram_write_step', .7),
        ('osram_counterfactual_readout', 'marginal'),
        ('paired_history_views', True),
    ):
        changed = dict(base, **{field: value})
        try:
            runner().candidate_config(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted competing/non-cfg84 field {field}')

