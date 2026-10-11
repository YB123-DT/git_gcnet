"""Contract for the approved Marginal seed67 continuation screen."""

from dataclasses import asdict
import importlib
import json
from pathlib import Path


def runner():
    return importlib.import_module(
        'experiments.osram_counterfactual_readouts_20261011.run_seed67')


def reference():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    path = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_67/config.json')
    return asdict(TrainConfig(**json.loads(path.read_text())))


def test_seed67_candidate_changes_only_marginal_readout():
    base = reference()
    candidate = runner().candidate_config(base)
    assert {key: (base[key], value) for key, value in candidate.items()
            if value != base[key]} == {
                'osram_counterfactual_readout': ('none', 'marginal')}


def test_seed67_task_and_schedule_are_fixed():
    assert runner().specs() == [dict(
        id='marginal_seed67', dataset='CMUMOSI', model='marginal', seed=67, fold=1)]
    assert runner().schedule() == ((100, .001), (150, .0001))

