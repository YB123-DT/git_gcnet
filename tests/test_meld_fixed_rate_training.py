from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from gcnet_missing_m3 import train_gcnet as trainer


@pytest.mark.parametrize('rate', [i / 10 for i in range(8)])
def test_fixed_acc_selection_uses_exactly_one_protocol_rate(rate):
    cfg = trainer.TrainConfig(dataset='MELD', backbone_type='osram',
        training_objective='emotion-only', train_rate_mode='fixed', fixed_missing_rate=rate,
        train_missing_rates=(rate,), checkpoint_selection='test-oracle-per-rate',
        missing_mask_protocol='gcnet-random-mask-v1', preserve_selection_artifacts=True)
    assert trainer._protocol_rates(cfg) == (rate,)
    assert cfg.osram_readout_fusion == 'flat'
    assert trainer.TrainConfig(**asdict(cfg)) == cfg


def test_complete_state_fixed_passes_the_training_entry_guard(monkeypatch, tmp_path):
    class ReachedDataLoader(Exception):
        pass

    def loader(**kwargs):
        raise ReachedDataLoader

    monkeypatch.setattr(trainer, 'get_loaders', loader)
    cfg = trainer.TrainConfig(dataset='MELD', fold=1, backbone_type='osram', device='cpu',
        training_objective='emotion-only', train_rate_mode='fixed', fixed_missing_rate=.7,
        checkpoint_selection='test-oracle-per-rate')
    with pytest.raises(ReachedDataLoader):
        trainer.run_experiment(cfg, 'a', 't', 'v', tmp_path, training_state=object())


def test_fixed_runner_only_changes_training_rate_config_and_registers_eight_jobs(tmp_path):
    module_path = Path(__file__).parents[1] / 'experiments/osram_meld_fixed_rates_20261018/run.py'
    assert module_path.exists(), 'fixed-rate runner is not implemented'
    spec = importlib.util.spec_from_file_location('fixed_rate_runner', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = json.loads((module.ROOT / 'experiments/osram_meld_mmgcn_features_20261018/CONFIG.json').read_text())
    jobs = module.build_jobs(tmp_path)
    assert len(jobs) == 8 and len({j['output'] for j in jobs}) == 8
    for i, job in enumerate(jobs):
        cfg = module.make_config(i / 10)
        assert job['config'] == cfg
        expected = dict(base, train_rate_mode='fixed', fixed_missing_rate=i / 10,
                        train_missing_rates=[i / 10])
        assert cfg == expected
    scheduler = module.FixedRateScheduler(tmp_path / 'queue')
    assert scheduler.capacity == {0: 4, 5: 4}
