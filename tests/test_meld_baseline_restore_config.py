from dataclasses import asdict, dataclass

import pytest

from experiments.osram_meld_write_retention_20261014.reproduce import validate_config


@dataclass
class Config:
    seed: int
    dataset: str = 'MELD'
    epochs: int = 50
    missing_mask_protocol: str = 'gcnet-random-mask-v1'
    checkpoint_selection: str = 'test-oracle-per-rate'
    backbone_type: str = 'osram'
    osram_readout_fusion: str = 'flat'
    training_objective: str = 'emotion-only'
    osram_bidirectional: bool = False


def test_restore_accepts_exact_original_configs_for_the_three_seeds():
    for seed in (66, 67, 68):
        config = Config(seed)
        assert validate_config(config, asdict(config)) == asdict(config)
    config = Config(65)
    with pytest.raises(ValueError, match='seed'):
        validate_config(config, asdict(config))
