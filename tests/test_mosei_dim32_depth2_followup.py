import json
from pathlib import Path

from experiments.osram_mosei_dim32_depth2_followup_20261012.run import METHOD, SPECS, candidate_config


def baseline(seed):
    row = json.loads(Path("experiments/osram_core20_20261005/REFERENCE_CONFIG.json").read_text())
    row.update(dataset="CMUMOSEI", seed=seed, fold=1, epochs=100, batch_size=32,
               learning_rate=1e-3, training_objective="emotion-only", backbone_type="osram",
               train_rate_mode="cyclic", checkpoint_selection="test-oracle-per-rate",
               osram_readout_fusion="flat", osram_output_dim=1600, osram_num_heads=8,
               osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
               osram_meaningful_block=None)
    return row


def test_locked_followup_matrix():
    assert set(SPECS) == {"seed67_lr1e3_e60", "seed68_lr1e3_e60", "seed66_lr1e4_e60"}
    for name, spec in SPECS.items():
        cfg = candidate_config(baseline(spec["seed"]), name)
        assert cfg["seed"] == spec["seed"]
        assert cfg["learning_rate"] == spec["learning_rate"]
        assert cfg["epochs"] == spec["epochs"]
        assert cfg["osram_meaningful_block"] == METHOD
