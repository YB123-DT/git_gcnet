import json
from pathlib import Path

from experiments.osram_lr_batch_screen_20261011.run import SPECS, NESTED, candidate_config


def baseline(dataset):
    path = Path("experiments/osram_core20_20261005/REFERENCE_CONFIG.json")
    row = json.loads(path.read_text())
    row.update(dataset=dataset, seed=66, fold=1, epochs=100, batch_size=32,
               learning_rate=1e-3, training_objective="emotion-only", backbone_type="osram",
               train_rate_mode="cyclic", checkpoint_selection="test-oracle-per-rate",
               osram_readout_fusion="flat", osram_output_dim=1600, osram_num_heads=8,
               osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
               osram_meaningful_block=None)
    return row


def test_locked_five_run_matrix():
    assert len(SPECS) == 5
    assert SPECS["mosei_flat_b64_lr2e3"] == dict(dataset="CMUMOSEI", model="flat", batch_size=64, learning_rate=2e-3, epochs=60, gpu=0)
    for name, spec in SPECS.items():
        cfg = candidate_config(baseline(spec["dataset"]), name)
        assert cfg["batch_size"] == spec["batch_size"]
        assert cfg["learning_rate"] == spec["learning_rate"]
        assert cfg["epochs"] == spec["epochs"]
        assert cfg["osram_meaningful_block"] == (NESTED if spec["model"] == "nested" else None)
