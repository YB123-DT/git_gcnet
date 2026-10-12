import pickle
from collections import Counter

import numpy as np


def _meld_pickle(path):
    ids = [10, 11, 12]
    speakers = {
        10: np.eye(9, dtype=np.int64)[[1, 3]],
        11: np.eye(9, dtype=np.int64)[[2]],
        12: np.eye(9, dtype=np.int64)[[0, 8, 0]],
    }
    labels = {10: [0, 1], 11: [2], 12: [3, 4, 6]}
    text = {key: np.full((len(labels[key]), 4), key, dtype=np.float32) for key in ids}
    audio = {key: np.full((len(labels[key]), 3), key + 1, dtype=np.float32) for key in ids}
    visual = {key: np.full((len(labels[key]), 2), key + 2, dtype=np.float32) for key in ids}
    value = (speakers, labels, labels, text, text, text, text, text, text, text,
             text, audio, visual, {key: ["x"] * len(labels[key]) for key in ids},
             [10], [12], [11])
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def test_meld_dataset_exposes_gcnet_conversation_contract(tmp_path):
    from gcnet_modality_jepa.dataloader_meld import MELDDataset

    source = tmp_path / "meld.pkl"
    _meld_pickle(source)
    dataset = MELDDataset(source)

    assert dataset.trainVids == {10}
    assert dataset.valVids == {11}
    assert dataset.testVids == {12}
    assert dataset.get_featDim() == (3, 4, 2)
    item = dataset[0]
    assert item[0].shape == item[3].shape
    np.testing.assert_array_equal(item[0], item[3])
    assert item[6].tolist() == [1.0, 3.0]


def test_meld_loaders_preserve_official_train_valid_test(tmp_path, monkeypatch):
    import config
    from gcnet_modality_jepa.train_gcnet import get_loaders

    source = tmp_path / "meld.pkl"
    _meld_pickle(source)
    monkeypatch.setitem(config.PATH_TO_LABEL, "MELD", str(source))
    train, valid, test, adim, tdim, vdim = get_loaders(
        audio_root="unused", text_root="unused", video_root="unused",
        num_folder=1, dataset="MELD", batch_size=2, num_workers=0,
        seed=66, validation_fraction=0.1, evaluation_protocol="official",
    )

    assert (adim, tdim, vdim) == (3, 4, 2)
    assert list(train[0].sampler.indices) == [0]
    assert list(valid[0].sampler.indices) == [1]
    assert list(test[0].sampler.indices) == [2]


def test_missing_m3_declares_meld_as_seven_class_nine_speaker_task():
    from gcnet_missing_m3.train_gcnet import _dataset_shape

    assert _dataset_shape("MELD") == {
        "num_folds": 1,
        "num_classes": 7,
        "num_speakers": 9,
        "task": "classification",
    }


def test_meld_nested_runner_keeps_selected_cfg84_protocol():
    from types import SimpleNamespace
    from experiments.osram_ecerc_features_20261012.run_meld import build_config

    config = build_config(66, config_class=SimpleNamespace)
    assert (config.dataset, config.fold, config.epochs) == ("MELD", 1, 100)
    assert config.osram_meaningful_block == "nested_gnn_rooted_evidence"
    assert config.checkpoint_selection == "test-oracle-per-rate"
    assert (config.osram_num_heads, config.osram_key_dim, config.osram_value_dim) == (8, 64, 64)
    assert config.osram_output_dim == 1600
    assert config.osram_write_step == 0.6


def test_meld_runner_accepts_bounded_epoch_and_nested_sweep_overrides():
    from types import SimpleNamespace
    from experiments.osram_ecerc_features_20261012.run_meld import build_config

    config = build_config(
        66,
        config_class=SimpleNamespace,
        epochs=50,
        meaningful_block="nested_dim128",
    )
    assert config.epochs == 50
    assert config.osram_meaningful_block == "nested_dim128"


def test_meld_nested_grid_has_64_balanced_unique_configs():
    from gcnet_missing_m3.meaningful_new40_registry import MELD_NESTED_GRID

    assert len(MELD_NESTED_GRID) == 64
    configs = [tuple(sorted(value.items())) for value in MELD_NESTED_GRID.values()]
    assert len(set(configs)) == 64
    expected = {
        "dim": {32, 64, 128, 256},
        "depth": {1, 2, 3, 4},
        "groups": {1, 2, 4, 8},
        "mlp_hidden": {64, 128, 256, 512},
    }
    for factor, levels in expected.items():
        counts = Counter(config[factor] for config in MELD_NESTED_GRID.values())
        assert set(counts) == levels
        assert set(counts.values()) == {16}


def test_meld_nested_local14_is_unique_and_disjoint_from_broad_grid():
    from gcnet_missing_m3.meaningful_new40_registry import (
        MELD_NESTED_GRID,
        MELD_NESTED_LOCAL14,
    )

    broad = {tuple(sorted(value.items())) for value in MELD_NESTED_GRID.values()}
    local = {tuple(sorted(value.items())) for value in MELD_NESTED_LOCAL14.values()}
    assert len(MELD_NESTED_LOCAL14) == len(local) == 14
    assert broad.isdisjoint(local)


def test_meld_screen_checkpoint_cleanup_is_scoped_to_best_checkpoints(tmp_path):
    from experiments.osram_ecerc_features_20261012.run_meld import (
        discard_screen_checkpoints,
    )

    first = tmp_path / "best_miss_0p0.pt"
    second = tmp_path / "best_miss_0p7.pt"
    retained = tmp_path / "metrics.json"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    retained.write_text("{}")

    summary = discard_screen_checkpoints(tmp_path)

    assert summary == {"count": 2, "bytes": 6}
    assert not first.exists() and not second.exists()
    assert retained.exists()


def test_meld_flat_variant_changes_only_the_meaningful_block():
    from types import SimpleNamespace
    from experiments.osram_ecerc_features_20261012.run_meld import build_config

    nested = vars(build_config(66, config_class=SimpleNamespace, model_variant="nested"))
    flat = vars(build_config(66, config_class=SimpleNamespace, model_variant="flat"))
    assert nested["osram_meaningful_block"] == "nested_gnn_rooted_evidence"
    assert flat["osram_meaningful_block"] == "none"
    assert {key: value for key, value in nested.items() if key != "osram_meaningful_block"} == {
        key: value for key, value in flat.items() if key != "osram_meaningful_block"
    }
