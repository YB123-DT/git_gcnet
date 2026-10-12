import pickle
from types import SimpleNamespace

import numpy as np
import pytest


from experiments.osram_ecerc_features_20261012.prepare_features import (
    export_ecerc_features,
)
from experiments.osram_ecerc_features_20261012.run import build_config, tasks


def _dump(path, value):
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def test_export_ecerc_features_preserves_utterance_alignment(tmp_path):
    ids = {"dialog": ["u1", "u2"]}
    speakers = {"dialog": ["F", "M"]}
    labels = {"dialog": [1, 3]}
    audio = {"dialog": np.arange(6, dtype=np.float32).reshape(2, 3)}
    visual = {"dialog": np.arange(4, dtype=np.float64).reshape(2, 2)}
    base = (ids, speakers, labels, {}, audio, visual, {"dialog": ["a", "b"]},
            ["dialog"], [])
    text = {"dialog": np.arange(8, dtype=np.float32).reshape(2, 4)}
    roberta = (speakers, labels, text, text, text, text, {}, ["dialog"], [], [])
    base_path, text_path = tmp_path / "base.pkl", tmp_path / "text.pkl"
    _dump(base_path, base)
    _dump(text_path, roberta)

    manifest = export_ecerc_features(base_path, text_path, tmp_path / "features")

    assert manifest["utterances"] == 2
    assert manifest["dimensions"] == {"audio": 3, "text": 4, "visual": 2}
    np.testing.assert_array_equal(
        np.load(tmp_path / "features/audio/u2.npy"), audio["dialog"][1]
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "features/text/u1.npy"), text["dialog"][0]
    )
    assert np.load(tmp_path / "features/visual/u1.npy").dtype == np.float32


def test_export_ecerc_features_rejects_misaligned_dialogues(tmp_path):
    base = ({"dialog": ["u1"]}, {"dialog": ["F"]}, {"dialog": [0]}, {},
            {"dialog": np.zeros((1, 2))}, {"dialog": np.zeros((1, 3))}, {},
            ["dialog"], [])
    text = {"other": np.zeros((1, 4))}
    roberta = ({}, {}, text, text, text, text, {}, [], [], [])
    base_path, text_path = tmp_path / "base.pkl", tmp_path / "text.pkl"
    _dump(base_path, base)
    _dump(text_path, roberta)

    with pytest.raises(ValueError, match="dialogue ids"):
        export_ecerc_features(base_path, text_path, tmp_path / "features")


def test_ecerc_run_uses_selected_nested_cfg84_protocol():
    config = build_config(seed=66, fold=3, config_class=SimpleNamespace)

    assert config.dataset == "IEMOCAPSix"
    assert config.fold == 3
    assert config.epochs == 100
    assert config.backbone_type == "osram"
    assert config.osram_meaningful_block == "nested_gnn_rooted_evidence"
    assert (config.osram_num_heads, config.osram_key_dim, config.osram_value_dim) == (8, 64, 64)
    assert config.osram_output_dim == 1600
    assert config.osram_write_step == 0.6
    assert not config.osram_bidirectional
    assert config.training_objective == "emotion-only"
    assert config.checkpoint_selection == "test-oracle-per-rate"


def test_ecerc_run_covers_all_five_iemocap_sessions_once():
    assert tasks(seed=66) == [
        {"seed": 66, "fold": fold} for fold in range(1, 6)
    ]
