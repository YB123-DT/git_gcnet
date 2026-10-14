import numpy as np
import torch

from experiments.osram_meld_information_availability_20261014.features import DonorMatch
from experiments.osram_meld_information_availability_20261014.probe import (
    PROBE_PAIRS,
    MeldProbe,
    build_probe_inputs,
    classification_metrics,
    fit_normalizer,
    transform,
)


def _cache(n=6):
    rng = np.random.default_rng(3)
    availability = np.asarray([[1, 1, 1], [1, 0, 1], [0, 1, 1]] * 2)
    return {
        "local": rng.normal(size=(n, 256)).astype(np.float32),
        "base": rng.normal(size=(n, 512)).astype(np.float32),
        "gap": rng.normal(size=(n, 3, 512)).astype(np.float32),
        "all_mean": rng.normal(size=(n, 256)).astype(np.float32),
        "same_speaker_mean": rng.normal(size=(n, 256)).astype(np.float32),
        "availability": availability.astype(np.float32),
        "all_present": np.asarray([0, 1, 1, 0, 1, 1]),
        "same_speaker_present": np.asarray([0, 0, 1, 0, 0, 1]),
        "all_count": np.asarray([0, 1, 2, 0, 1, 2]),
        "same_speaker_count": np.asarray([0, 0, 1, 0, 0, 1]),
    }


def test_probe_inputs_have_expected_families_and_exact_width_controls():
    data = _cache()
    donor = DonorMatch(
        indices=np.asarray([3, 4, 5, 0, 1, 2]),
        included=np.ones(6, dtype=bool), total=6, matched=6,
    )
    inputs = build_probe_inputs(data, donor)
    assert set(inputs) == {
        "Local", "Base", "BaseDonor", "Memory", "MemoryDonor",
        "LocalBase", "LocalBaseDonor", "LocalMemory", "LocalMemoryDonor",
        "RawHistory", "RawHistoryDonor", "LocalRawHistory", "LocalRawHistoryDonor",
    }
    for real, control in PROBE_PAIRS.items():
        assert inputs[real].shape == inputs[control].shape
    np.testing.assert_array_equal(inputs["LocalBase"][:, :256], data["local"])
    np.testing.assert_array_equal(inputs["LocalBaseDonor"][:, :256], data["local"])
    assert not np.array_equal(inputs["LocalBase"][:, 256:768], inputs["LocalBaseDonor"][:, 256:768])


def test_normalization_is_fit_on_train_only_and_stable_for_constant_columns():
    train = np.asarray([[1, 2, 7], [3, 4, 7]], dtype=np.float32)
    test = np.asarray([[1000, -1000, 7]], dtype=np.float32)
    normalizer = fit_normalizer(train)
    np.testing.assert_allclose(normalizer["mean"], [2, 3, 7])
    assert normalizer["scale"][2] == 1
    transformed = transform(test, normalizer)
    assert transformed[0, 0] == 998
    assert transformed[0, 2] == 0


def test_probe_architecture_and_gradients_are_finite():
    probe = MeldProbe(input_dim=19, hidden_dim=64, classes=7)
    assert isinstance(probe.network[0], torch.nn.LayerNorm)
    assert probe.network[1].in_features == 19
    assert probe.network[1].out_features == 64
    assert probe.network[-1].out_features == 7
    loss = torch.nn.functional.cross_entropy(probe(torch.randn(8, 19)), torch.arange(8) % 7)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in probe.parameters())


def test_classification_metrics_match_known_predictions():
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    predictions = np.asarray([0, 1, 1, 1, 2, 0])
    result = classification_metrics(labels, predictions)
    assert result["n"] == 6
    assert result["accuracy"] == 4 / 6
    assert 0 <= result["weighted_f1"] <= 1
