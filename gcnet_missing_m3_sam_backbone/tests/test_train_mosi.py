import inspect
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from gcnet_missing_m3_sam_backbone.train_mosi import (
    SAMTrainConfig,
    complete_batch,
    regression_metrics,
    select_best_epoch,
    train_model,
    write_json,
)


def test_nonzero_metrics_match_existing_implementation():
    labels = np.array([-1.0, 0.0, 1.0, 2.0])
    predictions = np.array([-0.2, -1.0, 0.1, -0.3])

    metrics = regression_metrics(labels, predictions)
    nonzero = labels != 0
    binary_labels = labels[nonzero] > 0
    binary_predictions = predictions[nonzero] > 0

    assert metrics["sample_count"] == 3
    assert metrics["accuracy"] == accuracy_score(binary_labels, binary_predictions)
    assert metrics["weighted_f1"] == f1_score(
        binary_labels,
        binary_predictions,
        average="weighted",
    )


def test_best_epoch_is_selected_only_by_validation_loss():
    records = [
        {
            "epoch": 1,
            "validation": {"loss": 0.5},
            "test": {"weighted_f1": 0.9},
        },
        {
            "epoch": 2,
            "validation": {"loss": 0.4},
            "test": {"weighted_f1": 0.7},
        },
    ]

    assert select_best_epoch(records) == 2


def test_train_model_has_no_test_loader_argument():
    assert "test_loader" not in inspect.signature(train_model).parameters


def test_complete_batch_uses_host_features_and_explicit_full_availability():
    length, batch = 3, 2
    host = [torch.randn(length, batch, dim) for dim in (4, 6, 8)]
    guest = [torch.full_like(value, 999.0) for value in host]
    qmask = torch.zeros(batch, length)
    umask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    labels = torch.randn(batch, length)
    raw = host + guest + [qmask, umask, labels, ["a", "b"]]

    view = complete_batch(raw, torch.device("cpu"))

    for actual, expected in zip(view["features"], host):
        torch.testing.assert_close(actual, expected)
    expected_availability = umask.transpose(0, 1).unsqueeze(-1).repeat(1, 1, 3)
    assert torch.equal(view["availability"], expected_availability)
    assert torch.equal(view["labels"], labels)


def test_write_json_is_complete_and_sorted(tmp_path):
    target = tmp_path / "nested" / "metrics.json"

    write_json(target, {"z": 1, "a": 2})

    assert json.loads(target.read_text()) == {"a": 2, "z": 1}
    assert not (target.parent / "metrics.json.tmp").exists()


def test_formal_config_is_locked_to_complete_mosi_protocol():
    config = SAMTrainConfig()

    assert config.dataset == "CMUMOSI"
    assert config.missing_rate == 0.0
    assert config.evaluation_protocol == "official"
    assert config.checkpoint_selection == "validation_loss"
    assert config.width == 120
    assert config.heads == 4
