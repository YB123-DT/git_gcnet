import torch

from gcnet_missing_m3.train_gcnet import (
    TrainConfig,
    _emotion_loss,
    _task_loss,
)


def _batch():
    # [L, B, 1] logits and [B, L] continuous MOSI targets.
    logits = torch.tensor(
        [[[0.0], [2.0]], [[1.0], [-1.0]], [[2.0], [0.5]]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, -1.0, 0.5]], dtype=torch.float32
    )
    umask = torch.ones(2, 3, dtype=torch.bool)
    # A=4, T=2, V=1.  The six valid rows contain patterns 1, 2, and 7.
    availability = torch.tensor(
        [
            [[0, 0, 1], [1, 1, 1]],
            [[0, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [0, 1, 0]],
        ],
        dtype=torch.float32,
    )
    return logits, labels, umask, availability


def test_sample_mean_is_legacy_task_loss():
    logits, labels, umask, availability = _batch()
    expected = _task_loss("CMUMOSI", logits, labels, umask)
    actual, diagnostics = _emotion_loss(
        "CMUMOSI", logits, labels, umask, availability, "sample-mean"
    )
    torch.testing.assert_close(actual, expected)
    assert diagnostics == {}


def test_pattern_balanced_averages_present_pattern_losses_only():
    logits, labels, umask, availability = _batch()
    group_losses = []
    pattern_ids = (
        availability[..., 0].long() * 4
        + availability[..., 1].long() * 2
        + availability[..., 2].long()
    )
    valid = umask.T
    for pattern_id in (1, 2, 7):
        selected = valid & pattern_ids.eq(pattern_id)
        group_losses.append(
            _task_loss("CMUMOSI", logits, labels, selected.T)
        )
    expected = torch.stack(group_losses).mean()
    actual, diagnostics = _emotion_loss(
        "CMUMOSI", logits, labels, umask, availability, "pattern-balanced"
    )
    torch.testing.assert_close(actual, expected)
    assert set(diagnostics) == {"1", "2", "7"}


def test_pattern_groupdro_updates_persistent_weights_toward_harder_group():
    logits, labels, umask, availability = _batch()
    weights = torch.ones(7, dtype=torch.float64)
    _, diagnostics = _emotion_loss(
        "CMUMOSI",
        logits * 4.0,
        labels,
        umask,
        availability,
        "pattern-groupdro",
        group_dro_weights=weights,
        group_dro_eta=0.1,
    )
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(
        weights.sum(), torch.ones((), dtype=weights.dtype), atol=1e-12, rtol=1e-12
    )
    assert weights[0] != weights[1] or weights[0] != weights[6]
    assert set(diagnostics) == {"1", "2", "7"}


def test_default_config_keeps_sample_mean():
    assert TrainConfig().emotion_loss_mode == "sample-mean"
