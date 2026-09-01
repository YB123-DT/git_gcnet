import torch

from gcnet_missing_m3_sam_backbone.attention import (
    MaskedTrackPooling,
    SafeDirectedAttention,
)


def test_masked_key_value_cannot_change_output():
    torch.manual_seed(3)
    layer = SafeDirectedAttention(8, 2, 0.0).eval()
    query = torch.randn(3, 1, 8)
    key = torch.randn(3, 1, 8)
    changed = key.clone()
    changed[1] = 10000.0
    query_valid = torch.ones(3, 1, dtype=torch.bool)
    key_valid = torch.tensor([[1], [0], [1]], dtype=torch.bool)

    left, left_valid, left_weights = layer(
        query,
        key,
        query_valid,
        key_valid,
        return_weights=True,
    )
    right, right_valid, right_weights = layer(
        query,
        changed,
        query_valid,
        key_valid,
        return_weights=True,
    )

    torch.testing.assert_close(left, right)
    torch.testing.assert_close(left_weights, right_weights)
    assert torch.equal(left_valid, right_valid)
    assert torch.equal(left_weights[..., 1], torch.zeros_like(left_weights[..., 1]))


def test_conversation_without_keys_returns_zero_track():
    layer = SafeDirectedAttention(8, 2, 0.0).eval()
    output, valid, weights = layer(
        torch.randn(2, 1, 8),
        torch.randn(2, 1, 8),
        torch.ones(2, 1, dtype=torch.bool),
        torch.zeros(2, 1, dtype=torch.bool),
        return_weights=True,
    )

    assert not valid.any()
    assert torch.equal(output, torch.zeros_like(output))
    assert torch.equal(weights, torch.zeros_like(weights))
    assert torch.isfinite(output).all()


def test_invalid_query_is_zero_and_inactive():
    layer = SafeDirectedAttention(8, 2, 0.0).eval()
    query_valid = torch.tensor([[1], [0], [1]], dtype=torch.bool)
    output, valid = layer(
        torch.randn(3, 1, 8),
        torch.randn(3, 1, 8),
        query_valid,
        torch.ones(3, 1, dtype=torch.bool),
    )

    assert torch.equal(valid, query_valid)
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_track_pooling_ignores_invalid_values_and_normalizes_valid_weights():
    torch.manual_seed(5)
    pool = MaskedTrackPooling(8).eval()
    tracks = torch.randn(2, 1, 3, 8)
    track_valid = torch.tensor(
        [[[1, 0, 1]], [[0, 0, 0]]],
        dtype=torch.bool,
    )
    changed = tracks.clone()
    changed[:, :, 1] = 10000.0

    left, left_weights = pool(tracks, track_valid)
    right, right_weights = pool(changed, track_valid)

    torch.testing.assert_close(left, right)
    torch.testing.assert_close(left_weights, right_weights)
    torch.testing.assert_close(left_weights[0].sum(), torch.tensor(1.0))
    assert torch.equal(left_weights[0, 0, 1], torch.tensor(0.0))
    assert torch.equal(left[1], torch.zeros_like(left[1]))
    assert torch.equal(left_weights[1], torch.zeros_like(left_weights[1]))
