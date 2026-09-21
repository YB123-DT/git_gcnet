"""Tests for the Base--Gap delta emotion readout."""

from unittest.mock import patch

import torch
from torch import nn

from gcnet_missing_m3 import osram


def _inputs():
    torch.manual_seed(123)
    length, batch, latent, context = 4, 2, 8, 12
    node = torch.randn(length, batch, latent)
    latents = {name: torch.randn_like(node) for name in osram.MODALITIES}
    availability = torch.tensor(
        [
            [[1, 1, 1], [1, 0, 1]],
            [[0, 1, 0], [1, 1, 0]],
            [[0, 0, 1], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    qmask = torch.zeros(batch, length, dtype=torch.long)
    umask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.float32)
    return node, latents, availability, qmask, umask, [3, 3]


def _readout():
    adapter = nn.Sequential(
        nn.LayerNorm(8 + 4 * 12),
        nn.Linear(8 + 4 * 12, 10),
        nn.GELU(),
        nn.Dropout(0),
        nn.Linear(10, 10),
    )
    nn.init.zeros_(adapter[-1].weight)
    nn.init.zeros_(adapter[-1].bias)
    return adapter, nn.Linear(8, 10), nn.LayerNorm(10)


def test_delta_fusion_masks_observed_gaps_and_keeps_padding_zero():
    fusion = osram.BaseGapDeltaFusion(8, 12, 10, dropout=0).eval()
    adapter, local_skip, emotion_norm = _readout()
    torch.nn.init.normal_(adapter[-1].weight, std=0.1)
    adapter.eval(); local_skip.eval(); emotion_norm.eval()
    local = torch.randn(4, 2, 8)
    base = torch.randn(4, 2, 12)
    gap = torch.randn(4, 2, 3, 12)
    availability = _inputs()[2]
    umask = _inputs()[4]
    output = fusion(
        local, base, gap, availability, umask,
        emotion_adapter=adapter, local_skip=local_skip, emotion_norm=emotion_norm,
    )

    changed = gap.clone()
    valid = umask.T.bool()
    observed_gap = valid.unsqueeze(-1) & availability.bool()
    changed[observed_gap] = float("nan")
    torch.testing.assert_close(
        output,
        fusion(
            local, base, changed, availability, umask,
            emotion_adapter=adapter, local_skip=local_skip, emotion_norm=emotion_norm,
        ),
    )
    assert not output[~valid].count_nonzero()


def test_delta_fusion_uses_base_anchor_and_missing_gap_differences():
    fusion = osram.BaseGapDeltaFusion(8, 12, 10, dropout=0).eval()
    adapter, local_skip, emotion_norm = _readout()
    adapter.eval(); local_skip.eval(); emotion_norm.eval()
    torch.nn.init.zeros_(fusion.base_projection.weight)
    torch.nn.init.zeros_(fusion.base_projection.bias)
    torch.nn.init.zeros_(fusion.gap_projection.weight)
    torch.nn.init.zeros_(fusion.gap_projection.bias)
    captured = []
    hook = adapter.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    args = (
        torch.randn(4, 2, 8),
        torch.randn(4, 2, 12),
        torch.randn(4, 2, 3, 12),
        _inputs()[2],
        _inputs()[4],
    )
    fusion(
        *args,
        emotion_adapter=adapter, local_skip=local_skip, emotion_norm=emotion_norm,
    )
    hook.remove()
    valid = args[-1].T.bool()
    expected_base = torch.zeros_like(args[1])
    expected_gap = torch.zeros_like(args[2])
    safe_local = torch.where(valid.unsqueeze(-1), args[0], torch.zeros_like(args[0]))
    expected = torch.cat(
        (safe_local, expected_base, expected_gap.reshape(4, 2, -1),), dim=-1
    )
    torch.testing.assert_close(captured[0], expected)
    assert fusion.last_diagnostics["active_evidence_count_mean"] == float(
        (1 + (1 - args[3]).sum(dim=-1))[valid].mean()
    )


def test_delta_readout_preserves_scan_and_initial_flat_anchor():
    kwargs = dict(
        latent_dim=8,
        output_dim=10,
        num_heads=2,
        key_dim=3,
        value_dim=3,
        dropout=0,
        n_speakers=1,
        bidirectional=False,
        write_step=0.6,
    )
    torch.manual_seed(31)
    flat = osram.OSRAMBackbone(**kwargs).eval()
    torch.manual_seed(31)
    delta = osram.OSRAMBackbone(**kwargs, osram_readout_fusion="base-gap-delta").eval()
    node, latents, availability, qmask, umask, lengths = _inputs()
    args = (node, latents, availability, qmask, umask, lengths)
    flat_states, delta_states = [], []
    old_flat, old_delta = flat.block_write, delta.block_write

    def capture_flat(*values, **kwargs):
        state = old_flat(*values, **kwargs)
        flat_states.append(state.detach().clone())
        return state

    def capture_delta(*values, **kwargs):
        state = old_delta(*values, **kwargs)
        delta_states.append(state.detach().clone())
        return state

    with patch.object(flat, "block_write", side_effect=capture_flat), patch.object(
        delta, "block_write", side_effect=capture_delta
    ):
        flat_output = flat(*args)
        delta_output = delta(*args)
    assert len(flat_states) == len(delta_states)
    for left, right in zip(flat_states, delta_states):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for name in ("local", "base", "gap"):
        torch.testing.assert_close(flat_output[1][name], delta_output[1][name], rtol=0, atol=0)
    torch.testing.assert_close(flat_output[0], delta_output[0], rtol=0, atol=0)
    for key, value in flat.state_dict().items():
        assert torch.equal(value, delta.state_dict()[key]), key


def test_train_config_accepts_base_gap_delta_readout():
    from gcnet_missing_m3.train_gcnet import TrainConfig

    config = TrainConfig(backbone_type="osram", osram_readout_fusion="base-gap-delta")
    assert config.osram_readout_fusion == "base-gap-delta"
