from unittest.mock import patch

import torch

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


def test_modality_track_flat_fusion_masks_missing_slots_and_padding():
    fusion = osram.ModalityTrackFlatFusion(8, 12, 10, dropout=0).eval()
    torch.manual_seed(7)
    tracks = torch.randn(4, 2, 3, 8)
    base = torch.randn(4, 2, 12)
    gap = torch.randn(4, 2, 3, 12)
    _, _, availability, _, umask, _ = _inputs()
    output = fusion(tracks, base, gap, availability, umask)
    changed = tracks.clone()
    changed[~availability.bool()] = float("nan")
    gap_changed = gap.clone()
    gap_changed[availability.bool()] = float("nan")
    torch.testing.assert_close(
        output, fusion(changed, base, gap_changed, availability, umask)
    )
    assert output.shape == (4, 2, 10)
    assert not output[~umask.T.bool()].count_nonzero()


def test_modality_tracks_keep_scan_and_contexts_unchanged():
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
    tracks = osram.OSRAMBackbone(
        **kwargs, osram_readout_fusion="modality-tracks"
    ).eval()
    node, latents, availability, qmask, umask, lengths = _inputs()
    embeddings = torch.randn(3, 8)
    args = (node, latents, availability, qmask, umask, lengths)
    flat_states, track_states = [], []
    old_flat, old_tracks = flat.block_write, tracks.block_write

    def capture_flat(*values, **kwargs):
        state = old_flat(*values, **kwargs)
        flat_states.append(state.detach().clone())
        return state

    def capture_tracks(*values, **kwargs):
        state = old_tracks(*values, **kwargs)
        track_states.append(state.detach().clone())
        return state

    with patch.object(flat, "block_write", side_effect=capture_flat), patch.object(
        tracks, "block_write", side_effect=capture_tracks
    ):
        flat_output = flat(*args)
        track_output = tracks(*args, modality_embeddings=embeddings)
    assert len(flat_states) == len(track_states)
    for left, right in zip(flat_states, track_states):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for name in ("local", "base", "gap"):
        torch.testing.assert_close(
            flat_output[1][name], track_output[1][name], rtol=0, atol=0
        )
    assert not torch.equal(flat_output[0], track_output[0])


def test_modality_tracks_requires_embeddings_only_for_new_readout():
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
    model = osram.OSRAMBackbone(**kwargs, osram_readout_fusion="modality-tracks")
    with torch.no_grad():
        node, latents, availability, qmask, umask, lengths = _inputs()
        try:
            model(node, latents, availability, qmask, umask, lengths)
        except ValueError as error:
            assert "modality_embeddings" in str(error)
        else:
            raise AssertionError("modality-tracks must require modality embeddings")


def test_modality_track_residual_is_zero_initialized_on_flat_anchor():
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
    torch.manual_seed(71)
    flat = osram.OSRAMBackbone(**kwargs).eval()
    torch.manual_seed(71)
    residual = osram.OSRAMBackbone(
        **kwargs, osram_readout_fusion="modality-track-residual"
    ).eval()
    node, latents, availability, qmask, umask, lengths = _inputs()
    embeddings = torch.randn(3, 8)
    args = (node, latents, availability, qmask, umask, lengths)
    flat_hidden, flat_context = flat(*args)
    residual_hidden, residual_context = residual(
        *args, modality_embeddings=embeddings
    )
    torch.testing.assert_close(flat_hidden, residual_hidden, rtol=0, atol=0)
    for name in ("local", "base", "gap"):
        torch.testing.assert_close(
            flat_context[name], residual_context[name], rtol=0, atol=0
        )
    assert torch.all(residual.modality_track_residual.residual[-1].weight == 0)
    assert torch.all(residual.modality_track_residual.residual[-1].bias == 0)
    assert torch.all(residual.modality_track_residual.gate.bias == -2)


def test_modality_track_residual_changes_readout_not_scan():
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
    torch.manual_seed(73)
    model = osram.OSRAMBackbone(
        **kwargs, osram_readout_fusion="modality-track-residual"
    ).eval()
    node, latents, availability, qmask, umask, lengths = _inputs()
    embeddings = torch.randn(3, 8)
    args = (node, latents, availability, qmask, umask, lengths)
    states_before = []
    old_write = model.block_write

    def capture_before(*values, **call_kwargs):
        state = old_write(*values, **call_kwargs)
        states_before.append(state.detach().clone())
        return state

    with patch.object(model, "block_write", side_effect=capture_before):
        base_hidden, base_context = model(*args, modality_embeddings=embeddings)
    states_after = []

    def capture_after(*values, **call_kwargs):
        state = old_write(*values, **call_kwargs)
        states_after.append(state.detach().clone())
        return state

    with patch.object(model, "block_write", side_effect=capture_after):
        with torch.no_grad():
            model.modality_track_residual.residual[-1].bias.fill_(0.25)
        changed_hidden, changed_context = model(
            *args, modality_embeddings=embeddings
        )
    assert not torch.equal(base_hidden, changed_hidden)
    for name in ("local", "base", "gap"):
        torch.testing.assert_close(
            base_context[name], changed_context[name], rtol=0, atol=0
        )
    assert len(states_before) == len(states_after)
    for before, after in zip(states_before, states_after):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
