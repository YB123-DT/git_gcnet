import torch

from gcnet_missing_m3.osram import OSRAMBackbone
from gcnet_missing_m3.text_core import TextCore


def _inputs(length=4, batch=2, latent_dim=8, context_dim=16):
    torch.manual_seed(7)
    latents = {
        name: torch.randn(length, batch, latent_dim)
        for name in ("audio", "text", "visual")
    }
    availability = torch.tensor(
        [
            [[1, 0, 1], [1, 1, 1]],
            [[1, 1, 0], [1, 0, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[1, 0, 1], [1, 1, 0]],
        ],
        dtype=torch.float32,
    )
    umask = torch.ones(batch, length)
    base = torch.randn(length, batch, context_dim)
    gap = torch.randn(length, batch, 3, context_dim)
    return latents, availability, umask, base, gap


def test_text_core_uses_only_audio_visual_for_prediction_and_masks_padding():
    latents, availability, umask, base, gap = _inputs()
    module = TextCore(8, 16, dropout=0.0).eval()
    output = module(latents, availability, umask, base, gap)
    changed = dict(latents)
    changed["text"] = torch.randn_like(changed["text"]) * 100.0
    changed_output = module(changed, availability, umask, base, gap)
    torch.testing.assert_close(output["u_pred"], changed_output["u_pred"])
    assert output["u_pred"].shape == (4, 2, 64)
    assert output["read_residual"].shape == (4, 2, 8)


def test_text_core_zero_initialized_slot_preserves_read_node():
    latents, availability, umask, base, gap = _inputs(latent_dim=8, context_dim=16)
    module = TextCore(8, 16, dropout=0.0).eval()
    output = module(latents, availability, umask, base, gap)
    assert torch.equal(output["read_residual"], torch.zeros_like(output["read_residual"]))


def test_osram_context_read_residual_is_optional_and_shape_checked():
    torch.manual_seed(11)
    model = OSRAMBackbone(
        latent_dim=8,
        output_dim=12,
        num_heads=2,
        key_dim=4,
        value_dim=4,
        n_speakers=1,
        dropout=0.0,
        bidirectional=False,
        write_step=0.6,
    ).eval()
    length, batch = 3, 1
    node = torch.randn(length, batch, 8)
    latents = {name: torch.randn_like(node) for name in ("audio", "text", "visual")}
    availability = torch.tensor(
        [[[1, 0, 1]], [[1, 1, 0]], [[1, 1, 1]]], dtype=torch.float32
    )
    qmask = torch.zeros(batch, length)
    umask = torch.ones(batch, length)
    plain, _ = model(node, latents, availability, qmask, umask, [length])
    with_zero, _ = model(
        node,
        latents,
        availability,
        qmask,
        umask,
        [length],
        context_read_residual=lambda base, gap: torch.zeros_like(node),
    )
    torch.testing.assert_close(plain, with_zero)


def test_osram_rejects_wrong_text_core_residual_shape():
    model = OSRAMBackbone(
        latent_dim=8,
        output_dim=12,
        num_heads=2,
        key_dim=4,
        value_dim=4,
        n_speakers=1,
        dropout=0.0,
        bidirectional=False,
        write_step=0.6,
    )
    node = torch.randn(2, 1, 8)
    latents = {name: torch.randn_like(node) for name in ("audio", "text", "visual")}
    availability = torch.ones(2, 1, 3)
    qmask = torch.zeros(1, 2)
    umask = torch.ones(1, 2)
    try:
        model(
            node,
            latents,
            availability,
            qmask,
            umask,
            [2],
            context_read_residual=lambda base, gap: torch.zeros(2, 1, 7),
        )
    except ValueError as error:
        assert "context_read_residual" in str(error)
    else:
        raise AssertionError("wrong context residual shape was accepted")
