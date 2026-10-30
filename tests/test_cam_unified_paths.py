"""Existing native switches must implement the requested task ablations."""
from unittest.mock import patch

import pytest
import torch

from gcnet_missing_m3.osram import OSRAMBackbone


def make(mode="full", gap="residual"):
    torch.manual_seed(163)
    model = OSRAMBackbone(
        latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
        value_dim=4, n_speakers=1, dropout=0, bidirectional=False,
        write_step=.6, osram_emotion_ablation=mode, osram_gap_read=gap,
    ).eval()
    # Do not mistake the initial zero adapter for proof of ablation isolation.
    torch.nn.init.normal_(model.emotion_adapter[-1].weight, std=.1)
    return model


def data():
    torch.manual_seed(14)
    node = torch.randn(3, 1, 8)
    latents = {name: torch.randn_like(node) for name in ("audio", "text", "visual")}
    availability = torch.tensor([[[1, 1, 0]], [[1, 0, 1]], [[0, 1, 0]]]).float()
    return node, latents, availability, torch.zeros(1, 3).long(), torch.ones(1, 3), [3]


@pytest.mark.parametrize("mode", ["local-only", "local-base", "full"])
def test_task_cannot_receive_suppressed_memory_slots(mode):
    model = make(mode)
    recorded = {}

    def retain(_, args):
        recorded["input"] = args[0]
        args[0].retain_grad()

    handle = model.emotion_adapter.register_forward_pre_hook(retain)
    try:
        hidden, _ = model(*data())
        (hidden * torch.arange(1, 11)).sum().backward()
    finally:
        handle.remove()
    value = recorded["input"]
    assert value.shape[-1] == 72  # toy Local8 + four Context16 slots
    if mode == "local-only":
        assert value[..., 8:].count_nonzero() == 0
        writer = [p.grad for p in model.value_projectors.parameters()]
        assert all(g is None or not g.count_nonzero() for g in writer)
    elif mode == "local-base":
        assert value[..., 24:].count_nonzero() == 0
        assert value[1:, ..., 8:24].count_nonzero() > 0
    else:
        assert value[1:, ..., 24:].count_nonzero() > 0
    assert model.local_skip.weight.grad is not None


def test_raw_query_ignores_residualized_direction_but_keeps_joint_writes():
    full, raw = make(), make(gap="raw")
    raw.load_state_dict(full.state_dict(), strict=True)
    outputs = []
    writes = []
    for model in (full, raw):
        states = []
        original = model.block_write

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            states.append(result.detach().clone())
            return result

        with patch.object(model, "_address_residual", side_effect=lambda k, q: torch.zeros_like(q)):
            with patch.object(model, "block_write", side_effect=observe):
                outputs.append(model(*data())[1])
        writes.append(states)
    assert len(writes[0]) == len(writes[1]) == 3
    assert all(torch.equal(a, b) for a, b in zip(*writes))
    assert torch.equal(outputs[0]["base"], outputs[1]["base"])
    assert outputs[0]["gap"].count_nonzero() == 0
    assert outputs[1]["gap"].count_nonzero() > 0
