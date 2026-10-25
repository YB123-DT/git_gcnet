from types import SimpleNamespace
import torch
from torch import nn
from osram_four_block_wiring.frozen_flat import FrozenFlatReader


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.osram = nn.Module()
        self.osram.emotion_norm = nn.LayerNorm(1600)
        self.smax_fc = nn.Linear(1600, 1)
        self.scans = 0

    def forward(self, features, a, qmask, umask, lengths, predict_missing=False):
        self.scans += 1
        t, b = a.shape[:2]
        self.last_osram_context = dict(local=features[0][..., :256],
            base=features[0].new_zeros(t, b, 1024), gap=features[0].new_zeros(t, b, 3, 1024))
        h = self.osram.emotion_norm(features[0]) * umask.T[..., None]
        latents = {k: features[0][..., :256] * a[..., i:i+1] for i, k in enumerate(('audio','text','visual'))}
        return self.smax_fc(h), h, latents, None


def test_reader_one_forward_frozen_head_input_gradients():
    parent = Parent()
    reader = FrozenFlatReader(parent)
    x = torch.randn(2, 1, 1600)
    frame = reader(x, torch.ones(2, 1, 3), torch.zeros(1, 2), torch.ones(1, 2), [2])
    assert parent.scans == 1 and frame.context.shape == (2, 1, 2304)
    assert not frame.anchor.requires_grad
    assert torch.equal(reader.decode_delta(frame, torch.zeros_like(frame.anchor)), frame.prediction)
    delta = torch.randn_like(frame.anchor, requires_grad=True)
    reader.decode_delta(frame, delta).sum().backward()
    assert delta.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in parent.parameters())


def test_hypothesis_decode_uses_same_head():
    parent = Parent()
    reader = FrozenFlatReader(parent)
    frame = reader(torch.randn(1, 1, 1600), torch.ones(1, 1, 3), torch.zeros(1, 1), torch.ones(1, 1), [1])
    delta = torch.randn(1, 1, 4, 1600)
    actual = reader.decode_delta(frame, delta)
    expected = torch.stack([reader.decode_delta(frame, delta[:, :, k]) for k in range(4)], 2)
    torch.testing.assert_close(actual, expected)
