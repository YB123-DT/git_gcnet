import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def test_delta_off_retains_memory_skip_and_masks_inactive_gaps():
    path = Path(__file__).resolve().parents[1]/'experiments/osram_nested_delta_off_missing_text_20261010/readout.py'
    assert path.exists(), 'Paired frozen readout helper is missing'
    spec = importlib.util.spec_from_file_location('delta_readout', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    osram = SimpleNamespace(local_skip=nn.Linear(2, 3),
        emotion_adapter=nn.Linear(18, 3), emotion_norm=nn.LayerNorm(3))
    model = SimpleNamespace(osram=osram, smax_fc=nn.Linear(3, 1))
    local = torch.randn(2, 1, 2)
    base = torch.randn(2, 1, 4)
    gap = torch.full((2, 1, 3, 4), float('nan'))
    gap[0, 0, 1] = 1.
    availability = torch.tensor([[[1., 0., 1.]], [[0., 0., 0.]]])
    umask = torch.tensor([[1., 0.]])
    gap_safe = torch.zeros_like(gap)
    gap_safe[0, 0, 1] = 1.
    valid = umask.T.bool()
    safe_local = torch.where(valid[..., None], local, 0.)
    safe_base = torch.where(valid[..., None], base, 0.)
    expected_hidden = osram.emotion_norm(osram.local_skip(safe_local) +
        osram.emotion_adapter(torch.cat((safe_local, safe_base, gap_safe.flatten(2)), -1)))
    expected = torch.where(valid, model.smax_fc(expected_hidden).squeeze(-1), 0.)
    actual = module.flat_prediction(model, local, base, gap, availability, umask)
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all() and actual[1].eq(0).all()
    zero_delta_on = module.flat_prediction(model, local, base, gap, availability, umask,
                                          readout=(local, base, gap))
    assert torch.equal(zero_delta_on, actual)
