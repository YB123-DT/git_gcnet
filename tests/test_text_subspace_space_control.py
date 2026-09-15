import importlib.util
from types import SimpleNamespace

import torch


def test_control_available():
    assert importlib.util.find_spec('experiments.osram_text_subspace_20260915.compare_target_spaces') is not None


def test_same_forward_identity_projection_is_exact_control():
    from experiments.osram_text_subspace_20260915.compare_target_spaces import compare_losses
    encoder = torch.nn.Linear(4, 8)
    x = encoder(torch.randn(6,4))
    pred = SimpleNamespace(reg_predictions=x[:,None,None,:].expand(-1,1,3,-1),
                           cl_predictions=(x*2)[:,None,None,:].expand(-1,1,3,-1),
                           target_mask=torch.ones(6,1,3,dtype=torch.bool))
    target = {'text': torch.randn(6,1,8)}
    identity = torch.nn.Linear(8,8, bias=False).requires_grad_(False)
    with torch.no_grad():identity.weight.copy_(torch.eye(8))
    result = compare_losses(pred, target, identity, x.square().mean(),
                            {'combined': list(encoder.named_parameters())})
    assert result['full-text'] == result['predictable-subspace']
    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is None for p in identity.parameters())
