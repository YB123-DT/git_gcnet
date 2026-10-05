import torch

from gcnet_missing_m3.osram import LocalConditionedEvidenceGate


def test_gap_only_identity_mask_and_penalty():
    gate = LocalConditionedEvidenceGate(8, 16, gap_only=True)
    local = torch.randn(2, 2, 8)
    base = torch.randn(2, 2, 16)
    gap = torch.randn(2, 2, 3, 16)
    availability = torch.tensor([[[1., 1., 1.], [1., 0., 0.]],
                                 [[0., 1., 1.], [0., 0., 0.]]])
    umask = torch.tensor([[1., 1.], [1., 0.]])
    active = torch.cat((umask.T.bool()[..., None],
                        umask.T.bool()[..., None] & ~availability.bool()), -1)
    assert torch.equal(gate(local, base, gap, availability, umask), active.float())
    with torch.no_grad():
        gate.output.bias.fill_(.5)
    gates = gate(local, base, gap, availability, umask)
    assert torch.equal(gates[..., 0], umask.T)
    assert torch.equal(gates[~active], torch.zeros_like(gates[~active]))
    expected = (.2 * torch.tanh(torch.tensor(.5))).square()
    torch.testing.assert_close(gate.regularization, expected)
    gate.regularization.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in gate.parameters())
    all_observed = torch.ones_like(availability)
    gate(local, base, gap, all_observed, umask)
    assert gate.regularization.item() == 0


def test_default_unchanged_parameter_rng_and_gap_values():
    torch.manual_seed(66)
    original = LocalConditionedEvidenceGate(8, 16)
    torch.manual_seed(66)
    gap_only = LocalConditionedEvidenceGate(8, 16, gap_only=True)
    assert all(torch.equal(v, gap_only.state_dict()[k]) for k, v in original.state_dict().items())
    with torch.no_grad():
        original.output.weight.fill_(.01)
        gap_only.load_state_dict(original.state_dict())
    args = (torch.randn(2, 2, 8), torch.randn(2, 2, 16), torch.randn(2, 2, 3, 16),
            torch.tensor([[[1., 0., 0.], [0., 1., 0.]], [[0., 0., 1.], [1., 1., 1.]]]),
            torch.ones(2, 2))
    torch.testing.assert_close(original(*args)[..., 1:], gap_only(*args)[..., 1:], rtol=0, atol=0)
