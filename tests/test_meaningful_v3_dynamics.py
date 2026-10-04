"""CPU contracts and distinguishing mechanism checks for dynamics readouts."""
import pytest
import torch

from gcnet_missing_m3.meaningful_v3_dynamics import METHODS, TreeMachine, build


@pytest.mark.parametrize("method", METHODS)
def test_contract_mask_identity_reset_and_gradients(method):
    torch.manual_seed(31)
    model = build(method).eval()
    local = torch.randn(3, 256)
    evidence = torch.randn(3, 4, 512)
    active = torch.tensor([[True, True, False, True], [False] * 4, [True] * 4])
    availability = torch.zeros(3, 3)
    dirty = evidence.masked_fill(~active[..., None], float("nan"))
    clean = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
    out_local, out_evidence = model(local, dirty, active, availability)
    torch.testing.assert_close(out_local, local, rtol=0, atol=0)
    torch.testing.assert_close(out_evidence, clean, rtol=0, atol=0)
    (out_local.square().sum() + out_evidence.square().sum()).backward()
    assert model.evidence_decoder.weight.grad.abs().sum() > 0
    # Zero-init bridges deliberately block core gradients at initialization.
    with torch.no_grad():
        model.local_decoder.weight.normal_(std=.03)
        model.evidence_decoder.weight.normal_(std=.03)
    model.zero_grad(set_to_none=True)
    first = model(local, dirty, active, availability)
    reference = model(local, clean, active, availability)
    for a, b in zip(first, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all()
    model(local * 2, evidence * 3, active, availability)
    repeated = model(local, dirty, active, availability)
    for a, b in zip(first, repeated):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    loss = sum(x.square().mean() for x in first)
    loss.backward()
    core_gradients = [p.grad for p in model.core.parameters() if p.grad is not None]
    assert core_gradients and all(torch.isfinite(g).all() for g in core_gradients)
    assert sum(g.abs().sum() for g in core_gradients) > 0
    assert torch.equal(first[1][~active], torch.zeros_like(first[1][~active]))


def test_tree_car_cdr_cons_preserve_reachable_subtrees():
    core = TreeMachine(dim=8)
    left = torch.zeros(1, core.nodes, 8)
    right = torch.zeros_like(left)
    left[:, :7] = torch.randn(1, 7, 8)
    right[:, :7] = torch.randn(1, 7, 8)
    root = torch.randn(1, 8)
    memory = torch.stack((left, right), 1)
    arguments = torch.tensor([[[1., 0., 1., 0.], [0., 1., 0., 1.]]])
    cons = core.interpret(memory, arguments, torch.tensor([[0., 0., 1.]]), root)
    torch.testing.assert_close(cons[:, 0], root)
    torch.testing.assert_close(core.left.T @ cons, left)
    torch.testing.assert_close(core.right.T @ cons, right)


def test_srwm_updates_program_not_only_output_weights():
    model = build(METHODS[0], latent_dim=8, num_heads=1, value_dim=8)
    tokens = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    output = model.core(tokens, mask)
    output[:, 0].square().sum().backward()
    gradient = model.core.initial.grad
    assert all(gradient[a:b].abs().sum() > 0 for a, b in ((0, 32), (32, 64), (64, 96), (96, 100)))


def test_shuffle_uses_eight_ports_and_five_switch_stages():
    model = build(METHODS[2], latent_dim=8, num_heads=1, value_dim=8)
    counts = []
    hooks = [switch.register_forward_pre_hook(lambda _, args: counts.append(args[0].shape[1]))
             for switch in (model.core.forward_switch, model.core.reverse_switch, model.core.last_switch)]
    model.core(torch.randn(2, 5, 32), torch.ones(2, 5, dtype=torch.bool))
    for hook in hooks:
        hook.remove()
    assert counts == [8] * 5
