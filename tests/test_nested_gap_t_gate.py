"""Gap-T-only residual gating must leave Gap-A/V on their old Nested paths."""
import torch

from gcnet_missing_m3.meaningful_input_new40 import build_new40

OLD = 'nested_gnn_rooted_evidence'
NEW = 'nested_gnn_gap_t_residual_gate'
ALL = 'nested_gnn_gap_residual_gate'
torch.set_num_threads(1)


def models():
    torch.manual_seed(66)
    old = build_new40(OLD, 16, 2, 4)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    new = build_new40(NEW, 16, 2, 4)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    with torch.no_grad():
        for model in (old, new):
            model.local_decoder.bias.fill_(.2)
            for decoder in model.memory_decoders:
                decoder.bias.fill_(.4)
    return old, new


def inputs():
    # A, V, AV: T missing; T and ATV: T present.
    av = torch.tensor([[1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1]])
    active = torch.cat((torch.ones(5, 1, dtype=torch.bool), ~av.bool()), -1)
    return torch.randn(5, 16), torch.randn(5, 4, 8), active, av


def test_registered_and_same_capacity_as_all_gap_gate():
    from gcnet_missing_m3.meaningful_new40_registry import NEW40_VARIANTS
    assert NEW in NEW40_VARIANTS
    _, model = models()
    all_gate = build_new40(ALL, 16, 2, 4)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in all_gate.parameters())
    assert model.gap_residual_gate.gated_modalities == (1,)


def test_initial_one_is_exact_original_with_nonzero_residual():
    old, new = models()
    args = inputs()
    for a, b in zip(old(*args), new(*args)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_half_gate_changes_only_text_delta():
    old, new = models()
    args = inputs()
    with torch.no_grad():
        new.gap_residual_gate.network[-1].bias.fill_(torch.logit(torch.tensor(.25)))
    old_l, old_m = old(*args)
    new_l, new_m = new(*args)
    torch.testing.assert_close(old_l, new_l, rtol=0, atol=0)
    for slot in (0, 1, 3):  # Base, Gap-A, Gap-V unchanged exactly.
        torch.testing.assert_close(old_m[:, slot], new_m[:, slot], rtol=0, atol=0)
    expected = torch.where(args[2][:, 2, None], args[1][:, 2] + .5 * (old_m[:, 2] - args[1][:, 2]), 0)
    torch.testing.assert_close(new_m[:, 2], expected)
    torch.testing.assert_close(new_m[3:], old_m[3:], rtol=0, atol=0)
    gates = new.gap_residual_gate.last_gates
    assert torch.equal(gates[:, (0, 2)], args[2][:, (1, 3)].float())


def test_no_text_gap_means_no_gate_gradient():
    _, model = models()
    local, memory, active, av = inputs()
    local, memory = model(local[3:], memory[3:], active[3:], av[3:])
    memory.square().sum().backward()
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.gap_residual_gate.parameters())


def test_text_gate_updates_finitely():
    _, model = models()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    initial = model.gap_residual_gate.network[-1].weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        _, memory = model(*inputs())
        memory[:, 2].square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(initial, model.gap_residual_gate.network[-1].weight)
