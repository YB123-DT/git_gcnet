"""Contracts for a scalar Gate on Nested's decoded Gap residual only."""
import torch
import pytest

from gcnet_missing_m3.meaningful_input_new40 import build_new40


OLD = 'nested_gnn_rooted_evidence'
NEW = 'nested_gnn_gap_residual_gate'
torch.set_num_threads(1)


def pair():
    torch.manual_seed(66)
    old = build_new40(OLD, 16, 2, 4)
    after = torch.get_rng_state().clone()
    torch.manual_seed(66)
    new = build_new40(NEW, 16, 2, 4)
    assert torch.equal(after, torch.get_rng_state())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])
    # Nonzero decoders make the gate test meaningful, not a zero-residual identity.
    with torch.no_grad():
        for model in (old, new):
            model.local_decoder.bias.fill_(.2)
            for decoder in model.memory_decoders:
                decoder.bias.fill_(.4)
    return old, new


def inputs():
    av = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    active = torch.cat((torch.ones(4, 1, dtype=torch.bool), ~av.bool()), -1)
    return torch.randn(4, 16), torch.randn(4, 4, 8), active, av


def test_registration_preserves_catalog():
    from gcnet_missing_m3.meaningful_new40_registry import NEW40_METHODS, NEW40_VARIANTS
    assert len(NEW40_METHODS) == 40
    assert NEW in NEW40_VARIANTS
    assert NEW not in NEW40_METHODS


def test_training_cli_accepts_new_method_without_other_gate_flags():
    # Exercise the actual parser without importing optional graph-training packages
    # missing in the lightweight local CPU environment.
    import argparse
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'gcnet_missing_m3/train_gcnet.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_parser')
    namespace = dict(argparse=argparse, __package__='gcnet_missing_m3')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    args = namespace['build_parser']().parse_args(['--output-dir', '/tmp/not-started-nested-gap-gate',
                                     '--audio-feature', 'test-a', '--text-feature', 'test-t',
                                     '--video-feature', 'test-v',
                                     '--osram-meaningful-block', NEW])
    assert args.osram_meaningful_block == NEW
    assert not args.osram_local_evidence_gate


def test_old_weights_load_only_with_explicit_gate_key_difference():
    old, new = pair()
    result = new.load_state_dict(old.state_dict(), strict=False)
    assert not result.unexpected_keys
    expected = {'gap_residual_gate.' + k for k in new.gap_residual_gate.state_dict()}
    assert set(result.missing_keys) == expected
    # The original architecture still loads original checkpoints strictly.
    old.load_state_dict(old.state_dict(), strict=True)


def test_identity_gate_keeps_old_parameters_rng_and_nonzero_outputs():
    old, new = pair()
    args = inputs()
    for a, b in zip(old(*args), new(*args)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    gates = new.gap_residual_gate.last_gates
    torch.testing.assert_close(gates, args[2][:, 1:].float(), rtol=0, atol=0)


@pytest.mark.parametrize('scale', [.5, .75, 1.5])
def test_scales_gap_delta_not_original_gap_or_base_or_local(scale):
    old, new = pair()
    args = inputs()
    with torch.no_grad():
        # A deterministic non-unit gate isolates the residual arithmetic.
        gate = new.gap_residual_gate
        gate.network[-1].weight.zero_()
        gate.network[-1].bias.fill_(torch.logit(torch.tensor(scale / 2)))
    local, evidence, active, _ = args
    old_l, old_m = old(*args)
    new_l, new_m = new(*args)
    torch.testing.assert_close(new_l, old_l, rtol=0, atol=0)
    torch.testing.assert_close(new_m[:, 0], old_m[:, 0], rtol=0, atol=0)
    expected = torch.where(active[:, 1:, None],
                           evidence[:, 1:] + scale * (old_m[:, 1:] - evidence[:, 1:]), 0)
    torch.testing.assert_close(new_m[:, 1:], expected)


def test_gate_learns_finitely_without_extra_loss():
    _, model = pair()
    args = inputs()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = model.gap_residual_gate.network[-1].weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        local, memory = model(*args)
        (local.square().mean() + memory.square().mean()).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(initial, model.gap_residual_gate.network[-1].weight)


def test_public_adapter_masks_first_turn_padding_inactive_and_backward_half():
    from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
    adapter = MeaningfulInputAdapter(256, 1024, 1600, NEW, 8, 64)
    local, base, gap = torch.randn(3, 2, 256), torch.randn(3, 2, 1024), torch.randn(3, 2, 3, 1024)
    av = torch.tensor([[[1, 0, 0], [0, 1, 1]]]).expand(3, -1, -1)
    umask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    valid = umask.T.bool()
    clean = (torch.where(valid[..., None], local, 0),
             torch.where(valid[..., None], base, 0),
             torch.where((valid[..., None] & ~av.bool())[..., None], gap, 0))
    with torch.no_grad():
        for decoder in adapter.core.memory_decoders:
            decoder.bias.fill_(.4)
    output = adapter(*clean, av, umask)
    dirty = (local.masked_fill(~valid[..., None], float('nan')),
             base.masked_fill(~valid[..., None], float('inf')),
             gap.masked_fill((~valid[..., None] | av.bool())[..., None], float('nan')))
    for a, b in zip(output, adapter(*dirty, av, umask)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all()
        assert a[~valid].count_nonzero() == 0
    for a, b in zip(output, clean):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    assert output[2][av.bool()].count_nonzero() == 0
    assert torch.equal(output[1][..., 512:], clean[1][..., 512:])
    assert torch.equal(output[2][..., 512:], clean[2][..., 512:])
    assert 'gap_residual_gate' in adapter.last_diagnostics
