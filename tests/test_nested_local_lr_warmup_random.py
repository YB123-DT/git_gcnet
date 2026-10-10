import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import pytest


def policy():
    path = Path(__file__).resolve().parents[1]/'experiments/osram_nested_local_lr_warmup_random_20261010/variant.py'
    assert path.exists(), 'Isolated Nested policy is missing'
    spec = importlib.util.spec_from_file_location('nested_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warmup_only_changes_nested_group():
    apply = policy().apply_nested_lr
    optimizer = SimpleNamespace(param_groups=[{'lr': .001}, {'lr': .0005, 'name': 'nested'}])
    values = []
    for epoch in range(7):
        apply(optimizer, epoch, .0005, 5)
        assert optimizer.param_groups[0]['lr'] == .001
        values.append(optimizer.param_groups[1]['lr'])
    assert values == pytest.approx([.0001, .0002, .0003, .0004, .0005, .0005, .0005])


def test_nested_partition_has_no_duplicates_or_omissions():
    split = policy().separate_nested_group
    model = nn.Module()
    model.osram = nn.Module()
    model.osram.meaningful_block = nn.Linear(2, 2)
    model.osram.flat = nn.Linear(2, 2)
    original = [{'params': list(model.parameters()), 'lr': .001}]
    groups, provenance = split(original, {'backbone': {'learning_rate': .001,
        'parameter_count': 12}}, model, .0005)
    params = [p for group in groups for p in group['params']]
    assert len(params) == len(set(map(id, params))) == len(list(model.parameters()))
    assert {id(p) for p in groups[-1]['params']} == {id(p) for p in model.osram.meaningful_block.parameters()}
    assert groups[0]['lr'] == .001 and groups[-1]['lr'] == .0005
    assert provenance['backbone']['parameter_count'] == 6
    assert provenance['nested']['parameter_count'] == 6
    assert len(original[0]['params']) == 4


def test_random_decoder_preserves_rng_core_and_safe_residual_interface():
    from gcnet_missing_m3.meaningful_input_new40 import TokenAdapter
    from gcnet_missing_m3.meaningful_new40_structure import NestedGNN
    torch.manual_seed(66)
    old = TokenAdapter(NestedGNN())
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(66)
    new = policy().random_nested_adapter(256, 8, 64)
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, parameter in old.core.named_parameters():
        assert torch.equal(parameter, dict(new.core.named_parameters())[name])
    assert new.residual is True
    assert new.local_decoder.weight.ne(0).any()
    assert all(layer.weight.ne(0).any() for layer in new.memory_decoders)
    local = torch.randn(1, 256)
    evidence = torch.randn(1, 4, 512)
    active = torch.tensor([[True, False, False, False]])
    output_local, output_memory = new(local, evidence, active, torch.ones(1, 3))
    assert output_local.shape == local.shape and output_memory.shape == evidence.shape
    assert output_memory[:, 1:].eq(0).all()
    assert not torch.equal(output_memory[:, 0], evidence[:, 0])
    (output_local.square().mean() + output_memory.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
