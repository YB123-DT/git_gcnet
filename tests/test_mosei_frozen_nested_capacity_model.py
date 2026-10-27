"""Only the approved Nested architecture factor changes; Flat stays frozen."""
import copy
import importlib
from unittest.mock import patch

import pytest
import torch

from tests.test_frozen_nested_gcnet_model import pretrained_parent, open_decoder, parent_call
from tests.test_three_block_joint_model import view


def model(config, parent=None):
    module = importlib.import_module('experiments.osram_mosei_frozen_nested_capacity_20261027.model')
    return module.VariantFrozenNestedModel(pretrained_parent() if parent is None else parent, config)


@pytest.mark.parametrize('config,dim,depth,hidden', [
    ('dim32', 32, 3, 64), ('mlp128', 64, 3, 128), ('depth2', 64, 2, 64)])
def test_approved_architecture_changes_only_one_factor_and_seeded_identity(config, dim, depth, hidden):
    torch.set_num_threads(1)
    parent = pretrained_parent()
    torch.manual_seed(66)
    first = model(config, copy.deepcopy(parent))
    torch.manual_seed(66)
    second = model(config, copy.deepcopy(parent))
    block = first.block.core
    assert block.num_heads == 8 and block.value_dim == 64
    assert block.local_decoder.in_features == dim
    assert len(block.core.layers) == depth
    assert all(layer[0].out_features == hidden for layer in block.core.layers)
    assert block.core.pool[0].out_features == hidden
    assert block.core.readout[0].out_features == hidden
    assert all(torch.equal(tensor, second.block.state_dict()[name]) for name, tensor in first.block.state_dict().items())
    assert not any(p.requires_grad for p in first.parent.parameters())
    assert all(p.requires_grad for p in first.block.parameters())
    # Construction draws exactly the requested adapter, with no discarded
    # default architecture before it.
    from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter
    torch.manual_seed(66)
    adapter = MeaningfulInputAdapter(256, 1024, 1600, first.block.method, 8, 64)
    assert all(torch.equal(tensor, adapter.state_dict()[name]) for name, tensor in first.block.state_dict().items())


@pytest.mark.parametrize('config', ['dim32', 'mlp128', 'depth2'])
def test_variant_zero_prediction_one_scan_raw_skip_rng_and_strict_restore(config):
    torch.set_num_threads(1)
    original = pretrained_parent()
    nested = model(config, copy.deepcopy(original)).eval()
    data = view(2)
    expected = parent_call(original, data)[0]
    rng = torch.random.get_rng_state()
    with patch.object(nested.parent.osram, '_scan', wraps=nested.parent.osram._scan) as scan:
        result = nested(data)
    assert scan.call_count == 1
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(result['prediction'], expected)
    assert torch.equal(result['base_prediction'], expected)
    open_decoder(nested)
    nested.train()
    nested(view(2))['prediction'].square().mean().backward()
    gradients = [p.grad for p in nested.block.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0
    assert all(p.grad is None for p in nested.parent.parameters())
    restored = model(config).eval()
    restored.load_state_dict(nested.state_dict(), strict=True)
    nested.eval()
    assert torch.equal(restored(data)['prediction'], nested(data)['prediction'])


def test_original_and_default_sweep_parameter_draws_and_nonzero_outputs_are_exact():
    from gcnet_missing_m3.meaningful_input_new40 import TokenAdapter
    from gcnet_missing_m3.meaningful_new40_structure import NestedGNN
    from gcnet_missing_m3.nested_sweep import NestedSweepGNN
    torch.set_num_threads(1)
    torch.manual_seed(66)
    original = TokenAdapter(NestedGNN())
    torch.manual_seed(66)
    sweep = TokenAdapter(NestedSweepGNN(dim=64, depth=3, mlp_hidden=64))
    assert set(original.state_dict()) == set(sweep.state_dict())
    assert all(torch.equal(tensor, sweep.state_dict()[name]) for name, tensor in original.state_dict().items())
    for decoder in [original.local_decoder, *original.memory_decoders]:
        torch.nn.init.normal_(decoder.weight, std=.002)
    sweep.load_state_dict(original.state_dict(), strict=True)
    args = (torch.randn(2, 256), torch.randn(2, 4, 512),
            torch.tensor([[1, 1, 0, 1], [1, 0, 1, 0]]).bool(),
            torch.tensor([[0, 1, 0], [1, 0, 1]]).bool())
    assert all(torch.equal(a, b) for a, b in zip(original(*args), sweep(*args)))


def test_unapproved_architecture_is_rejected():
    with pytest.raises(ValueError):
        model('dim128')
