"""Read-conditioned B2 must not modify real-observation write trajectories."""

import base64
import copy
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import torch

from gcnet_missing_m3.osram import MODALITIES, OSRAMBackbone


def _model(cls=OSRAMBackbone, **kwargs):
    return cls(latent_dim=8, output_dim=10, num_heads=2, key_dim=3,
               value_dim=4, n_speakers=2, dropout=kwargs.pop('dropout', 0.),
               bidirectional=kwargs.pop('bidirectional', False),
               write_step=.6, **kwargs)


def _inputs():
    torch.manual_seed(73)
    node = torch.randn(5, 2, 8)
    latents = {name: torch.randn_like(node) for name in MODALITIES}
    availability = torch.tensor([[[1, 0, 1], [0, 1, 0]],
                                 [[0, 1, 1], [1, 0, 1]],
                                 [[1, 1, 0], [0, 0, 0]],
                                 [[0, 1, 0], [0, 0, 0]],
                                 [[0, 0, 0], [0, 0, 0]]]).float()
    return (node, latents, availability, torch.zeros(2, 5).long(),
            torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]]).float(), [4, 2])


def _run_trace(model, args, **kwargs):
    trajectory = []
    original = model.block_write

    def capture(memory, keys, values, availability, beta=None):
        result = original(memory, keys, values, availability, beta)
        trajectory.append(tuple(x.detach().clone() for x in
                                (memory, keys, values, result)))
        return result

    with patch.object(model, 'block_write', side_effect=capture):
        output = model(*args, **kwargs)
    return output, trajectory


@pytest.mark.parametrize('bidirectional', [False, True])
@pytest.mark.parametrize('variant', ['original', 'zero', 'random', 'shuffled'])
def test_read_node_preserves_every_write_and_real_slot_projection(variant, bidirectional):
    args = _inputs()
    model = _model(bidirectional=bidirectional).eval()
    node = args[0]
    reads = {'original': node, 'zero': torch.zeros_like(node),
             'random': torch.randn_like(node), 'shuffled': node.flip(0)}
    read = reads[variant]
    (_, context), reference = _run_trace(model, args)
    (_, changed), trajectory = _run_trace(model, args, read_node=read, write_node=node)
    assert len(reference) == len(trajectory) == (8 if bidirectional else 4)
    for expected, actual in zip(reference, trajectory):
        for left, right in zip(expected, actual):
            assert torch.equal(left, right)
    keys, values, query = model._project_sequence(*args[:4])
    new_keys, new_values, new_query = model._project_sequence(
        *args[:4], read_node=read, write_node=node)
    for name in MODALITIES:
        assert torch.equal(keys[name], new_keys[name])
        assert torch.equal(values[name], new_values[name])
    assert torch.equal(query, new_query) == (variant == 'original')
    assert torch.equal(context['local'], changed['local']) == (variant == 'original')


@pytest.mark.parametrize('bidirectional', [False, True])
def test_default_is_exact_historical_forward_backward_parameters_and_rng(bidirectional):
    encoded = os.environ.get('OSRAM_B2_HISTORICAL_SOURCE_B64')
    source = base64.b64decode(encoded).decode() if encoded else subprocess.check_output(
        ['git', 'show', 'fca2aa3890a2fddf9b08aa200a1a8372b675bd1e:gcnet_missing_m3/osram.py'],
        cwd=Path(__file__).resolve().parents[1], text=True)
    namespace = {'__name__': 'historical_osram_b2'}
    exec(compile(source, 'historical_osram_b2.py', 'exec'), namespace)
    torch.manual_seed(29)
    historical = _model(namespace['OSRAMBackbone'], dropout=.2, bidirectional=bidirectional)
    initialization_rng = torch.get_rng_state().clone()
    torch.manual_seed(29)
    model = _model(dropout=.2, bidirectional=bidirectional)
    assert torch.equal(initialization_rng, torch.get_rng_state())
    assert list(model.state_dict()) == list(historical.state_dict())
    for name, value in historical.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    torch.nn.init.normal_(historical.emotion_adapter[-1].weight, std=.1)
    model.load_state_dict(historical.state_dict(), strict=True)
    args = _inputs()
    for tensor in (args[0], *args[1].values()):
        tensor.requires_grad_()
    new_args = copy.deepcopy(args)
    state = torch.get_rng_state().clone()
    expected, expected_context = historical(*args)
    expected.square().sum().backward()
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    actual, actual_context = model(*new_args)
    actual.square().sum().backward()
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert torch.equal(expected, actual)
    for name in expected_context:
        assert torch.equal(expected_context[name], actual_context[name])
    for name, parameter in model.named_parameters():
        grad = dict(historical.named_parameters())[name].grad
        assert (grad is None) == (parameter.grad is None)
        if grad is not None:
            assert torch.equal(grad, parameter.grad), name
    for old, new in zip((args[0], *args[1].values()), (new_args[0], *new_args[1].values())):
        assert torch.equal(old.grad, new.grad)


def test_read_before_write_causality_padding_and_write_side_gap_keys():
    args = _inputs()
    model = _model().eval()
    read = torch.randn_like(args[0])
    (hidden, context), trajectory = _run_trace(model, args, read_node=read)
    assert torch.count_nonzero(context['base'][0]) == 0
    assert torch.count_nonzero(context['gap'][0]) == 0
    valid = args[4].T.bool()
    for value in (hidden, *context.values()):
        assert torch.count_nonzero(value[~valid]) == 0
    assert torch.equal(trajectory[2][0][1], trajectory[1][3][1])
    assert torch.equal(trajectory[3][0][1], trajectory[2][3][1])
    changed = copy.deepcopy(args)
    for value in (changed[0], *changed[1].values()):
        value[2:] = torch.randn_like(value[2:])
    other_read = read.clone()
    other_read[2:] = torch.randn_like(other_read[2:])
    other_hidden, other_context = model(*changed, read_node=other_read)
    assert torch.equal(hidden[:2], other_hidden[:2])
    for name in context:
        assert torch.equal(context[name][:2], other_context[name][:2])
    write_keys, _, _ = model._project_sequence(*args[:4])
    captured = []
    original = model._address_residual

    def capture(keys, query):
        captured.append(keys.detach().clone())
        return original(keys, query)

    with patch.object(model, '_address_residual', side_effect=capture):
        model(*args, read_node=read)
    for time in range(4):
        expected = torch.stack([write_keys[n][time] for n in MODALITIES], -1)
        expected *= args[2][time, :, None, None, :]
        for observed in captured[time * 3: time * 3 + 3]:
            assert torch.equal(observed, expected)


def test_write_node_changes_only_key_projection():
    args = _inputs()
    model = _model().eval()
    keys, values, queries = model._project_sequence(*args[:4])
    new_keys, new_values, new_queries = model._project_sequence(
        *args[:4], write_node=torch.randn_like(args[0]))
    assert torch.equal(queries, new_queries)
    for name in MODALITIES:
        assert not torch.equal(keys[name], new_keys[name])
        assert torch.equal(values[name], new_values[name])
