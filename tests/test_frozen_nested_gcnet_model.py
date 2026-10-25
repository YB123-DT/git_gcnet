"""Original Nested changes only the Adapter input on one frozen Flat scan."""
import copy
import importlib
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from osram_four_block_wiring.frozen_flat import FrozenFlatReader
from tests.test_completion_memory_write import trace
from tests.test_three_block_joint_model import parent_model, view


def nested_model(parent=None):
    name = 'experiments.osram_frozen_nested_gcnet_20261025.model'
    assert importlib.util.find_spec(name) is not None, 'frozen Nested wrapper is missing'
    return importlib.import_module(name).FrozenNestedModel(
        pretrained_parent() if parent is None else parent)


def pretrained_parent():
    parent = parent_model()
    # A real pretrained Flat head is not the original zero-output initialization.
    torch.nn.init.normal_(parent.osram.emotion_adapter[-1].weight, std=.01)
    torch.nn.init.normal_(parent.osram.emotion_adapter[-1].bias, std=.01)
    return parent.eval()


def open_decoder(model):
    for layer in (model.block.core.local_decoder, *model.block.core.memory_decoders):
        torch.nn.init.normal_(layer.weight, std=.002)


def parent_call(parent, data):
    return parent([data['incomplete']], data['availability'], data['qmask'],
                  data['umask'], data['lengths'], predict_missing=False)


def read_frame(parent, data):
    return FrozenFlatReader(parent)(data['incomplete'], data['availability'],
                                   data['qmask'], data['umask'], data['lengths'])


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(74)


def test_only_original_nested_parameters_train_and_strict_restore():
    model = nested_model().train()
    assert model.training and model.block.training
    assert not any(module.training for module in model.parent.modules())
    assert all(not p.requires_grad for p in model.parent.parameters())
    assert all(p.requires_grad for p in model.block.parameters())
    assert sum(p.numel() for p in model.block.parameters()) == 159235
    assert {name.split('.')[0] for name in model.state_dict()} == {'parent', 'block'}
    open_decoder(model)
    restored = nested_model()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert all(torch.equal(value, restored.state_dict()[key])
               for key, value in model.state_dict().items())
    model.eval()
    restored.eval()
    data = view()
    assert torch.equal(model(data)['prediction'], restored(data)['prediction'])


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('batch_size', [1, 2, 32])
def test_zero_initialization_exact_original_one_scan_rng_and_state(training, batch_size):
    original = pretrained_parent()
    model = nested_model(copy.deepcopy(original)).train(training)
    data = view(batch_size)
    state = {key: value.clone() for key, value in model.parent.state_dict().items()}
    torch.manual_seed(91)
    expected = parent_call(original, data)[0]
    expected_rng = torch.random.get_rng_state()
    scans = []
    handle = model.parent.osram.register_forward_hook(lambda *args: scans.append(1))
    torch.manual_seed(91)
    with patch.object(model.parent.osram, '_scan', wraps=model.parent.osram._scan) as scan:
        result = model(data)
    handle.remove()
    assert scans == [1] and scan.call_count == 1
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    assert torch.equal(result['prediction'], expected)
    assert torch.equal(result['base_prediction'], expected)
    assert result['aux_loss'].item() == 0 and result['aux_loss'].ndim == 0
    for key, value in model.parent.state_dict().items():
        assert torch.equal(value, state[key]), key


def test_frozen_head_backpropagates_only_to_nested_and_updates_upstream():
    model = nested_model().train()
    open_decoder(model)
    before_parent = {key: value.clone() for key, value in model.parent.state_dict().items()}
    before_block = {key: value.clone() for key, value in model.block.state_dict().items()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    result = model(view(2))
    result['prediction'].square().mean().backward()
    assert all(p.grad is None for p in model.parent.parameters())
    for module in (model.block.core.local_decoder, model.block.core.tokenizer,
                   model.block.core.core):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    optimizer.step()
    assert any(not torch.equal(value, before_block[key])
               for key, value in model.block.state_dict().items())
    assert all(torch.equal(value, before_parent[key])
               for key, value in model.parent.state_dict().items())


def test_training_uses_one_decode_eval_branch_off_and_cached_live_parity():
    parent = pretrained_parent()
    model = nested_model(copy.deepcopy(parent))
    open_decoder(model)
    data = view(2)
    model.train()
    with patch.object(model.parent.osram.emotion_adapter, 'forward',
                      wraps=model.parent.osram.emotion_adapter.forward) as decode:
        result = model(data)
    assert decode.call_count == 1
    assert result['base_prediction'] is result['prediction']
    model.eval()
    with patch.object(model.parent.osram.emotion_adapter, 'forward',
                      wraps=model.parent.osram.emotion_adapter.forward) as decode:
        result = model(data)
    assert decode.call_count == 2
    assert torch.equal(result['base_prediction'], parent_call(parent, data)[0])
    frame = read_frame(parent, data)
    with patch.object(model.parent.osram, '_scan', side_effect=AssertionError('cached scan')):
        cached = model.predict_frame(frame)
    assert torch.equal(cached, result['prediction'])
    assert not torch.equal(cached, result['base_prediction'])
    model.train()
    model.zero_grad()
    model.predict_frame(frame).square().mean().backward()
    assert model.block.core.local_decoder.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parent.parameters())


def test_original_raw_skip_write_read_and_active_slot_boundaries():
    original = pretrained_parent()
    model = nested_model(copy.deepcopy(original)).eval()
    open_decoder(model)
    data = view(2)
    data['umask'][1, 2] = 0
    data['lengths'][1] = 2
    data['availability'][2, 1] = 0
    before_skip, after_skip, changed_inputs = [], [], []
    before_layout, after_layout = [], []

    def capture_skip(values, layouts):
        def capture(module, args):
            values.append(args[0].clone())
            layouts.append(args[0].stride())
        return capture

    handles = [original.osram.local_skip.register_forward_pre_hook(
        capture_skip(before_skip, before_layout)),
        model.parent.osram.local_skip.register_forward_pre_hook(
        capture_skip(after_skip, after_layout)),
        model.block.register_forward_hook(lambda module, args, output: changed_inputs.append(output))]
    args = ([data['incomplete']], data['availability'], data['qmask'], data['umask'], data['lengths'])
    _, expected_writes, expected_addresses = trace(original, args)
    class LiveCall:
        osram = model.parent.osram

        def __call__(self, *args, **kwargs):
            return model(data)

    try:
        live, actual_writes, actual_addresses = trace(LiveCall(), args)
    finally:
        for handle in handles:
            handle.remove()
    assert all(torch.equal(before_skip[0], value) for value in after_skip)
    assert all(layout == before_layout[0] for layout in after_layout)
    assert all(torch.equal(a, b) for left, right in zip(expected_writes, actual_writes)
               for a, b in zip(left, right))
    assert all(torch.equal(a, b) for a, b in zip(expected_addresses, actual_addresses))
    frame = live['frame']
    local, base, gap = changed_inputs[0]
    valid = data['umask'].T.bool()
    active = valid[..., None] & ~data['availability'].bool()
    assert torch.equal(base[..., 512:], frame.base[..., 512:])
    assert torch.equal(gap[..., 512:], frame.gap[..., 512:])
    assert gap[~active].count_nonzero() == 0
    assert local[~valid].count_nonzero() == base[~valid].count_nonzero() == 0
    assert torch.equal(local[0], frame.local[0])
    assert torch.equal(base[0], frame.base[0])
    assert torch.equal(gap[0], frame.gap[0])
    assert not torch.equal(local[1, 0], frame.local[1, 0])
    assert torch.equal(live['prediction'][~valid], model.parent.smax_fc.bias[None])


def test_causal_future_invariance_and_no_missing_feature_leakage():
    model = nested_model().eval()
    open_decoder(model)
    data = view(2)
    expected = model(data)['prediction']
    changed = copy.deepcopy(data)
    changed['incomplete'][2] = torch.randn_like(changed['incomplete'][2]) * 100
    changed['availability'][2] = 1
    assert torch.equal(model(changed)['prediction'][:2], expected[:2])
    changed = copy.deepcopy(data)
    offset = 0
    for index, width in enumerate((3, 4, 5)):
        changed['incomplete'][..., offset:offset + width][~data['availability'][..., index].bool()] = float('nan')
        offset += width
    assert torch.equal(model(changed)['prediction'], expected)


def test_cached_nonzero_backward_and_empty_history_are_preserved():
    model = nested_model().eval()
    open_decoder(model)
    availability = torch.tensor([[[1, 0, 1]], [[0, 1, 0]], [[1, 0, 1]], [[0, 0, 0]]]).bool()
    valid = torch.tensor([[1], [1], [1], [0]]).bool()
    frame = SimpleNamespace(local=torch.randn(4, 1, 256),
                            base=torch.randn(4, 1, 1024),
                            gap=torch.randn(4, 1, 3, 1024),
                            availability=availability, valid=valid)
    # The second utterance has no forward evidence, despite nonzero backward
    # slots. It must not acquire synthetic history from graph embeddings.
    frame.base[:2, ..., :512] = 0
    frame.gap[:2, ..., :512] = 0
    frame.gap[availability | ~valid[..., None]] = 0
    frame.local[~valid] = 0
    frame.base[~valid] = 0
    outputs = []
    handle = model.block.register_forward_hook(lambda module, args, output: outputs.append(output))
    try:
        model.predict_frame(frame)
    finally:
        handle.remove()
    local, base, gap = outputs[0]
    assert torch.equal(base[..., 512:], frame.base[..., 512:])
    assert torch.equal(gap[..., 512:], frame.gap[..., 512:])
    assert torch.equal(local[:2], frame.local[:2])
    assert torch.equal(base[:2], frame.base[:2])
    assert torch.equal(gap[:2], frame.gap[:2])
    assert not torch.equal(local[2], frame.local[2])
    assert gap[availability | ~valid[..., None]].count_nonzero() == 0


def test_adapter_hook_is_removed_when_branch_raises():
    model = nested_model().train()
    adapter = model.parent.osram.emotion_adapter
    before = tuple(adapter._forward_pre_hooks)
    with patch.object(model.block, 'forward', side_effect=RuntimeError('nested failure')):
        with pytest.raises(RuntimeError, match='nested failure'):
            model(view())
    assert tuple(adapter._forward_pre_hooks) == before
    assert model(view())['prediction'].shape == (3, 1, 1)
