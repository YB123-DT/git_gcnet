"""Locked recent NoArea history and one-live-trajectory readout contracts."""
import copy
import pytest
import torch

from experiments.osram_recent_noarea_serial_20261025.blocks import (
    NoAreaSignatureBlock, SerialNoAreaBlock,
)
from experiments.osram_recent_noarea_serial_20261025.model import RecentNoAreaModel
from tests.test_signature_order_audit_controls import make_frame
from tests.test_three_block_joint_model import parent_model, view


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def randomize_exit(block, mode):
    exits = [block.out[-1]] if mode == 'parallel' else block.decoders
    for layer in exits:
        torch.nn.init.normal_(layer.weight, std=.01)


def test_descriptor_exact_old_noarea_symmetric_second_level_and_no_unused_bridge():
    block, reference, frame = SerialNoAreaBlock(), NoAreaSignatureBlock(), make_frame()
    block.descriptor.load_state_dict(reference.state_dict(), strict=False)
    descriptor = block.history_descriptor(frame)
    expected = reference.descriptor(frame)
    assert torch.equal(descriptor, torch.cat((expected[..., :270], expected[..., -3:]), -1))
    signature = descriptor[..., :270].reshape(12, 2, 3, 90)
    first, second = signature[..., :9], signature[..., 9:].reshape(12, 2, 3, 9, 9)
    assert torch.equal(second, .5 * first[..., :, None] * first[..., None, :])
    assert not any('descriptor.out' in name for name, _ in block.named_parameters())
    assert sum(p.numel() for p in reference.parameters()) == 281856


def test_descriptor_strict_past_real_observed_window_eight():
    block, frame = SerialNoAreaBlock(), make_frame(t=14, b=2)
    frame.valid[3, 0] = False
    frame.availability[::2, 0, 1] = False
    original = block.history_descriptor(frame)
    changed = copy.deepcopy(frame)
    changed.latents[8:, 0] += 100
    changed.latents[3, 0] += 100
    changed.latents[:, 1] -= 100
    changed.latents[~frame.availability] += 100
    assert torch.equal(original[:9, 0], block.history_descriptor(changed)[:9, 0])
    changed = copy.deepcopy(frame)
    changed.latents[:5, 0] += 100
    assert torch.equal(original[13, 0], block.history_descriptor(changed)[13, 0])
    assert torch.equal(original[0, :, -3:], torch.zeros(2, 3))


def test_serial_masks_zero_counterfactual_backward_half_and_local_unchanged():
    block, frame = SerialNoAreaBlock(), make_frame(t=12, b=2)
    randomize_exit(block, 'serial')
    frame.availability[:3, 0] = False
    frame.availability[5, 0] = torch.tensor([True, False, True])
    frame.valid[-1, 1] = False
    original_local = frame.local.clone()
    delta = block(frame)
    assert delta.shape == (12, 2, 4, 512)
    assert not delta[:4, 0].any()
    assert not delta[-1, 1].any()
    assert not delta[5, 0, [1, 3]].any()
    assert delta[5, 0, [0, 2]].abs().sum() > 0
    assert torch.equal(frame.local, original_local)
    zero = torch.zeros(12, 2, 64)
    assert torch.equal(block(frame, history=zero), torch.zeros_like(delta))
    updated_base, updated_gap = block.updated_reads(frame)
    assert torch.equal(updated_base[..., 512:], frame.base[..., 512:])
    assert torch.equal(updated_gap[..., 512:], frame.gap[..., 512:])


@pytest.mark.parametrize('mode', ['parallel', 'serial'])
@pytest.mark.parametrize('stage', ['joint', 'frozen'])
@pytest.mark.parametrize('training', [False, True])
def test_initial_identity_rng_and_single_scan(mode, stage, training):
    torch.manual_seed(4)
    original = parent_model().train(training and stage == 'joint')
    # An already trained Flat bridge must also remain exactly unchanged.
    torch.nn.init.normal_(original.osram.emotion_adapter[-1].weight, std=.01)
    wrapped = RecentNoAreaModel(copy.deepcopy(original), mode, stage).train(training)
    data = view(2)
    scans = []
    handle = wrapped.parent.osram.register_forward_hook(lambda *args: scans.append(1))
    torch.manual_seed(91)
    expected = original([data['incomplete']], data['availability'], data['qmask'],
                        data['umask'], data['lengths'])[0]
    rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    result = wrapped(data)
    handle.remove()
    assert scans == [1]
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(result['prediction'], expected)
    assert torch.equal(result['base_prediction'], expected)
    assert result['aux_loss'].item() == 0
    assert all(p.requires_grad == (stage == 'joint') for p in wrapped.parent.parameters())
    assert wrapped.parent.training == (training and stage == 'joint')


def test_parallel_is_the_exact_existing_class_and_serial_locked_parameter_count():
    from experiments.osram_signature_noarea_iemocap68_20261025.block import NoAreaSignatureBlock as Original
    parallel = RecentNoAreaModel(parent_model(), 'parallel')
    serial = RecentNoAreaModel(parent_model(), 'serial')
    assert type(parallel.block) is Original
    assert sum(p.numel() for p in serial.block.parameters()) == 238798
    assert all(layer.bias is None and not layer.weight.any() for layer in serial.block.decoders)
    assert serial.block.history_projection[0].bias is None
    assert serial.block.interaction[0].normalized_shape == (203,)
    assert serial.block.roles.weight.shape == (4, 8)
    assert not any(isinstance(module, torch.nn.Dropout) for module in serial.block.modules())


@pytest.mark.parametrize('mode', ['parallel', 'serial'])
@pytest.mark.parametrize('stage', ['joint', 'frozen'])
def test_gradients_restore_and_cached_decode(mode, stage):
    torch.manual_seed(44)
    parent = parent_model()
    torch.nn.init.normal_(parent.osram.emotion_adapter[-1].weight, std=.01)
    wrapped = RecentNoAreaModel(parent, mode, stage).train()
    randomize_exit(wrapped.block, mode)
    # Torch 1.13 MKLDNN has a known noncontiguous LSTM backward failure.
    with torch.backends.mkldnn.flags(enabled=False):
        result = wrapped(view(2))
        result['prediction'].square().mean().backward()
    for name, parameter in wrapped.block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    slots = wrapped.block.descriptor.slots if mode == 'serial' else wrapped.block.slots
    assert sum(p.grad.abs().sum().item() for p in slots.parameters()) > 0
    if stage == 'frozen':
        assert all(p.grad is None for p in parent.parameters())
    else:
        assert sum(p.grad.abs().sum().item() for p in parent.parameters() if p.grad is not None) > 0
    wrapped.eval()
    data = view(2)
    from osram_four_block_wiring.frozen_flat import FrozenFlatReader
    cached_parent = parent_model()
    cached_parent.load_state_dict(parent.state_dict())
    frame = FrozenFlatReader(cached_parent)(
        data['incomplete'], data['availability'], data['qmask'], data['umask'], data['lengths'])
    expected = wrapped(data)['prediction']
    torch.testing.assert_close(wrapped.predict_frame(frame), expected, rtol=0, atol=0)
    restored = RecentNoAreaModel(parent_model(), mode, stage).eval()
    restored.load_state_dict(wrapped.state_dict())
    assert torch.equal(restored(data)['prediction'], expected)


@pytest.mark.parametrize('mode', ['parallel', 'serial'])
def test_exception_cleans_all_temporary_hooks(mode):
    wrapped = RecentNoAreaModel(parent_model(), mode)
    modules = [wrapped.parent.observed_set, wrapped.parent.osram.emotion_adapter,
               wrapped.parent.osram.emotion_norm]
    counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]
    def fail(*args):
        raise RuntimeError('injected failure')
    wrapped.block.forward = fail
    with pytest.raises(RuntimeError, match='injected failure'):
        wrapped(view())
    assert counts == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]
