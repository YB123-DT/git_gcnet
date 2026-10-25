"""Frozen interventions preserve endpoints/support while isolating path order."""
from types import SimpleNamespace

import pytest
import torch

from osram_four_block_wiring.blocks import SignatureBlock
from osram_four_block_wiring.frozen_flat import FlatFrame


def make_frame(t=12, b=2):
    generator = torch.Generator().manual_seed(901)
    frame = SimpleNamespace(
        latents=torch.randn(t, b, 3, 256, generator=generator),
        local=torch.randn(t, b, 256, generator=generator),
        availability=torch.ones(t, b, 3, dtype=torch.bool),
        valid=torch.ones(t, b, dtype=torch.bool),
    )
    zeros = torch.zeros(t, b, 1)
    return FlatFrame(**vars(frame), base=torch.zeros(t, b, 1024), gap=torch.zeros(t, b, 3, 1024),
                     context=torch.zeros(t, b, 2304), anchor=zeros.expand(t, b, 1600),
                     hidden=zeros.expand(t, b, 1600), prediction=zeros)


def control(block, frame, mode, seed=66):
    from experiments.osram_signature_order_audit_20261025.controls import descriptor_control
    return descriptor_control(block, frame, mode, shuffle_seed=seed)


def levels(descriptor):
    signatures = descriptor[..., :270].reshape(*descriptor.shape[:2], 3, 90)
    return signatures[..., :9], signatures[..., 9:].reshape(*descriptor.shape[:2], 3, 9, 9)


def test_normal_is_bitexact_original_descriptor_with_padding_and_missing():
    block, frame = SignatureBlock(), make_frame()
    frame.valid[2, 1] = False
    frame.valid[9:, 1] = False
    frame.availability[::2, :, 1] = False
    descriptor, eligible, diagnostics = control(block, frame, "normal")
    assert torch.equal(descriptor, block.descriptor(frame))
    assert eligible.shape == frame.valid.shape
    assert torch.equal(eligible, diagnostics["path_eligible"].any(-1) & frame.valid)
    assert not eligible[~frame.valid].any()


def test_shuffle_preserves_time_endpoints_observed_set_counts_and_first_level():
    block, frame = SignatureBlock(), make_frame()
    frame.availability[::3, :, 1] = False
    original, eligible, before = control(block, frame, "normal")
    changed, shuffled_eligible, after = control(block, frame, "interior_shuffle")
    assert torch.equal(eligible, shuffled_eligible)
    assert torch.equal(before["observed"], after["observed"])
    assert torch.equal(before["counts"], after["counts"])
    assert torch.equal(original[..., 270:], changed[..., 270:])
    torch.testing.assert_close(levels(original)[0], levels(changed)[0], atol=1e-6, rtol=1e-6)
    points, permuted = after["original_points"], after["controlled_points"]
    assert torch.equal(points[..., -1], permuted[..., -1])
    observed = after["observed"]
    ordinal = observed.long().cumsum(-1)
    endpoints = observed & ((ordinal == 1) | (ordinal == observed.sum(-1, keepdim=True)))
    assert torch.equal(points[endpoints], permuted[endpoints])
    assert torch.equal(points[~observed], permuted[~observed])
    identity = torch.arange(block.window).expand_as(after["permutation"])
    changed_permutation = (after["permutation"] != identity).any(-1)
    assert torch.equal(changed_permutation, after["path_eligible"])
    for t, b, modality in after["path_eligible"].nonzero().tolist():
        slots = observed[t, b, modality]
        p = after["permutation"][t, b, modality]
        assert sorted(p[slots].tolist()) == slots.nonzero().flatten().tolist()
    assert not torch.equal(levels(original)[1][eligible], levels(changed)[1][eligible])


def test_shuffle_is_reproducible_and_does_not_touch_global_rng():
    block, frame = SignatureBlock(), make_frame()
    before = torch.random.get_rng_state().clone()
    a, _, _ = control(block, frame, "interior_shuffle", 66)
    assert torch.equal(before, torch.random.get_rng_state())
    b, _, _ = control(block, frame, "interior_shuffle", 66)
    c, _, _ = control(block, frame, "interior_shuffle", 67)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_no_area_preserves_first_level_symmetric_second_level_and_all_other_inputs():
    block, frame = SignatureBlock(), make_frame()
    original, eligible, _ = control(block, frame, "normal")
    changed, changed_eligible, _ = control(block, frame, "degree2_no_area")
    first, second = levels(original)
    new_first, new_second = levels(changed)
    assert torch.equal(eligible, changed_eligible)
    assert torch.equal(first, new_first)
    assert torch.equal(original[..., 270:], changed[..., 270:])
    torch.testing.assert_close(second + second.transpose(-1, -2),
                               new_second + new_second.transpose(-1, -2), atol=2e-5, rtol=2e-5)
    assert torch.equal(new_second, new_second.transpose(-1, -2))
    assert (second - second.transpose(-1, -2)).abs().max() > .01
    torch.testing.assert_close(new_second, .5 * first[..., :, None] * first[..., None, :])


@pytest.mark.parametrize("mode", ["normal", "interior_shuffle", "degree2_no_area"])
def test_strict_past_future_padding_and_other_conversation_do_not_leak(mode):
    block, frame = SignatureBlock(), make_frame()
    frame.valid[1, 0] = False
    baseline, _, _ = control(block, frame, mode)
    altered = SimpleNamespace(**{key: value.clone() for key, value in vars(frame).items()})
    altered.latents[7:, 0] += 100
    altered.latents[1, 0] += 100
    altered.latents[:, 1] -= 100
    replay, _, _ = control(block, altered, mode)
    assert torch.equal(baseline[:8, 0], replay[:8, 0])


def test_empty_input_short_history_and_invalid_mode():
    block = SignatureBlock()
    descriptor, eligible, diagnostics = control(block, make_frame(t=0), "normal")
    assert descriptor.shape == (0, 2, 532)
    assert eligible.shape == (0, 2)
    assert diagnostics["observed"].shape == (0, 2, 3, 8)
    frame = make_frame(t=4)
    baseline, eligible, _ = control(block, frame, "normal")
    changed, _, _ = control(block, frame, "interior_shuffle")
    assert not eligible.any()
    assert torch.equal(baseline, changed)
    with pytest.raises(ValueError, match="mode"):
        control(block, frame, "unknown")


def test_window_is_valid_utterance_window_not_per_modality_observation_window():
    block, frame = SignatureBlock(window=3), make_frame(t=9, b=1)
    frame.valid[5, 0] = False
    frame.availability[:, :, 1] = False
    frame.availability[1, :, 1] = True
    frame.availability[3, :, 1] = True
    _, _, details = control(block, frame, "normal")
    assert details["counts"][8, 0, 1].item() == 0  # past valid positions are 4,6,7.


@pytest.mark.parametrize("window", [1, 2])
def test_tiny_windows_cannot_permute_interiors(window):
    block, frame = SignatureBlock(window=window), make_frame()
    original, _, _ = control(block, frame, "normal")
    changed, eligible, _ = control(block, frame, "interior_shuffle")
    assert torch.equal(original, changed)
    assert not eligible.any()


def test_controls_match_independent_piecewise_linear_reference():
    from osram_four_block_wiring.blocks import degree_two_signature
    block, frame = SignatureBlock(), make_frame()
    frame.valid[3, 0] = False
    frame.availability[1::3, :, 2] = False
    for mode in ("normal", "interior_shuffle", "degree2_no_area"):
        descriptor, _, details = control(block, frame, mode)
        for t in range(len(frame.valid)):
            for b in range(frame.valid.shape[1]):
                for modality in range(3):
                    observed = details["observed"][t, b, modality]
                    reference = degree_two_signature(details["controlled_points"][t, b, modality][observed])
                    if mode == "degree2_no_area":
                        reference = torch.cat([reference[:9],
                                               (.5 * torch.outer(reference[:9], reference[:9])).flatten()])
                    actual = descriptor[t, b, modality * 90:(modality + 1) * 90]
                    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)


def test_adding_future_steps_does_not_change_past_shuffle_draws():
    block, frame = SignatureBlock(), make_frame()
    original, _, _ = control(block, frame, "interior_shuffle")
    truncated = FlatFrame(**{key: value[:8] for key, value in vars(frame).items()})
    shorter, _, _ = control(block, truncated, "interior_shuffle")
    assert torch.equal(original[:8], shorter)
