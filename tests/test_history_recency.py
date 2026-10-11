"""Strict-past modality history metadata for the original Flat readout."""

import importlib

import pytest
import torch
from unittest.mock import patch


def metadata(availability, umask):
    module = importlib.import_module('gcnet_missing_m3.history_recency')
    return module.strict_past_history_recency(availability, umask)


def test_history_recency_is_strictly_past_and_marks_never_seen():
    availability = torch.tensor([
        [[1, 0, 0]],
        [[0, 1, 0]],
        [[0, 0, 1]],
        [[1, 0, 0]],
        [[0, 0, 0]],
    ])
    actual = metadata(availability, torch.ones(1, 5))[:, 0]
    expected = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 1, 0, 2, 1, 0],
        [1, 1, 1, 3, 2, 1],
        [1, 1, 1, 1, 3, 2],
    ], dtype=actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_current_availability_cannot_change_current_metadata():
    prefix = torch.tensor([[[1, 0, 1]], [[0, 1, 0]]])
    left = torch.cat((prefix, torch.tensor([[[1, 1, 1]]])), 0)
    right = torch.cat((prefix, torch.tensor([[[0, 0, 1]]])), 0)
    umask = torch.ones(1, 3)
    torch.testing.assert_close(metadata(left, umask)[2], metadata(right, umask)[2])


def test_padding_is_zero_and_does_not_update_history():
    availability = torch.tensor([
        [[1, 0, 0]],
        [[0, 1, 1]],  # padded and must not enter history
        [[0, 0, 0]],
    ])
    umask = torch.tensor([[1., 0., 1.]])
    actual = metadata(availability, umask)[:, 0]
    torch.testing.assert_close(actual[1], torch.zeros(6))
    torch.testing.assert_close(actual[2], torch.tensor([1., 0., 0., 2., 0., 0.]))


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError):
        metadata(torch.ones(2, 3), torch.ones(1, 2))
    with pytest.raises(ValueError):
        metadata(torch.ones(2, 1, 3), torch.ones(2, 1))


def backbone(enabled=False):
    from gcnet_missing_m3.osram import OSRAMBackbone
    return OSRAMBackbone(
        latent_dim=8, output_dim=10, num_heads=2, key_dim=3, value_dim=3,
        dropout=0, n_speakers=1, bidirectional=False,
        osram_history_recency=enabled,
    )


def backbone_inputs():
    node = torch.randn(4, 1, 8)
    latents = {name: torch.randn_like(node) for name in ('audio', 'text', 'visual')}
    availability = torch.tensor([
        [[1, 0, 1]], [[0, 1, 1]], [[1, 1, 0]], [[0, 0, 0]],
    ])
    umask = torch.tensor([[1., 1., 1., 0.]])
    return node, latents, availability, torch.zeros(1, 4, dtype=torch.long), umask, [3]


def test_backbone_switch_only_extends_flat_adapter_and_preserves_memory_scan():
    torch.manual_seed(31)
    original = backbone(False).eval()
    original_rng = torch.random.get_rng_state()
    torch.manual_seed(31)
    changed = backbone(True).eval()
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    original_dim = 8 + 4 * original.context_dim
    assert original.emotion_adapter[0].normalized_shape == (original_dim,)
    assert changed.emotion_adapter[0].normalized_shape == (original_dim + 6,)
    for key, value in original.state_dict().items():
        if not key.startswith('emotion_adapter.'):
            assert torch.equal(value, changed.state_dict()[key]), key

    states = [[], []]
    outputs = []
    inputs = backbone_inputs()
    for index, model in enumerate((original, changed)):
        write = model.block_write
        def capture(*args, _index=index, _write=write, **kwargs):
            state = _write(*args, **kwargs)
            states[_index].append(state.detach().clone())
            return state
        with patch.object(model, 'block_write', side_effect=capture):
            outputs.append(model(*inputs)[0])
    for left, right in zip(*states):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.count_nonzero(outputs[1][-1]) == 0
    assert changed.last_history_recency.shape == (4, 1, 6)
    torch.testing.assert_close(changed.last_history_recency[-1], torch.zeros(1, 6))


def test_history_recency_path_has_finite_gradients():
    model = backbone(True).train()
    output = model(*backbone_inputs())[0]
    output.square().sum().backward()
    for parameter in model.emotion_adapter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_train_config_accepts_only_independent_cfg84_history_recency():
    from gcnet_missing_m3.train_gcnet import TrainConfig
    common = dict(
        backbone_type='osram', osram_bidirectional=False,
        training_objective='emotion-only', disable_unused_aux_modules=True,
        osram_output_dim=1600, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
    )
    assert TrainConfig(**common, osram_history_recency=True).osram_history_recency
    for field, value in (
        ('osram_counterfactual_readout', 'marginal'),
        ('osram_relation_block', True),
        ('paired_history_views', True),
    ):
        with pytest.raises(ValueError, match='history recency'):
            TrainConfig(**common, osram_history_recency=True, **{field: value})

