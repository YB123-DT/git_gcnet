import pytest
import torch
from torch.nn import functional as F

import gcnet_missing_m3_raw_sdr as raw_sdr_package
from gcnet_missing_m3.model import RawResidualObservedEncoder
from gcnet_missing_m3_sdr_backbone.model import MissingM3SDRModel
from gcnet_missing_m3_raw_sdr.model import MissingM3RawSDRModel


PATTERNS = torch.tensor(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.float32,
)


def _model_arguments(dimensions=(2, 3, 4), **overrides):
    arguments = {
        "base_model": "LSTM",
        "adim": dimensions[0],
        "tdim": dimensions[1],
        "vdim": dimensions[2],
        "D_e": 3,
        "graph_hidden_size": 5,
        "n_speakers": 2,
        "window_past": 2,
        "window_future": 2,
        "n_classes": 1,
        "dropout": 0.0,
        "time_attn": False,
        "no_cuda": True,
        "latent_dim": 8,
        "num_experts": 2,
        "top_k": 1,
        "projector_dropout": 0.0,
        "predictor_dropout": 0.0,
        "graph_branch_mode": "both",
        "classification_completion": False,
        "local_context_residual": False,
        "node_interaction_residual": False,
        "readout_type": "shared",
        "mmoe_variant": "dual-gate",
        "recurrent_padding_mode": "legacy",
        "postgraph_sequence_mode": "independent",
        "graph_message_calibration": "none",
    }
    arguments.update(overrides)
    return arguments


def _model(dimensions=(2, 3, 4), **overrides):
    return MissingM3RawSDRModel(
        **_model_arguments(dimensions=dimensions, **overrides)
    )


def _all_pattern_batch(dimensions=(2, 3, 4)):
    total_length = 4
    lengths = [4, 3]
    features = torch.randn(
        total_length,
        2,
        sum(dimensions),
        dtype=torch.float32,
    )
    umask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0]],
        dtype=torch.float32,
    )
    valid = umask.T.bool()
    availability = torch.zeros(total_length, 2, 3, dtype=torch.float32)
    availability[valid] = PATTERNS
    qmask = torch.zeros(2, total_length, dtype=torch.long)
    for batch_index, length in enumerate(lengths):
        qmask[batch_index, :length] = torch.arange(length) % 2
    return features, availability, qmask, umask, lengths


def _expanded_observed_mask(availability, umask, dimensions):
    valid = umask.T.bool().unsqueeze(-1)
    blocks = [
        availability[..., index : index + 1].bool().expand(
            *availability.shape[:2], width
        )
        for index, width in enumerate(dimensions)
    ]
    return torch.cat(blocks, dim=-1) & valid


def _gradient_total(module):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in gradients)
    return sum(gradient.abs().sum().item() for gradient in gradients)


def test_raw_sdr_model_is_public_thin_subclass_with_fixed_identity():
    model = _model()

    assert raw_sdr_package.MissingM3RawSDRModel is MissingM3RawSDRModel
    assert issubclass(MissingM3RawSDRModel, MissingM3SDRModel)
    assert model.sdr_input_type == "raw-residual"
    assert model.sdr_variant == "sdr-public"
    assert model.representation_type == "slot"
    assert isinstance(model.observed_set, RawResidualObservedEncoder)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("fusion_type", "slot"),
        ("representation_type", "track"),
        ("sdr_input_type", "slot"),
        ("sdr_variant", "sdr-paper"),
    ],
)
def test_raw_sdr_model_rejects_explicit_identity_conflicts(name, value):
    with pytest.raises(ValueError, match=name):
        _model(**{name: value})


def test_formal_raw_sdr_uses_raw_width_public_backbone_without_legacy_modules():
    dimensions = (512, 1024, 1024)
    model = _model(dimensions=dimensions, n_speakers=1)
    state_keys = tuple(model.state_dict())

    assert isinstance(model.observed_set, RawResidualObservedEncoder)
    assert model.conversation_backbone.input_dim == 2560
    assert model.conversation_backbone.variant == "sdr-public"
    assert not hasattr(model.conversation_backbone, "speaker_branch")
    assert not any("speaker_branch" in key for key in state_keys)
    forbidden = (
        "lstm.",
        "gru.",
        "graph_net_temporal.",
        "graph_net_speaker.",
    )
    assert not any(key.startswith(forbidden) for key in state_keys)


def test_zero_adapters_preserve_all_seven_observed_raw_patterns_exactly():
    dimensions = (512, 1024, 1024)
    torch.manual_seed(801)
    model = _model(
        dimensions=dimensions,
        n_speakers=1,
        latent_dim=256,
    ).eval()
    features, availability, _, umask, _ = _all_pattern_batch(dimensions)

    encoded, latents = model.observed_set(features, availability, umask)
    observed = _expanded_observed_mask(availability, umask, dimensions)

    assert set(map(tuple, availability[umask.T.bool()].tolist())) == set(
        map(tuple, PATTERNS.tolist())
    )
    assert all(
        torch.count_nonzero(adapter[-1].weight).item() == 0
        and torch.count_nonzero(adapter[-1].bias).item() == 0
        for adapter in model.observed_set.adapters.values()
    )
    assert torch.equal(encoded, features.masked_fill(~observed, 0.0))
    assert torch.count_nonzero(encoded[~observed]).item() == 0
    assert set(latents) == {"audio", "text", "visual"}
    assert all(value.shape == (4, 2, 256) for value in latents.values())
    for index, name in enumerate(("audio", "text", "visual")):
        unavailable = ~availability[..., index].bool() | ~umask.T.bool()
        assert torch.count_nonzero(latents[name][unavailable]).item() == 0


def test_missing_raw_blocks_cannot_change_encoding_latents_or_inference():
    torch.manual_seed(802)
    model = _model().eval()
    features, availability, qmask, umask, lengths = _all_pattern_batch()
    changed = features.clone()
    start = 0
    valid = umask.T.bool()
    for index, width in enumerate(model.dimensions):
        stop = start + width
        missing = valid & ~availability[..., index].bool()
        block = changed[..., start:stop]
        block[missing] = block[missing] + 1000.0
        start = stop
    assert not torch.equal(changed, features)

    baseline_encoded, baseline_latents = model.observed_set(
        features, availability, umask
    )
    changed_encoded, changed_latents = model.observed_set(
        changed, availability, umask
    )
    with torch.no_grad():
        baseline = model(
            [features],
            availability,
            qmask,
            umask,
            lengths,
            predict_missing=False,
        )
        altered = model(
            [changed],
            availability,
            qmask,
            umask,
            lengths,
            predict_missing=False,
        )

    assert torch.equal(baseline_encoded, changed_encoded)
    for name in baseline_latents:
        assert torch.equal(baseline_latents[name], changed_latents[name])
    assert baseline[3] is altered[3] is None
    assert torch.equal(baseline[0], altered[0])
    assert torch.equal(baseline[1], altered[1])


def test_real_backward_reaches_student_adapter_sdr_graph_heads_not_teacher():
    torch.manual_seed(803)
    model = _model().train()
    features, availability, qmask, umask, lengths = _all_pattern_batch()

    logits, hidden, _, predictions = model(
        [features],
        availability,
        qmask,
        umask,
        lengths,
        predict_missing=True,
    )
    valid = umask.T.bool()
    targets = predictions.target_mask
    loss = (
        F.mse_loss(logits[valid], torch.full_like(logits[valid], 0.25))
        + F.mse_loss(
            predictions.reg_predictions[targets],
            torch.full_like(predictions.reg_predictions[targets], -0.5),
        )
        + F.mse_loss(
            predictions.cl_predictions[targets],
            torch.full_like(predictions.cl_predictions[targets], 0.5),
        )
    )
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(logits).all().item()
    assert torch.isfinite(hidden).all().item()
    assert _gradient_total(model.observed_set.projectors) > 0.0
    assert _gradient_total(model.observed_set.adapters) > 0.0
    assert _gradient_total(model.conversation_backbone.pre_graph_bigru) > 0.0
    assert _gradient_total(
        model.conversation_backbone.temporal_branch.rgcn
    ) > 0.0
    assert _gradient_total(model.smax_fc) > 0.0
    assert _gradient_total(model.missing_predictor) > 0.0
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
