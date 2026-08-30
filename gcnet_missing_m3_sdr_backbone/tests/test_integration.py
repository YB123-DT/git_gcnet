import inspect

import pytest
import torch

import gcnet_missing_m3_sdr_backbone as sdr_package
import gcnet_missing_m3_sdr_backbone.model as sdr_model
from gcnet_missing_m3.model import MissingM3GraphModel


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

LOCKED_CONFIGURATION_FAILURES = (
    ("base_model", "GRU"),
    ("fusion_type", "mean"),
    ("representation_type", "track"),
    ("classification_completion", True),
    ("graph_branch_mode", "temporal-only"),
    ("time_attn", True),
    ("local_context_residual", True),
    ("node_interaction_residual", True),
    ("readout_type", "availability-low-rank"),
    ("mmoe_variant", "paper-faithful"),
    ("recurrent_padding_mode", "packed"),
    ("postgraph_sequence_mode", "shared-bilstm"),
    ("graph_message_calibration", "branch-layernorm-residual"),
)


def _model_arguments(**overrides):
    arguments = {
        "base_model": "LSTM",
        "adim": 2,
        "tdim": 3,
        "vdim": 4,
        "D_e": 200,
        "graph_hidden_size": 100,
        "n_speakers": 2,
        "window_past": 2,
        "window_future": 2,
        "n_classes": 6,
        "dropout": 0.0,
        "time_attn": False,
        "no_cuda": True,
        "latent_dim": 8,
        "num_experts": 2,
        "top_k": 1,
        "projector_dropout": 0.0,
        "predictor_dropout": 0.0,
        "fusion_type": "slot",
        "graph_branch_mode": "both",
        "classification_completion": False,
        "representation_type": "slot",
    }
    arguments.update(overrides)
    return arguments


def _candidate_class():
    candidate_class = getattr(sdr_model, "MissingM3SDRModel", None)
    assert candidate_class is not None, "MissingM3SDRModel is not implemented"
    return candidate_class


def _candidate(variant="sdr-public", **overrides):
    return _candidate_class()(
        **_model_arguments(**overrides),
        sdr_variant=variant,
    )


def _parent_positional_arguments(**overrides):
    values_by_name = _model_arguments(**overrides)
    values = []
    parameters = inspect.signature(MissingM3GraphModel.__init__).parameters
    for parameter in tuple(parameters.values())[1:]:
        if parameter.name in values_by_name:
            values.append(values_by_name[parameter.name])
        else:
            assert parameter.default is not inspect.Parameter.empty
            values.append(parameter.default)
    return values


def _all_pattern_batch():
    total_length = 4
    lengths = [4, 3]
    features = torch.randn(total_length, 2, 9, dtype=torch.float32)
    umask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0]],
        dtype=torch.float32,
    )
    valid = umask.T.bool()
    availability = torch.zeros(total_length, 2, 3, dtype=torch.float32)
    availability[valid] = PATTERNS
    qmask = torch.zeros(2, total_length, dtype=torch.long)
    for batch_index, length in enumerate(lengths):
        qmask[batch_index, :length] = (
            torch.arange(length, dtype=torch.long) % 2
        )
    return features, availability, qmask, umask, lengths


def test_missing_m3_sdr_model_is_public_and_subclasses_control():
    candidate_class = _candidate_class()

    assert sdr_package.MissingM3SDRModel is candidate_class
    assert issubclass(candidate_class, MissingM3GraphModel)
    parameter = inspect.signature(candidate_class.__init__).parameters[
        "sdr_variant"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("variant", ["sdr-public", "sdr-paper"])
def test_shared_initialization_is_bitwise_equal_to_lstm_control(variant):
    shared_prefixes = (
        "observed_set.",
        "teacher.",
        "missing_predictor.",
        "smax_fc.",
    )
    torch.manual_seed(701)
    control = MissingM3GraphModel(**_model_arguments())
    torch.manual_seed(701)
    candidate = _candidate(variant)

    control_state = {
        key: value
        for key, value in control.state_dict().items()
        if key.startswith(shared_prefixes)
    }
    candidate_state = {
        key: value
        for key, value in candidate.state_dict().items()
        if key.startswith(shared_prefixes)
    }

    assert control_state
    assert candidate_state.keys() == control_state.keys()
    for key, control_value in control_state.items():
        assert torch.equal(control_value, candidate_state[key]), key


@pytest.mark.parametrize("variant", ["sdr-public", "sdr-paper"])
def test_legacy_conversation_modules_are_not_registered(variant):
    candidate = _candidate(variant)
    state_keys = tuple(candidate.state_dict())
    removed_prefixes = (
        "lstm.",
        "gru.",
        "graph_net_temporal.",
        "graph_net_speaker.",
    )

    for prefix in removed_prefixes:
        assert not any(key.startswith(prefix) for key in state_keys)
    assert any(key.startswith("conversation_backbone.") for key in state_keys)
    assert candidate.base_model == "LSTM"
    assert isinstance(candidate.conversation_backbone.pre_graph_bigru, torch.nn.GRU)
    assert candidate.conversation_backbone.variant == variant
    assert candidate.conversation_backbone.input_dim == candidate.latent_dim
    assert candidate.conversation_backbone.recurrent_hidden == 200
    assert candidate.conversation_backbone.graph_hidden == 100


@pytest.mark.parametrize(("name", "invalid"), LOCKED_CONFIGURATION_FAILURES)
def test_candidate_rejects_non_experimental_keyword_configuration(name, invalid):
    with pytest.raises(ValueError, match=name):
        _candidate(**{name: invalid})


@pytest.mark.parametrize(("name", "invalid"), LOCKED_CONFIGURATION_FAILURES)
def test_candidate_rejects_non_experimental_positional_configuration(name, invalid):
    candidate_class = _candidate_class()

    with pytest.raises(ValueError, match=name):
        candidate_class(
            *_parent_positional_arguments(**{name: invalid}),
            sdr_variant="sdr-public",
        )


def test_parent_signature_binding_rejects_duplicate_arguments():
    candidate_class = _candidate_class()

    with pytest.raises(TypeError, match="multiple values.*base_model"):
        candidate_class(
            "LSTM",
            **_model_arguments(),
            sdr_variant="sdr-public",
        )


def test_parent_signature_binding_rejects_unknown_arguments():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _candidate(unknown_parent_option=True)


def test_sdr_variant_remains_keyword_only_after_full_parent_positional_binding():
    candidate = _candidate_class()(
        *_parent_positional_arguments(),
        sdr_variant="sdr-paper",
    )

    assert candidate.sdr_variant == "sdr-paper"
    assert candidate.conversation_backbone.variant == "sdr-paper"


@pytest.mark.parametrize("dropout", [False, True])
def test_boolean_keyword_dropout_is_rejected_before_conversion(dropout):
    with pytest.raises(TypeError, match="dropout.*bool"):
        _candidate(dropout=dropout)


@pytest.mark.parametrize("dropout", [False, True])
def test_boolean_positional_dropout_is_rejected_before_conversion(dropout):
    candidate_class = _candidate_class()

    with pytest.raises(TypeError, match="dropout.*bool"):
        candidate_class(
            "LSTM",
            2,
            3,
            4,
            200,
            100,
            2,
            2,
            2,
            6,
            dropout,
            time_attn=False,
            no_cuda=True,
            latent_dim=8,
            num_experts=2,
            top_k=1,
            projector_dropout=0.0,
            predictor_dropout=0.0,
            fusion_type="slot",
            graph_branch_mode="both",
            classification_completion=False,
            representation_type="slot",
            sdr_variant="sdr-public",
        )


def test_candidate_rejects_pre_graph_residual():
    candidate = _candidate()
    values = torch.randn(4, 2, candidate.latent_dim)
    _, _, qmask, umask, lengths = _all_pattern_batch()

    with pytest.raises(ValueError, match="pre_graph_residual"):
        candidate.encode_hidden(
            [values],
            qmask,
            umask,
            lengths,
            pre_graph_residual=torch.zeros_like(values),
        )


@pytest.mark.parametrize("variant", ["sdr-public", "sdr-paper"])
def test_predict_missing_false_skips_predictor_and_teacher_and_zero_pads(
    variant,
    monkeypatch,
):
    torch.manual_seed(702)
    candidate = _candidate(variant).eval()
    features, availability, qmask, umask, lengths = _all_pattern_batch()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("training-only module was executed")

    monkeypatch.setattr(candidate.missing_predictor, "forward", fail_if_called)
    monkeypatch.setattr(candidate.teacher, "forward", fail_if_called)
    for projector in candidate.teacher.values():
        monkeypatch.setattr(projector, "forward", fail_if_called)

    with torch.no_grad():
        logits, hidden, latents, predictions = candidate(
            [features],
            availability,
            qmask,
            umask,
            lengths,
            predict_missing=False,
        )

    valid = umask.T.bool()
    assert logits.shape == (4, 2, 6)
    assert hidden.shape == (4, 2, 500)
    assert set(latents) == {"audio", "text", "visual"}
    assert predictions is None
    assert torch.equal(hidden[~valid], torch.zeros_like(hidden[~valid]))
    assert set(map(tuple, availability[valid].tolist())) == set(
        map(tuple, PATTERNS.tolist())
    )


@pytest.mark.parametrize("variant", ["sdr-public", "sdr-paper"])
def test_predict_missing_true_preserves_control_masks_and_width_contract(variant):
    torch.manual_seed(703)
    control = MissingM3GraphModel(**_model_arguments()).eval()
    torch.manual_seed(703)
    candidate = _candidate(variant).eval()
    features, availability, qmask, umask, lengths = _all_pattern_batch()
    predictor_widths = []
    handle = candidate.missing_predictor.register_forward_pre_hook(
        lambda _module, inputs: predictor_widths.append(inputs[1].shape[-1])
    )

    try:
        with torch.no_grad():
            control_output = control(
                [features],
                availability,
                qmask,
                umask,
                lengths,
                predict_missing=True,
            )
            candidate_output = candidate(
                [features],
                availability,
                qmask,
                umask,
                lengths,
                predict_missing=True,
            )
    finally:
        handle.remove()

    logits, hidden, latents, predictions = candidate_output
    control_predictions = control_output[3]
    assert len(candidate_output) == 4
    assert logits.shape == (4, 2, 6)
    assert hidden.shape == (4, 2, 500)
    assert candidate.smax_fc.in_features == 500
    assert candidate.conversation_backbone.output_dim == 500
    assert all(value.shape == (4, 2, 8) for value in latents.values())
    assert predictions.reg_predictions.shape == (4, 2, 3, 8)
    assert predictions.cl_predictions.shape == (4, 2, 3, 8)
    assert predictions.target_mask.shape == (4, 2, 3)
    assert predictions.source_counts.shape == (4, 2, 3)
    assert torch.equal(
        predictions.target_mask,
        control_predictions.target_mask,
    )
    assert torch.equal(
        predictions.source_counts,
        control_predictions.source_counts,
    )
    assert predictor_widths == [500]


@pytest.mark.parametrize("variant", ["sdr-public", "sdr-paper"])
def test_cpu_forward_backward_is_finite_and_backbone_receives_gradients(variant):
    torch.manual_seed(704)
    candidate = _candidate(variant).train()
    features, availability, qmask, umask, lengths = _all_pattern_batch()
    features.requires_grad_(True)

    logits, hidden, _, predictions = candidate(
        [features],
        availability,
        qmask,
        umask,
        lengths,
        predict_missing=True,
    )
    valid = umask.T.bool()
    loss = (
        logits[valid].square().mean()
        + hidden[valid].square().mean()
        + predictions.reg_predictions[predictions.target_mask].square().mean()
        + predictions.cl_predictions[predictions.target_mask].square().mean()
    )
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(logits).all().item()
    assert torch.isfinite(hidden).all().item()
    assert torch.equal(hidden[~valid], torch.zeros_like(hidden[~valid]))
    assert features.grad is not None
    assert torch.isfinite(features.grad).all().item()
    gradients = [
        parameter.grad
        for parameter in candidate.conversation_backbone.parameters()
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all().item() for gradient in gradients)


@pytest.mark.parametrize("variant", ["sdr-public", "sdr-paper"])
def test_teacher_ema_update_remains_available(variant):
    candidate = _candidate(variant)
    before = {
        key: value.detach().clone()
        for key, value in candidate.teacher.state_dict().items()
    }
    with torch.no_grad():
        for parameter in candidate.observed_set.projectors.parameters():
            parameter.add_(1.0)
    students = {
        key: value.detach().clone()
        for key, value in candidate.observed_set.projectors.state_dict().items()
    }
    tau = 0.75

    candidate.update_teacher(tau)

    assert candidate.ema_step == 1
    assert all(
        not parameter.requires_grad for parameter in candidate.teacher.parameters()
    )
    for key, actual in candidate.teacher.state_dict().items():
        expected = before[key].clone().mul_(tau).add_(
            students[key], alpha=1.0 - tau
        )
        assert torch.equal(actual, expected), key


@pytest.mark.parametrize("variant", ["sdr-public", "sdr-paper"])
def test_parameter_provenance_separates_backbone_and_whole_model(variant):
    candidate = _candidate(
        variant,
        adim=512,
        tdim=1024,
        vdim=1024,
        latent_dim=256,
        n_speakers=1,
        n_classes=1,
    )
    backbone_registered = sum(
        parameter.numel()
        for parameter in candidate.conversation_backbone.parameters()
    )
    backbone_trainable = sum(
        parameter.numel()
        for parameter in candidate.conversation_backbone.parameters()
        if parameter.requires_grad
    )
    model_registered = sum(parameter.numel() for parameter in candidate.parameters())
    model_trainable = sum(
        parameter.numel()
        for parameter in candidate.parameters()
        if parameter.requires_grad
    )

    provenance = {
        "backbone_registered": backbone_registered,
        "backbone_trainable": backbone_trainable,
        "model_registered": model_registered,
        "model_trainable": model_trainable,
    }
    expected = {
        "sdr-public": {
            "backbone_registered": 9_444_901,
            "backbone_trainable": 9_444_901,
            "model_registered": 12_222_238,
            "model_trainable": 11_362_078,
        },
        "sdr-paper": {
            "backbone_registered": 18_038_302,
            "backbone_trainable": 18_038_302,
            "model_registered": 20_815_639,
            "model_trainable": 19_955_479,
        },
    }

    assert provenance == expected[variant]
