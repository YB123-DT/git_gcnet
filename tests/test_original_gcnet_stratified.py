from __future__ import annotations

import importlib

import pytest
import torch

from gcnet_modality_jepa.model import GraphModel


ASSERT_CLOSE = getattr(torch.testing, "assert_close", torch.testing.assert_allclose)


def _model_arguments() -> dict:
    return {
        "base_model": "LSTM",
        "adim": 2,
        "tdim": 3,
        "vdim": 4,
        "D_e": 4,
        "graph_hidden_size": 2,
        "n_speakers": 2,
        "window_past": 1,
        "window_future": 1,
        "n_classes": 6,
        "dropout": 0.0,
        "no_cuda": True,
    }


def _locked_reference() -> GraphModel:
    return GraphModel(
        **_model_arguments(),
        time_attn=False,
        enable_reconstruction=True,
        graph_branch_mode="both",
        recurrent_padding_mode="legacy",
        postgraph_sequence_mode="independent",
        graph_message_calibration="none",
    )


def _control_class():
    try:
        module = importlib.import_module("gcnet_original_stratified")
    except ModuleNotFoundError:
        pytest.fail("OriginalGCNetControl has not been implemented")
    return module.OriginalGCNetControl


def _inputs(*, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(19)
    features = torch.randn(3, 2, 9, generator=generator)
    features.requires_grad_(requires_grad)
    availability = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    qmask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    return [features], availability, qmask, umask, [3, 2]


def _scalar_loss(outputs) -> torch.Tensor:
    logits, reconstruction, hidden = outputs[:3]
    return (
        logits.square().mean()
        + reconstruction[0].square().mean()
        + hidden.square().mean()
    )


def test_control_is_an_exact_zero_parameter_locked_graph_model() -> None:
    control_class = _control_class()
    torch.manual_seed(66)
    reference = _locked_reference()
    torch.manual_seed(66)
    control = control_class(**_model_arguments())

    assert control.reconstruction_loss_variant == "corrected-formal-repo"
    assert control.enable_reconstruction is True
    assert control.graph_branch_mode == "both"
    assert control.recurrent_padding_mode == "legacy"
    assert control.postgraph_sequence_mode == "independent"
    assert control.graph_message_calibration == "none"
    assert control.time_attn is False
    assert list(control.state_dict()) == list(reference.state_dict())
    assert sum(parameter.numel() for parameter in control.parameters()) == sum(
        parameter.numel() for parameter in reference.parameters()
    )
    for name, tensor in reference.state_dict().items():
        ASSERT_CLOSE(control.state_dict()[name], tensor, rtol=0, atol=0)

    forbidden = {"teacher", "projector", "mmoe", "predictor", "completion"}
    assert not any(
        token in name.lower()
        for name, _ in control.named_parameters()
        for token in forbidden
    )


def test_control_forward_matches_reference_and_ignores_availability() -> None:
    control_class = _control_class()
    torch.manual_seed(67)
    reference = _locked_reference().eval()
    torch.manual_seed(67)
    control = control_class(**_model_arguments()).eval()
    inputfeats, availability, qmask, umask, lengths = _inputs()

    expected = reference(inputfeats, qmask, umask, lengths)
    actual = control(
        inputfeats,
        availability,
        qmask,
        umask,
        lengths,
        predict_missing=False,
    )
    alternate = control(
        inputfeats,
        torch.zeros_like(availability),
        qmask,
        umask,
        lengths,
    )

    assert actual[3] is None
    assert alternate[3] is None
    for expected_tensor, actual_tensor, alternate_tensor in zip(
        (expected[0], expected[1][0], expected[2]),
        (actual[0], actual[1][0], actual[2]),
        (alternate[0], alternate[1][0], alternate[2]),
    ):
        ASSERT_CLOSE(actual_tensor, expected_tensor, rtol=0, atol=0)
        ASSERT_CLOSE(alternate_tensor, expected_tensor, rtol=0, atol=0)


def test_control_backward_matches_reference_for_inputs_and_all_parameters() -> None:
    control_class = _control_class()
    torch.manual_seed(68)
    reference = _locked_reference().eval()
    torch.manual_seed(68)
    control = control_class(**_model_arguments()).eval()
    reference_inputs = _inputs(requires_grad=True)
    control_inputs = _inputs(requires_grad=True)

    reference_outputs = reference(
        reference_inputs[0],
        reference_inputs[2],
        reference_inputs[3],
        reference_inputs[4],
    )
    control_outputs = control(
        control_inputs[0],
        control_inputs[1],
        control_inputs[2],
        control_inputs[3],
        control_inputs[4],
    )
    _scalar_loss(reference_outputs).backward()
    _scalar_loss(control_outputs).backward()

    ASSERT_CLOSE(
        control_inputs[0][0].grad,
        reference_inputs[0][0].grad,
        rtol=0,
        atol=1e-6,
    )
    reference_parameters = dict(reference.named_parameters())
    control_parameters = dict(control.named_parameters())
    assert list(control_parameters) == list(reference_parameters)
    for name, reference_parameter in reference_parameters.items():
        control_gradient = control_parameters[name].grad
        reference_gradient = reference_parameter.grad
        assert (control_gradient is None) == (reference_gradient is None), name
        if reference_gradient is not None:
            ASSERT_CLOSE(
                control_gradient,
                reference_gradient,
                rtol=0,
                atol=1e-6,
            )


def test_control_rejects_missing_prediction() -> None:
    control = _control_class()(**_model_arguments())
    inputfeats, availability, qmask, umask, lengths = _inputs()

    with pytest.raises(ValueError, match="predict_missing"):
        control(
            inputfeats,
            availability,
            qmask,
            umask,
            lengths,
            predict_missing=True,
        )
