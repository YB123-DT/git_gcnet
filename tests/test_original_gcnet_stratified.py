from __future__ import annotations

import hashlib
import importlib

import pytest
import torch

from gcnet_missing_m3.mixed_rate import MISSING_RATES, stratified_rates_for_batch
from gcnet_modality_jepa.model import GraphModel
from gcnet_original_stratified.train_gcnet import (
    OriginalTrainConfig,
    original_control_loss,
    train_epoch,
)


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


def test_original_control_loss_uses_corrected_missing_only_modality_mse() -> None:
    logits = torch.randn(2, 2, 6, requires_grad=True)
    prediction = torch.zeros(2, 2, 6, requires_grad=True)
    complete = torch.tensor(
        [
            [[1.0, 3.0, 20.0, 30.0, 40.0, 50.0], [99.0] * 6],
            [[8.0, 9.0, 2.0, 4.0, 6.0, 7.0], [99.0] * 6],
        ]
    )
    availability = torch.tensor(
        [
            [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    labels = torch.tensor([[0, 1], [2, 3]])
    umask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    total, task, reconstruction = original_control_loss(
        logits=logits,
        reconstruction=[prediction],
        complete_features=complete,
        availability=availability,
        labels=labels,
        umask=umask,
        dataset="IEMOCAPSix",
        dimensions=(2, 3, 1),
    )

    audio_missing_mse = complete[0, 0, :2].square().mean()
    text_missing_mse = complete[1, 0, 2:5].square().mean()
    expected = torch.stack((audio_missing_mse, text_missing_mse)).mean()
    ASSERT_CLOSE(reconstruction, expected, rtol=0, atol=0)
    ASSERT_CLOSE(total, task + expected, rtol=0, atol=0)


def test_original_control_loss_is_differentiable_exact_zero_for_atv() -> None:
    logits = torch.randn(2, 2, 6, requires_grad=True)
    prediction = torch.randn(2, 2, 6, requires_grad=True)
    complete = torch.randn(2, 2, 6)
    umask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    availability = umask.T.unsqueeze(-1).expand(-1, -1, 3).clone()

    _, _, reconstruction = original_control_loss(
        logits=logits,
        reconstruction=[prediction],
        complete_features=complete,
        availability=availability,
        labels=torch.tensor([[0, 1], [2, 3]]),
        umask=umask,
        dataset="IEMOCAPSix",
        dimensions=(2, 3, 1),
    )
    reconstruction.backward()

    assert reconstruction.item() == 0.0
    assert reconstruction.requires_grad
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad).item() == 0


def test_original_train_config_rejects_non_stratified_modes() -> None:
    for mode in ("fixed", "cyclic", "all"):
        with pytest.raises(ValueError, match="stratified"):
            OriginalTrainConfig(train_rate_mode=mode)


class _SingleBatchLoader:
    def __init__(self, conversation_ids):
        self.dataset = tuple(conversation_ids)
        self.sampler = tuple(range(len(conversation_ids)))
        self._batch = (tuple(conversation_ids),)

    def __iter__(self):
        yield self._batch


class _CountingOriginalControl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.forward_count = 0

    def forward(
        self,
        inputfeats,
        availability,
        qmask,
        umask,
        seq_lengths,
        predict_missing=False,
    ):
        assert predict_missing is False
        self.forward_count += 1
        length, batch = inputfeats[0].shape[:2]
        class_axis = torch.arange(
            6, device=self.scale.device, dtype=self.scale.dtype
        ).view(1, 1, 6)
        logits = self.scale * class_axis.expand(length, batch, -1)
        reconstruction = [self.scale * torch.ones_like(inputfeats[0])]
        hidden = self.scale * torch.ones(length, batch, 2)
        return logits, reconstruction, hidden, None


def test_train_epoch_uses_one_view_update_and_exact_stratified_hash(
    monkeypatch,
) -> None:
    import gcnet_original_stratified.train_gcnet as trainer

    conversation_ids = tuple(f"dialogue-{index}" for index in range(32))
    loader = _SingleBatchLoader(conversation_ids)
    config = OriginalTrainConfig(dataset="IEMOCAPSix", fold=5, seed=66)
    model = _CountingOriginalControl()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    seen_rates = []

    monkeypatch.setattr(trainer, "_move_batch", lambda raw, _device: raw)

    def fake_prepare(data, schedules, rates, epoch, dimensions):
        del schedules, epoch
        seen_rates.extend(rates)
        batch = len(data[-1])
        availability = torch.ones(1, batch, 3)
        for index, rate in enumerate(rates):
            if rate > 0:
                availability[0, index, 0] = 0
            if rate >= 0.4:
                availability[0, index, 1] = 0
        complete = torch.arange(
            1, 1 + batch * sum(dimensions), dtype=torch.float32
        ).reshape(1, batch, sum(dimensions))
        expanded = torch.repeat_interleave(
            availability,
            torch.tensor(dimensions),
            dim=-1,
        )
        return {
            "incomplete": complete * expanded,
            "complete": complete,
            "availability": availability,
            "qmask": torch.zeros(batch, 1),
            "umask": torch.ones(batch, 1),
            "lengths": [1] * batch,
            "labels": torch.zeros(batch, 1, dtype=torch.long),
        }

    monkeypatch.setattr(trainer, "_prepare_stratified_view", fake_prepare)
    metrics = train_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        config=config,
        schedules={rate: object() for rate in MISSING_RATES},
        epoch=0,
        dimensions=(2, 3, 1),
        device=torch.device("cpu"),
    )

    expected_assignment = stratified_rates_for_batch(
        MISSING_RATES,
        master_seed=66,
        dataset="IEMOCAPSix",
        fold=5,
        epoch=0,
        batch_index=0,
        epoch_size=32,
        conversations_seen=0,
        conversation_ids=conversation_ids,
    )
    expected_digest = hashlib.sha256()
    expected_digest.update(b"\0")
    expected_digest.update(expected_assignment.assignment_hash.encode("ascii"))

    assert tuple(seen_rates) == expected_assignment.rates
    assert model.forward_count == 1
    assert metrics["source_conversation_count"] == 32
    assert metrics["masked_view_count"] == 32
    assert metrics["model_forward_count"] == 1
    assert metrics["optimizer_steps"] == 1
    assert metrics["rate_conversation_counts"] == {
        str(rate): 4 for rate in MISSING_RATES
    }
    assert metrics["jepa_target_count"] == 0
    assert metrics["reconstruction_target_count"] == sum(
        metrics["rate_missing_modality_counts"].values()
    )
    assert metrics["stratified_assignment_hash"] == expected_digest.hexdigest()
