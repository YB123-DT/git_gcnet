from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from gcnet_modality_jepa import train_gcnet
from gcnet_modality_jepa.train_gcnet import (
    _backward_and_optimizer_step,
    build_argument_parser,
    build_model,
    validate_training_args,
)
from gcnet_plci_jepa.model import PLCIJEPAGraphModel
from gcnet_plci_single_view.model import SingleViewPLCIJEPAGraphModel


def _args(**overrides):
    values = vars(build_argument_parser().parse_args(["--base-model", "LSTM"]))
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parser_keeps_independent_default_and_exposes_plci_options():
    args = build_argument_parser().parse_args(["--base-model", "LSTM"])
    assert args.jepa_architecture == "independent"
    assert args.plci_latent_dim > 0
    assert args.plci_ema_tau < 1.0


def test_build_model_routes_plci_architecture():
    model = build_model(_args(jepa_architecture="plci", hidden=8), 2, 3, 4)
    assert isinstance(model, PLCIJEPAGraphModel)


def test_build_model_routes_single_view_plci_architecture():
    parsed = build_argument_parser().parse_args(
        ["--base-model", "LSTM", "--jepa-architecture", "plci-single"]
    )
    assert parsed.jepa_architecture == "plci-single"

    model = build_model(
        _args(jepa_architecture="plci-single", hidden=8), 2, 3, 4
    )

    assert isinstance(model, SingleViewPLCIJEPAGraphModel)
    assert type(model) is SingleViewPLCIJEPAGraphModel


def test_single_view_loss_uses_natural_state_without_balanced_sampler():
    model = mock.Mock()
    predictions = mock.sentinel.predictions
    teacher_targets = mock.sentinel.teacher_targets
    model.predict_natural.return_value = predictions
    model.encode_teacher_targets.return_value = teacher_targets
    natural_latents = mock.sentinel.natural_latents
    natural_hidden = mock.sentinel.natural_hidden
    natural_availability = torch.tensor([[[1.0, 0.0, 1.0]]])
    umask = torch.ones(1, 1)
    expected_loss = torch.tensor(0.25)
    expected_counts = {"utterances": 1, "targets": 1, "paths": 2}

    with mock.patch.object(
        train_gcnet, "sample_balanced_patterns"
    ) as balanced_sampler, mock.patch.object(
        train_gcnet,
        "plci_jepa_loss",
        return_value=(expected_loss, expected_counts),
    ) as loss_fn:
        loss, counts, selected_availability = (
            train_gcnet._compute_plci_training_loss(
                mode="plci-single",
                model=model,
                full_features=torch.randn(1, 1, 9),
                natural_availability=natural_availability,
                qmask=torch.zeros(1, 1),
                umask=umask,
                lengths=[1],
                natural_latents=natural_latents,
                natural_hidden=natural_hidden,
                dimensions=(2, 3, 4),
                auxiliary_generator=None,
            )
        )

    balanced_sampler.assert_not_called()
    model.forward_auxiliary.assert_not_called()
    model.predict_natural.assert_called_once_with(
        natural_latents,
        natural_hidden,
        natural_availability,
        umask,
    )
    model.encode_teacher_targets.assert_called_once()
    loss_fn.assert_called_once_with(predictions, teacher_targets)
    assert loss is expected_loss
    assert counts is expected_counts
    assert selected_availability is natural_availability


@pytest.mark.parametrize("architecture", ("plci", "plci-single"))
@pytest.mark.parametrize(
    "override",
    [
        {"reccls_flag": True},
        {"lower_bound": True},
        {"reconstruction_target": "full_fused"},
        {"all_modal_recon_weight": 0.1},
        {"stability_recon_weight": 0.1},
        {"model_variant": "replacement"},
        {"plci_latent_dim": 0},
        {"plci_context_cap": float("inf")},
        {"plci_ema_tau": 1.0},
    ],
)
def test_plci_rejects_incompatible_or_invalid_options(architecture, override):
    options = {"jepa_architecture": architecture, "loss_recon": True}
    options.update(override)
    with pytest.raises(ValueError):
        validate_training_args(_args(**options))


def test_dual_view_still_requires_inherited_reconstruction_objective():
    with pytest.raises(ValueError, match="--loss-recon is required"):
        validate_training_args(
            _args(jepa_architecture="plci", loss_recon=False)
        )


def test_single_view_allows_latent_prediction_to_replace_reconstruction():
    validate_training_args(
        _args(jepa_architecture="plci-single", loss_recon=False)
    )


def test_ema_callback_runs_strictly_after_optimizer_step():
    events = []
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append("optimizer")
            return super().step(closure)

    optimizer = RecordingSGD([parameter], lr=0.1)
    _backward_and_optimizer_step(
        parameter.square(),
        torch.nn.ModuleList([torch.nn.ParameterList([parameter])]),
        optimizer,
        0.0,
        post_step=lambda: events.append("ema"),
    )
    assert events == ["optimizer", "ema"]


def test_plci_optimizer_filter_excludes_frozen_teacher_parameters():
    model = build_model(_args(jepa_architecture="plci", hidden=8), 2, 3, 4)
    optimized = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert optimized
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())
    assert not any(parameter is teacher for parameter in optimized for teacher in model.teacher.parameters())
