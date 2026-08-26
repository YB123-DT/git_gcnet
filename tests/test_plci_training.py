from types import SimpleNamespace

import pytest
import torch

from gcnet_modality_jepa.train_gcnet import (
    _backward_and_optimizer_step,
    build_argument_parser,
    build_model,
    validate_training_args,
)
from gcnet_plci_jepa.model import PLCIJEPAGraphModel


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


@pytest.mark.parametrize(
    "override",
    [
        {"loss_recon": False},
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
def test_plci_rejects_incompatible_or_invalid_options(override):
    options = {"jepa_architecture": "plci", "loss_recon": True}
    options.update(override)
    with pytest.raises(ValueError):
        validate_training_args(_args(**options))


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
