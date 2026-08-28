import copy
import json
from dataclasses import asdict

import pytest
import torch

import gcnet_missing_m3.train_gcnet as train_gcnet
from gcnet_missing_m3.loss import MissingM3Loss, missing_m3_loss
from gcnet_missing_m3.mixed_rate import (
    MISSING_RATES,
    BalancedBatchRateSchedule,
    mean_validation_weighted_f1,
    select_best_epoch,
)
from gcnet_missing_m3.model import (
    ContextualM3Predictor,
    LocalContextResidualFusion,
    MissingM3GraphModel,
    ObservedSetEncoder,
    RawResidualObservedEncoder,
)
from gcnet_missing_m3.train_gcnet import (
    TrainConfig,
    _dataset_shape,
    _metrics,
    _task_loss,
    build_parser,
)


ASSERT_CLOSE = getattr(torch.testing, "assert_close", torch.testing.assert_allclose)


def _all_patterns():
    return torch.tensor(
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
    ).unsqueeze(1)


def test_observed_set_encoder_supports_seven_patterns_and_ignores_missing_values():
    torch.manual_seed(11)
    encoder = ObservedSetEncoder((2, 3, 4), latent_dim=8, dropout=0.0).eval()
    availability = _all_patterns()
    umask = torch.ones(1, 7)
    features = torch.randn(7, 1, 9)
    changed = features.clone()
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    changed[expanded == 0] += 10_000.0

    first_node, first_latents = encoder(features, availability, umask)
    second_node, second_latents = encoder(changed, availability, umask)

    assert first_node.shape == (7, 1, 8)
    assert set(first_latents) == {"audio", "text", "visual"}
    ASSERT_CLOSE(first_node, second_node, rtol=0, atol=0)
    for index, name in enumerate(("audio", "text", "visual")):
        ASSERT_CLOSE(first_latents[name], second_latents[name], rtol=0, atol=0)
        assert torch.count_nonzero(first_latents[name][~availability[..., index].bool()]) == 0


def test_observed_set_encoder_zeros_padding():
    encoder = ObservedSetEncoder((2, 3, 4), latent_dim=8, dropout=0.0).eval()
    availability = torch.tensor(
        [[[1, 0, 0]], [[1, 1, 1]], [[0, 0, 0]]], dtype=torch.float32
    )
    umask = torch.tensor([[1.0, 1.0, 0.0]])

    node, latents = encoder(torch.randn(3, 1, 9), availability, umask)

    assert torch.count_nonzero(node[2]) == 0
    assert all(torch.count_nonzero(value[2]) == 0 for value in latents.values())


def test_slot_fusion_supports_seven_patterns_without_missing_value_leakage():
    torch.manual_seed(13)
    encoder = ObservedSetEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0, fusion_type="slot"
    ).eval()
    availability = _all_patterns()
    umask = torch.ones(1, 7)
    features = torch.randn(7, 1, 9)
    changed = features.clone()
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    changed[expanded == 0] += 10_000.0

    first, _ = encoder(features, availability, umask)
    second, _ = encoder(changed, availability, umask)

    assert first.shape == (7, 1, 8)
    ASSERT_CLOSE(first, second, rtol=0, atol=0)


def test_slot_fusion_keeps_audio_text_and_visual_in_fixed_distinct_slots():
    torch.manual_seed(19)
    encoder = ObservedSetEncoder(
        (1, 1, 1), latent_dim=4, dropout=0.0, fusion_type="slot"
    ).eval()
    availability = torch.tensor(
        [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]], dtype=torch.float32
    )
    captured = []
    handle = encoder.fusion[0].register_forward_pre_hook(
        lambda _module, values: captured.append(values[0].detach().clone())
    )

    encoder(torch.ones(3, 1, 3), availability, torch.ones(1, 3))
    handle.remove()

    fusion_input = captured[0]
    assert fusion_input.shape == (3, 16)
    for row, active_slot in enumerate((0, 1, 2)):
        modality_slots = fusion_input[row, :12].reshape(3, 4)
        assert torch.count_nonzero(modality_slots[active_slot]) > 0
        inactive = [index for index in range(3) if index != active_slot]
        assert torch.count_nonzero(modality_slots[inactive]) == 0


def test_default_mean_fusion_is_exactly_backward_compatible():
    torch.manual_seed(23)
    default = ObservedSetEncoder((2, 3, 4), latent_dim=8, dropout=0.0).eval()
    torch.manual_seed(23)
    explicit = ObservedSetEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0, fusion_type="mean"
    ).eval()
    availability = _all_patterns()
    features = torch.randn(7, 1, 9)
    umask = torch.ones(1, 7)

    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        ASSERT_CLOSE(value, explicit.state_dict()[key], rtol=0, atol=0)
    first, _ = default(features, availability, umask)
    second, _ = explicit(features, availability, umask)
    ASSERT_CLOSE(first, second, rtol=0, atol=0)


def test_fusion_type_is_validated_and_exposed_by_cli():
    with pytest.raises(ValueError, match="fusion_type"):
        ObservedSetEncoder((2, 3, 4), latent_dim=8, fusion_type="attention")

    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--fusion-type",
            "slot",
        ]
    )
    assert args.fusion_type == "slot"


def test_raw_residual_encoder_starts_as_exact_masked_raw_input():
    torch.manual_seed(29)
    encoder = RawResidualObservedEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0
    ).eval()
    availability = _all_patterns()
    umask = torch.ones(1, 7)
    features = torch.randn(7, 1, 9)
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )

    output, latents = encoder(features, availability, umask)

    ASSERT_CLOSE(output, features * expanded, rtol=0, atol=0)
    assert set(latents) == {"audio", "text", "visual"}
    assert all(value.shape == (7, 1, 8) for value in latents.values())


def test_raw_residual_encoder_ignores_missing_values_and_zeros_padding():
    torch.manual_seed(31)
    encoder = RawResidualObservedEncoder(
        (2, 3, 4), latent_dim=8, dropout=0.0
    ).eval()
    availability = torch.cat(
        [_all_patterns(), torch.zeros(1, 1, 3)], dim=0
    )
    umask = torch.tensor([[1.0] * 7 + [0.0]])
    features = torch.randn(8, 1, 9)
    changed = features.clone()
    expanded = torch.repeat_interleave(
        availability, torch.tensor((2, 3, 4)), dim=-1
    )
    changed[expanded == 0] += 10_000.0

    first_output, first_latents = encoder(features, availability, umask)
    second_output, second_latents = encoder(changed, availability, umask)

    ASSERT_CLOSE(first_output, second_output, rtol=0, atol=0)
    for name in ("audio", "text", "visual"):
        ASSERT_CLOSE(first_latents[name], second_latents[name], rtol=0, atol=0)
        assert torch.count_nonzero(first_latents[name][-1]) == 0
    assert torch.count_nonzero(first_output[-1]) == 0


def test_contextual_m3_selects_true_missing_targets_and_averages_two_sources(monkeypatch):
    predictor = ContextualM3Predictor(
        latent_dim=4,
        context_dim=5,
        num_experts=2,
        top_k=1,
        dropout=0.0,
    )
    latents = {
        "audio": torch.full((2, 1, 4), 1.0),
        "text": torch.full((2, 1, 4), 3.0),
        "visual": torch.full((2, 1, 4), 5.0),
    }
    hidden = torch.zeros(2, 1, 5)
    availability = torch.tensor(
        [[[1, 1, 0]], [[1, 0, 0]]], dtype=torch.float32
    )
    umask = torch.ones(1, 2)

    def fake_direction(source, context, source_index, target_index):
        value = source + float(source_index * 10 + target_index)
        return value, value + 100.0

    monkeypatch.setattr(predictor, "direction_forward", fake_direction)
    output = predictor(latents, hidden, availability, umask)

    assert output.target_mask[:, 0].tolist() == [
        [False, False, True],
        [False, True, True],
    ]
    assert output.source_counts[:, 0].tolist() == [
        [0, 0, 2],
        [0, 1, 1],
    ]
    # AT -> V averages A->V (1 + 2) and T->V (3 + 12).
    ASSERT_CLOSE(output.reg_predictions[0, 0, 2], torch.full((4,), 9.0))
    # A -> T is a single direction.
    ASSERT_CLOSE(output.reg_predictions[1, 0, 1], torch.full((4,), 2.0))


def test_missing_m3_loss_is_zero_for_complete_atv_and_finite_for_missing_targets():
    predictor = ContextualM3Predictor(4, 5, num_experts=2, top_k=1, dropout=0.0)
    latents = {name: torch.randn(3, 2, 4) for name in ("audio", "text", "visual")}
    hidden = torch.randn(3, 2, 5, requires_grad=True)
    umask = torch.ones(2, 3)
    complete = torch.ones(3, 2, 3)
    teacher = {name: torch.randn(3, 2, 4) for name in latents}

    complete_predictions = predictor(latents, hidden, complete, umask)
    complete_loss = missing_m3_loss(complete_predictions, teacher, temperature=0.1)

    assert complete_loss.target_count == 0
    assert complete_loss.total.item() == 0.0

    missing = complete.clone()
    missing[:, :, 2] = 0
    predictions = predictor(latents, hidden, missing, umask)
    result = missing_m3_loss(predictions, teacher, temperature=0.1)
    result.total.backward()

    assert result.target_count == 6
    assert torch.isfinite(result.total)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def _model_arguments():
    return dict(
        base_model="LSTM",
        adim=2,
        tdim=3,
        vdim=4,
        D_e=4,
        graph_hidden_size=2,
        n_speakers=2,
        window_past=1,
        window_future=1,
        n_classes=6,
        dropout=0.0,
        time_attn=False,
        no_cuda=True,
        latent_dim=8,
        num_experts=2,
        top_k=1,
        predictor_dropout=0.0,
    )


def _model_inputs():
    features = torch.randn(3, 2, 9)
    availability = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    qmask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    return features, availability, qmask, umask, [3, 2]


def test_local_context_residual_has_expected_shape_zero_padding_and_no_missing_leakage():
    torch.manual_seed(41)
    fusion = LocalContextResidualFusion(
        latent_dim=4, context_dim=6, hidden_dim=7, dropout=0.0
    ).eval()
    with torch.no_grad():
        torch.nn.init.normal_(fusion.fusion[-1].weight)
        torch.nn.init.normal_(fusion.fusion[-1].bias)
    availability = torch.tensor(
        [
            [[1, 0, 1], [0, 1, 0]],
            [[1, 1, 0], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    umask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    latents = {
        name: torch.randn(2, 2, 4)
        for name in ("audio", "text", "visual")
    }
    changed = {name: value.clone() for name, value in latents.items()}
    for index, name in enumerate(("audio", "text", "visual")):
        changed[name][~availability[..., index].bool()] += 10_000.0

    first = fusion(latents, availability, umask)
    second = fusion(changed, availability, umask)

    assert first.shape == (2, 2, 6)
    ASSERT_CLOSE(first, second, rtol=0, atol=0)
    assert torch.count_nonzero(first[~umask.T.bool()]) == 0


def test_local_context_residual_preserves_same_seed_shared_state_and_zero_init_outputs():
    torch.manual_seed(43)
    base = MissingM3GraphModel(
        **_model_arguments(), fusion_type="slot"
    ).eval()
    torch.manual_seed(43)
    local = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=12,
        local_fusion_dropout=0.0,
    ).eval()

    base_state = base.state_dict()
    local_state = local.state_dict()
    assert set(base_state).issubset(local_state)
    for key, value in base_state.items():
        ASSERT_CLOSE(value, local_state[key], rtol=0, atol=0)

    torch.manual_seed(47)
    inputs = _model_inputs()
    base_outputs = base([inputs[0]], *inputs[1:], predict_missing=True)
    local_outputs = local([inputs[0]], *inputs[1:], predict_missing=True)

    for base_value, local_value in zip(base_outputs[:3], local_outputs[:3]):
        if isinstance(base_value, dict):
            for name in base_value:
                ASSERT_CLOSE(base_value[name], local_value[name], rtol=0, atol=0)
        else:
            ASSERT_CLOSE(base_value, local_value, rtol=0, atol=0)
    assert base_outputs[3] is not None and local_outputs[3] is not None
    for field in ("reg_predictions", "cl_predictions", "target_mask", "source_counts"):
        ASSERT_CLOSE(
            getattr(base_outputs[3], field),
            getattr(local_outputs[3], field),
            rtol=0,
            atol=0,
        )


def test_local_context_residual_preserves_train_mode_rng_and_zero_init_outputs():
    arguments = _model_arguments()
    arguments["predictor_dropout"] = 0.4
    torch.manual_seed(67)
    base = MissingM3GraphModel(
        **arguments, fusion_type="slot"
    ).train()
    torch.manual_seed(67)
    local = MissingM3GraphModel(
        **arguments,
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=12,
        local_fusion_dropout=0.5,
    ).train()
    torch.manual_seed(71)
    inputs = _model_inputs()
    forward_rng = torch.random.get_rng_state()

    torch.random.set_rng_state(forward_rng)
    base_outputs = base([inputs[0]], *inputs[1:], predict_missing=True)
    torch.random.set_rng_state(forward_rng)
    local_outputs = local([inputs[0]], *inputs[1:], predict_missing=True)

    ASSERT_CLOSE(base_outputs[0], local_outputs[0], rtol=0, atol=0)
    ASSERT_CLOSE(base_outputs[1], local_outputs[1], rtol=0, atol=0)
    assert base_outputs[3] is not None and local_outputs[3] is not None
    for field in ("reg_predictions", "cl_predictions", "target_mask", "source_counts"):
        ASSERT_CLOSE(
            getattr(base_outputs[3], field),
            getattr(local_outputs[3], field),
            rtol=0,
            atol=0,
        )


def test_local_context_residual_keeps_missing_predictor_on_graph_hidden(monkeypatch):
    torch.manual_seed(53)
    model = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=12,
        local_fusion_dropout=0.0,
    ).eval()
    with torch.no_grad():
        model.local_context_fusion.fusion[-1].bias.fill_(1.0)
    captured = {}

    def capture_predictor(latents, hidden, availability, umask):
        captured["hidden"] = hidden.detach().clone()
        return None

    monkeypatch.setattr(model.missing_predictor, "forward", capture_predictor)
    torch.manual_seed(59)
    inputs = _model_inputs()
    _, classification_hidden, _, _ = model(
        [inputs[0]], *inputs[1:], predict_missing=True
    )

    valid = inputs[3].T.bool()
    ASSERT_CLOSE(
        captured["hidden"][valid] + 1.0,
        classification_hidden.detach()[valid],
        rtol=0,
        atol=0,
    )


def test_local_context_residual_disabled_has_no_parameters_and_rejects_non_slot():
    disabled = MissingM3GraphModel(
        **_model_arguments(), fusion_type="slot", local_context_residual=False
    )

    assert not any(
        name.startswith("local_context_fusion.")
        for name, _ in disabled.named_parameters()
    )
    assert not hasattr(disabled, "local_context_fusion")
    for fusion_type in ("mean", "raw-residual"):
        with pytest.raises(ValueError, match="local_context_residual.*slot"):
            MissingM3GraphModel(
                **_model_arguments(),
                fusion_type=fusion_type,
                local_context_residual=True,
            )


def test_local_context_residual_cli_and_config_route_all_options():
    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--fusion-type",
            "slot",
            "--local-context-residual",
            "--local-fusion-hidden-dim",
            "96",
            "--local-fusion-dropout",
            "0.15",
        ]
    )
    config = TrainConfig(
        fusion_type=args.fusion_type,
        local_context_residual=args.local_context_residual,
        local_fusion_hidden_dim=args.local_fusion_hidden_dim,
        local_fusion_dropout=args.local_fusion_dropout,
    )

    assert {
        key: asdict(config)[key]
        for key in (
            "local_context_residual",
            "local_fusion_hidden_dim",
            "local_fusion_dropout",
        )
    } == {
        "local_context_residual": True,
        "local_fusion_hidden_dim": 96,
        "local_fusion_dropout": 0.15,
    }


def test_local_context_residual_persists_in_config_json_and_checkpoint(tmp_path):
    config = TrainConfig(
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=96,
        local_fusion_dropout=0.15,
    )
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "best.pt"

    train_gcnet._write_run_config(config_path, config)
    train_gcnet._save_best_checkpoint(
        checkpoint_path,
        model_state={"weight": torch.tensor([1.0])},
        config_value=config,
        epoch=3,
        validation_mean_weighted_f1=0.75,
    )

    json_config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_config = torch.load(checkpoint_path, map_location="cpu")["config"]
    expected = {
        "local_context_residual": True,
        "local_fusion_hidden_dim": 96,
        "local_fusion_dropout": 0.15,
    }
    assert {key: json_config[key] for key in expected} == expected
    assert {key: checkpoint_config[key] for key in expected} == expected


def test_local_context_residual_backward_reaches_all_training_paths():
    torch.manual_seed(61)
    model = MissingM3GraphModel(
        **_model_arguments(),
        fusion_type="slot",
        local_context_residual=True,
        local_fusion_hidden_dim=12,
        local_fusion_dropout=0.0,
    ).train()
    inputs = _model_inputs()

    logits, _, _, predictions = model(
        [inputs[0]], *inputs[1:], predict_missing=True
    )
    assert predictions is not None and bool(predictions.target_mask.any())
    predicted = predictions.reg_predictions[predictions.target_mask]
    (logits.square().mean() + predicted.square().mean()).backward()

    parameter_groups = {
        "student": model.observed_set.projectors.parameters(),
        "local": model.local_context_fusion.parameters(),
        "graph": model.graph_net_temporal.parameters(),
        "predictor": model.missing_predictor.parameters(),
    }
    for name, parameters in parameter_groups.items():
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        assert gradients, name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients), name


def test_raw_residual_model_keeps_original_recurrent_width_and_cli():
    raw = MissingM3GraphModel(
        **_model_arguments(), fusion_type="raw-residual"
    )
    slot = MissingM3GraphModel(**_model_arguments(), fusion_type="slot")

    assert raw.lstm.input_size == 9
    assert slot.lstm.input_size == 8

    args = build_parser().parse_args(
        [
            "--audio-feature",
            "a",
            "--text-feature",
            "t",
            "--video-feature",
            "v",
            "--output-dir",
            "out",
            "--fusion-type",
            "raw-residual",
        ]
    )
    assert args.fusion_type == "raw-residual"


def test_raw_residual_backward_reaches_online_encoder_graph_and_predictor():
    torch.manual_seed(37)
    model = MissingM3GraphModel(
        **_model_arguments(), fusion_type="raw-residual"
    ).train()
    features = torch.randn(3, 2, 9)
    availability = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    qmask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    logits, _, _, predictions = model(
        [features], availability, qmask, umask, [3, 2], predict_missing=True
    )
    assert predictions is not None and bool(predictions.target_mask.any())
    predicted = predictions.reg_predictions[predictions.target_mask]
    loss = logits.square().mean() + predicted.square().mean()
    loss.backward()

    parameter_groups = {
        "student": model.observed_set.projectors.parameters(),
        "adapter": model.observed_set.adapters.parameters(),
        "graph": model.graph_net_temporal.parameters(),
        "predictor": model.missing_predictor.parameters(),
    }
    for name, parameters in parameter_groups.items():
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        assert gradients, name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients), name


def test_missing_m3_graph_model_uses_one_gcnet_forward_and_updates_teacher_exactly():
    torch.manual_seed(17)
    model = MissingM3GraphModel(**_model_arguments()).eval()
    features = torch.randn(3, 2, 9)
    availability = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    qmask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    lengths = [3, 2]

    logits, hidden, latents, predictions = model(
        [features], availability, qmask, umask, lengths, predict_missing=True
    )

    assert logits.shape == (3, 2, 6)
    assert hidden.shape == (3, 2, 10)
    assert predictions is not None
    assert not hasattr(model, "linear_rec")
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())

    before = copy.deepcopy(model.teacher.state_dict())
    with torch.no_grad():
        for parameter in model.observed_set.projectors.parameters():
            parameter.add_(1.0)
    students = model.observed_set.projectors.state_dict()
    model.update_teacher(0.9)
    for key, value in model.teacher.state_dict().items():
        ASSERT_CLOSE(value, before[key] * 0.9 + students[key] * 0.1)


def test_balanced_rate_schedule_covers_all_rates_and_rotates_by_epoch():
    schedule = BalancedBatchRateSchedule(MISSING_RATES)

    first = tuple(schedule.rate_for(epoch=0, batch_index=index) for index in range(8))
    second = tuple(schedule.rate_for(epoch=1, batch_index=index) for index in range(8))

    assert first == MISSING_RATES
    assert second == MISSING_RATES[1:] + MISSING_RATES[:1]
    assert set(first) == set(MISSING_RATES)


class _CountingOptimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.zero_grad_calls = 0
        self.step_gradients = []

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls += 1
        if set_to_none:
            self.parameter.grad = None
        elif self.parameter.grad is not None:
            self.parameter.grad.zero_()

    def step(self):
        self.step_gradients.append(self.parameter.grad.detach().clone())


class _LifecycleModel(torch.nn.Module):
    def __init__(self, events=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_rates = []
        self.teacher_calls = 0
        self.ema_calls = 0
        self.events = events

    def forward(
        self,
        features,
        availability,
        qmask,
        umask,
        lengths,
        predict_missing,
    ):
        del features, qmask, umask, lengths
        assert predict_missing is True
        rate_index = int(availability.item())
        rate = MISSING_RATES[rate_index]
        self.forward_rates.append(rate)
        if self.events is not None:
            self.events.append(("forward", rate))
        logits = (self.weight * (rate_index + 1)).reshape(1, 1, 1)
        return logits, None, None, logits

    def encode_teacher_targets(self, complete):
        self.teacher_calls += 1
        return {"complete": complete[0]}

    def update_teacher(self, tau):
        assert tau == pytest.approx(0.996)
        self.ema_calls += 1


def _install_train_epoch_lifecycle_fakes(monkeypatch):
    prepared_rates = []
    clipped_gradients = []
    events = []

    def prepare_view(data, schedule, epoch, dimensions):
        del data, epoch, dimensions
        rate_index = MISSING_RATES.index(schedule)
        prepared_rates.append(schedule)
        events.append(("prepare", schedule))
        return {
            "complete": torch.ones(1, 1, 1),
            "incomplete": torch.ones(1, 1, 1),
            "availability": torch.tensor(float(rate_index)),
            "qmask": torch.ones(1, 1),
            "umask": torch.ones(1, 1),
            "labels": torch.zeros(1, 1),
            "lengths": [1],
        }

    def jepa_loss(predictions, teacher, temperature):
        assert teacher.keys() == {"complete"}
        assert temperature == pytest.approx(0.03)
        rate_index = int(predictions.detach().item()) - 1
        zero = predictions.sum() * 0.0
        return MissingM3Loss(
            total=zero,
            regression=zero,
            contrastive=zero,
            target_count=int(rate_index > 0),
        )

    def clip_grad_norm(parameters, max_norm):
        assert max_norm == pytest.approx(1.0)
        gradients = [
            parameter.grad.detach().clone()
            for parameter in parameters
            if parameter.grad is not None
        ]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        clipped_gradients.append(gradients)

    monkeypatch.setattr(train_gcnet, "_move_batch", lambda raw, device: raw)
    monkeypatch.setattr(train_gcnet, "_prepare_view", prepare_view)
    monkeypatch.setattr(
        train_gcnet, "_task_loss", lambda dataset, logits, labels, umask: logits.sum()
    )
    monkeypatch.setattr(train_gcnet, "missing_m3_loss", jepa_loss)
    monkeypatch.setattr(
        train_gcnet,
        "_collect_predictions",
        lambda dataset, logits, labels, umask: (
            train_gcnet.np.array([0.0]),
            train_gcnet.np.array([0.0]),
        ),
    )
    monkeypatch.setattr(
        train_gcnet,
        "_metrics",
        lambda dataset, labels, predictions: {"weighted_f1": 1.0},
    )
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_grad_norm)
    return prepared_rates, clipped_gradients, events


def test_all_train_rate_mode_averages_eight_views_with_one_update_per_batch(monkeypatch):
    prepared_rates, clipped_gradients, events = _install_train_epoch_lifecycle_fakes(
        monkeypatch
    )
    model = _LifecycleModel(events)
    optimizer = _CountingOptimizer(model.weight)
    config = TrainConfig(train_rate_mode="all")

    metrics = train_gcnet.train_epoch(
        model=model,
        loader=[["first"], ["second"]],
        optimizer=optimizer,
        config=config,
        schedules={rate: rate for rate in MISSING_RATES},
        epoch=0,
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
    )

    assert prepared_rates == list(MISSING_RATES) * 2
    assert model.forward_rates == list(MISSING_RATES) * 2
    assert events == [
        event
        for _ in range(2)
        for rate in MISSING_RATES
        for event in (("prepare", rate), ("forward", rate))
    ]
    assert model.teacher_calls == 2
    assert optimizer.zero_grad_calls == 2
    assert len(clipped_gradients) == 2
    assert len(optimizer.step_gradients) == 2
    assert all(torch.isfinite(value) for value in optimizer.step_gradients)
    assert all(value.item() == pytest.approx(4.5) for value in optimizer.step_gradients)
    assert model.ema_calls == 2
    assert metrics["loss"] == pytest.approx(4.5)
    assert metrics["classification_loss"] == pytest.approx(4.5)
    assert metrics["jepa_target_count"] == 14
    assert metrics["optimizer_steps"] == 2
    assert metrics["rate_batch_counts"] == {str(rate): 2 for rate in MISSING_RATES}


def test_default_train_rate_mode_preserves_single_cyclic_view(monkeypatch):
    prepared_rates, clipped_gradients, events = _install_train_epoch_lifecycle_fakes(
        monkeypatch
    )
    model = _LifecycleModel(events)
    optimizer = _CountingOptimizer(model.weight)
    config = TrainConfig()

    metrics = train_gcnet.train_epoch(
        model=model,
        loader=[["only"]],
        optimizer=optimizer,
        config=config,
        schedules={rate: rate for rate in MISSING_RATES},
        epoch=2,
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
    )

    assert config.train_rate_mode == "cyclic"
    assert prepared_rates == [0.2]
    assert model.forward_rates == [0.2]
    assert events == [("prepare", 0.2), ("forward", 0.2)]
    assert model.teacher_calls == 1
    assert optimizer.zero_grad_calls == 1
    assert len(clipped_gradients) == 1
    assert len(optimizer.step_gradients) == 1
    assert optimizer.step_gradients[0].item() == pytest.approx(3.0)
    assert model.ema_calls == 1
    assert metrics["loss"] == pytest.approx(3.0)
    assert metrics["optimizer_steps"] == 1
    assert sum(metrics["rate_batch_counts"].values()) == 1
    assert metrics["rate_batch_counts"]["0.2"] == 1


def test_train_rate_mode_cli_defaults_and_persists_in_run_artifacts(tmp_path):
    required = [
        "--audio-feature",
        "a",
        "--text-feature",
        "t",
        "--video-feature",
        "v",
        "--output-dir",
        "out",
    ]
    parser = build_parser()
    assert parser.parse_args(required).train_rate_mode == "cyclic"
    args = parser.parse_args(required + ["--train-rate-mode", "all"])
    config = TrainConfig(train_rate_mode=args.train_rate_mode)
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "best.pt"

    train_gcnet._write_run_config(config_path, config)
    train_gcnet._save_best_checkpoint(
        checkpoint_path,
        model_state={"weight": torch.tensor([1.0])},
        config_value=config,
        epoch=3,
        validation_mean_weighted_f1=0.75,
    )

    assert json.loads(config_path.read_text(encoding="utf-8"))["train_rate_mode"] == "all"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["config"]["train_rate_mode"] == "all"
    with pytest.raises(SystemExit):
        parser.parse_args(required + ["--train-rate-mode", "invalid"])


def test_train_rate_mode_preserves_legacy_positional_config_order():
    legacy_field_names = (
        "dataset",
        "fold",
        "seed",
        "base_model",
        "window_past",
        "window_future",
        "hidden",
        "dropout",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "latent_dim",
        "num_experts",
        "top_k",
        "projector_dropout",
        "predictor_dropout",
        "fusion_type",
        "local_context_residual",
        "local_fusion_hidden_dim",
        "local_fusion_dropout",
        "jepa_weight",
        "temperature",
        "ema_tau",
        "gradient_clip_norm",
        "time_attention",
        "evaluation_protocol",
        "validation_fraction",
        "device",
    )
    legacy_values = (
        "CMUMOSI",
        1,
        77,
        "GRU",
        -1,
        3,
        128,
        0.2,
        16,
        12,
        2e-3,
        3e-5,
        64,
        3,
        1,
        0.05,
        0.06,
        "slot",
        True,
        96,
        0.15,
        0.2,
        0.07,
        0.99,
        0.5,
        True,
        "strict",
        0.2,
        "cpu",
    )

    config = TrainConfig(*legacy_values)

    serialized = asdict(config)
    assert tuple(serialized[name] for name in legacy_field_names) == legacy_values
    assert config.train_rate_mode == "cyclic"


def test_checkpoint_score_is_equal_mean_of_eight_validation_rates():
    metrics = {
        rate: {"weighted_f1": float(index)}
        for index, rate in enumerate(MISSING_RATES)
    }

    assert mean_validation_weighted_f1(metrics) == pytest.approx(3.5)
    with pytest.raises(ValueError, match="all eight"):
        mean_validation_weighted_f1(dict(list(metrics.items())[:-1]))


def test_best_epoch_uses_mixed_validation_mean_not_any_single_rate_peak():
    first = {
        rate: {"weighted_f1": 0.8 if rate == 0.0 else 0.4}
        for rate in MISSING_RATES
    }
    second = {
        rate: {"weighted_f1": 0.5}
        for rate in MISSING_RATES
    }

    assert select_best_epoch([
        {"epoch": 1, "validation": first},
        {"epoch": 2, "validation": second},
    ]) == 2


@pytest.mark.parametrize(
    "dataset,num_folds,num_classes,num_speakers,task",
    [
        ("IEMOCAPFour", 5, 4, 2, "classification"),
        ("IEMOCAPSix", 5, 6, 2, "classification"),
        ("CMUMOSI", 1, 1, 1, "regression"),
        ("CMUMOSEI", 1, 1, 1, "regression"),
    ],
)
def test_dataset_shape_matches_gcnet_task_contract(
    dataset, num_folds, num_classes, num_speakers, task
):
    actual = _dataset_shape(dataset)
    assert actual == {
        "num_folds": num_folds,
        "num_classes": num_classes,
        "num_speakers": num_speakers,
        "task": task,
    }


def test_task_loss_uses_cross_entropy_for_erc_and_mse_for_sentiment():
    logits = torch.tensor([[[2.0, -1.0]], [[-1.0, 2.0]]])
    labels = torch.tensor([[0, 1]])
    umask = torch.ones(1, 2)
    expected_ce = torch.nn.functional.cross_entropy(
        logits.transpose(0, 1).reshape(-1, 2), labels.reshape(-1)
    )
    ASSERT_CLOSE(_task_loss("IEMOCAPFour", logits, labels, umask), expected_ce)

    prediction = torch.tensor([[[0.5]], [[-1.0]]])
    target = torch.tensor([[1.5, 1.0]])
    expected_mse = torch.tensor(((0.5 - 1.5) ** 2 + (-1.0 - 1.0) ** 2) / 2)
    ASSERT_CLOSE(_task_loss("CMUMOSI", prediction, target, umask), expected_mse)


def test_sentiment_metrics_match_gcnet_binary_nonzero_protocol():
    labels = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).numpy()
    predictions = torch.tensor([-1.0, 0.5, -3.0, 0.1, 2.0]).numpy()

    result = _metrics("CMUMOSI", labels, predictions)

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["weighted_f1"] == pytest.approx(0.7333333333333334)
    assert result["mae"] == pytest.approx(1.28)
