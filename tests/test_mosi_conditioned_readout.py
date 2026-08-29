import pytest
import torch
from dataclasses import asdict

import gcnet_missing_m3.model as missing_m3_model
import gcnet_missing_m3.train_gcnet as train_gcnet
from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser


ASSERT_CLOSE = getattr(torch.testing, "assert_close", torch.testing.assert_allclose)


def _readout_class():
    readout = getattr(
        missing_m3_model, "AvailabilityConditionedLowRankReadout", None
    )
    assert readout is not None, "conditioned readout is not implemented"
    return readout


def _affine_readout_class():
    readout = getattr(
        missing_m3_model, "AvailabilityConditionedAffineReadout", None
    )
    assert readout is not None, "affine readout is not implemented"
    return readout


def _seven_patterns():
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


def test_conditioned_readout_starts_as_exact_zero_for_all_patterns():
    readout = _readout_class()(
        input_dim=5,
        output_dim=2,
        rank=3,
        route_type="availability-low-rank",
    ).eval()
    hidden = torch.randn(7, 1, 5)

    residual = readout(hidden, _seven_patterns(), torch.ones(1, 7))

    assert residual.shape == (7, 1, 2)
    ASSERT_CLOSE(residual, torch.zeros_like(residual), rtol=0, atol=0)
    assert readout.input_norm.elementwise_affine is False
    assert torch.count_nonzero(readout.basis.weight) > 0
    assert torch.count_nonzero(readout.pattern_factor) == 0
    assert torch.count_nonzero(readout.pattern_bias) == 0


def test_conditioned_readout_routes_the_seven_binary_patterns_exactly():
    readout = _readout_class()(
        input_dim=4,
        output_dim=1,
        rank=2,
        route_type="availability-low-rank",
    ).eval()
    with torch.no_grad():
        readout.basis.weight.zero_()
        readout.pattern_bias.copy_(
            torch.arange(1, 8, dtype=torch.float32).unsqueeze(-1)
        )

    residual = readout(
        torch.randn(7, 1, 4),
        _seven_patterns(),
        torch.ones(1, 7),
    )

    # Binary IDs are A=4, T=2, V=1, AT=6, AV=5, TV=3, ATV=7.
    ASSERT_CLOSE(
        residual[:, 0, 0],
        torch.tensor([4.0, 2.0, 1.0, 6.0, 5.0, 3.0, 7.0]),
        rtol=0,
        atol=0,
    )


def test_conditioned_readout_excludes_padding_and_rejects_empty_valid_pattern():
    readout = _readout_class()(
        input_dim=4,
        output_dim=1,
        rank=2,
        route_type="availability-low-rank",
    ).eval()
    with torch.no_grad():
        readout.pattern_bias.fill_(1.0)
    availability = torch.tensor(
        [[[1, 0, 0]], [[0, 1, 1]], [[0, 0, 0]]], dtype=torch.float32
    )
    umask = torch.tensor([[1.0, 1.0, 0.0]])
    hidden = torch.randn(3, 1, 4)
    changed = hidden.clone()
    changed[-1] += 10_000.0

    first = readout(hidden, availability, umask)
    second = readout(changed, availability, umask)

    ASSERT_CLOSE(first, second, rtol=0, atol=0)
    assert torch.count_nonzero(first[-1]) == 0

    invalid = availability.clone()
    invalid[0] = 0
    with pytest.raises(ValueError, match="nonempty"):
        readout(hidden, invalid, umask)


def test_asymmetric_initialization_reaches_factors_then_basis():
    readout = _readout_class()(
        input_dim=4,
        output_dim=1,
        rank=2,
        route_type="availability-low-rank",
    )
    hidden = torch.tensor(
        [[[0.0, 1.0, 2.0, 4.0]], [[4.0, 1.0, -1.0, 2.0]]]
    )
    availability = torch.tensor(
        [[[1, 0, 0]], [[1, 0, 0]]], dtype=torch.float32
    )
    umask = torch.ones(1, 2)

    readout(hidden, availability, umask).sum().backward()

    selected_row = 3  # A has binary ID 4 and therefore table row 3.
    assert torch.count_nonzero(readout.pattern_factor.grad[selected_row]) > 0
    assert torch.count_nonzero(readout.pattern_bias.grad[selected_row]) > 0
    assert torch.count_nonzero(readout.pattern_factor.grad[:selected_row]) == 0
    assert torch.count_nonzero(readout.pattern_factor.grad[selected_row + 1 :]) == 0
    assert torch.count_nonzero(readout.basis.weight.grad) == 0

    with torch.no_grad():
        readout.pattern_factor -= 0.1 * readout.pattern_factor.grad
        readout.pattern_bias -= 0.1 * readout.pattern_bias.grad
    readout.zero_grad(set_to_none=True)
    readout(hidden, availability, umask).sum().backward()

    assert torch.count_nonzero(readout.basis.weight.grad) > 0
    assert torch.isfinite(readout.basis.weight.grad).all()


def test_parameter_matched_route_is_pattern_invariant_and_uses_every_row():
    readout = _readout_class()(
        input_dim=4,
        output_dim=1,
        rank=2,
        route_type="shared-low-rank-parammatch",
    )
    with torch.no_grad():
        readout.pattern_factor.copy_(
            torch.arange(14, dtype=torch.float32).reshape(7, 2, 1) / 10.0
        )
        readout.pattern_bias.copy_(
            torch.arange(7, dtype=torch.float32).unsqueeze(-1) / 10.0
        )
    hidden = torch.tensor([[[0.0, 1.0, 2.0, 4.0]]]).repeat(7, 1, 1)

    residual = readout(hidden, _seven_patterns(), torch.ones(1, 7))

    ASSERT_CLOSE(
        residual,
        residual[0:1].expand_as(residual),
        rtol=0,
        atol=1e-7,
    )
    residual.sum().backward()
    assert torch.count_nonzero(readout.pattern_factor.grad.sum(dim=(1, 2))) == 7
    assert torch.count_nonzero(readout.pattern_bias.grad.sum(dim=1)) == 7


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"rank": 0}, "rank"),
        ({"route_type": "seven-independent-heads"}, "route_type"),
    ],
)
def test_conditioned_readout_rejects_invalid_configuration(kwargs, match):
    arguments = {
        "input_dim": 4,
        "output_dim": 1,
        "rank": 2,
        "route_type": "availability-low-rank",
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=match):
        _readout_class()(**arguments)


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
        n_classes=1,
        dropout=0.0,
        time_attn=False,
        no_cuda=True,
        latent_dim=8,
        num_experts=2,
        top_k=1,
        predictor_dropout=0.0,
        fusion_type="slot",
    )


def _model_inputs():
    features = torch.randn(4, 2, 9)
    availability = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[1, 0, 1], [0, 1, 0]],
            [[1, 1, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    qmask = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    )
    umask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    return features, availability, qmask, umask, [4, 3]


def test_explicit_shared_readout_preserves_default_keys_and_outputs():
    torch.manual_seed(211)
    default = missing_m3_model.MissingM3GraphModel(**_model_arguments()).eval()
    torch.manual_seed(211)
    explicit = missing_m3_model.MissingM3GraphModel(
        **_model_arguments(), readout_type="shared", readout_rank=8
    ).eval()

    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        ASSERT_CLOSE(value, explicit.state_dict()[name], rtol=0, atol=0)
    inputs = _model_inputs()
    default_outputs = default([inputs[0]], *inputs[1:], predict_missing=True)
    explicit_outputs = explicit([inputs[0]], *inputs[1:], predict_missing=True)
    for first, second in zip(default_outputs[:3], explicit_outputs[:3]):
        if isinstance(first, dict):
            for name in first:
                ASSERT_CLOSE(first[name], second[name], rtol=0, atol=0)
        else:
            ASSERT_CLOSE(first, second, rtol=0, atol=0)
    for field in (
        "reg_predictions",
        "cl_predictions",
        "target_mask",
        "source_counts",
    ):
        ASSERT_CLOSE(
            getattr(default_outputs[3], field),
            getattr(explicit_outputs[3], field),
            rtol=0,
            atol=0,
        )


def test_conditioned_model_preserves_shared_initialization_and_initial_output():
    torch.manual_seed(223)
    control = missing_m3_model.MissingM3GraphModel(**_model_arguments()).eval()
    torch.manual_seed(223)
    treatment = missing_m3_model.MissingM3GraphModel(
        **_model_arguments(),
        readout_type="availability-low-rank",
        readout_rank=8,
    ).eval()

    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    assert set(control_state).issubset(treatment_state)
    for name, value in control_state.items():
        ASSERT_CLOSE(value, treatment_state[name], rtol=0, atol=0)
    assert any(name.startswith("conditioned_readout.") for name in treatment_state)

    inputs = _model_inputs()
    control_outputs = control([inputs[0]], *inputs[1:], predict_missing=True)
    treatment_outputs = treatment(
        [inputs[0]], *inputs[1:], predict_missing=True
    )
    ASSERT_CLOSE(control_outputs[0], treatment_outputs[0], rtol=0, atol=0)
    ASSERT_CLOSE(control_outputs[1], treatment_outputs[1], rtol=0, atol=0)
    for name in control_outputs[2]:
        ASSERT_CLOSE(
            control_outputs[2][name], treatment_outputs[2][name], rtol=0, atol=0
        )
    for field in (
        "reg_predictions",
        "cl_predictions",
        "target_mask",
        "source_counts",
    ):
        ASSERT_CLOSE(
            getattr(control_outputs[3], field),
            getattr(treatment_outputs[3], field),
            rtol=0,
            atol=0,
        )


def test_conditioned_model_has_the_exact_low_rank_parameter_delta():
    control = missing_m3_model.MissingM3GraphModel(**_model_arguments())
    rank = 8
    treatment = missing_m3_model.MissingM3GraphModel(
        **_model_arguments(),
        readout_type="availability-low-rank",
        readout_rank=rank,
    )

    control_count = sum(parameter.numel() for parameter in control.parameters())
    treatment_count = sum(
        parameter.numel() for parameter in treatment.parameters()
    )
    hidden_dim = 2 * _model_arguments()["D_e"] + _model_arguments()[
        "graph_hidden_size"
    ]
    output_dim = _model_arguments()["n_classes"]
    expected = hidden_dim * rank + 7 * rank * output_dim + 7 * output_dim

    assert treatment_count - control_count == expected
    assert sum(
        parameter.numel()
        for parameter in treatment.conditioned_readout.parameters()
    ) == expected


def test_shared_checkpoint_loading_is_strict_and_conditioned_missing_keys_are_audited():
    shared = missing_m3_model.MissingM3GraphModel(**_model_arguments())
    state = shared.state_dict()
    explicit_shared = missing_m3_model.MissingM3GraphModel(
        **_model_arguments(), readout_type="shared"
    )

    strict_result = explicit_shared.load_state_dict(state, strict=True)

    assert strict_result.missing_keys == []
    assert strict_result.unexpected_keys == []

    conditioned = missing_m3_model.MissingM3GraphModel(
        **_model_arguments(), readout_type="availability-low-rank"
    )
    migration = conditioned.load_state_dict(state, strict=False)
    assert set(migration.missing_keys) == {
        "conditioned_readout.basis.weight",
        "conditioned_readout.pattern_factor",
        "conditioned_readout.pattern_bias",
    }
    assert migration.unexpected_keys == []


def test_graph_model_rejects_nonpositive_readout_rank_even_in_shared_mode():
    with pytest.raises(ValueError, match="readout_rank"):
        missing_m3_model.MissingM3GraphModel(
            **_model_arguments(), readout_type="shared", readout_rank=0
        )


def test_readout_provenance_reports_variant_rank_and_exact_parameter_count():
    provenance = getattr(train_gcnet, "_readout_provenance", None)
    assert provenance is not None, "readout provenance is not implemented"
    shared = missing_m3_model.MissingM3GraphModel(**_model_arguments())
    conditioned = missing_m3_model.MissingM3GraphModel(
        **_model_arguments(),
        readout_type="availability-low-rank",
        readout_rank=8,
    )

    assert provenance(shared) == {
        "readout_type": "shared",
        "readout_rank": 8,
        "readout_parameter_count": 0,
    }
    record = provenance(conditioned)
    assert record["readout_type"] == "availability-low-rank"
    assert record["readout_rank"] == 8
    assert record["readout_parameter_count"] == sum(
        parameter.numel()
        for parameter in conditioned.conditioned_readout.parameters()
    )


def test_mosi_regression_metrics_expose_machine_checkable_collapse_diagnostics():
    metrics = train_gcnet._metrics(
        "CMUMOSI",
        torch.tensor([-2.0, -1.0, 1.0, 2.0]).numpy(),
        torch.tensor([-0.5, 0.2, 0.3, 0.8]).numpy(),
    )

    assert metrics["prediction_std"] == pytest.approx(
        torch.tensor([-0.5, 0.2, 0.3, 0.8]).numpy().std()
    )
    assert metrics["predicted_sign_count"] == 2


def test_conditioned_module_construction_preserves_the_shared_cpu_rng_state():
    torch.manual_seed(911)
    missing_m3_model.MissingM3GraphModel(**_model_arguments())
    shared_rng = torch.get_rng_state()
    torch.manual_seed(911)
    missing_m3_model.MissingM3GraphModel(
        **_model_arguments(),
        readout_type="availability-low-rank",
        readout_rank=8,
    )
    conditioned_rng = torch.get_rng_state()

    ASSERT_CLOSE(shared_rng, conditioned_rng, rtol=0, atol=0)


def test_evaluation_artifact_availability_matches_prediction_flattening_order(
    monkeypatch,
):
    logits = torch.tensor(
        [
            [[10.0], [20.0]],
            [[11.0], [21.0]],
            [[12.0], [22.0]],
        ]
    )
    availability = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 1, 0]],
            [[1, 0, 1], [0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    labels = torch.tensor(
        [[-1.0, 1.0, 2.0], [-2.0, 2.0, 0.0]], dtype=torch.float32
    )
    view = {
        "incomplete": torch.zeros(3, 2, 1),
        "availability": availability,
        "qmask": torch.zeros(2, 3),
        "umask": umask,
        "lengths": [3, 2],
        "labels": labels,
    }

    class EvaluationModel:
        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            return logits, torch.zeros(3, 2, 1), {}, None

    monkeypatch.setattr(train_gcnet, "_move_batch", lambda raw, device: raw)
    monkeypatch.setattr(
        train_gcnet,
        "_prepare_view",
        lambda data, schedule, epoch, dimensions: view,
    )

    _, artifacts = train_gcnet.evaluate_rate(
        model=EvaluationModel(),
        loader=[["batch"]],
        schedule=object(),
        dataset="CMUMOSI",
        dimensions=(1, 1, 1),
        device=torch.device("cpu"),
        collect=True,
    )

    assert artifacts is not None
    assert artifacts["predictions"].tolist() == [10.0, 11.0, 12.0, 20.0, 21.0]
    assert artifacts["availability"].tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]


def test_affine_readout_starts_at_zero_and_routes_all_seven_patterns():
    readout = _affine_readout_class()(hidden_dim=4).eval()
    hidden = torch.randn(7, 1, 4)
    availability = _seven_patterns()
    umask = torch.ones(1, 7)

    initial = readout(hidden, availability, umask)

    ASSERT_CLOSE(initial, torch.zeros_like(initial), rtol=0, atol=0)
    assert readout.input_norm.elementwise_affine is False
    with torch.no_grad():
        readout.beta.copy_(
            torch.arange(1, 8, dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
        )
    routed = readout(hidden, availability, umask)
    ASSERT_CLOSE(
        routed[:, 0, 0],
        torch.tensor([4.0, 2.0, 1.0, 6.0, 5.0, 3.0, 7.0]),
        rtol=0,
        atol=0,
    )


def test_affine_readout_excludes_padding_and_updates_only_the_selected_row():
    readout = _affine_readout_class()(hidden_dim=4)
    hidden = torch.tensor(
        [[[0.0, 1.0, 2.0, 4.0]], [[4.0, 1.0, -1.0, 2.0]]]
    )
    availability = torch.tensor(
        [[[1, 0, 0]], [[0, 0, 0]]], dtype=torch.float32
    )
    umask = torch.tensor([[1.0, 0.0]])

    residual = readout(hidden, availability, umask)
    residual.sum().backward()

    assert torch.count_nonzero(residual[-1]) == 0
    selected_row = 3
    assert torch.count_nonzero(readout.gamma.grad[selected_row]) > 0
    assert torch.count_nonzero(readout.beta.grad[selected_row]) > 0
    assert torch.count_nonzero(readout.gamma.grad[:selected_row]) == 0
    assert torch.count_nonzero(readout.gamma.grad[selected_row + 1 :]) == 0


def test_affine_model_preserves_shared_initialization_rng_and_initial_output():
    torch.manual_seed(947)
    control = missing_m3_model.MissingM3GraphModel(**_model_arguments()).eval()
    control_rng = torch.get_rng_state()
    torch.manual_seed(947)
    treatment = missing_m3_model.MissingM3GraphModel(
        **_model_arguments(), readout_type="availability-affine"
    ).eval()
    treatment_rng = torch.get_rng_state()

    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    assert set(control_state).issubset(treatment_state)
    for name, value in control_state.items():
        ASSERT_CLOSE(value, treatment_state[name], rtol=0, atol=0)
    ASSERT_CLOSE(control_rng, treatment_rng, rtol=0, atol=0)
    inputs = _model_inputs()
    first = control([inputs[0]], *inputs[1:])
    second = treatment([inputs[0]], *inputs[1:])
    ASSERT_CLOSE(first[0], second[0], rtol=0, atol=0)
    ASSERT_CLOSE(first[1], second[1], rtol=0, atol=0)


def test_affine_model_conditions_only_the_final_shared_mapping(monkeypatch):
    model = missing_m3_model.MissingM3GraphModel(
        **_model_arguments(), readout_type="availability-affine"
    ).eval()
    inputs = _model_inputs()
    fixed_hidden = torch.ones(4, 2, 10)
    monkeypatch.setattr(
        model, "encode_hidden", lambda *_args, **_kwargs: fixed_hidden
    )
    with torch.no_grad():
        model.smax_fc.weight.zero_()
        model.smax_fc.weight[0, 0] = 1.0
        model.smax_fc.bias.zero_()
        model.affine_readout.beta.zero_()
        for row in range(7):
            model.affine_readout.beta[row, 0] = float(row + 1)

    logits, hidden, _, _ = model([inputs[0]], *inputs[1:])

    expected = torch.tensor(
        [[5.0, 4.0], [7.0, 2.0], [6.0, 3.0], [8.0, 1.0]]
    ).unsqueeze(-1)
    ASSERT_CLOSE(logits, expected, rtol=0, atol=0)
    ASSERT_CLOSE(hidden, fixed_hidden, rtol=0, atol=0)


def test_affine_readout_is_exposed_by_cli_and_stays_below_one_percent():
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
            "--readout-type",
            "availability-affine",
        ]
    )
    formal_arguments = _model_arguments()
    formal_arguments.update(
        D_e=100,
        graph_hidden_size=50,
        latent_dim=256,
        num_experts=4,
        top_k=2,
    )
    model = missing_m3_model.MissingM3GraphModel(
        **formal_arguments, readout_type=args.readout_type
    )
    affine_count = sum(
        parameter.numel() for parameter in model.affine_readout.parameters()
    )
    total_count = sum(parameter.numel() for parameter in model.parameters())

    assert args.readout_type == "availability-affine"
    assert affine_count == 2 * 7 * 250
    assert affine_count / total_count < 0.01


def test_conditioned_model_adds_the_target_utterance_pattern_residual(monkeypatch):
    model = missing_m3_model.MissingM3GraphModel(
        **_model_arguments(),
        readout_type="availability-low-rank",
        readout_rank=2,
    ).eval()
    inputs = _model_inputs()
    fixed_hidden = torch.ones(4, 2, 10)
    monkeypatch.setattr(
        model, "encode_hidden", lambda *_args, **_kwargs: fixed_hidden
    )
    with torch.no_grad():
        model.smax_fc.weight.zero_()
        model.smax_fc.bias.zero_()
        model.conditioned_readout.basis.weight.zero_()
        model.conditioned_readout.pattern_bias.copy_(
            torch.arange(1, 8, dtype=torch.float32).unsqueeze(-1)
        )

    logits, _, _, _ = model([inputs[0]], *inputs[1:])

    expected = torch.tensor(
        [[4.0, 3.0], [6.0, 1.0], [5.0, 2.0], [7.0, 0.0]]
    ).unsqueeze(-1)
    ASSERT_CLOSE(logits, expected, rtol=0, atol=0)


def test_readout_variant_is_exposed_by_cli_and_config():
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
            "--readout-type",
            "availability-low-rank",
            "--readout-rank",
            "8",
        ]
    )
    config = TrainConfig(
        readout_type=args.readout_type,
        readout_rank=args.readout_rank,
    )

    assert args.readout_type == "availability-low-rank"
    assert args.readout_rank == 8
    assert asdict(config)["readout_type"] == "availability-low-rank"
    assert asdict(config)["readout_rank"] == 8
    assert TrainConfig().readout_type == "shared"


def test_validation_only_training_is_explicit_and_opt_in():
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
            "--skip-test-evaluation",
        ]
    )

    assert args.skip_test_evaluation is True
    assert TrainConfig().evaluate_test is True
    assert TrainConfig(evaluate_test=False).evaluate_test is False
