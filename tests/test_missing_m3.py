import copy

import pytest
import torch

from gcnet_missing_m3.loss import missing_m3_loss
from gcnet_missing_m3.mixed_rate import (
    MISSING_RATES,
    BalancedBatchRateSchedule,
    mean_validation_weighted_f1,
    select_best_epoch,
)
from gcnet_missing_m3.model import (
    ContextualM3Predictor,
    MissingM3GraphModel,
    ObservedSetEncoder,
)
from gcnet_missing_m3.train_gcnet import (
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
