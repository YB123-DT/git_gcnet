import copy

import pytest
import torch

from gcnet_missing_m3.loss import missing_m3_loss
from gcnet_missing_m3.mixed_rate import (
    MISSING_RATES,
    BalancedBatchRateSchedule,
    mean_validation_weighted_f1,
)
from gcnet_missing_m3.model import (
    ContextualM3Predictor,
    MissingM3GraphModel,
    ObservedSetEncoder,
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
