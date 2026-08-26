import pytest
import torch

from gcnet_plci_jepa.loss import plci_jepa_loss
from gcnet_plci_jepa.modules import (
    MODALITIES,
    PLCIPredictions,
    PLCITargetPrediction,
    SourceAnchoredPredictor,
)


ASSERT_CLOSE = getattr(torch.testing, "assert_close", torch.testing.assert_allclose)


def make_predictor():
    torch.manual_seed(7)
    return SourceAnchoredPredictor(
        latent_dim=4,
        hidden_dim=6,
        source_dim=3,
        context_rank=2,
        innovation_rank=2,
        context_cap=0.2,
        innovation_cap=0.3,
        embedding_dim=2,
    )


def make_inputs(availability):
    length, batch = availability.shape[:2]
    torch.manual_seed(11)
    latents = {
        name: torch.randn(length, batch, 4) for name in MODALITIES
    }
    hidden = torch.randn(length, batch, 6)
    umask = torch.ones(batch, length)
    return latents, hidden, availability.float(), umask


def test_six_patterns_route_expected_targets_and_ordered_paths():
    availability = torch.tensor(
        [
            [[1, 0, 0]],
            [[0, 1, 0]],
            [[0, 0, 1]],
            [[1, 1, 0]],
            [[1, 0, 1]],
            [[0, 1, 1]],
        ]
    )
    predictions = make_predictor()(*make_inputs(availability))

    by_utterance = {}
    for target in predictions.targets:
        by_utterance.setdefault(target.utterance_index, []).append(target)
    expected = {
        0: [(1, (0,), 1), (2, (0,), 1)],
        1: [(0, (1,), 1), (2, (1,), 1)],
        2: [(0, (2,), 1), (1, (2,), 1)],
        3: [(2, (0, 1), 2)],
        4: [(1, (0, 2), 2)],
        5: [(0, (1, 2), 2)],
    }
    assert set(by_utterance) == set(expected)
    for utterance, records in by_utterance.items():
        observed = [
            (r.target_modality, r.anchor_modalities, r.paths.shape[0])
            for r in records
        ]
        assert observed == expected[utterance]
        assert all(r.source_pattern == utterance for r in records)
        assert all(r.paths.shape[1:] == (4,) for r in records)
        assert all(torch.isfinite(r.paths).all() for r in records)
        for record in records:
            ASSERT_CLOSE(torch.norm(record.paths, dim=-1), torch.ones(record.paths.shape[0]))


def test_dual_paths_compute_one_shared_context_and_start_with_zero_innovation():
    predictor = make_predictor()
    calls = []
    handle = predictor.context_outputs["visual"].register_forward_hook(
        lambda _module, _inputs, output: calls.append(output)
    )
    try:
        prediction = predictor(
            *make_inputs(torch.tensor([[[1, 1, 0]]]))
        ).targets[0]
    finally:
        handle.remove()

    assert prediction.paths.shape == (2, 4)
    assert len(calls) == 1
    assert prediction.context_norm.ndim == 0
    ASSERT_CLOSE(prediction.innovation_norms, torch.zeros(2))


def test_context_and_innovation_output_factors_are_zero_initialized():
    predictor = make_predictor()

    for output in predictor.context_outputs.values():
        assert torch.count_nonzero(output.weight) == 0
        assert torch.count_nonzero(output.bias) == 0
    for output in predictor.innovation_outputs.values():
        assert torch.count_nonzero(output.weight) == 0
        assert torch.count_nonzero(output.bias) == 0


def test_context_and_innovation_residual_norms_respect_caps():
    predictor = make_predictor()
    for output in predictor.context_outputs.values():
        torch.nn.init.constant_(output.weight, 20.0)
        torch.nn.init.constant_(output.bias, 20.0)
    for output in predictor.innovation_outputs.values():
        torch.nn.init.constant_(output.weight, 20.0)
        torch.nn.init.constant_(output.bias, 20.0)

    records = predictor(
        *make_inputs(torch.tensor([[[1, 0, 0]], [[1, 1, 0]]]))
    ).targets

    assert all(record.context_norm <= 0.2 + 1e-6 for record in records)
    assert all(
        torch.all(record.innovation_norms <= 0.3 + 1e-6)
        for record in records
    )


def test_padding_is_ignored_and_valid_invalid_patterns_are_rejected():
    predictor = make_predictor()
    availability = torch.tensor([[[1, 0, 0]], [[0, 0, 0]]])
    latents, hidden, availability, umask = make_inputs(availability)
    umask[0, 1] = 0
    assert len(predictor(latents, hidden, availability, umask).targets) == 2

    for invalid in (torch.tensor([[[0, 0, 0]]]), torch.tensor([[[1, 1, 1]]])):
        with pytest.raises(ValueError, match="active pattern"):
            predictor(*make_inputs(invalid))
    with pytest.raises(ValueError, match="binary"):
        predictor(*make_inputs(torch.tensor([[[2.0, 0.0, 0.0]]])))


@pytest.mark.parametrize(
    "latents,hidden,availability,umask,message",
    [
        ({name: torch.randn(2, 1, 4) for name in MODALITIES}, torch.randn(2, 1, 6), torch.ones(2, 1, 2), torch.ones(1, 2), r"\[L, B, 3\]"),
        ({name: torch.randn(2, 1, 4) for name in MODALITIES}, torch.randn(2, 5), torch.ones(2, 1, 3), torch.ones(1, 2), r"\[L, B, H\]"),
        ({name: torch.randn(2, 1, 4) for name in MODALITIES}, torch.randn(2, 1, 6), torch.ones(2, 1, 3), torch.ones(2, 1), r"\[B, L\]"),
        ({"audio": torch.randn(2, 1, 4)}, torch.randn(2, 1, 6), torch.ones(2, 1, 3), torch.ones(1, 2), "student_latents"),
    ],
)
def test_predictor_rejects_invalid_shapes(latents, hidden, availability, umask, message):
    with pytest.raises(ValueError, match=message):
        make_predictor()(latents, hidden, availability, umask)


def test_teacher_values_affect_only_loss_not_prediction_paths():
    predictor = make_predictor()
    predictions = predictor(*make_inputs(torch.tensor([[[1, 0, 0]]])))
    before = [record.paths.detach().clone() for record in predictions.targets]
    teacher_a = {name: torch.randn(1, 1, 4) for name in MODALITIES}
    teacher_b = {name: value.clone() for name, value in teacher_a.items()}
    teacher_b["text"].mul_(-1)

    loss_a, _ = plci_jepa_loss(predictions, teacher_a)
    loss_b, _ = plci_jepa_loss(predictions, teacher_b)

    assert not torch.allclose(loss_a, loss_b)
    for old, record in zip(before, predictions.targets):
        assert torch.equal(old, record.paths)


def test_loss_weights_utterances_not_targets_or_paths_and_detaches_teacher():
    prediction = torch.tensor([1.0, 0.0], requires_grad=True)
    predictions = PLCIPredictions(
        [
            PLCITargetPrediction(0, 1, 0, (0,), prediction[None], torch.tensor(0.0), torch.zeros(1)),
            PLCITargetPrediction(0, 2, 0, (0,), prediction[None], torch.tensor(0.0), torch.zeros(1)),
            PLCITargetPrediction(1, 2, 3, (0, 1), torch.stack((prediction, prediction)), torch.tensor(0.0), torch.zeros(2)),
        ]
    )
    # Utterance 0 has two perfect targets (loss 0); utterance 1 has one opposite
    # target with two paths (loss 2). Correct hierarchical averaging is 1.
    teacher = {
        "audio": torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]], requires_grad=True),
        "text": torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], requires_grad=True),
        "visual": torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]], requires_grad=True),
    }

    loss, counts = plci_jepa_loss(predictions, teacher)
    loss.backward()

    ASSERT_CLOSE(loss.detach(), torch.tensor(1.0))
    assert counts["utterances"] == 2
    assert counts["targets"] == 3
    assert counts["paths"] == 4
    assert counts["text_targets"] == 1
    assert counts["visual_targets"] == 2
    assert prediction.grad is not None
    assert all(value.grad is None for value in teacher.values())


def test_empty_loss_is_finite_differentiable_and_device_linked():
    predictor = make_predictor()
    latents, probe, availability, umask = make_inputs(
        torch.tensor([[[0, 0, 0]]])
    )
    probe.requires_grad_()
    umask.zero_()
    predictions = predictor(latents, probe, availability, umask)
    loss, counts = plci_jepa_loss(predictions, {})

    loss.backward()

    assert loss.item() == 0.0
    assert probe.grad is not None
    assert counts == {"utterances": 0, "targets": 0, "paths": 0,
                      "audio_targets": 0, "text_targets": 0, "visual_targets": 0,
                      "audio_paths": 0, "text_paths": 0, "visual_paths": 0}


def test_gradients_reach_base_and_zero_initialized_output_factors():
    predictor = make_predictor()
    predictions = predictor(
        *make_inputs(torch.tensor([[[1, 0, 0]], [[1, 1, 0]]]))
    )
    teacher = {name: torch.randn(2, 1, 4) for name in MODALITIES}

    loss, _ = plci_jepa_loss(predictions, teacher)
    loss.backward()

    assert predictor.base_trunk[0].weight.grad is not None
    assert torch.count_nonzero(predictor.base_trunk[0].weight.grad) > 0
    assert predictor.context_outputs["text"].weight.grad is not None
    assert torch.count_nonzero(predictor.context_outputs["text"].weight.grad) > 0
    assert predictor.innovation_outputs["visual"].weight.grad is not None
    assert torch.count_nonzero(predictor.innovation_outputs["visual"].weight.grad) > 0
