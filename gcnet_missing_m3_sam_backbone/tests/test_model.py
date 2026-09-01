import pytest
import torch

from gcnet_missing_m3_sam_backbone.model import MaskAwareSAMModel


def _model():
    torch.manual_seed(11)
    return MaskAwareSAMModel(
        audio_dim=4,
        text_dim=6,
        visual_dim=8,
        width=12,
        heads=3,
        dropout=0.0,
    )


def _features(length=5, batch=2):
    return [torch.randn(length, batch, dim) for dim in (4, 6, 8)]


def test_complete_forward_shape_and_backward():
    model = _model()
    features = _features()
    availability = torch.ones(5, 2, 3)
    umask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
        dtype=torch.float32,
    )

    prediction, hidden, audit = model(features, availability, umask)

    assert prediction.shape == (5, 2, 1)
    assert hidden.shape == (5, 2, 12)
    assert audit["track_valid"].shape == (5, 2, 9)
    prediction.sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.equal(prediction[3:, 1], torch.zeros_like(prediction[3:, 1]))


def test_missing_local_feature_has_no_effect():
    model = _model().eval()
    features = _features(length=3, batch=1)
    availability = torch.ones(3, 1, 3)
    availability[1, 0, 1] = 0
    changed = [value.clone() for value in features]
    changed[1][1, 0] = 10000.0
    umask = torch.ones(1, 3)

    left = model(features, availability, umask)[0]
    right = model(changed, availability, umask)[0]

    torch.testing.assert_close(left, right)


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ((1, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0, 0)),
        ((0, 1, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0)),
        ((0, 0, 1), (0, 0, 1, 0, 0, 0, 0, 0, 0)),
        ((1, 1, 0), (1, 1, 0, 1, 1, 0, 0, 0, 0)),
        ((1, 0, 1), (1, 0, 1, 0, 0, 1, 1, 0, 0)),
        ((0, 1, 1), (0, 1, 1, 0, 0, 0, 0, 1, 1)),
        ((1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1)),
    ],
)
def test_local_pattern_activates_only_available_tracks(pattern, expected):
    model = _model().eval()
    availability = torch.tensor(pattern, dtype=torch.float32).view(1, 1, 3)

    _, _, audit = model(_features(length=1, batch=1), availability, torch.ones(1, 1))

    assert tuple(audit["track_valid"][0, 0].int().tolist()) == expected


def test_observed_query_can_use_neighbor_observed_target():
    model = _model().eval()
    availability = torch.tensor(
        [[[1, 0, 0]], [[0, 1, 0]]],
        dtype=torch.float32,
    )

    _, _, audit = model(
        _features(length=2, batch=1),
        availability,
        torch.ones(1, 2),
    )

    # Track order: A, T, V, A->T, T->A, A->V, V->A, T->V, V->T.
    assert bool(audit["track_valid"][0, 0, 3])
    assert bool(audit["track_valid"][1, 0, 4])


def test_all_missing_effective_utterance_is_rejected():
    model = _model().eval()
    availability = torch.ones(2, 1, 3)
    availability[1, 0] = 0

    with pytest.raises(ValueError, match="at least one observed modality"):
        model(_features(length=2, batch=1), availability, torch.ones(1, 2))


def test_padding_feature_values_do_not_change_valid_predictions():
    model = _model().eval()
    features = _features(length=3, batch=1)
    changed = [value.clone() for value in features]
    for value in changed:
        value[2] = 10000.0
    availability = torch.ones(3, 1, 3)
    umask = torch.tensor([[1, 1, 0]], dtype=torch.float32)

    left = model(features, availability, umask)[0]
    right = model(changed, availability, umask)[0]

    torch.testing.assert_close(left[:2], right[:2])
    assert torch.equal(left[2], torch.zeros_like(left[2]))
