import pytest
import torch

from gcnet_missing_m3_text_anchor.model import TextAnchoredResidualModel


def _batch():
    features = [torch.randn(4, 2, width) for width in (5, 7, 9)]
    umask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
    availability = umask.transpose(0, 1).unsqueeze(-1).repeat(1, 1, 3)
    return features, availability, umask


def test_complete_forward_backward_and_two_cross_paths():
    model = TextAnchoredResidualModel(5, 7, 9, width=12, heads=3, dropout=0.0)
    features, availability, umask = _batch()
    prediction, hidden, audit = model(features, availability, umask)
    assert prediction.shape == (4, 2, 1)
    assert hidden.shape == (4, 2, 12)
    assert set(audit["cross_attention"]) == {"text-audio", "text-visual"}
    prediction.sum().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.parametrize("modality", (0, 2))
def test_missing_auxiliary_value_cannot_change_output(modality):
    torch.manual_seed(9)
    model = TextAnchoredResidualModel(5, 7, 9, width=12, heads=3, dropout=0.0).eval()
    features, availability, umask = _batch()
    availability[1, 0, modality] = 0
    changed = [value.clone() for value in features]
    changed[modality][1, 0] = 10000.0
    left = model(features, availability, umask)[0]
    right = model(changed, availability, umask)[0]
    torch.testing.assert_close(left, right)


def test_text_missing_uses_only_observed_fallback():
    torch.manual_seed(13)
    model = TextAnchoredResidualModel(5, 7, 9, width=12, heads=3, dropout=0.0).eval()
    features, availability, umask = _batch()
    availability[1, 0, 1] = 0
    changed = [value.clone() for value in features]
    changed[1][1, 0] = 10000.0
    left, hidden, audit = model(features, availability, umask)
    right = model(changed, availability, umask)[0]
    torch.testing.assert_close(left, right)
    assert torch.isfinite(hidden).all()
    assert audit["used_text_anchor"][1, 0].item() is False


def test_empty_valid_utterance_is_rejected():
    model = TextAnchoredResidualModel(5, 7, 9, width=12, heads=3, dropout=0.0)
    features, availability, umask = _batch()
    availability[0, 0] = 0
    with pytest.raises(ValueError, match="at least one"):
        model(features, availability, umask)
