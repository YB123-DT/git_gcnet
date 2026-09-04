import torch

from gcnet_mosi_text_only.model import TextOnlyTemporalModel


def test_forward_shape_padding_invariance_and_backward():
    torch.manual_seed(7)
    model = TextOnlyTemporalModel(text_dim=9, hidden_dim=12, dropout=0.0)
    text = torch.randn(5, 2, 9, requires_grad=True)
    umask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
    left, hidden_left = model(text, umask)
    changed = text.detach().clone()
    changed[3:, 1] = 1000.0
    right, hidden_right = model(changed, umask)
    assert left.shape == (5, 2, 1)
    assert hidden_left.shape == (5, 2, 12)
    torch.testing.assert_close(left[:3, 1], right[:3, 1])
    torch.testing.assert_close(hidden_left[:3, 1], hidden_right[:3, 1])
    assert torch.count_nonzero(left[3:, 1]) == 0
    left.sum().backward()
    assert text.grad is not None and torch.isfinite(text.grad).all()


def test_model_registers_only_text_temporal_and_regression_components():
    model = TextOnlyTemporalModel(text_dim=11, hidden_dim=20, dropout=0.1)
    names = set(dict(model.named_modules()))
    assert {"text_projection", "temporal", "regressor"} <= names
    forbidden = ("audio", "visual", "graph", "jepa", "mmoe", "teacher", "predictor")
    assert not any(token in name.lower() for name in names for token in forbidden)
