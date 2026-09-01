import torch

from gcnet_m3_complete_context.model import CompleteM3Regressor


def test_baseline_state_keys_match_locked_checkpoint_contract():
    model = CompleteM3Regressor((5, 7, 9), latent_dim=8, dropout=0.0)
    keys = set(model.state_dict())
    assert "projectors.audio.fc1.weight" in keys
    assert "projectors.text.output_norm.bias" in keys
    assert "projectors.visual.fc2.weight" in keys
    assert "fusion.network.0.weight" in keys
    assert "fusion.network.4.bias" in keys


def test_context_disabled_is_exact_baseline():
    torch.manual_seed(5)
    baseline = CompleteM3Regressor(
        (5, 7, 9), latent_dim=8, projector_dropout=0.0, dropout=0.0
    )
    candidate = CompleteM3Regressor(
        (5, 7, 9),
        latent_dim=8,
        projector_dropout=0.0,
        dropout=0.0,
        temporal_context=True,
    )
    candidate.load_baseline_state_dict(baseline.state_dict())
    features = [torch.randn(4, 2, width) for width in (5, 7, 9)]
    umask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(
        baseline(features, umask),
        candidate(features, umask),
        rtol=0,
        atol=0,
    )


def test_temporal_context_receives_gradient_after_zero_init_step():
    model = CompleteM3Regressor(
        (5, 7, 9), latent_dim=8, dropout=0.0, temporal_context=True
    )
    features = [torch.randn(4, 2, width) for width in (5, 7, 9)]
    umask = torch.ones(2, 4)
    model(features, umask).sum().backward()
    assert torch.isfinite(model.temporal_context.output.weight.grad).all()
    assert model.temporal_context.output.weight.grad.abs().sum() > 0
