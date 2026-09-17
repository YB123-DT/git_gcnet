"""PAM-A: explicit episodic Text memory with direct signed address supervision."""

import copy

import pytest
import torch

from gcnet_missing_m3.pam_address import ExplicitAddressTextMemory
from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser


def _pam(latent_dim=8, context_dim=10, key_dim=4):
    torch.manual_seed(5)
    return ExplicitAddressTextMemory(
        latent_dim=latent_dim,
        context_dim=context_dim,
        key_dim=key_dim,
        dropout=0.0,
    )


def _inputs(length=4, batch=2, latent_dim=8, context_dim=10):
    generator = torch.Generator().manual_seed(0)
    latents = {
        name: torch.randn(length, batch, latent_dim, generator=generator)
        for name in ("audio", "text", "visual")
    }
    availability = torch.zeros(length, batch, 3)
    availability[..., 0] = 1.0
    availability[..., 2] = 1.0
    availability[..., 1] = 1.0
    base = torch.randn(length, batch, context_dim, generator=generator)
    gap = torch.randn(length, batch, context_dim, generator=generator)
    umask = torch.ones(batch, length)
    return latents, availability, umask, base, gap


def test_bank_is_read_before_write_and_text_only_episodes_are_stored():
    pam = _pam()
    with torch.no_grad():
        pam.address_scorer[-1].weight.normal_(0.0, 0.1)
    latents, availability, umask, base, gap = _inputs(length=2)
    availability[0, :, 0] = 0.0
    availability[0, :, 2] = 0.0  # T-only first episode
    availability[1, :, 1] = 0.0  # Text-missing second episode
    target = torch.randn_like(latents["text"])
    outputs = pam(latents, availability, umask, base, gap, target_text=target)
    assert outputs["write_mask"].tolist() == [[True, True], [False, False]]
    assert outputs["text_missing_mask"].tolist() == [[False, False], [True, True]]
    assert outputs["bank_size"].tolist() == [[0, 0], [1, 1]]
    # The first read is always zero because the first episode is written after
    # the read.
    assert torch.allclose(
        outputs["z_hat_text"][0], torch.zeros_like(outputs["z_hat_text"][0])
    )
    # A second-step read comes from the explicit bank.
    assert outputs["z_hat_text"][1].abs().sum() > 0
    assert outputs["address_count"].item() > 0
    assert torch.isfinite(outputs["address_loss"])


def test_missing_text_never_writes_and_current_text_does_not_change_context():
    pam = _pam()
    latents, availability, umask, base, gap = _inputs(length=3)
    availability[:, :, 1] = 0.0
    baseline = pam(latents, availability, umask, base, gap)
    changed = {name: value.clone() for name, value in latents.items()}
    changed["text"] = changed["text"] * 100.0
    modified = pam(changed, availability, umask, base, gap)
    assert not baseline["write_mask"].any()
    assert torch.allclose(baseline["z_hat_text"], modified["z_hat_text"])


def test_padding_is_not_read_or_written():
    pam = _pam()
    latents, availability, umask, base, gap = _inputs(length=3)
    availability[2] = 0.0
    umask[0, 2] = 0.0
    umask[1, 2] = 0.0
    outputs = pam(latents, availability, umask, base, gap)
    assert not outputs["write_mask"][2].any()
    assert torch.allclose(
        outputs["z_hat_text"][2], torch.zeros_like(outputs["z_hat_text"][2])
    )


def test_address_loss_is_finite_and_scorer_receives_gradient():
    pam = _pam()
    latents, availability, umask, base, gap = _inputs(length=4)
    availability[1, :, 1] = 0.0
    availability[3, :, 1] = 0.0
    target = torch.randn_like(latents["text"])
    outputs = pam(latents, availability, umask, base, gap, target_text=target)
    assert outputs["address_count"].item() > 0
    loss = outputs["address_loss"]
    loss.backward()
    # First backward only moves the zero-initialized scorer output layer.
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in pam.address_scorer.parameters()
    )


def test_pam_address_config_and_cli_roundtrip():
    config = TrainConfig(
        training_objective="pam-address-text",
        completion_path="pam-address-text",
        backbone_type="osram",
        osram_bidirectional=False,
        osram_write_step=0.6,
        osram_readout_fusion="flat",
        osram_ablation="full",
        osram_emotion_ablation="full",
        osram_predictor_mode="structured",
        fusion_type="mean",
        representation_type="slot",
        teacher_mode="ema",
        pam_key_dim=64,
        pam_loss_weight=5.0,
    )
    assert config.completion_path == "pam-address-text"
    with pytest.raises(ValueError, match="matching training_objective"):
        TrainConfig(
            training_objective="pam-text",
            completion_path="pam-address-text",
            backbone_type="osram",
            osram_bidirectional=False,
            osram_write_step=0.6,
        )
    parser = build_parser()
    args = parser.parse_args([
        "--audio-feature", "a", "--text-feature", "t", "--video-feature", "v",
        "--output-dir", "out",
        "--completion-path", "pam-address-text",
        "--training-objective", "pam-address-text",
        "--pam-key-dim", "64",
        "--pam-loss-weight", "5.0",
    ])
    assert args.completion_path == "pam-address-text"
    assert args.training_objective == "pam-address-text"


def _model_args(**overrides):
    values = dict(
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
        backbone_type="osram",
        osram_output_dim=16,
        osram_num_heads=2,
        osram_key_dim=4,
        osram_value_dim=4,
        osram_bidirectional=False,
        osram_write_step=0.6,
        osram_readout_fusion="flat",
        osram_ablation="full",
        osram_emotion_ablation="full",
        osram_predictor_mode="structured",
        completion_path="pam-address-text",
        pam_key_dim=4,
        teacher_mode="ema",
        training_objective="pam-address-text",
        fusion_type="mean",
        representation_type="slot",
    )
    values.update(overrides)
    return values


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


def test_pam_address_model_forward_and_save_load():
    from gcnet_missing_m3.model import MissingM3GraphModel

    torch.manual_seed(17)
    model = MissingM3GraphModel(**_model_args()).eval()
    features, availability, qmask, umask, lengths = _model_inputs()
    with torch.no_grad():
        logits = model([features], availability, qmask, umask, lengths)[0]
    assert logits.shape == (3, 2, 6)
    assert torch.isfinite(logits).all()
    assert model.last_pam_outputs["address_loss"].item() == 0.0
    restored = MissingM3GraphModel(**_model_args()).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    with torch.no_grad():
        after = restored([features], availability, qmask, umask, lengths)[0]
    assert torch.allclose(logits, after)
