"""PAM-E: explicit cross-modal episodic Text memory."""

import copy

import pytest
import torch

from gcnet_missing_m3.pam_episodic import ExplicitEpisodicTextMemory
from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser


def _pam(latent_dim=8, context_dim=10, key_dim=4):
    torch.manual_seed(3)
    return ExplicitEpisodicTextMemory(
        latent_dim=latent_dim,
        context_dim=context_dim,
        key_dim=key_dim,
        dropout=0.0,
    )


def _inputs(length=3, batch=2, latent_dim=8, context_dim=10):
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


def test_observed_text_is_read_before_write_and_text_only_episodes_are_stored():
    pam = _pam()
    latents, availability, umask, base, gap = _inputs(length=2)
    # First utterance is Text-only (A/V missing), second is AV and Text-missing.
    availability[0, :, 0] = 0.0
    availability[0, :, 2] = 0.0
    availability[1, :, 1] = 0.0
    outputs = pam(latents, availability, umask, base, gap)
    assert outputs["write_mask"].tolist() == [[True, True], [False, False]]
    assert outputs["text_missing_mask"].tolist() == [[False, False], [True, True]]
    assert outputs["bank_size"].tolist() == [[0, 0], [1, 1]]
    # The second step must read the first (Text-only) episode, not its own
    # currently missing Text.  With one episode the signed ridge read is
    # exactly v * (k.q) / (k.k + lambda).
    k = outputs["query"][1] * 0.0
    # Recompute the expected first-step key/value with the public encoder.
    with torch.no_grad():
        value = latents["text"][0]
        key = torch.nn.functional.normalize(pam.text_key_encoder(value), dim=-1)
        query = outputs["query"][1]
        coefficient = (key * query).sum(-1, keepdim=True) / (
            (key * key).sum(-1, keepdim=True) + pam.ridge
        )
        expected = value * coefficient
    assert torch.allclose(outputs["z_hat_text"][1], expected, atol=1e-6)


def test_missing_text_never_writes_or_uses_current_text():
    pam = _pam()
    latents, availability, umask, base, gap = _inputs(length=3)
    availability[:, :, 1] = 0.0  # all Text missing but A/V observed
    baseline = pam(latents, availability, umask, base, gap)
    changed = {name: value.clone() for name, value in latents.items()}
    changed["text"] = changed["text"] * 100.0
    modified = pam(changed, availability, umask, base, gap)
    assert not baseline["write_mask"].any()
    assert torch.allclose(baseline["z_hat_text"], modified["z_hat_text"])
    assert torch.allclose(baseline["query"], modified["query"])
    assert torch.allclose(
        baseline["z_hat_text"], torch.zeros_like(baseline["z_hat_text"])
    )


def test_padding_does_not_enter_bank_or_read():
    pam = _pam()
    latents, availability, umask, base, gap = _inputs(length=4)
    availability[3] = 0.0
    umask[0, 3] = 0.0
    umask[1, 3] = 0.0
    outputs = pam(latents, availability, umask, base, gap)
    assert not outputs["write_mask"][3].any()
    assert torch.allclose(
        outputs["z_hat_text"][3], torch.zeros_like(outputs["z_hat_text"][3])
    )


def test_query_and_key_receive_gradients_and_values_keep_real_text_gradient():
    pam = _pam()
    latents, availability, umask, base, gap = _inputs(length=3)
    availability[1, :, 1] = 0.0
    latents["text"].requires_grad_()
    outputs = pam(latents, availability, umask, base, gap)
    loss = outputs["z_hat_text"][1].square().mean()
    loss.backward()
    assert latents["text"].grad is not None
    assert latents["text"].grad.abs().sum() > 0
    for module in (pam.query_encoder, pam.text_key_encoder):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_pam_episodic_config_and_cli_roundtrip():
    config = TrainConfig(
        training_objective="pam-episodic-text",
        completion_path="pam-episodic-text",
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
        pam_loss_weight=0.05,
    )
    assert config.completion_path == "pam-episodic-text"
    with pytest.raises(ValueError, match="matching training_objective"):
        TrainConfig(
            training_objective="pam-text",
            completion_path="pam-episodic-text",
            backbone_type="osram",
            osram_bidirectional=False,
            osram_write_step=0.6,
        )
    parser = build_parser()
    args = parser.parse_args([
        "--audio-feature", "a", "--text-feature", "t", "--video-feature", "v",
        "--output-dir", "out",
        "--completion-path", "pam-episodic-text",
        "--training-objective", "pam-episodic-text",
        "--pam-key-dim", "64",
        "--pam-loss-weight", "0.05",
    ])
    assert args.completion_path == "pam-episodic-text"
    assert args.training_objective == "pam-episodic-text"


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
        completion_path="pam-episodic-text",
        pam_key_dim=4,
        teacher_mode="ema",
        training_objective="pam-episodic-text",
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


def test_pam_episodic_model_forward_and_gradient_route():
    from gcnet_missing_m3.model import MissingM3GraphModel

    torch.manual_seed(11)
    model = MissingM3GraphModel(**_model_args())
    features, availability, qmask, umask, lengths = _model_inputs()
    logits, _, _, _ = model([features], availability, qmask, umask, lengths)
    assert logits.shape == (3, 2, 6)
    assert torch.isfinite(logits).all()
    pam = model.last_pam_outputs
    assert pam["z_hat_text"].shape == (3, 2, 8)
    target = torch.randn_like(pam["z_hat_text"])
    mask = pam["text_missing_mask"]
    loss = torch.nn.functional.smooth_l1_loss(pam["z_hat_text"][mask], target[mask])
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.pam_episodic_memory.query_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.pam_episodic_memory.text_key_encoder.parameters()
    )


def test_pam_episodic_save_load_outputs_are_equal():
    from gcnet_missing_m3.model import MissingM3GraphModel

    torch.manual_seed(13)
    model = MissingM3GraphModel(**_model_args()).eval()
    features, availability, qmask, umask, lengths = _model_inputs()
    with torch.no_grad():
        before = model([features], availability, qmask, umask, lengths)[0]
    restored = MissingM3GraphModel(**_model_args()).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    with torch.no_grad():
        after = restored([features], availability, qmask, umask, lengths)[0]
    assert torch.allclose(before, after)
