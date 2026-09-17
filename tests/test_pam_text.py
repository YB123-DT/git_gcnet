"""PAM-T: causal A/V-to-Text associative memory, first Text-only version."""

import copy

import pytest
import torch

from gcnet_missing_m3.b2 import CompletedReadFusion
from gcnet_missing_m3.pam import TextConditionedPAMMemory
from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser


def _pam_module(latent_dim=8, key_dim=4):
    torch.manual_seed(7)
    return TextConditionedPAMMemory(
        latent_dim=latent_dim, key_dim=key_dim, dropout=0.0
    )


def _latents(length=3, batch=2, dim=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "audio": torch.randn(length, batch, dim, generator=generator),
        "text": torch.randn(length, batch, dim, generator=generator),
        "visual": torch.randn(length, batch, dim, generator=generator),
    }


def _availability(length=3, batch=2, text_present=True):
    availability = torch.zeros(length, batch, 3)
    availability[..., 0] = 1.0
    availability[..., 2] = 1.0
    availability[..., 1] = 1.0 if text_present else 0.0
    return availability


def test_empty_memory_first_read_is_zero_and_same_step_read_precedes_write():
    pam = _pam_module()
    latents = _latents(length=1)
    availability = _availability(length=1, text_present=True)
    umask = torch.ones(2, 1)
    outputs = pam(latents, availability, umask)
    # The first read sees zero memory even though the current Text is
    # observed and will be written after the read.
    assert torch.allclose(
        outputs["z_hat_text"], torch.zeros_like(outputs["z_hat_text"])
    )
    assert outputs["write_mask"].all()

    # Changing the observed Text input must not change this pre-write read.
    changed = {name: value.clone() for name, value in latents.items()}
    changed["text"] = changed["text"] + 3.0
    second = pam(changed, availability, umask)
    assert torch.allclose(outputs["z_hat_text"], second["z_hat_text"])
    assert torch.allclose(outputs["query"], second["query"])


def test_previous_observed_text_can_be_read_at_later_step():
    pam = _pam_module()
    latents = _latents(length=2)
    latents["audio"][1] = latents["audio"][0]
    latents["visual"][1] = latents["visual"][0]
    availability = _availability(length=2, text_present=True)
    availability[1, :, 1] = 0.0  # second utterance has missing Text
    umask = torch.ones(2, 2)
    outputs = pam(latents, availability, umask)
    # Same A/V condition and same normalized query at both steps.
    assert torch.allclose(outputs["query"][0], outputs["query"][1])
    # beta=0.5: M = 0.5 * v q^T, so M q = 0.5 * v.
    assert torch.allclose(
        outputs["z_hat_text"][1], 0.5 * latents["text"][0], atol=1e-6
    )


def test_text_missing_never_updates_memory():
    pam = _pam_module()
    latents = _latents(length=2)
    availability = _availability(length=2, text_present=False)
    umask = torch.ones(2, 2)
    outputs = pam(latents, availability, umask)
    assert not outputs["write_mask"].any()
    assert torch.allclose(
        outputs["z_hat_text"], torch.zeros_like(outputs["z_hat_text"])
    )


def test_hidden_text_input_does_not_change_pam_when_text_is_missing():
    pam = _pam_module()
    latents = _latents(length=2)
    availability = _availability(length=2, text_present=False)
    umask = torch.ones(2, 2)
    baseline = pam(latents, availability, umask)
    changed = {name: value.clone() for name, value in latents.items()}
    changed["text"] = changed["text"] * 100.0
    modified = pam(changed, availability, umask)
    assert torch.allclose(baseline["z_hat_text"], modified["z_hat_text"])
    assert torch.allclose(baseline["query"], modified["query"])
    assert torch.allclose(baseline["pam_mask"], modified["pam_mask"])


def test_pam_write_value_keeps_real_text_gradient_and_loss_reaches_source_and_beta():
    pam = _pam_module()
    latents = _latents(length=2)
    latents["text"].requires_grad_()
    availability = _availability(length=2, text_present=True)
    availability[1, :, 1] = 0.0
    umask = torch.ones(2, 2)
    outputs = pam(latents, availability, umask)
    loss = outputs["z_hat_text"][1].square().mean()
    loss.backward()
    # The real observed Text latent is the write value and remains in the
    # differentiable memory trajectory for later missing-Text reads.
    assert latents["text"].grad is not None
    assert latents["text"].grad.abs().sum() > 0
    # beta and the source encoder receive gradients through future reads.
    assert pam.beta.grad is not None and pam.beta.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in pam.source_encoder.parameters()
    )


def test_pam_prediction_supervision_mask_is_text_missing_only():
    pam = _pam_module()
    latents = _latents(length=2, batch=1)
    availability = torch.tensor(
        [[[1, 1, 0]], [[1, 0, 1]]], dtype=torch.float32
    )
    umask = torch.ones(1, 2)
    outputs = pam(latents, availability, umask)
    assert outputs["text_missing_mask"].tolist() == [[False], [True]]


def test_padding_does_not_change_valid_pam_reads():
    pam = _pam_module()
    latents = _latents(length=4, batch=2)
    availability = _availability(length=4, text_present=True)
    availability[3, :, :] = 0.0
    umask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    outputs = pam(latents, availability, umask)
    # The incomplete-padding step does not contribute a read or write.
    assert torch.allclose(
        outputs["z_hat_text"][3], torch.zeros_like(outputs["z_hat_text"][3])
    )
    assert not outputs["pam_mask"][3].any()


def test_completed_read_fusion_can_restrict_active_positions():
    fusion = CompletedReadFusion(latent_dim=4, dropout=0.0)
    observed = torch.randn(3, 2, 4)
    latents = {
        name: torch.randn(3, 2, 4) for name in ("audio", "text", "visual")
    }
    availability = torch.ones(3, 2, 3)
    umask = torch.ones(2, 3)
    predictions = torch.zeros(3, 2, 3, 4)
    active = torch.zeros(3, 2, dtype=torch.bool)
    output = fusion(
        observed, latents, predictions, availability, umask, active_mask=active
    )
    assert torch.allclose(output, observed)
    # Legacy callers keep the old all-incomplete behavior.
    output_legacy = fusion(observed, latents, predictions, availability, umask)
    assert torch.allclose(output_legacy, observed)


def _pam_model_args(**overrides):
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
        completion_path="pam-text",
        pam_key_dim=4,
        teacher_mode="ema",
        training_objective="pam-text",
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


def test_default_model_has_no_pam_module():
    from gcnet_missing_m3.model import MissingM3GraphModel

    model = MissingM3GraphModel(**_pam_model_args(
        completion_path="none", training_objective="emotion-only"
    ))
    assert "pam_text_memory" not in dict(model.named_modules())


def test_pam_text_model_forward_and_pam_gradient_path():
    from gcnet_missing_m3.model import MissingM3GraphModel

    torch.manual_seed(11)
    model = MissingM3GraphModel(**_pam_model_args())
    features, availability, qmask, umask, lengths = _model_inputs()
    logits, hidden, _, _ = model(
        [features], availability, qmask, umask, lengths
    )
    assert logits.shape == (3, 2, 6)
    assert torch.isfinite(logits).all()
    pam = model.last_pam_outputs
    assert pam["z_hat_text"].shape == (3, 2, 8)
    target = torch.randn_like(pam["z_hat_text"])
    mask = pam["pam_mask"]
    pam_loss = torch.nn.functional.smooth_l1_loss(
        pam["z_hat_text"][mask], target[mask]
    )
    pam_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.pam_text_memory.source_encoder.parameters()
    )


def test_pam_text_save_load_outputs_are_equal():
    from gcnet_missing_m3.model import MissingM3GraphModel

    torch.manual_seed(13)
    model = MissingM3GraphModel(**_pam_model_args()).eval()
    features, availability, qmask, umask, lengths = _model_inputs()
    with torch.no_grad():
        before = model([features], availability, qmask, umask, lengths)[0]
    restored = MissingM3GraphModel(**_pam_model_args()).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    with torch.no_grad():
        after = restored([features], availability, qmask, umask, lengths)[0]
    assert torch.allclose(before, after)


def test_pam_text_config_and_cli_roundtrip():
    config = TrainConfig(
        training_objective="pam-text",
        completion_path="pam-text",
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
    assert config.pam_key_dim == 64
    assert config.pam_loss_weight == 0.05
    with pytest.raises(ValueError, match="pam-text requires both"):
        TrainConfig(
            training_objective="joint",
            completion_path="pam-text",
            backbone_type="osram",
            osram_bidirectional=False,
            osram_write_step=0.6,
        )
    parser = build_parser()
    args = parser.parse_args([
        "--audio-feature", "a", "--text-feature", "t", "--video-feature", "v",
        "--output-dir", "out",
        "--completion-path", "pam-text",
        "--training-objective", "pam-text",
        "--pam-key-dim", "64",
        "--pam-loss-weight", "0.05",
    ])
    assert args.completion_path == "pam-text"
    assert args.training_objective == "pam-text"
    assert args.pam_key_dim == 64
    assert args.pam_loss_weight == 0.05
