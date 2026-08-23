from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from gcnet_modality_jepa import train_gcnet
from gcnet_modality_jepa.loss import FullFusedReconLoss, MaskedReconLoss
from gcnet_modality_jepa.protocol import SeedBundle


def test_full_fused_loss_reconstructs_all_modalities_when_only_audio_is_missing():
    predicted = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    target = torch.zeros_like(predicted)
    availability = torch.tensor([[[0.0, 1.0, 1.0]]])

    loss = FullFusedReconLoss()(
        [predicted],
        [target],
        [availability],
        torch.ones(1, 1),
        adim=1,
        tdim=1,
        vdim=2,
    )

    expected = torch.tensor((1.0 + 4.0 + (9.0 + 16.0) / 2.0) / 3.0)
    assert torch.allclose(loss, expected)


def test_full_fused_loss_ignores_fully_observed_utterances_and_padding():
    predicted = torch.full((3, 1, 4), 100.0, requires_grad=True)
    target = torch.zeros_like(predicted)
    with torch.no_grad():
        predicted[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    availability = torch.tensor(
        [
            [[0.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    umask = torch.tensor([[1.0, 1.0, 0.0]])

    loss = FullFusedReconLoss()(
        [predicted], [target], [availability], umask, 1, 1, 2
    )

    expected = torch.tensor((1.0 + 4.0 + (9.0 + 16.0) / 2.0) / 3.0)
    assert torch.allclose(loss, expected)


def test_full_fused_loss_is_differentiable_zero_without_missing_targets():
    predicted = torch.randn(2, 1, 6, requires_grad=True)
    target = torch.randn_like(predicted, requires_grad=True)

    loss = FullFusedReconLoss()(
        [predicted],
        [target],
        [torch.ones(2, 1, 3)],
        torch.ones(1, 2),
        2,
        2,
        2,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))
    assert target.grad is None


def _model_args(reconstruction_target: str):
    return train_gcnet.build_argument_parser().parse_args(
        [
            "--base-model",
            "LSTM",
            "--hidden",
            "4",
            "--n_classes",
            "4",
            "--dropout",
            "0",
            "--predictor-dropout",
            "0",
            "--no-cuda",
            "--reconstruction-target",
            reconstruction_target,
        ]
    )


def test_reconstruction_target_cli_defaults_to_missing_and_accepts_full_fused():
    parser = train_gcnet.build_argument_parser()

    assert parser.parse_args([]).reconstruction_target == "missing"
    assert (
        parser.parse_args(["--reconstruction-target", "full_fused"])
        .reconstruction_target
        == "full_fused"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--reconstruction-target", "all"])


def test_full_fused_mode_routes_only_primary_reconstruction_loss():
    args = SimpleNamespace(
        seed=70,
        stability_recon_weight=0.0,
        reconstruction_target="full_fused",
    )
    masked_loss = MaskedReconLoss()
    full_fused_loss = FullFusedReconLoss()

    with mock.patch.object(
        train_gcnet, "_train_or_eval_model_impl", return_value=mock.sentinel.result
    ) as implementation:
        result = train_gcnet.train_or_eval_model(
            args,
            mock.sentinel.model,
            mock.sentinel.reg_loss,
            mock.sentinel.cls_loss,
            masked_loss,
            mock.sentinel.loader,
            mock.sentinel.means,
            full_fused_rec_loss=full_fused_loss,
        )

    assert result is mock.sentinel.result
    call = implementation.call_args
    assert call.args[4] is masked_loss
    assert call.kwargs["primary_rec_loss"] is full_fused_loss


def test_reconstruction_target_does_not_change_model_parameters():
    torch.manual_seed(66)
    missing_model = train_gcnet.build_model(_model_args("missing"), 2, 3, 4)
    torch.manual_seed(66)
    full_fused_model = train_gcnet.build_model(_model_args("full_fused"), 2, 3, 4)

    missing_parameters = list(missing_model.named_parameters())
    full_fused_parameters = list(full_fused_model.named_parameters())
    assert [name for name, _ in missing_parameters] == [
        name for name, _ in full_fused_parameters
    ]
    assert [parameter.shape for _, parameter in missing_parameters] == [
        parameter.shape for _, parameter in full_fused_parameters
    ]
    assert sum(parameter.numel() for _, parameter in missing_parameters) == sum(
        parameter.numel() for _, parameter in full_fused_parameters
    )


def test_fully_observed_modes_have_zero_reconstruction_and_gradient_parity():
    torch.manual_seed(66)
    missing_model = train_gcnet.build_model(_model_args("missing"), 1, 1, 1)
    full_fused_model = train_gcnet.build_model(_model_args("full_fused"), 1, 1, 1)
    full_fused_model.load_state_dict(missing_model.state_dict(), strict=True)
    missing_model.eval()
    full_fused_model.eval()

    features = [torch.randn(2, 1, 3)]
    qmask = torch.tensor([[0.0, 1.0]])
    umask = torch.ones(1, 2)
    availability = [torch.ones(2, 1, 3)]
    missing_logits, missing_reconstruction, _, _ = missing_model(
        features, qmask, umask, [2], predict_modalities=False
    )
    full_fused_logits, full_fused_reconstruction, _, _ = full_fused_model(
        features, qmask, umask, [2], predict_modalities=False
    )
    missing_reconstruction_loss = MaskedReconLoss()(
        missing_reconstruction, features, availability, umask, 1, 1, 1
    )
    full_fused_reconstruction_loss = FullFusedReconLoss()(
        full_fused_reconstruction, features, availability, umask, 1, 1, 1
    )

    (missing_logits.square().mean() + missing_reconstruction_loss).backward()
    (full_fused_logits.square().mean() + full_fused_reconstruction_loss).backward()

    assert missing_reconstruction_loss.item() == 0.0
    assert full_fused_reconstruction_loss.item() == 0.0
    assert torch.equal(missing_logits, full_fused_logits)
    for (missing_name, missing_parameter), (fused_name, fused_parameter) in zip(
        missing_model.named_parameters(), full_fused_model.named_parameters()
    ):
        assert missing_name == fused_name
        if missing_parameter.grad is None or fused_parameter.grad is None:
            assert missing_parameter.grad is None
            assert fused_parameter.grad is None
        else:
            assert torch.allclose(
                missing_parameter.grad, fused_parameter.grad, atol=1e-7, rtol=0.0
            ), missing_name


def test_fold_metrics_record_reconstruction_target_with_legacy_default():
    legacy = train_gcnet._with_gradient_clip_fold_metric(
        {"fold": 1}, SimpleNamespace(gradient_clip_norm=0.0)
    )
    full_fused = train_gcnet._with_gradient_clip_fold_metric(
        {"fold": 1},
        SimpleNamespace(
            gradient_clip_norm=0.0, reconstruction_target="full_fused"
        ),
    )

    assert legacy["reconstruction_target"] == "missing"
    assert full_fused["reconstruction_target"] == "full_fused"


def test_manifest_method_evidence_defaults_to_missing_for_legacy_args():
    split_hash = "d" * 64

    def metadata(split, indices):
        seed = SeedBundle(66).derive(
            "data_order:CMUMOSI:fold:1:{}".format(split)
        )
        return {
            "indices": list(indices),
            "split_hash": split_hash,
            "order_seed": seed,
            "order_signature": train_gcnet.sampler_signature(indices, seed),
        }

    args = SimpleNamespace(
        dataset="CMUMOSI",
        seed=66,
        model_variant="addon",
        jepa_weight=0.1,
        loss_recon=True,
    )
    manifest = train_gcnet.build_fold_run_manifest(
        args=args,
        fold=1,
        loader_metadata={
            "train": metadata("train", (0, 1)),
            "validation": metadata("validation", (2,)),
            "test": metadata("test", (3,)),
        },
        lifecycle_evidence={
            "evaluation_protocol": "strict",
            "epochs_completed": 1,
            "best_epoch": 1,
            "best_validation_f1": 0.7,
            "test_call_count": 1,
            "mask_schedule_hashes": {
                "train": "1" * 64,
                "validation": "2" * 64,
                "test": "3" * 64,
            },
            "realized_missing_rates": {
                "train": [0.2],
                "validation": 0.2,
                "test": 0.2,
            },
        },
        fold_record={"weighted_f1": 0.7, "accuracy": 0.6},
        feature_evidence={
            "audio": {"path": "/a", "metadata_sha256": "a" * 64},
            "text": {"path": "/t", "metadata_sha256": "b" * 64},
            "visual": {"path": "/v", "metadata_sha256": "c" * 64},
        },
        environment={
            "python": "3.10",
            "torch": "2.2",
            "cuda": None,
            "cudnn": None,
            "pyg": "2.0",
            "numpy": "1.0",
            "sklearn": "1.0",
            "gpu": {"index": None, "model": None, "driver": None},
        },
        provenance={
            "command": ["python", "train"],
            "cwd": "/repo",
            "git_revision": None,
            "git_status": "clean",
        },
        shared_init_hash="4" * 64,
        training_seed=SeedBundle(66).derive("training_stochasticity:fold:1"),
        mask_rate=0.2,
        output_paths={"result_archive": "/out/result.npz", "archive_fold_index": 0},
    )

    assert manifest["method"]["reconstruction_target"] == "missing"
