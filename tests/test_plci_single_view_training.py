from unittest import mock

import torch

from gcnet_modality_jepa import train_gcnet
from gcnet_modality_jepa.loss import MaskedCELoss, MaskedMSELoss, MaskedReconLoss
from gcnet_modality_jepa.targets import ModalityMeans


def training_args():
    return train_gcnet.build_argument_parser().parse_args(
        [
            "--dataset",
            "IEMOCAPSix",
            "--base-model",
            "LSTM",
            "--hidden",
            "4",
            "--n_classes",
            "6",
            "--n_speakers",
            "2",
            "--dropout",
            "0",
            "--no-cuda",
            "--loss-recon",
            "--jepa-architecture",
            "plci-single",
            "--evaluation-protocol",
            "strict",
            "--seed",
            "66",
        ]
    )


def one_batch():
    torch.manual_seed(5)
    length = 4
    batch = 1
    audio_host = torch.randn(length, batch, 2)
    text_host = torch.randn(length, batch, 3)
    visual_host = torch.randn(length, batch, 4)
    audio_guest = torch.randn(length, batch, 2)
    text_guest = torch.randn(length, batch, 3)
    visual_guest = torch.randn(length, batch, 4)
    qmask = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    umask = torch.ones(batch, length)
    labels = torch.tensor([[0, 1, 2, 3]])
    return (
        audio_host,
        text_host,
        visual_host,
        audio_guest,
        text_guest,
        visual_guest,
        qmask,
        umask,
        labels,
        ["conversation-a"],
    )


def test_single_view_training_batch_calls_gcnet_hidden_once_and_never_samples_auxiliary():
    args = training_args()
    train_gcnet.validate_training_args(args)
    model = train_gcnet.build_model(args, 2, 3, 4)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    modality_means = ModalityMeans(
        torch.zeros(2),
        torch.zeros(3),
        torch.zeros(4),
    )

    with mock.patch.object(
        model, "encode_hidden", wraps=model.encode_hidden
    ) as encode_hidden, mock.patch.object(
        train_gcnet, "sample_balanced_patterns"
    ) as balanced_sampler:
        train_gcnet.train_or_eval_model(
            args,
            model,
            MaskedMSELoss(),
            MaskedCELoss(),
            MaskedReconLoss(),
            [one_batch()],
            modality_means,
            mask_rate=0.5,
            optimizer=optimizer,
            train=True,
            split="train",
            fold=5,
            epoch=0,
            collect_artifacts=False,
            plci_aux_generator=None,
        )

    assert encode_hidden.call_count == 1
    balanced_sampler.assert_not_called()
    assert model.last_prediction_umask is not None
    assert model.last_prediction_umask.sum().item() > 0
