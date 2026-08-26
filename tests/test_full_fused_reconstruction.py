import torch

from versions.full_fused_reconstruction.variant import FullFusedReconLoss


def test_full_fused_loss_reconstructs_every_modality_for_incomplete_utterance():
    prediction = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    target = torch.zeros_like(prediction)
    availability = torch.tensor([[[0.0, 1.0, 1.0]]])

    loss = FullFusedReconLoss()(
        [prediction],
        [target],
        [availability],
        torch.ones(1, 1),
        adim=1,
        tdim=1,
        vdim=2,
    )

    expected = torch.tensor((1.0 + 4.0 + (9.0 + 16.0) / 2.0) / 3.0)
    assert torch.allclose(loss, expected)


def test_full_fused_loss_ignores_complete_and_padding_utterances():
    prediction = torch.full((3, 1, 4), 100.0, requires_grad=True)
    target = torch.zeros_like(prediction)
    with torch.no_grad():
        prediction[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    availability = torch.tensor(
        [
            [[0.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )

    loss = FullFusedReconLoss()(
        [prediction],
        [target],
        [availability],
        torch.tensor([[1.0, 1.0, 0.0]]),
        adim=1,
        tdim=1,
        vdim=2,
    )

    expected = torch.tensor((1.0 + 4.0 + (9.0 + 16.0) / 2.0) / 3.0)
    assert torch.allclose(loss, expected)


def test_full_fused_loss_is_differentiable_zero_without_missing_modalities():
    prediction = torch.randn(2, 1, 6, requires_grad=True)
    target = torch.randn_like(prediction)

    loss = FullFusedReconLoss()(
        [prediction],
        [target],
        [torch.ones(2, 1, 3)],
        torch.ones(1, 2),
        adim=2,
        tdim=2,
        vdim=2,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
