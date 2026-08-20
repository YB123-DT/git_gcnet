from __future__ import annotations

import unittest

import torch

from gcnet_modality_jepa.loss import MaskedReconLoss


class ReconstructionNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.loss = MaskedReconLoss()

    def test_equal_missing_errors_are_invariant_to_duplication(self) -> None:
        single_reconstruction = torch.zeros(2, 1, 3)
        single_target = torch.zeros_like(single_reconstruction)
        single_target[0, 0, 0] = 2.0
        single_availability = torch.ones(2, 1, 3)
        single_availability[0, 0, 0] = 0.0

        duplicated_reconstruction = torch.zeros(3, 1, 3)
        duplicated_target = torch.zeros_like(duplicated_reconstruction)
        duplicated_target[:2, 0, 0] = 2.0
        duplicated_availability = torch.ones(3, 1, 3)
        duplicated_availability[:2, 0, 0] = 0.0

        single_loss = self.loss(
            [single_reconstruction],
            [single_target],
            [single_availability],
            torch.ones(1, 2),
            1,
            1,
            1,
        )
        duplicated_loss = self.loss(
            [duplicated_reconstruction],
            [duplicated_target],
            [duplicated_availability],
            torch.ones(1, 3),
            1,
            1,
            1,
        )

        self.assertAlmostEqual(single_loss.item(), 4.0)
        self.assertTrue(torch.allclose(single_loss, duplicated_loss))

    def test_observed_and_padded_targets_do_not_affect_loss(self) -> None:
        reconstruction = torch.zeros(3, 2, 3)
        target = torch.zeros_like(reconstruction)
        target[2, 0, 0] = 2.0
        availability = torch.ones(3, 2, 3)
        availability[2, 0, 0] = 0.0
        availability[1, 1, 0] = 0.0
        umask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

        changed_target = target.clone()
        changed_target[0, 0] = 123.0
        changed_target[0, 1] = -321.0
        changed_target[1, 0] = 99.0
        changed_target[1, 1] = 1000.0
        changed_target[2, 1] = -1000.0

        original_loss = self.loss(
            [reconstruction], [target], [availability], umask, 1, 1, 1
        )
        changed_loss = self.loss(
            [reconstruction], [changed_target], [availability], umask, 1, 1, 1
        )

        self.assertAlmostEqual(original_loss.item(), 4.0)
        self.assertTrue(torch.allclose(original_loss, changed_loss))

    def test_loss_is_normalized_by_supervised_feature_dimension(self) -> None:
        reconstruction = torch.zeros(1, 1, 7, requires_grad=True)
        target = torch.tensor(
            [[[2.0, 2.0, 2.0, 2.0, 100.0, 100.0, 100.0]]],
            requires_grad=True,
        )
        availability = torch.ones(1, 1, 3)
        availability[0, 0, 0] = 0.0

        loss = self.loss(
            [reconstruction],
            [target],
            [availability],
            torch.ones(1, 1),
            4,
            2,
            1,
        )
        loss.backward()

        self.assertAlmostEqual(loss.item(), 4.0)
        self.assertIsNotNone(reconstruction.grad)
        self.assertIsNone(target.grad)

    def test_empty_selection_is_finite_graph_connected_zero(self) -> None:
        reconstruction = torch.randn(2, 1, 3, requires_grad=True)
        target = torch.randn(2, 1, 3, requires_grad=True)
        availability = torch.zeros(2, 1, 3)

        loss = self.loss(
            [reconstruction],
            [target],
            [availability],
            torch.zeros(1, 2),
            1,
            1,
            1,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(reconstruction.grad)
        self.assertTrue(torch.equal(reconstruction.grad, torch.zeros_like(reconstruction)))
        self.assertIsNone(target.grad)

    def test_nonempty_modalities_are_averaged_equally(self) -> None:
        reconstruction = torch.zeros(6, 1, 3)
        target = torch.zeros_like(reconstruction)
        availability = torch.ones(6, 1, 3)

        target[0, 0, 0] = 1.0
        availability[0, 0, 0] = 0.0
        target[1:3, 0, 1] = 2.0
        availability[1:3, 0, 1] = 0.0
        target[3:6, 0, 2] = 3.0
        availability[3:6, 0, 2] = 0.0

        loss = self.loss(
            [reconstruction],
            [target],
            [availability],
            torch.ones(1, 6),
            1,
            1,
            1,
        )

        self.assertAlmostEqual(loss.item(), (1.0 + 4.0 + 9.0) / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
