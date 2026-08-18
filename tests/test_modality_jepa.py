from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from gcnet_modality_jepa.loss import masked_centered_cosine_loss
from gcnet_modality_jepa.metrics import compute_modality_diagnostics
from gcnet_modality_jepa.model import ModalityPredictor
from gcnet_modality_jepa.targets import ModalityMeans, compute_modality_means


class ModalityJEPATest(unittest.TestCase):
    def test_fold_means_select_speaker_and_ignore_padding(self) -> None:
        # [seq=2, batch=2, dim=1]
        audio_host = torch.tensor([[[1.0], [2.0]], [[3.0], [99.0]]])
        audio_guest = torch.tensor([[[10.0], [20.0]], [[30.0], [999.0]]])
        text_host = audio_host + 100.0
        text_guest = audio_guest + 100.0
        visual_host = audio_host + 200.0
        visual_guest = audio_guest + 200.0
        qmask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # [batch, seq]
        umask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        labels = torch.zeros((2, 2), dtype=torch.long)
        batch = (
            audio_host,
            text_host,
            visual_host,
            audio_guest,
            text_guest,
            visual_guest,
            qmask,
            umask,
            labels,
            ["Ses01", "Ses02"],
        )

        means = compute_modality_means([batch])

        # Valid selected values are host 1, guest 20, guest 30.
        torch.testing.assert_close(means.audio, torch.tensor([17.0]))
        torch.testing.assert_close(means.text, torch.tensor([117.0]))
        torch.testing.assert_close(means.visual, torch.tensor([217.0]))

    def test_predictor_outputs_one_independent_tensor_per_modality(self) -> None:
        predictor = ModalityPredictor(
            hidden_dim=5,
            audio_dim=2,
            text_dim=3,
            visual_dim=4,
            dropout=0.0,
        )
        hidden = torch.randn(7, 3, 5)

        predictions = predictor(hidden)

        self.assertEqual(predictions.audio.shape, (7, 3, 2))
        self.assertEqual(predictions.text.shape, (7, 3, 3))
        self.assertEqual(predictions.visual.shape, (7, 3, 4))
        self.assertIsNot(predictor.audio_head, predictor.text_head)
        self.assertIsNot(predictor.text_head, predictor.visual_head)

    def test_loss_uses_only_missing_real_utterances_and_stops_target_gradient(self) -> None:
        torch.manual_seed(66)
        predictor = ModalityPredictor(5, 2, 3, 4, dropout=0.0)
        hidden = torch.randn(2, 2, 5, requires_grad=True)
        predictions = predictor(hidden)
        full_features = torch.randn(2, 2, 9, requires_grad=True)
        availability = torch.ones(2, 2, 3)
        availability[0, 0, 0] = 0  # only audio at one real utterance is missing
        umask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        means = ModalityMeans(
            audio=torch.zeros(2),
            text=torch.zeros(3),
            visual=torch.zeros(4),
        )

        loss, counts = masked_centered_cosine_loss(
            predictions, full_features, availability, umask, means
        )
        changed = full_features.detach().clone()
        changed[..., 2:] += 10_000.0  # text/visual are not supervised in this batch
        unchanged_loss, _ = masked_centered_cosine_loss(
            predictions, changed, availability, umask, means
        )
        loss.backward()

        torch.testing.assert_close(loss.detach(), unchanged_loss.detach())
        self.assertEqual(counts, {"audio": 1, "text": 0, "visual": 0})
        self.assertIsNotNone(hidden.grad)
        self.assertIsNone(full_features.grad)
        self.assertIsNotNone(predictor.audio_head[-1].weight.grad)
        self.assertIsNone(predictor.text_head[-1].weight.grad)

    def test_loss_is_finite_zero_when_no_modality_is_missing(self) -> None:
        predictor = ModalityPredictor(5, 2, 3, 4, dropout=0.0)
        predictions = predictor(torch.randn(2, 2, 5))
        full_features = torch.randn(2, 2, 9)
        availability = torch.ones(2, 2, 3)
        umask = torch.ones(2, 2)
        means = ModalityMeans(torch.zeros(2), torch.zeros(3), torch.zeros(4))

        loss, counts = masked_centered_cosine_loss(
            predictions, full_features, availability, umask, means
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(counts, {"audio": 0, "text": 0, "visual": 0})

    def test_metrics_detect_correct_predictions_and_rank_one_collapse(self) -> None:
        targets = torch.eye(4)
        predictions = targets.clone()
        metrics = compute_modality_diagnostics(predictions, targets, shuffle_seed=66)

        self.assertAlmostEqual(metrics["real_cosine"], 1.0, places=6)
        self.assertGreater(metrics["real_shuffle_gap"], 0.0)
        self.assertGreater(metrics["prediction_effective_rank"], 3.9)

        collapsed = torch.ones(4, 4)
        collapsed_metrics = compute_modality_diagnostics(
            collapsed, targets, shuffle_seed=66
        )
        self.assertTrue(math.isfinite(collapsed_metrics["prediction_effective_rank"]))
        self.assertAlmostEqual(
            collapsed_metrics["prediction_effective_rank"], 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
