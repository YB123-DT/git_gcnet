from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "gcnet"))

from model import GraphModel as BaselineGraphModel
from gcnet_modality_jepa.model import ModalityJEPAGraphModel
from gcnet_modality_jepa.parity import (
    compare_shared_gradients,
    compare_shared_tensors,
    load_shared_backbone,
    miss0_jepa_loss,
)


def _model_arguments() -> dict:
    return {
        "base_model": "LSTM",
        "adim": 2,
        "tdim": 3,
        "vdim": 4,
        "D_e": 4,
        "graph_hidden_size": 2,
        "n_speakers": 2,
        "window_past": 1,
        "window_future": 1,
        "n_classes": 6,
        "dropout": 0.0,
        "time_attn": False,
        "no_cuda": True,
    }


class MissZeroParityTest(unittest.TestCase):
    def test_shared_checkpoint_produces_identical_logits(self) -> None:
        torch.manual_seed(66)
        baseline = BaselineGraphModel(**_model_arguments()).eval()
        jepa = ModalityJEPAGraphModel(
            **_model_arguments(), predictor_dropout=0.1
        ).eval()
        load_shared_backbone(baseline, jepa)
        features = [torch.randn(3, 1, 9)]
        qmask = torch.tensor([[0.0, 1.0, 0.0]])
        umask = torch.ones(1, 3)

        baseline_logits, _, _ = baseline(features, qmask, umask, [3])
        jepa_logits, _, _, predictions = jepa(
            features, qmask, umask, [3], predict_modalities=False
        )

        self.assertIsNone(predictions)
        self.assertLess(
            torch.max(torch.abs(baseline_logits - jepa_logits)).item(), 1e-6
        )
        self.assertEqual(compare_shared_tensors(baseline, jepa), 0.0)

    def test_miss0_loss_is_detached_zero_with_zero_parameter_gradient(self) -> None:
        model = ModalityJEPAGraphModel(
            **_model_arguments(), predictor_dropout=0.1
        )

        loss, gradient_norm = miss0_jepa_loss(model)

        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(gradient_norm, 0.0)

    def test_one_optimizer_step_keeps_shared_parameters_identical(self) -> None:
        torch.manual_seed(66)
        baseline = BaselineGraphModel(**_model_arguments()).train()
        jepa = ModalityJEPAGraphModel(
            **_model_arguments(), predictor_dropout=0.1
        ).train()
        load_shared_backbone(baseline, jepa)
        baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
        jepa_optimizer = torch.optim.Adam(jepa.parameters(), lr=1e-3)
        features = [torch.randn(3, 1, 9)]
        qmask = torch.tensor([[0.0, 1.0, 0.0]])
        umask = torch.ones(1, 3)
        labels = torch.tensor([0, 1, 2])

        baseline_logits, _, _ = baseline(features, qmask, umask, [3])
        jepa_logits, _, _, _ = jepa(
            features, qmask, umask, [3], predict_modalities=False
        )
        baseline_loss = torch.nn.functional.cross_entropy(
            baseline_logits.view(-1, 6), labels
        )
        jepa_loss = torch.nn.functional.cross_entropy(
            jepa_logits.view(-1, 6), labels
        )
        baseline_loss.backward()
        jepa_loss.backward()

        self.assertLess(compare_shared_gradients(baseline, jepa), 1e-6)
        self.assertTrue(
            all(parameter.grad is None for parameter in jepa.modality_predictor.parameters())
        )
        baseline_optimizer.step()
        jepa_optimizer.step()
        self.assertLess(compare_shared_tensors(baseline, jepa), 1e-6)


if __name__ == "__main__":
    unittest.main()
