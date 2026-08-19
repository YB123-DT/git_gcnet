from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from gcnet_jepa_replacement.model import ReplacementJEPAGraphModel
from gcnet_modality_jepa.train_gcnet import build_model


def model_arguments() -> dict:
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
        "predictor_dropout": 0.0,
    }


class ReplacementModelTest(unittest.TestCase):
    def test_trainer_builds_replacement_variant(self) -> None:
        args = SimpleNamespace(
            hidden=4,
            base_model="LSTM",
            n_speakers=2,
            windowp=1,
            windowf=1,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
            predictor_dropout=0.0,
            model_variant="replacement",
        )

        model = build_model(args, adim=2, tdim=3, vdim=4)

        self.assertIsInstance(model, ReplacementJEPAGraphModel)

    def test_reconstruction_head_is_not_instantiated(self) -> None:
        model = ReplacementJEPAGraphModel(**model_arguments())

        self.assertFalse(any(name.startswith("linear_rec") for name, _ in model.named_parameters()))
        self.assertNotIn("linear_rec", dict(model.named_modules()))

    def test_forward_returns_no_reconstruction_tensor(self) -> None:
        model = ReplacementJEPAGraphModel(**model_arguments()).eval()
        features = [torch.randn(3, 1, 9)]
        qmask = torch.tensor([[0.0, 1.0, 0.0]])
        umask = torch.ones(1, 3)

        logits, reconstruction, _, predictions = model(
            features, qmask, umask, [3], predict_modalities=True
        )

        self.assertEqual(tuple(logits.shape), (3, 1, 6))
        self.assertEqual(reconstruction, [])
        self.assertIsNotNone(predictions)


if __name__ == "__main__":
    unittest.main()
