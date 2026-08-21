from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

from gcnet_modality_jepa import train_gcnet


class GradientClippingTest(unittest.TestCase):
    def test_cli_defaults_to_disabled_and_rejects_invalid_norms(self) -> None:
        parser = train_gcnet.build_argument_parser()
        args = parser.parse_args([])

        self.assertEqual(args.gradient_clip_norm, 0.0)
        train_gcnet.validate_training_args(args)

        for invalid in (-0.1, math.inf, -math.inf, math.nan):
            with self.subTest(value=invalid):
                candidate = parser.parse_args(
                    ["--gradient-clip-norm={}".format(invalid)]
                )
                with self.assertRaisesRegex(
                    ValueError, "--gradient-clip-norm.*finite.*non-negative"
                ):
                    train_gcnet.validate_training_args(candidate)

    def test_disabled_path_does_not_call_clipper_or_change_update(self) -> None:
        candidate = torch.nn.Linear(2, 1, bias=False)
        reference = torch.nn.Linear(2, 1, bias=False)
        reference.load_state_dict(candidate.state_dict())
        candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        inputs = torch.tensor([[2.0, -1.0]])

        with mock.patch.object(
            torch.nn.utils,
            "clip_grad_norm_",
            wraps=torch.nn.utils.clip_grad_norm_,
        ) as clip:
            returned_norm = train_gcnet._backward_and_optimizer_step(
                candidate(inputs).sum(),
                candidate,
                candidate_optimizer,
                gradient_clip_norm=0.0,
            )

        reference(inputs).sum().backward()
        expected_norm = float(reference.weight.grad.norm().item())
        reference_optimizer.step()

        clip.assert_not_called()
        self.assertAlmostEqual(returned_norm, expected_norm)
        self.assertTrue(torch.equal(candidate.weight, reference.weight))

    def test_enabled_path_clips_between_backward_and_optimizer_step(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        events: list[str] = []
        model.weight.register_hook(
            lambda gradient: events.append("backward") or gradient
        )
        optimizer = mock.Mock()
        optimizer.step.side_effect = lambda: events.append("step")

        def clip(parameters, norm):
            events.append("clip")
            self.assertEqual(norm, 1.0)
            self.assertEqual(list(parameters), [model.weight])
            return torch.tensor(2.5)

        with mock.patch.object(
            torch.nn.utils, "clip_grad_norm_", side_effect=clip
        ) as clip_mock:
            returned_norm = train_gcnet._backward_and_optimizer_step(
                model(torch.ones(1, 1)).sum(),
                model,
                optimizer,
                gradient_clip_norm=1.0,
            )

        self.assertEqual(events, ["backward", "clip", "step"])
        self.assertEqual(returned_norm, 2.5)
        clip_mock.assert_called_once()

    def test_training_diagnostics_summarize_pre_clip_norms(self) -> None:
        diagnostics = train_gcnet._summarize_gradient_clipping(
            configured_norm=1.0,
            pre_clip_norms=[2.5, 0.5, 1.0],
        )

        self.assertEqual(
            diagnostics,
            {
                "configured_norm": 1.0,
                "optimizer_steps": 3,
                "clipped_steps": 1,
                "clipped_fraction": 1.0 / 3.0,
                "mean_pre_clip_total_gradient_norm": 4.0 / 3.0,
                "max_pre_clip_total_gradient_norm": 2.5,
            },
        )

    def test_nonfinite_clipper_result_raises_before_optimizer_step(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = mock.Mock()

        with mock.patch.object(
            torch.nn.utils,
            "clip_grad_norm_",
            return_value=torch.tensor(float("nan")),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "pre-clip total gradient norm must be finite"
            ):
                train_gcnet._backward_and_optimizer_step(
                    model(torch.ones(1, 1)).sum(),
                    model,
                    optimizer,
                    gradient_clip_norm=1.0,
                )

        optimizer.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
