from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from gcnet_jepa_replacement.model import ReplacementJEPAGraphModel
from gcnet_modality_jepa import train_gcnet
from gcnet_modality_jepa.model import ModalityJEPAGraphModel
from gcnet_modality_jepa.protocol import SeedBundle
from gcnet_modality_jepa.train_gcnet import (
    build_model,
    compute_stability_reconstruction_loss,
    validate_training_args,
)


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


def trainer_arguments(model_variant: str, stability_recon_weight: float) -> SimpleNamespace:
    return SimpleNamespace(
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
        model_variant=model_variant,
        stability_recon_weight=stability_recon_weight,
    )


class CommonStabilityHeadTest(unittest.TestCase):
    def test_enabling_stability_preserves_existing_parameters_and_torch_rng(self) -> None:
        for model_class in (ModalityJEPAGraphModel, ReplacementJEPAGraphModel):
            with self.subTest(model_class=model_class.__name__):
                torch.manual_seed(70)
                disabled = model_class(
                    **model_arguments(), enable_stability_reconstruction=False
                )
                disabled_state = {
                    name: value.detach().clone()
                    for name, value in disabled.state_dict().items()
                }
                rng_after_disabled = torch.get_rng_state().clone()

                torch.manual_seed(70)
                enabled = model_class(
                    **model_arguments(), enable_stability_reconstruction=True
                )
                rng_after_enabled = torch.get_rng_state().clone()

                self.assertTrue(torch.equal(rng_after_enabled, rng_after_disabled))
                for name, expected in disabled_state.items():
                    self.assertTrue(
                        torch.equal(enabled.state_dict()[name], expected),
                        msg="parameter changed when stability was enabled: {}".format(name),
                    )

        torch.manual_seed(70)
        disabled_addon = ModalityJEPAGraphModel(
            **model_arguments(), enable_stability_reconstruction=False
        )
        torch.manual_seed(70)
        enabled_addon = ModalityJEPAGraphModel(
            **model_arguments(), enable_stability_reconstruction=True
        )
        self.assertTrue(torch.equal(
            enabled_addon.linear_rec.weight, disabled_addon.linear_rec.weight
        ))
        self.assertTrue(torch.equal(
            enabled_addon.linear_rec.bias, disabled_addon.linear_rec.bias
        ))

    def test_variants_share_stability_head_initialization_after_seed_reset(self) -> None:
        torch.manual_seed(70)
        addon = ModalityJEPAGraphModel(
            **model_arguments(), enable_stability_reconstruction=True
        )
        torch.manual_seed(70)
        replacement = ReplacementJEPAGraphModel(
            **model_arguments(), enable_stability_reconstruction=True
        )

        self.assertEqual(
            addon.stability_rec_head.weight.shape,
            replacement.stability_rec_head.weight.shape,
        )
        self.assertTrue(torch.equal(
            addon.stability_rec_head.weight,
            replacement.stability_rec_head.weight,
        ))
        self.assertTrue(torch.equal(
            addon.stability_rec_head.bias,
            replacement.stability_rec_head.bias,
        ))

        hidden = torch.randn(3, 2, 10)
        addon_output = addon.reconstruct_stability(hidden)
        replacement_output = replacement.reconstruct_stability(hidden)
        self.assertEqual(addon_output.shape, (3, 2, 9))
        self.assertEqual(replacement_output.shape, addon_output.shape)
        self.assertTrue(torch.equal(addon_output, replacement_output))

    def test_replacement_keeps_method_specific_reconstruction_absent(self) -> None:
        model = ReplacementJEPAGraphModel(
            **model_arguments(), enable_stability_reconstruction=True
        )

        self.assertFalse(hasattr(model, "linear_rec"))
        self.assertIn("stability_rec_head", dict(model.named_modules()))

    def test_disabled_stability_head_is_absent_and_reports_clear_error(self) -> None:
        for model_class in (ModalityJEPAGraphModel, ReplacementJEPAGraphModel):
            with self.subTest(model_class=model_class.__name__):
                model = model_class(
                    **model_arguments(), enable_stability_reconstruction=False
                )
                self.assertFalse(hasattr(model, "stability_rec_head"))
                with self.assertRaisesRegex(RuntimeError, "stability reconstruction is disabled"):
                    model.reconstruct_stability(torch.randn(1, 1, 10))

    def test_build_model_enables_stability_head_only_for_positive_weight(self) -> None:
        for variant in ("addon", "replacement"):
            with self.subTest(variant=variant):
                enabled = build_model(
                    trainer_arguments(variant, stability_recon_weight=0.25),
                    adim=2,
                    tdim=3,
                    vdim=4,
                )
                disabled = build_model(
                    trainer_arguments(variant, stability_recon_weight=0.0),
                    adim=2,
                    tdim=3,
                    vdim=4,
                )
                self.assertTrue(hasattr(enabled, "stability_rec_head"))
                self.assertFalse(hasattr(disabled, "stability_rec_head"))


class StabilityTrainerGateTest(unittest.TestCase):
    def _run_helper(self, train: bool, weight: float):
        args = SimpleNamespace(
            stability_recon_weight=weight,
            stability_aux_mask_rate=0.5,
        )
        hidden = torch.randn(2, 1, 10)
        reconstruction = torch.randn(2, 1, 9)
        model = mock.Mock()
        model.return_value = (None, [], hidden, None)
        model.reconstruct_stability.return_value = reconstruction
        rec_loss = mock.Mock(return_value=torch.tensor(1.25))
        input_features = [torch.randn(2, 1, 9)]

        result = compute_stability_reconstruction_loss(
            args=args,
            model=model,
            rec_loss=rec_loss,
            input_features=input_features,
            qmask=torch.tensor([[0.0, 1.0]]),
            umask=torch.ones(1, 2),
            lengths=[2],
            dimensions=(2, 3, 4),
            train=train,
            stability_mask_rng=np.random.RandomState(1234),
        )
        return result, model, rec_loss

    def test_training_with_positive_weight_uses_aux_hidden_and_stability_head(self) -> None:
        result, model, rec_loss = self._run_helper(train=True, weight=0.25)

        self.assertEqual(result.item(), 1.25)
        model.assert_called_once()
        model.reconstruct_stability.assert_called_once()
        rec_loss.assert_called_once()
        self.assertIs(
            model.reconstruct_stability.call_args.args[0],
            model.return_value[2],
        )
        self.assertEqual(rec_loss.call_args.args[0], [model.reconstruct_stability.return_value])

    def test_evaluation_does_not_run_auxiliary_forward(self) -> None:
        result, model, rec_loss = self._run_helper(train=False, weight=0.25)

        self.assertIsNone(result)
        model.assert_not_called()
        model.reconstruct_stability.assert_not_called()
        rec_loss.assert_not_called()

    def test_zero_weight_does_not_run_auxiliary_forward_during_training(self) -> None:
        result, model, rec_loss = self._run_helper(train=True, weight=0.0)

        self.assertIsNone(result)
        model.assert_not_called()
        model.reconstruct_stability.assert_not_called()
        rec_loss.assert_not_called()

    def test_real_training_path_restores_torch_rng_and_keeps_encoder_gradients(self) -> None:
        args = SimpleNamespace(
            stability_recon_weight=0.25,
            stability_aux_mask_rate=0.5,
        )
        model_args = model_arguments()
        model_args["dropout"] = 0.5
        model = ModalityJEPAGraphModel(
            **model_args, enable_stability_reconstruction=True
        ).train()
        input_features = [torch.randn(4, 1, 9)]
        qmask = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        umask = torch.ones(1, 4)
        stability_mask_rng = np.random.RandomState(1234)
        rng_before = torch.get_rng_state().clone()

        loss = compute_stability_reconstruction_loss(
            args=args,
            model=model,
            rec_loss=train_gcnet.MaskedReconLoss(),
            input_features=input_features,
            qmask=qmask,
            umask=umask,
            lengths=[4],
            dimensions=(2, 3, 4),
            train=True,
            stability_mask_rng=stability_mask_rng,
        )

        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
        self.assertIsNotNone(loss.grad_fn)
        loss.backward()
        self.assertIsNotNone(model.stability_rec_head.weight.grad)
        self.assertGreater(model.stability_rec_head.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.lstm.weight_ih_l0.grad)
        self.assertGreater(model.lstm.weight_ih_l0.grad.abs().sum().item(), 0.0)

    def test_replacement_accepts_stability_but_rejects_original_reconstruction(self) -> None:
        stability_only = SimpleNamespace(
            model_variant="replacement",
            loss_recon=False,
            all_modal_recon_weight=0.0,
            stability_recon_weight=0.25,
            stability_aux_mask_rate=0.1,
        )
        validate_training_args(stability_only)

        for overrides in (
            {"loss_recon": True},
            {"all_modal_recon_weight": 0.1},
        ):
            values = vars(stability_only).copy()
            values.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    validate_training_args(SimpleNamespace(**values))


class StabilityMaskRNGTest(unittest.TestCase):
    def assert_numpy_state_equal(self, first, second) -> None:
        self.assertEqual(first[0], second[0])
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2:], second[2:])

    def test_auxiliary_masks_use_advancing_local_rng_without_touching_global(self) -> None:
        np.random.seed(9876)
        global_before = np.random.get_state()
        seed_bundle = SeedBundle(master_seed=70)
        local_rng = np.random.RandomState(seed_bundle.derive("stability_mask"))
        local_before = local_rng.get_state()
        full = torch.ones(16, 2, 9)

        _, first = train_gcnet.build_masked_auxiliary_view(
            full,
            missing_rate=0.5,
            dimensions=(2, 3, 4),
            rng=local_rng,
        )
        local_after_first = local_rng.get_state()
        _, second = train_gcnet.build_masked_auxiliary_view(
            full,
            missing_rate=0.5,
            dimensions=(2, 3, 4),
            rng=local_rng,
        )
        global_after = np.random.get_state()

        self.assert_numpy_state_equal(global_before, global_after)
        with self.assertRaises(AssertionError):
            self.assert_numpy_state_equal(local_before, local_after_first)
        self.assertFalse(torch.equal(first, second))

    def test_stability_rng_factory_replays_component_stream(self) -> None:
        first_rng = train_gcnet.create_stability_mask_rng(70)
        second_rng = train_gcnet.create_stability_mask_rng(70)

        first = train_gcnet.random_mask(3, 64, 0.5, rng=first_rng)
        second = train_gcnet.random_mask(3, 64, 0.5, rng=second_rng)

        self.assertTrue(np.array_equal(first, second))


class TrainerCompatibilityTest(unittest.TestCase):
    def test_loss_vector_keeps_legacy_five_field_prefix(self) -> None:
        legacy = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = train_gcnet.build_loss_vector(
            total=legacy[0],
            primary=legacy[1],
            missing_reconstruction=legacy[2],
            jepa=legacy[3],
            all_modal_reconstruction=legacy[4],
            stability_reconstruction=6.0,
        )

        self.assertEqual(result[:5], legacy)
        self.assertEqual(result[5], 6.0)

    def test_legacy_trainer_positional_arguments_use_compatibility_wrapper(self) -> None:
        args = SimpleNamespace(stability_recon_weight=0.0, seed=70)
        model = mock.sentinel.model
        reg_loss = mock.sentinel.reg_loss
        cls_loss = mock.sentinel.cls_loss
        rec_loss = mock.sentinel.rec_loss
        dataloader = mock.sentinel.dataloader
        modality_means = mock.sentinel.modality_means
        expected = mock.sentinel.result

        with mock.patch.object(
            train_gcnet, "_train_or_eval_model_impl", return_value=expected
        ) as implementation:
            result = train_gcnet.train_or_eval_model(
                args,
                model,
                reg_loss,
                cls_loss,
                rec_loss,
                dataloader,
                modality_means,
            )

        self.assertIs(result, expected)
        implementation.assert_called_once()
        call = implementation.call_args
        self.assertEqual(call.args[:7], (
            args,
            model,
            reg_loss,
            cls_loss,
            rec_loss,
            dataloader,
            modality_means,
        ))
        self.assertIsInstance(call.kwargs["all_modal_rec_loss"], train_gcnet.AllModalReconLoss)
        self.assertIsInstance(call.kwargs["stability_mask_rng"], np.random.RandomState)


if __name__ == "__main__":
    unittest.main()
