from __future__ import annotations

import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset

from gcnet_modality_jepa import train_gcnet
from gcnet_modality_jepa.mask_schedule import ConversationMaskSchedule
from gcnet_modality_jepa.protocol import EpochSeededSubsetSampler, SeedBundle


class _SyntheticDataset(Dataset):
    def __init__(self, vids, labels_by_vid=None):
        self.vids = list(vids)
        self.videoLabelsNew = labels_by_vid or {
            vid: [index % 2] for index, vid in enumerate(self.vids)
        }
        self.trainVids = set()
        self.valVids = set()
        self.testVids = set()

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(items):
        return items

    @staticmethod
    def get_featDim():
        return 2, 3, 4


class LoaderProtocolIntegrationTest(unittest.TestCase):
    def test_mosi_loaders_use_official_membership_and_seeded_samplers(self):
        dataset = _SyntheticDataset(
            ["test-a", "train-b", "validation-a", "train-a"]
        )
        dataset.trainVids = {"train-a", "train-b"}
        dataset.valVids = {"validation-a"}
        dataset.testVids = {"test-a"}

        with mock.patch.object(
            train_gcnet, "load_cmumosi_dataset", return_value=dataset
        ), mock.patch.object(
            train_gcnet, "build_official_split", wraps=train_gcnet.build_official_split
        ) as split_builder:
            train_loaders, val_loaders, test_loaders, *dimensions = (
                train_gcnet.get_loaders(
                    "audio",
                    "text",
                    "video",
                    num_folder=1,
                    dataset="CMUMOSI",
                    batch_size=2,
                    num_workers=0,
                    seed=17,
                )
            )

        split_builder.assert_called_once_with(
            dataset.vids,
            train_vids=dataset.trainVids,
            validation_vids=dataset.valVids,
            test_vids=dataset.testVids,
        )
        self.assertEqual(dimensions, [2, 3, 4])
        loaders = train_loaders + val_loaders + test_loaders
        self.assertTrue(
            all(isinstance(loader.sampler, EpochSeededSubsetSampler) for loader in loaders)
        )
        self.assertEqual(tuple(train_loaders[0].sampler.indices), (1, 3))
        self.assertEqual(tuple(val_loaders[0].sampler.indices), (2,))
        self.assertEqual(tuple(test_loaders[0].sampler.indices), (0,))
        bundle = SeedBundle(17)
        self.assertEqual(
            train_loaders[0].sampler.seed,
            bundle.derive("data_order:CMUMOSI:fold:1:train"),
        )
        self.assertIsNot(val_loaders[0], test_loaders[0])

    def test_iemocap_each_fold_uses_loso_split_and_non_test_validation(self):
        vids = []
        labels = {}
        for session in range(1, 6):
            for conversation in range(1, 5):
                vid = "Ses0{}F_impro{:02d}".format(session, conversation)
                vids.append(vid)
                labels[vid] = [conversation % 2]
        dataset = _SyntheticDataset(vids, labels)

        with mock.patch.object(
            train_gcnet, "load_iemocap_dataset", return_value=dataset
        ), mock.patch.object(
            train_gcnet,
            "build_iemocap_loso_split",
            wraps=train_gcnet.build_iemocap_loso_split,
        ) as split_builder:
            train_loaders, val_loaders, test_loaders, *_ = train_gcnet.get_loaders(
                "audio",
                "text",
                "video",
                num_folder=5,
                dataset="IEMOCAPFour",
                batch_size=4,
                num_workers=0,
                seed=23,
                validation_fraction=0.25,
                evaluation_protocol="strict",
            )

        self.assertEqual(split_builder.call_count, 5)
        expected_split_seed = SeedBundle(23).derive("split")
        for fold in range(5):
            call = split_builder.call_args_list[fold]
            self.assertEqual(call.kwargs["test_session"], fold + 1)
            self.assertEqual(call.kwargs["validation_fraction"], 0.25)
            self.assertEqual(call.kwargs["seed"], expected_split_seed)
            train_indices = set(train_loaders[fold].sampler.indices)
            validation_indices = set(val_loaders[fold].sampler.indices)
            test_indices = set(test_loaders[fold].sampler.indices)
            self.assertFalse(train_indices & validation_indices)
            self.assertFalse(train_indices & test_indices)
            self.assertFalse(validation_indices & test_indices)
            self.assertTrue(
                all(not vids[index].startswith("Ses0{}".format(fold + 1))
                    for index in validation_indices)
            )
            self.assertTrue(
                all(vids[index].startswith("Ses0{}".format(fold + 1))
                    for index in test_indices)
            )
            self.assertIsNot(val_loaders[fold], test_loaders[fold])
            self.assertIsInstance(
                train_loaders[fold].sampler, EpochSeededSubsetSampler
            )

    def test_validation_fraction_cli_default_is_point_one(self):
        parser = train_gcnet.build_argument_parser()

        args = parser.parse_args([])

        self.assertEqual(args.validation_fraction, 0.1)


class PrimaryMaskIntegrationTest(unittest.TestCase):
    def make_schedule(self, split="train", model_variant="addon"):
        args = SimpleNamespace(
            dataset="CMUMOSI",
            seed=71,
            model_variant=model_variant,
            evaluation_protocol="strict",
        )
        return train_gcnet.build_mask_schedule(
            args=args,
            split=split,
            fold=1,
            mask_rate=0.5,
        )

    def test_batch_masks_are_conversation_keyed_and_padding_stays_zero(self):
        schedule = self.make_schedule()
        umask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        host, guest = train_gcnet.build_primary_mask_tensors(
            schedule,
            conversation_ids=["conversation-a", "conversation-b"],
            umask=umask,
            epoch=2,
        )

        self.assertEqual(tuple(host.shape), (5, 2, 3))
        self.assertEqual(tuple(guest.shape), (5, 2, 3))
        self.assertTrue(torch.equal(host[3:, 0], torch.zeros_like(host[3:, 0])))
        self.assertTrue(
            torch.equal(guest[3:, 0], torch.zeros_like(guest[3:, 0]))
        )
        self.assertTrue(torch.all(host[:3, 0].sum(dim=-1) >= 1))
        expected = schedule.generate(
            "conversation-a", length=5, valid_length=3, side="host", epoch=2
        )
        self.assertTrue(
            torch.equal(host[:, 0], torch.as_tensor(expected.availability))
        )

    def test_train_changes_by_epoch_but_evaluation_is_fixed(self):
        umask = torch.ones(1, 128)
        train_schedule = self.make_schedule("train")
        validation_schedule = self.make_schedule("validation")

        train_epoch_one = train_gcnet.build_primary_mask_tensors(
            train_schedule, ["conversation-a"], umask, epoch=1
        )[0]
        train_epoch_two = train_gcnet.build_primary_mask_tensors(
            train_schedule, ["conversation-a"], umask, epoch=2
        )[0]
        validation_epoch_zero = train_gcnet.build_primary_mask_tensors(
            validation_schedule, ["conversation-a"], umask, epoch=0
        )[0]
        validation_later = train_gcnet.build_primary_mask_tensors(
            validation_schedule, ["conversation-a"], umask, epoch=99
        )[0]

        self.assertFalse(torch.equal(train_epoch_one, train_epoch_two))
        self.assertTrue(torch.equal(validation_epoch_zero, validation_later))

    def test_model_variant_does_not_change_primary_masks(self):
        umask = torch.ones(2, 64)
        addon = self.make_schedule(model_variant="addon")
        replacement = self.make_schedule(model_variant="replacement")

        addon_masks = train_gcnet.build_primary_mask_tensors(
            addon, ["a", "b"], umask, epoch=4
        )
        replacement_masks = train_gcnet.build_primary_mask_tensors(
            replacement, ["a", "b"], umask, epoch=4
        )

        self.assertEqual(addon.config_hash, replacement.config_hash)
        for first, second in zip(addon_masks, replacement_masks):
            self.assertTrue(torch.equal(first, second))

    def test_wrapper_passes_explicit_protocol_context_to_implementation(self):
        args = SimpleNamespace(
            seed=70,
            dataset="CMUMOSI",
            stability_recon_weight=0.0,
        )
        schedule = ConversationMaskSchedule(
            dataset="CMUMOSI",
            split="test",
            fold=1,
            requested_missing_rate=0.5,
            mask_seed=SeedBundle(70).derive("missing_mask"),
        )

        with mock.patch.object(
            train_gcnet, "_train_or_eval_model_impl", return_value=mock.sentinel.result
        ) as implementation:
            result = train_gcnet.train_or_eval_model(
                args,
                mock.sentinel.model,
                mock.sentinel.reg_loss,
                mock.sentinel.cls_loss,
                mock.sentinel.rec_loss,
                mock.sentinel.loader,
                mock.sentinel.means,
                mask_rate=0.5,
                split="test",
                fold=1,
                epoch=0,
                mask_schedule=schedule,
                collect_artifacts=False,
            )

        self.assertIs(result, mock.sentinel.result)
        call = implementation.call_args
        self.assertEqual(call.kwargs["split"], "test")
        self.assertEqual(call.kwargs["fold"], 1)
        self.assertEqual(call.kwargs["epoch"], 0)
        self.assertIs(call.kwargs["mask_schedule"], schedule)
        self.assertFalse(call.kwargs["collect_artifacts"])

    def test_collect_artifacts_defaults_true_for_legacy_wrapper(self):
        args = SimpleNamespace(
            seed=70,
            dataset="CMUMOSI",
            stability_recon_weight=0.0,
        )

        with mock.patch.object(
            train_gcnet, "_train_or_eval_model_impl", return_value=mock.sentinel.result
        ) as implementation:
            train_gcnet.train_or_eval_model(
                args,
                mock.sentinel.model,
                mock.sentinel.reg_loss,
                mock.sentinel.cls_loss,
                mock.sentinel.rec_loss,
                mock.sentinel.loader,
                mock.sentinel.means,
            )

        self.assertTrue(implementation.call_args.kwargs["collect_artifacts"])


class SharedInitializationCliTest(unittest.TestCase):
    def test_shared_initialization_cli_options_are_available(self):
        parser = train_gcnet.build_argument_parser()

        args = parser.parse_args(
            [
                "--shared-init-checkpoint",
                "/tmp/shared.pt",
                "--require-shared-init-hash",
                "abc123",
            ]
        )

        self.assertEqual(args.shared_init_checkpoint, "/tmp/shared.pt")
        self.assertEqual(args.require_shared_init_hash, "abc123")

    def test_checkpoint_is_loaded_and_required_hash_validated(self):
        model = torch.nn.Linear(2, 1)

        with mock.patch.object(
            train_gcnet, "load_shared_checkpoint", return_value="shared-hash"
        ) as loader:
            result = train_gcnet.prepare_shared_initialization(
                model,
                checkpoint_path="shared.pt",
                required_hash="shared-hash",
            )

        self.assertEqual(result, "shared-hash")
        loader.assert_called_once_with(
            "shared.pt", model, expected_hash="shared-hash"
        )

    def test_required_hash_without_checkpoint_validates_current_shared_state(self):
        model = torch.nn.Linear(2, 1)

        with mock.patch.object(
            train_gcnet, "shared_state_hash", return_value="actual"
        ):
            with self.assertRaisesRegex(ValueError, "required shared initialization hash"):
                train_gcnet.prepare_shared_initialization(
                    model, checkpoint_path=None, required_hash="expected"
                )


class TrainingStochasticityResetTest(unittest.TestCase):
    def setUp(self):
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

    def tearDown(self):
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
        if self.cuda_states is not None:
            torch.cuda.set_rng_state_all(self.cuda_states)

    def assert_numpy_state_equal(self, first, second):
        self.assertEqual(first[0], second[0])
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2:], second[2:])

    def test_reset_replays_global_rngs_after_unrelated_variant_draws(self):
        master_seed = 70
        fold = 3
        expected_seed = SeedBundle(master_seed).derive(
            "training_stochasticity:fold:3"
        )

        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        random.random()
        np.random.rand(11)
        torch.rand(13)
        first_seed = train_gcnet.reset_training_stochasticity(
            master_seed, fold, strict_deterministic=False
        )
        first_states = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
        )
        first_draws = (random.random(), np.random.rand(4), torch.rand(4))

        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        for _ in range(5):
            random.random()
            np.random.rand(7)
            torch.rand(9)
        second_seed = train_gcnet.reset_training_stochasticity(
            master_seed, fold, strict_deterministic=False
        )
        second_states = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
        )
        second_draws = (random.random(), np.random.rand(4), torch.rand(4))

        self.assertEqual(first_seed, expected_seed)
        self.assertEqual(second_seed, expected_seed)
        self.assertEqual(first_states[0], second_states[0])
        self.assert_numpy_state_equal(first_states[1], second_states[1])
        self.assertTrue(torch.equal(first_states[2], second_states[2]))
        self.assertEqual(len(first_states[3]), len(second_states[3]))
        for first, second in zip(first_states[3], second_states[3]):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_draws[0], second_draws[0])
        self.assertTrue(np.array_equal(first_draws[1], second_draws[1]))
        self.assertTrue(torch.equal(first_draws[2], second_draws[2]))

    def test_reset_preserves_strict_deterministic_handling(self):
        expected_seed = SeedBundle(70).derive(
            "training_stochasticity:fold:2"
        )

        with mock.patch.object(train_gcnet, "set_random_seed") as seed_setter:
            actual_seed = train_gcnet.reset_training_stochasticity(
                70, 2, strict_deterministic=True
            )

        self.assertEqual(actual_seed, expected_seed)
        seed_setter.assert_called_once_with(
            expected_seed, strict_deterministic=True
        )

    def test_main_resets_after_setup_and_immediately_before_lifecycle(self):
        source = Path(train_gcnet.__file__).read_text(encoding="utf-8")

        model_setup = source.index("shared_init_hash = prepare_shared_initialization")
        means_setup = source.index("modality_means = compute_modality_means")
        reset = source.index("training_seed = reset_training_stochasticity")
        lifecycle = source.index("lifecycle = run_training_fold", reset)

        self.assertLess(model_setup, reset)
        self.assertLess(means_setup, reset)
        between_reset_and_lifecycle = source[reset:lifecycle]
        self.assertNotIn("torch.rand", between_reset_and_lifecycle)
        self.assertNotIn("np.random", between_reset_and_lifecycle)
        self.assertNotIn("random.", between_reset_and_lifecycle)


if __name__ == "__main__":
    unittest.main()
