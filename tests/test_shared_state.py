from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from gcnet_jepa_replacement.model import ReplacementJEPAGraphModel
from gcnet_modality_jepa import shared_state as shared_state_module
from gcnet_modality_jepa.model import ModalityJEPAGraphModel
from gcnet_modality_jepa.shared_state import (
    SharedStateParityError,
    assert_shared_state_parity,
    compare_shared_state,
    extract_shared_state,
    load_shared_checkpoint,
    save_shared_checkpoint,
    shared_state_hash,
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
        "enable_stability_reconstruction": True,
    }


def build_variants(seed: int = 66):
    torch.manual_seed(seed)
    addon = ModalityJEPAGraphModel(**model_arguments())
    torch.manual_seed(seed)
    replacement = ReplacementJEPAGraphModel(**model_arguments())
    return addon, replacement


def clone_variant_heads(model: nn.Module):
    excluded_prefixes = (
        "linear_rec.",
        "modality_predictor.",
        "stability_rec_head.",
    )
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(excluded_prefixes)
    }


def clone_model_state(model: nn.Module):
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


class SharedStateTest(unittest.TestCase):
    def assert_model_state_equal(self, model: nn.Module, expected) -> None:
        self.assertEqual(set(model.state_dict()), set(expected))
        for name, tensor in expected.items():
            self.assertTrue(torch.equal(model.state_dict()[name], tensor), name)

    def test_seed_reset_gives_addon_and_replacement_equal_shared_hashes(self) -> None:
        addon, replacement = build_variants()

        addon_state = extract_shared_state(addon)
        replacement_state = extract_shared_state(replacement)

        self.assertTrue(compare_shared_state(addon_state, replacement_state))
        assert_shared_state_parity(addon, replacement)
        self.assertEqual(
            shared_state_hash(addon_state), shared_state_hash(replacement_state)
        )

    def test_extract_includes_only_encoder_graph_and_classifier_cpu_clones(self) -> None:
        addon, _ = build_variants()
        addon.target_encoder = nn.Linear(3, 3)
        model_state = addon.state_dict()

        shared = extract_shared_state(addon)

        self.assertIsInstance(shared, OrderedDict)
        self.assertEqual(list(shared), sorted(shared))
        for prefix in (
            "lstm.",
            "gru.",
            "graph_net_temporal.",
            "graph_net_speaker.",
            "smax_fc.",
        ):
            self.assertTrue(any(name.startswith(prefix) for name in shared), prefix)
        for prefix in (
            "linear_rec.",
            "modality_predictor.",
            "stability_rec_head.",
            "target_encoder.",
        ):
            self.assertFalse(any(name.startswith(prefix) for name in shared), prefix)
        for name, tensor in shared.items():
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)
            self.assertNotEqual(tensor.data_ptr(), model_state[name].data_ptr())

    def test_hash_canonicalizes_mapping_order_and_tensor_layout(self) -> None:
        contiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        noncontiguous = contiguous.t()
        first = OrderedDict((("z", noncontiguous), ("a", torch.tensor([2]))))
        second = OrderedDict(
            (("a", torch.tensor([2])), ("z", noncontiguous.contiguous()))
        )

        self.assertEqual(shared_state_hash(first), shared_state_hash(second))
        self.assertEqual(len(shared_state_hash(first)), 64)
        changed_dtype = OrderedDict(second)
        changed_dtype["a"] = changed_dtype["a"].float()
        self.assertNotEqual(
            shared_state_hash(second), shared_state_hash(changed_dtype)
        )

    def test_hash_changes_with_tensor_name_shape_and_one_raw_byte(self) -> None:
        tensor = torch.arange(6, dtype=torch.uint8).reshape(2, 3)
        original = OrderedDict((("tensor", tensor),))
        renamed = OrderedDict((("renamed", tensor),))
        reshaped = OrderedDict((("tensor", tensor.reshape(3, 2)),))
        changed_tensor = tensor.clone()
        changed_tensor[0, 0] += 1
        changed_value = OrderedDict((("tensor", changed_tensor),))

        original_hash = shared_state_hash(original)
        self.assertNotEqual(original_hash, shared_state_hash(renamed))
        self.assertNotEqual(original_hash, shared_state_hash(reshaped))
        self.assertNotEqual(original_hash, shared_state_hash(changed_value))

    def test_assert_parity_reports_the_first_mismatched_tensor(self) -> None:
        addon, replacement = build_variants()
        first_name = next(iter(extract_shared_state(replacement)))
        replacement.state_dict()[first_name].view(-1)[0] += 1.0

        self.assertFalse(compare_shared_state(addon, replacement))
        with self.assertRaisesRegex(
            SharedStateParityError, "{}.*values differ".format(first_name)
        ):
            assert_shared_state_parity(addon, replacement)

    def test_atomic_checkpoint_roundtrip_leaves_variant_heads_unchanged(self) -> None:
        source, _ = build_variants(seed=66)
        torch.manual_seed(91)
        destination = ModalityJEPAGraphModel(**model_arguments())
        heads_before = clone_variant_heads(destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "shared.pt"
            saved_hash = save_shared_checkpoint(path, source, seed=66)
            loaded_hash = load_shared_checkpoint(path, destination)

            self.assertTrue(path.is_file())
            self.assertEqual(list(path.parent.glob(".shared.pt.*.tmp")), [])

        self.assertEqual(loaded_hash, saved_hash)
        assert_shared_state_parity(source, destination)
        self.assertEqual(set(clone_variant_heads(destination)), set(heads_before))
        for name, expected in heads_before.items():
            self.assertTrue(torch.equal(destination.state_dict()[name], expected), name)

    def test_saved_checkpoint_has_exact_versioned_schema(self) -> None:
        source, _ = build_variants(seed=66)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            saved_hash = save_shared_checkpoint(path, source, seed=66)
            payload = torch.load(str(path), map_location="cpu")

        self.assertEqual(
            set(payload), {"format", "version", "seed", "shared_hash", "tensors"}
        )
        self.assertEqual(
            payload["format"], shared_state_module.SHARED_CHECKPOINT_FORMAT
        )
        self.assertEqual(
            payload["version"], shared_state_module.SHARED_CHECKPOINT_VERSION
        )
        self.assertEqual(payload["format"], "gcnet-shared-state")
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["seed"], 66)
        self.assertEqual(payload["shared_hash"], saved_hash)
        self.assertEqual(payload["shared_hash"], shared_state_hash(payload["tensors"]))

    def test_save_rejects_boolean_seed(self) -> None:
        source, _ = build_variants(seed=66)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            with self.assertRaisesRegex(TypeError, "seed.*integer"):
                save_shared_checkpoint(path, source, seed=True)
            self.assertFalse(path.exists())

    def test_load_rejects_wrong_format_and_version_before_mutation(self) -> None:
        source, _ = build_variants(seed=66)
        destination, _ = build_variants(seed=91)
        destination_before = clone_model_state(destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            malformed_path = Path(directory) / "malformed.pt"
            save_shared_checkpoint(path, source, seed=66)
            original = torch.load(str(path), map_location="cpu")

            cases = (
                ("format", "not-gcnet-shared-state", "format"),
                ("version", 2, "version"),
            )
            for field, value, message in cases:
                with self.subTest(field=field):
                    malformed = dict(original)
                    malformed[field] = value
                    torch.save(malformed, str(malformed_path))
                    with self.assertRaisesRegex(ValueError, message):
                        load_shared_checkpoint(malformed_path, destination)
                    self.assert_model_state_equal(destination, destination_before)

    def test_load_rejects_each_missing_top_level_field_before_mutation(self) -> None:
        source, _ = build_variants(seed=66)
        destination, _ = build_variants(seed=91)
        destination_before = clone_model_state(destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            malformed_path = Path(directory) / "malformed.pt"
            save_shared_checkpoint(path, source, seed=66)
            original = torch.load(str(path), map_location="cpu")

            for field in ("format", "version", "seed", "shared_hash", "tensors"):
                with self.subTest(field=field):
                    malformed = dict(original)
                    del malformed[field]
                    torch.save(malformed, str(malformed_path))
                    with self.assertRaisesRegex(ValueError, "missing.*{}".format(field)):
                        load_shared_checkpoint(malformed_path, destination)
                    self.assert_model_state_equal(destination, destination_before)

    def test_load_rejects_unexpected_top_level_field_before_mutation(self) -> None:
        source, _ = build_variants(seed=66)
        destination, _ = build_variants(seed=91)
        destination_before = clone_model_state(destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            malformed_path = Path(directory) / "malformed.pt"
            save_shared_checkpoint(path, source, seed=66)
            malformed = torch.load(str(path), map_location="cpu")
            malformed["unexpected"] = True
            torch.save(malformed, str(malformed_path))

            with self.assertRaisesRegex(ValueError, "unexpected.*unexpected"):
                load_shared_checkpoint(malformed_path, destination)

        self.assert_model_state_equal(destination, destination_before)

    def test_load_rejects_malformed_seed_hash_and_tensors_before_mutation(self) -> None:
        source, _ = build_variants(seed=66)
        destination, _ = build_variants(seed=91)
        destination_before = clone_model_state(destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            malformed_path = Path(directory) / "malformed.pt"
            save_shared_checkpoint(path, source, seed=66)
            original = torch.load(str(path), map_location="cpu")

            cases = (
                ("seed", "66", "seed.*integer"),
                ("shared_hash", 123, "shared_hash.*string"),
                ("tensors", [], "tensors.*mapping"),
            )
            for field, value, message in cases:
                with self.subTest(field=field):
                    malformed = dict(original)
                    malformed[field] = value
                    torch.save(malformed, str(malformed_path))
                    with self.assertRaisesRegex(ValueError, message):
                        load_shared_checkpoint(malformed_path, destination)
                    self.assert_model_state_equal(destination, destination_before)

    def test_load_rejects_boolean_seed_before_mutation(self) -> None:
        source, _ = build_variants(seed=66)
        destination, _ = build_variants(seed=91)
        destination_before = clone_model_state(destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            malformed_path = Path(directory) / "boolean-seed.pt"
            save_shared_checkpoint(path, source, seed=66)
            malformed = torch.load(str(path), map_location="cpu")
            malformed["seed"] = True
            torch.save(malformed, str(malformed_path))

            with self.assertRaisesRegex(ValueError, "seed.*integer"):
                load_shared_checkpoint(malformed_path, destination)

        self.assert_model_state_equal(destination, destination_before)

    def test_load_rejects_corrupted_tensor_before_mutating_model(self) -> None:
        source, _ = build_variants(seed=66)
        torch.manual_seed(91)
        destination = ModalityJEPAGraphModel(**model_arguments())
        destination_before = {
            name: tensor.detach().clone()
            for name, tensor in destination.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            corrupt_path = Path(directory) / "corrupt.pt"
            save_shared_checkpoint(path, source, seed=66)
            payload = torch.load(str(path), map_location="cpu")
            first_name = next(iter(payload["tensors"]))
            payload["tensors"][first_name].view(-1)[0] += 1.0
            torch.save(payload, str(corrupt_path))

            with self.assertRaisesRegex(ValueError, "hash.*corrupt"):
                load_shared_checkpoint(corrupt_path, destination)

        for name, expected in destination_before.items():
            self.assertTrue(torch.equal(destination.state_dict()[name], expected), name)

    def test_load_rejects_missing_and_unexpected_shared_keys(self) -> None:
        source, _ = build_variants()
        destination, _ = build_variants(seed=91)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            save_shared_checkpoint(path, source, seed=66)
            original = torch.load(str(path), map_location="cpu")

            missing = dict(original)
            missing["tensors"] = OrderedDict(original["tensors"])
            missing_name, _ = missing["tensors"].popitem(last=False)
            missing["shared_hash"] = shared_state_hash(missing["tensors"])
            missing_path = Path(directory) / "missing.pt"
            torch.save(missing, str(missing_path))
            with self.assertRaisesRegex(ValueError, "missing.*{}".format(missing_name)):
                load_shared_checkpoint(missing_path, destination)

            unexpected = dict(original)
            unexpected["tensors"] = OrderedDict(original["tensors"])
            unexpected_name = "smax_fc.unexpected"
            unexpected["tensors"][unexpected_name] = torch.zeros(1)
            unexpected["shared_hash"] = shared_state_hash(unexpected["tensors"])
            unexpected_path = Path(directory) / "unexpected.pt"
            torch.save(unexpected, str(unexpected_path))
            with self.assertRaisesRegex(
                ValueError, "unexpected.*{}".format(unexpected_name)
            ):
                load_shared_checkpoint(unexpected_path, destination)


if __name__ == "__main__":
    unittest.main()
