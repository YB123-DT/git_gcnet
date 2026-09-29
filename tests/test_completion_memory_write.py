"""Predicted writes are opt-in, causal, and distinct from real availability."""
import copy
from dataclasses import fields
import unittest
from unittest.mock import patch

import torch

from gcnet_missing_m3.model import MissingM3GraphModel, MODALITIES
from gcnet_missing_m3.train_gcnet import TrainConfig, build_parser


def model_kwargs():
    return dict(
        base_model="LSTM", adim=3, tdim=4, vdim=5, D_e=4,
        graph_hidden_size=2, n_speakers=1, window_past=2, window_future=2,
        n_classes=1, latent_dim=8, num_experts=2, top_k=1, dropout=0.,
        projector_dropout=0., predictor_dropout=0., time_attn=False,
        no_cuda=True, backbone_type="osram", osram_output_dim=10,
        osram_num_heads=2, osram_key_dim=3, osram_value_dim=4,
        osram_bidirectional=False, osram_write_step=.6,
        completion_path="pre_osram_joint_dual_projector",
        training_objective="emotion-only", disable_unused_aux_modules=True,
    )


def inputs():
    availability = torch.tensor([
        [[1, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0]],
        [[1, 1, 0], [0, 0, 0]], [[0, 1, 1], [0, 0, 0]],
        [[1, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
    ]).float()
    umask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0]]).float()
    return (torch.randn(8, 2, 12), availability,
            torch.zeros(2, 8, dtype=torch.long), umask, [7, 2])


def trace(model, args, **kwargs):
    writes, addresses = [], []
    original_write = model.osram.block_write
    original_address = model.osram._address_residual

    def capture_write(memory, keys, values, availability, beta=None):
        result = original_write(memory, keys, values, availability, beta)
        writes.append(tuple(x.detach().clone() for x in (keys, values, availability, result)))
        return result

    def capture_address(keys, query):
        addresses.append(keys.detach().clone())
        return original_address(keys, query)

    with patch.object(model.osram, "block_write", side_effect=capture_write), \
            patch.object(model.osram, "_address_residual", side_effect=capture_address):
        result = model(*args, **kwargs)
    return result, writes, addresses


class CompletionMemoryWriteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(184)

    def test_option_defaults_off_and_has_cli_flag(self):
        names = {field.name: field for field in fields(TrainConfig)}
        self.assertIn("completion_write_to_memory", names)
        self.assertFalse(names["completion_write_to_memory"].default)
        parsed = build_parser().parse_args([
            "--completion-write-to-memory", "--audio-feature", "a",
            "--text-feature", "t", "--video-feature", "v", "--output-dir", "unused",
        ])
        self.assertTrue(parsed.completion_write_to_memory)

    def models(self):
        read = MissingM3GraphModel(**model_kwargs()).eval()
        torch.nn.init.normal_(read.osram.emotion_adapter[-1].weight, std=.1)
        write = MissingM3GraphModel(**model_kwargs(), completion_write_to_memory=True).eval()
        write.load_state_dict(read.state_dict(), strict=True)
        return read, write

    def test_no_new_parameters_or_initialization_rng(self):
        torch.manual_seed(18)
        read = MissingM3GraphModel(**model_kwargs())
        state = torch.get_rng_state().clone()
        torch.manual_seed(18)
        write = MissingM3GraphModel(**model_kwargs(), completion_write_to_memory=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(list(read.state_dict()), list(write.state_dict()))
        self.assertTrue(all(torch.equal(v, write.state_dict()[k]) for k, v in read.state_dict().items()))

    def test_only_missing_write_slots_change_not_real_addresses_or_availability(self):
        read, write = self.models()
        args = inputs()
        original_mask = args[1].clone()
        _, expected, expected_addresses = trace(read, args)
        _, actual, actual_addresses = trace(write, args)
        self.assertTrue(torch.equal(args[1], original_mask))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected_addresses, actual_addresses)))
        changed_memory = False
        for t, (left, right) in enumerate(zip(expected, actual)):
            observed = args[1][t].bool()[:, None, None, :]
            missing = (~args[1][t].bool() & args[3][:, t, None].bool())[:, None, None, :]
            for original, completed in zip(left[:2], right[:2]):
                self.assertTrue(torch.equal(original[observed.expand_as(original)], completed[observed.expand_as(completed)]))
                if missing.any():
                    self.assertGreater(completed[missing.expand_as(completed)].abs().sum().item(), 0.)
            expected_write_mask = args[3][:, t, None].expand(-1, 3)
            self.assertTrue(torch.equal(right[2], expected_write_mask))
            changed_memory |= not torch.equal(left[3], right[3])
            if t >= 2:
                self.assertTrue(torch.equal(right[3][1], actual[1][3][1]))
        self.assertTrue(changed_memory)
        self.assertTrue(torch.equal(read.last_osram_context["local"], write.last_osram_context["local"]))
        self.assertEqual(write.last_osram_context["base"][0].count_nonzero().item(), 0)
        self.assertEqual(write.last_osram_context["gap"][0].count_nonzero().item(), 0)
        self.assertFalse(torch.equal(read.last_osram_context["base"][1:], write.last_osram_context["base"][1:]))

    def test_complete_sequences_are_bitwise_unchanged(self):
        read, write = self.models()
        args = list(inputs())
        args[1] = args[3].T.unsqueeze(-1).expand(-1, -1, 3).clone()
        before, history, _ = trace(read, args)
        after, new_history, _ = trace(write, args)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(all(torch.equal(x, y) for a, b in zip(history, new_history) for x, y in zip(a, b)))

    def test_prediction_write_is_after_current_read_and_changes_later_memory(self):
        _, model = self.models()
        args = inputs()
        result, history, _ = trace(model, args, predict_missing=True)
        old_context = {k: v.clone() for k, v in model.last_osram_context.items()}
        prediction = result[3].reg_predictions.detach().clone()
        prediction[2, 0, 1] = torch.randn(8) * 10
        altered, changed_history, _ = trace(model, args, completion_predictions_override=prediction)
        for name in ("base", "gap"):
            self.assertTrue(torch.equal(old_context[name][:3], model.last_osram_context[name][:3]))
        self.assertTrue(torch.equal(result[0][:3], altered[0][:3]))
        self.assertFalse(torch.equal(history[2][3], changed_history[2][3]))
        self.assertFalse(torch.equal(old_context["base"][3:], model.last_osram_context["base"][3:]))

    def test_no_missing_feature_leakage_or_future_dependency(self):
        _, model = self.models()
        args = inputs()
        expected = model(*args)[0]
        changed = copy.deepcopy(args)
        start = 0
        for i, width in enumerate(model.dimensions):
            changed[0][..., start:start + width][~args[1][..., i].bool()] = float("nan")
            start += width
        actual = model(*changed)[0]
        self.assertTrue(torch.equal(expected, actual))
        changed = copy.deepcopy(args)
        changed[0][3:] = torch.randn_like(changed[0][3:]) * 30
        self.assertTrue(torch.equal(expected[:3], model(*changed)[0][:3]))
        # The existing classifier may emit its bias on padding; only contexts
        # and writes are masked, and the task loss/evaluator exclude padding.
        for context in model.last_osram_context.values():
            self.assertEqual(context[~args[3].T.bool()].count_nonzero().item(), 0)

    def test_gradients_train_online_not_frozen_predictor(self):
        _, model = self.models()
        model.train()
        model(*inputs())[0].square().sum().backward()
        for module in (model.completion_projectors, model.source_only_predictor):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.observed_set.projectors.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.osram.value_projectors.parameters()))

    def test_rejects_unrelated_paths(self):
        kwargs = model_kwargs()
        kwargs["completion_path"] = "none"
        with self.assertRaisesRegex(ValueError, "dual-projector"):
            MissingM3GraphModel(**kwargs, completion_write_to_memory=True)
        kwargs = model_kwargs()
        kwargs["osram_bidirectional"] = True
        with self.assertRaises(ValueError):
            MissingM3GraphModel(**kwargs, completion_write_to_memory=True)


if __name__ == "__main__":
    unittest.main()
