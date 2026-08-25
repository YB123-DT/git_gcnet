import unittest

import torch
from torch_geometric.nn import GraphConv, RGCNConv

from model import GraphModel, GraphNetwork
from second_graph_aggregation import (
    GenAggGraphConv,
    SoftMedoidGraphConv,
    SSMAGraphConv,
)


class SecondGraphAggregationIntegrationTests(unittest.TestCase):
    @staticmethod
    def _model(selector=None, **overrides):
        arguments = dict(
            base_model="LSTM",
            adim=2,
            tdim=2,
            vdim=2,
            D_e=4,
            graph_hidden_size=2,
            n_speakers=2,
            window_past=1,
            window_future=1,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
            graph_conv_variant="original",
        )
        arguments.update(overrides)
        if selector is not None:
            arguments["second_graph_aggregation"] = selector
        return GraphModel(**arguments)

    @staticmethod
    def _inputs(requires_grad=False):
        values = torch.tensor(
            [
                [[0.2, -0.1, 0.3, 0.7, -0.4, 0.5],
                 [0.1, 0.4, -0.2, 0.6, 0.3, -0.5]],
                [[-0.3, 0.8, 0.5, -0.1, 0.2, 0.4],
                 [0.7, -0.6, 0.1, 0.3, -0.2, 0.9]],
                [[0.6, 0.2, -0.7, 0.4, 0.8, -0.1],
                 [-0.5, 0.3, 0.2, -0.4, 0.1, 0.7]],
            ],
            dtype=torch.float32,
            requires_grad=requires_grad,
        )
        modality_mask = torch.ones(3, 2, 3)
        qmask = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        umask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        return [values], modality_mask, qmask, umask, [3, 2]

    @staticmethod
    def _loss(outputs):
        return outputs[0].square().sum() + outputs[1][0].square().sum() + outputs[2].square().sum()

    @staticmethod
    def _native_parameter_name(candidate_name):
        return candidate_name.replace(
            ".conv2.lin_l.", ".conv2.lin_rel."
        ).replace(".conv2.lin_r.", ".conv2.lin_root.")

    def test_implicit_legacy_and_explicit_add_are_exactly_identical(self):
        seed = 101
        torch.manual_seed(seed)
        implicit = self._model()
        implicit_rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        explicit = self._model("add")
        explicit_rng = torch.get_rng_state().clone()

        self.assertIs(type(implicit.graph_net_temporal.conv2), GraphConv)
        self.assertIs(type(explicit.graph_net_temporal.conv2), GraphConv)
        self.assertTrue(torch.equal(implicit_rng, explicit_rng))
        self.assertEqual(list(implicit.state_dict()), list(explicit.state_dict()))
        for name, expected in implicit.state_dict().items():
            self.assertTrue(torch.equal(expected, explicit.state_dict()[name]), name)
        self.assertEqual(
            sum(parameter.numel() for parameter in implicit.parameters()),
            sum(parameter.numel() for parameter in explicit.parameters()),
        )

        implicit.eval()
        explicit.eval()
        implicit_inputs = self._inputs(requires_grad=True)
        explicit_inputs = self._inputs(requires_grad=True)
        implicit_outputs = implicit(*implicit_inputs)
        explicit_outputs = explicit(*explicit_inputs)
        self.assertEqual(len(implicit_outputs), 3)
        for expected, actual in zip(
            (implicit_outputs[0], implicit_outputs[1][0], implicit_outputs[2]),
            (explicit_outputs[0], explicit_outputs[1][0], explicit_outputs[2]),
        ):
            self.assertTrue(torch.equal(expected, actual))

        self._loss(implicit_outputs).backward()
        self._loss(explicit_outputs).backward()
        self.assertTrue(
            torch.equal(implicit_inputs[0][0].grad, explicit_inputs[0][0].grad)
        )
        for (expected_name, expected), (actual_name, actual) in zip(
            implicit.named_parameters(), explicit.named_parameters()
        ):
            self.assertEqual(expected_name, actual_name)
            if expected.grad is None:
                self.assertIsNone(actual.grad, expected_name)
            else:
                self.assertTrue(torch.equal(expected.grad, actual.grad), expected_name)

    def test_candidates_replace_only_conv2_and_preserve_shared_initialization(self):
        seed = 109
        torch.manual_seed(seed)
        original = self._model("add")
        original_rng = torch.get_rng_state().clone()
        original_parameters = dict(original.named_parameters())

        expected_extra_suffixes = {
            "alpha",
            "beta",
            "forward_map.layers.0.weight",
            "forward_map.layers.0.bias",
            "forward_map.layers.1.weight",
            "forward_map.layers.1.bias",
            "forward_map.layers.3.weight",
            "forward_map.layers.3.bias",
            "forward_map.layers.4.weight",
            "forward_map.layers.4.bias",
            "forward_map.layers.6.weight",
            "forward_map.layers.6.bias",
            "inverse_map.layers.0.weight",
            "inverse_map.layers.0.bias",
            "inverse_map.layers.1.weight",
            "inverse_map.layers.1.bias",
            "inverse_map.layers.3.weight",
            "inverse_map.layers.3.bias",
            "inverse_map.layers.4.weight",
            "inverse_map.layers.4.bias",
            "inverse_map.layers.6.weight",
            "inverse_map.layers.6.bias",
        }

        for selector, conv_type in (
            ("genagg", GenAggGraphConv),
            ("soft_medoid", SoftMedoidGraphConv),
            ("ssma", SSMAGraphConv),
        ):
            with self.subTest(selector=selector):
                torch.manual_seed(seed)
                candidate = self._model(selector)
                self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
                for branch in (
                    candidate.graph_net_temporal,
                    candidate.graph_net_speaker,
                ):
                    self.assertIs(type(branch.conv1), RGCNConv)
                    self.assertIsInstance(branch.conv2, conv_type)

                normalized = {
                    self._native_parameter_name(name): parameter
                    for name, parameter in candidate.named_parameters()
                }
                for name, expected in original_parameters.items():
                    self.assertIn(name, normalized)
                    self.assertTrue(torch.equal(expected, normalized[name]), name)
                extras = set(normalized) - set(original_parameters)
                if selector == "soft_medoid":
                    self.assertEqual(extras, set())
                elif selector == "ssma":
                    self.assertEqual(
                        extras,
                        {
                            "graph_net_{}.conv2.compressor.{}".format(
                                branch, suffix
                            )
                            for branch in ("temporal", "speaker")
                            for suffix in ("weight", "bias")
                        },
                    )
                else:
                    self.assertEqual(
                        extras,
                        {
                            "graph_net_{}.conv2.{}".format(branch, suffix)
                            for branch in ("temporal", "speaker")
                            for suffix in expected_extra_suffixes
                        },
                    )

    def test_locked_full_size_parameter_counts(self):
        dimensions = dict(
            adim=1024,
            tdim=1024,
            vdim=512,
            D_e=200,
            graph_hidden_size=100,
        )
        for selector, expected_delta in (
            ("add", 0),
            ("soft_medoid", 0),
            ("genagg", 118),
            ("ssma", 595_400),
        ):
            with self.subTest(selector=selector):
                model = self._model(selector, **dimensions)
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    36_419_816 + expected_delta,
                )
                self.assertEqual(
                    model.selected_path_parameter_count(),
                    34_140_166 + expected_delta,
                )

    def test_selector_is_factory_validated_at_both_constructor_levels(self):
        with self.assertRaisesRegex(ValueError, "unknown second graph aggregation"):
            GraphNetwork(8, 3, False, second_graph_aggregation="median")
        with self.assertRaisesRegex(ValueError, "unknown second graph aggregation"):
            self._model("median")

    def test_auxiliary_loss_is_safe_zero_for_non_genagg_and_before_forward(self):
        for selector in ("add", "soft_medoid", "ssma", "genagg"):
            with self.subTest(selector=selector):
                model = self._model(selector).to(dtype=torch.float64)
                loss = model.second_graph_auxiliary_loss()
                self.assertEqual(loss.shape, torch.Size([]))
                self.assertEqual(loss.dtype, torch.float64)
                self.assertEqual(loss.device, next(model.parameters()).device)
                self.assertEqual(loss.item(), 0.0)

    def test_ssma_selected_path_forward_and_backward_are_finite(self):
        torch.manual_seed(131)
        model = self._model("ssma")
        inputs = self._inputs(requires_grad=True)

        outputs = model(*inputs)
        self.assertTrue(all(torch.isfinite(value).all() for value in (
            outputs[0], outputs[1][0], outputs[2]
        )))
        self._loss(outputs).backward()

        self.assertTrue(torch.isfinite(inputs[0][0].grad).all())
        for branch_name in ("temporal", "speaker"):
            conv = getattr(model, "graph_net_{}".format(branch_name)).conv2
            self.assertIsNotNone(conv.compressor.weight.grad)
            self.assertTrue(torch.isfinite(conv.compressor.weight.grad).all())

    def test_genagg_auxiliary_is_exact_branch_sum_and_reaches_both_maps(self):
        torch.manual_seed(127)
        model = self._model("genagg")
        outputs = model(*self._inputs())
        temporal = model.graph_net_temporal.conv2.inverse_consistency_loss()
        speaker = model.graph_net_speaker.conv2.inverse_consistency_loss()
        auxiliary = model.second_graph_auxiliary_loss()
        self.assertTrue(torch.equal(auxiliary, temporal + speaker))

        (self._loss(outputs) + auxiliary).backward()
        for branch_name in ("temporal", "speaker"):
            conv = getattr(model, "graph_net_{}".format(branch_name)).conv2
            for name, parameter in conv.named_parameters():
                if name.startswith(("forward_map", "inverse_map")) or name in {
                    "alpha",
                    "beta",
                }:
                    self.assertIsNotNone(parameter.grad, "{}.{}".format(branch_name, name))
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
