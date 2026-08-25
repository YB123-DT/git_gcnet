import unittest

import torch
from torch_geometric.nn import GraphConv, RGCNConv

from cp_lecc_rgcn import CompletePreservingLowRankECCConv
from model import GraphModel
from mpfilm_rgcn import MissingPatternFiLMRGCNConv
from sequence_aff import MaskConditionedSequenceAFF


class GraphModelMPFiLMIntegrationTests(unittest.TestCase):
    def _model(self, variant, branch_fusion="addition"):
        return GraphModel(
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
            graph_conv_variant=variant,
            branch_fusion=branch_fusion,
        )

    def _forward_arguments(self, modality_mask=None):
        sequence_length, batch_size = 3, 2
        inputs = [torch.randn(sequence_length, batch_size, 6)]
        if modality_mask is None:
            modality_mask = torch.ones(sequence_length, batch_size, 3)
        qmask = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        umask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        return inputs, modality_mask, qmask, umask, [3, 2]

    def test_branch_fusion_module_is_always_present_and_choices_are_validated(self):
        for mode in ("addition", "mask_sequence_aff"):
            with self.subTest(mode=mode):
                model = self._model("original", branch_fusion=mode)
                self.assertEqual(model.branch_fusion, mode)
                self.assertIsInstance(
                    model.branch_fusion_module, MaskConditionedSequenceAFF
                )

        with self.assertRaisesRegex(ValueError, "branch_fusion"):
            self._model("original", branch_fusion="attention")

    def test_default_addition_preserves_explicit_addition_parameters_and_rng(self):
        seed = 131
        torch.manual_seed(seed)
        default = GraphModel(
            base_model="LSTM", adim=2, tdim=2, vdim=2, D_e=4,
            graph_hidden_size=2, n_speakers=2, window_past=1,
            window_future=1, n_classes=6, dropout=0.0, time_attn=False,
            no_cuda=True, graph_conv_variant="original",
        )
        default_rng = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        explicit = self._model("original", branch_fusion="addition")

        self.assertTrue(torch.equal(default_rng, torch.get_rng_state()))
        for name, expected in default.named_parameters():
            self.assertTrue(
                torch.equal(expected, dict(explicit.named_parameters())[name]), name
            )

        default.eval()
        explicit.eval()
        arguments = self._forward_arguments()
        expected = default(*arguments)
        actual = explicit(*arguments)
        for expected_tensor, actual_tensor in zip(
            (expected[0], expected[1][0], expected[2]),
            (actual[0], actual[1][0], actual[2]),
        ):
            self.assertTrue(torch.equal(expected_tensor, actual_tensor))

    def test_all_atv_mask_sequence_aff_matches_addition_forward_and_gradients(self):
        torch.manual_seed(137)
        addition = self._model("original", branch_fusion="addition")
        torch.manual_seed(137)
        candidate = self._model("original", branch_fusion="mask_sequence_aff")
        candidate.load_state_dict(addition.state_dict())
        addition.eval()
        candidate.eval()
        addition_arguments = self._forward_arguments()
        candidate_arguments = (
            [addition_arguments[0][0].detach().clone().requires_grad_()],
            addition_arguments[1], addition_arguments[2], addition_arguments[3],
            addition_arguments[4],
        )
        addition_arguments = (
            [addition_arguments[0][0].detach().clone().requires_grad_()],
            addition_arguments[1], addition_arguments[2], addition_arguments[3],
            addition_arguments[4],
        )

        expected = addition(*addition_arguments)
        actual = candidate(*candidate_arguments)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1][0], expected[1][0]))
        self.assertTrue(torch.equal(actual[2], expected[2]))
        expected[0].sum().backward()
        actual[0].sum().backward()
        self.assertTrue(torch.equal(
            addition_arguments[0][0].grad, candidate_arguments[0][0].grad
        ))
        candidate_parameters = dict(candidate.named_parameters())
        for name, parameter in addition.named_parameters():
            if not name.startswith("branch_fusion_module."):
                candidate_gradient = candidate_parameters[name].grad
                if parameter.grad is None:
                    self.assertIsNone(candidate_gradient, name)
                else:
                    self.assertTrue(
                        torch.equal(parameter.grad, candidate_gradient), name
                    )

    def test_mask_sequence_aff_is_addition_at_init_then_changes_incomplete_output(self):
        torch.manual_seed(139)
        addition = self._model("original", branch_fusion="addition")
        torch.manual_seed(139)
        candidate = self._model("original", branch_fusion="mask_sequence_aff")
        candidate.load_state_dict(addition.state_dict())
        addition.eval()
        candidate.eval()
        modality_mask = torch.ones(3, 2, 3)
        modality_mask[0, 0] = torch.tensor([1, 0, 0])
        arguments = self._forward_arguments(modality_mask)

        expected = addition(*arguments)
        initialized = candidate(*arguments)
        self.assertTrue(torch.equal(initialized[0], expected[0]))
        self.assertTrue(torch.equal(initialized[1][0], expected[1][0]))
        self.assertTrue(torch.equal(initialized[2], expected[2]))

        with torch.no_grad():
            candidate.branch_fusion_module.local_context[-1].bias.fill_(1.0)
        changed = candidate(*arguments)
        self.assertFalse(torch.equal(changed[2][0, 0], expected[2][0, 0]))

    def test_original_keeps_both_pyg_rgcn_layers(self):
        model = self._model("original")

        self.assertIsInstance(model.graph_net_temporal.conv1, RGCNConv)
        self.assertIsInstance(model.graph_net_speaker.conv1, RGCNConv)

    def test_full_replaces_both_first_layers_only(self):
        model = self._model("full")

        self.assertIsInstance(
            model.graph_net_temporal.conv1, MissingPatternFiLMRGCNConv
        )
        self.assertIsInstance(
            model.graph_net_speaker.conv1, MissingPatternFiLMRGCNConv
        )
        self.assertEqual(model.graph_net_temporal.conv1.num_relations, 3)
        self.assertEqual(model.graph_net_speaker.conv1.num_relations, 4)

    def test_faithful_edgewise_replaces_both_first_layers(self):
        model = self._model("faithful_edgewise")

        self.assertEqual(
            model.graph_net_temporal.conv1.variant, "faithful_edgewise"
        )
        self.assertEqual(
            model.graph_net_speaker.conv1.variant, "faithful_edgewise"
        )

    def test_cp_lecc_replaces_both_first_layers_only(self):
        model = self._model("cp_lecc")

        self.assertIsInstance(
            model.graph_net_temporal.conv1, CompletePreservingLowRankECCConv
        )
        self.assertIsInstance(
            model.graph_net_speaker.conv1, CompletePreservingLowRankECCConv
        )
        self.assertEqual(model.graph_net_temporal.conv1.num_relations, 3)
        self.assertEqual(model.graph_net_speaker.conv1.num_relations, 4)
        self.assertIsInstance(model.graph_net_temporal.conv2, GraphConv)
        self.assertIsInstance(model.graph_net_speaker.conv2, GraphConv)

    def test_cp_lecc_locked_model_parameter_count(self):
        dimensions = dict(
            base_model="LSTM",
            adim=1024,
            tdim=1024,
            vdim=512,
            D_e=200,
            graph_hidden_size=100,
            n_speakers=2,
            window_past=2,
            window_future=2,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
        )
        original = GraphModel(**dimensions, graph_conv_variant="original")
        candidate = GraphModel(**dimensions, graph_conv_variant="cp_lecc")
        aff = GraphModel(
            **dimensions,
            graph_conv_variant="original",
            branch_fusion="mask_sequence_aff",
        )
        cp_aff = GraphModel(
            **dimensions,
            graph_conv_variant="cp_lecc",
            branch_fusion="mask_sequence_aff",
        )

        original_count = sum(parameter.numel() for parameter in original.parameters())
        candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
        self.assertEqual(original_count, 36_419_816)
        self.assertEqual(candidate_count, 36_480_488)
        self.assertEqual(candidate_count - original_count, 60_672)
        self.assertEqual(original.selected_path_parameter_count(), 34_140_166)
        self.assertEqual(candidate.selected_path_parameter_count(), 34_200_838)
        self.assertEqual(
            sum(parameter.numel() for parameter in aff.parameters()), 36_419_816
        )
        self.assertEqual(aff.selected_path_parameter_count(), 34_393_416)
        self.assertEqual(
            sum(parameter.numel() for parameter in cp_aff.parameters()), 36_480_488
        )
        self.assertEqual(cp_aff.selected_path_parameter_count(), 34_454_088)
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in aff.branch_fusion_module.parameters()
            ),
            253_250,
        )
        self.assertEqual(
            candidate.selected_path_parameter_count()
            - original.selected_path_parameter_count(),
            60_672,
        )

    def test_cp_lecc_construction_preserves_every_common_parameter(self):
        seed = 83
        torch.manual_seed(seed)
        original = self._model("original")
        torch.manual_seed(seed)
        candidate = self._model("cp_lecc")
        original_parameters = dict(original.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())

        for name, expected in original_parameters.items():
            self.assertIn(name, candidate_parameters)
            actual = candidate_parameters[name]
            self.assertEqual(expected.shape, actual.shape, name)
            self.assertTrue(torch.equal(expected, actual), name)

        dynamic_names = {
            "target_content",
            "source_content",
            "relation_embedding",
            "generator_hidden_weight",
            "generator_hidden_bias",
            "generator_output_weight",
            "generator_output_bias",
            "basis_left",
            "basis_right",
        }
        extra_names = set(candidate_parameters) - set(original_parameters)
        self.assertEqual(
            extra_names,
            {
                f"graph_net_{graph}.conv1.{name}"
                for graph in ("temporal", "speaker")
                for name in dynamic_names
            },
        )

    def test_cp_lecc_all_atv_forward_is_bitwise_original_fast_path(self):
        seed = 97
        torch.manual_seed(seed)
        original = self._model("original")
        torch.manual_seed(seed)
        candidate = self._model("cp_lecc")
        with torch.no_grad():
            for graph_network in (
                candidate.graph_net_temporal,
                candidate.graph_net_speaker,
            ):
                for name, parameter in graph_network.conv1.named_parameters():
                    if name not in {"weight", "root", "bias"}:
                        parameter.fill_(0.25)
                        self.assertGreater(torch.count_nonzero(parameter).item(), 0)

        original.eval()
        candidate.eval()
        sequence_length, batch_size = 3, 2
        inputs = [torch.randn(sequence_length, batch_size, 6)]
        modality_mask = torch.ones(sequence_length, batch_size, 3)
        qmask = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        umask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        arguments = (inputs, modality_mask, qmask, umask, [3, 2])

        expected = original(*arguments)
        actual = candidate(*arguments)

        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1][0], expected[1][0]))
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_forward_accepts_selected_modality_mask(self):
        torch.manual_seed(3)
        model = self._model("full")
        sequence_length, batch_size = 3, 2
        inputs = [torch.randn(sequence_length, batch_size, 6)]
        modality_mask = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0]],
                [[1, 1, 0], [0, 0, 1]],
                [[1, 1, 1], [1, 0, 1]],
            ],
            dtype=torch.float32,
        )
        qmask = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        umask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)

        log_prob, reconstruction, hidden = model(
            inputs, modality_mask, qmask, umask, [3, 2]
        )

        self.assertEqual(tuple(log_prob.shape), (3, 2, 6))
        self.assertEqual(tuple(reconstruction[0].shape), (3, 2, 6))
        self.assertEqual(tuple(hidden.shape[:2]), (3, 2))


if __name__ == "__main__":
    unittest.main()
