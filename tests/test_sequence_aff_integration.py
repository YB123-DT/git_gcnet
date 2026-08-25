import unittest

import torch

from model import GraphModel
from sequence_aff import MaskConditionedSequenceAFF


class SequenceAFFIntegrationTests(unittest.TestCase):
    def _model(self, branch_fusion="addition"):
        return GraphModel(
            base_model="LSTM", adim=2, tdim=2, vdim=2, D_e=4,
            graph_hidden_size=2, n_speakers=2, window_past=1,
            window_future=1, n_classes=6, dropout=0.0, time_attn=False,
            no_cuda=True, graph_conv_variant="original",
            branch_fusion=branch_fusion,
        )

    def _forward_arguments(self, modality_mask=None):
        if modality_mask is None:
            modality_mask = torch.ones(3, 2, 3)
        return (
            [torch.randn(3, 2, 6)],
            modality_mask,
            torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32),
            torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32),
            [3, 2],
        )

    def test_registered_mode_owns_the_sequence_aff_module(self):
        model = self._model("mask_sequence_aff")

        self.assertEqual(model.branch_fusion, "mask_sequence_aff")
        self.assertIsInstance(model.branch_fusion_module, MaskConditionedSequenceAFF)
        with self.assertRaisesRegex(ValueError, "branch_fusion"):
            self._model("attention")

    def test_default_addition_preserves_parameters_rng_and_output(self):
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
        explicit = self._model("addition")

        self.assertTrue(torch.equal(default_rng, torch.get_rng_state()))
        for name, expected in default.named_parameters():
            self.assertTrue(
                torch.equal(expected, dict(explicit.named_parameters())[name]), name
            )
        default.eval()
        explicit.eval()
        torch.manual_seed(0)
        arguments = self._forward_arguments()
        default_input = arguments[0][0]
        explicit_input = default_input.clone()
        default_args = ([default_input],) + arguments[1:]
        explicit_args = ([explicit_input],) + arguments[1:]
        expected = default(*default_args)
        actual = explicit(*explicit_args)
        for left, right in zip(
            (expected[0], expected[1][0], expected[2]),
            (actual[0], actual[1][0], actual[2]),
        ):
            self.assertTrue(torch.equal(left, right))

    def test_incomplete_mask_uses_aff_after_neutral_initialization(self):
        torch.manual_seed(139)
        addition = self._model("addition")
        torch.manual_seed(139)
        candidate = self._model("mask_sequence_aff")
        candidate.load_state_dict(addition.state_dict())
        addition.eval()
        candidate.eval()
        modality_mask = torch.ones(3, 2, 3)
        modality_mask[0, 0] = torch.tensor([1, 0, 0])
        arguments = self._forward_arguments(modality_mask)

        expected = addition(*arguments)
        initialized = candidate(*arguments)
        self.assertTrue(torch.equal(initialized[2], expected[2]))
        with torch.no_grad():
            candidate.branch_fusion_module.local_context[-1].bias.fill_(1.0)
        changed = candidate(*arguments)
        self.assertFalse(torch.equal(changed[2][0, 0], expected[2][0, 0]))


if __name__ == "__main__":
    unittest.main()
