import unittest

import torch
from torch_geometric.nn import RGCNConv

from model import GraphModel
from mpfilm_rgcn import MissingPatternFiLMRGCNConv


class GraphModelMPFiLMIntegrationTests(unittest.TestCase):
    def _model(self, variant):
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
        )

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
