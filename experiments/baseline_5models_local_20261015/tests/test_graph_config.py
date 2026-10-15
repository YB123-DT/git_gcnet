"""CPU regressions for unified and official graph configuration names."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.graph_models import build_model


class GraphConfigTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def build(self, model='SDR-GNN', **options):
        return build_model(dict(model=model, hidden=4, **options), (2, 3, 1), 4, 2)

    def assert_reconstruction_weight(self, model, weight):
        outputs = dict(logits=torch.zeros(2, 1, 4),
                       reconstruction=torch.ones(2, 1, 6))
        availability = torch.tensor([[[0., 1., 1.]], [[0., 1., 1.]]])
        total, parts = model.loss(
            outputs, torch.zeros(1, 2, dtype=torch.long), torch.zeros(2, 1, 6),
            availability, torch.ones(1, 2, dtype=torch.bool))
        expected = ((1 - weight) * parts['classification']
                    + weight * parts['reconstruction'])
        torch.testing.assert_close(total, expected)

    def test_unified_keys_change_actual_dropout_and_loss(self):
        model = self.build(drop_rate=.1, recon_weight=.2)
        self.assertEqual(model.model.gru.dropout, .1)
        self.assertEqual(model.model.graph_net_temporal.grufusion.dropout, .1)
        self.assertEqual(model.model.multihead_attn.dropout, .1)
        self.assert_reconstruction_weight(model, .2)
        gcnet = self.build('GCNet', drop_rate=.1)
        self.assertEqual(gcnet.model.lstm.dropout, .1)

    def test_official_keys_remain_supported(self):
        model = self.build(dropout=.1, recon_loss_weight=.2)
        self.assertEqual(model.model.gru.dropout, .1)
        self.assert_reconstruction_weight(model, .2)

    def test_matching_aliases_are_accepted(self):
        model = self.build(drop_rate=.1, dropout=.1,
                           recon_weight=.2, recon_loss_weight=.2)
        self.assertEqual(model.model.gru.dropout, .1)
        self.assert_reconstruction_weight(model, .2)

    def test_conflicting_dropout_names_are_rejected(self):
        for name in ('GCNet', 'SDR-GNN'):
            with self.subTest(model=name):
                with self.assertRaisesRegex(ValueError, 'drop_rate.*dropout'):
                    self.build(name, drop_rate=.1, dropout=.5)

    def test_conflicting_reconstruction_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'recon_weight.*recon_loss_weight'):
            self.build(recon_weight=.2, recon_loss_weight=.5)

    def test_official_default_values_are_preserved(self):
        model = self.build()
        self.assertEqual(model.model.gru.dropout, .5)
        self.assert_reconstruction_weight(model, .5)


if __name__ == '__main__':
    unittest.main()
