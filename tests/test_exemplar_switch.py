import unittest

import numpy as np
import torch
from torch import nn


class SwitchTests(unittest.TestCase):
    def test_report_decomposition_uses_same_on_selected_weights(self):
        from experiments.osram_exemplar_switch_20261023.report import aggregate
        rates = {f'{i/10:.1f}': dict(ON=dict(ACC=80., WF1=79., MSE=1.),
            OFF=dict(ACC=78., WF1=77., MSE=1.2), Flat=dict(ACC=81., WF1=80., MSE=.9),
            corrections=4, harms=2) for i in range(8)}
        result = aggregate(rates, tuple(rates))
        self.assertEqual(result['direct_WF1'], 2.)
        self.assertEqual(result['main_path_WF1'], -3.)
        self.assertEqual(result['total_WF1'], -1.)
        self.assertEqual(result['direct_WF1'] + result['main_path_WF1'], result['total_WF1'])

    def test_metrics_have_explicit_off_to_on_orientation_and_exclude_neutrals(self):
        from experiments.osram_exemplar_switch_20261023.run import paired_metrics
        result = paired_metrics(np.array([1., -1., 0.]),
                                np.array([.2, .3, -.1]), np.array([-.2, -.3, .1]))
        self.assertEqual(result['N_binary'], 2)
        self.assertEqual(result['corrections'], 1)
        self.assertEqual(result['harms'], 1)
        self.assertEqual(result['ON']['ACC'], 50.)
        self.assertEqual(result['OFF']['ACC'], 50.)
        self.assertAlmostEqual(result['ON']['MSE'], (0.8**2 + 1.3**2 + .1**2)/3)

    def test_alignment_rejects_same_count_different_mask_or_target(self):
        from experiments.osram_exemplar_switch_20261023.run import assert_aligned
        a = dict(labels=np.array([1., -1.]), speakers=np.array([0, 1]),
                 conversation_ids=np.array(['a', 'b']), utterance_indices=np.array([0, 0]),
                 availability=np.array([[1, 0, 0], [0, 1, 0]]))
        assert_aligned(a, a)
        for key in ('labels', 'availability'):
            b = {k: v.copy() for k, v in a.items()}
            b[key].flat[0] += 1
            with self.assertRaises(AssertionError):
                assert_aligned(a, b)

    def test_single_scan_capture_equals_existing_disabled_readout(self):
        from experiments.osram_exemplar_switch_20261023.run import frozen_readouts
        from types import SimpleNamespace
        torch.manual_seed(8)

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.osram = nn.Module()
                self.osram.local_skip = nn.Linear(3, 5)
                self.osram.emotion_adapter = nn.Linear(7, 5)
                self.osram.emotion_norm = nn.LayerNorm(5)
                self.smax_fc = nn.Linear(5, 1)

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()
                self.enabled = True
                self.calls = 0

            def forward(self, view):
                self.calls += 1
                b = self.backbone
                anchor = b.osram.local_skip(view['local']) + b.osram.emotion_adapter(view['flat'])
                if self.enabled:
                    anchor = anchor + view['delta']
                return b.smax_fc(b.osram.emotion_norm(anchor))

        m = Wrapper().eval().requires_grad_(False)
        v = dict(local=torch.randn(4, 2, 3), flat=torch.randn(4, 2, 7),
                 delta=torch.randn(4, 2, 5), umask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
        state = torch.get_rng_state().clone()
        result = frozen_readouts(m, v)
        self.assertEqual(m.calls, 1)
        m.enabled = False
        explicit = m(v)
        valid = v['umask'].T.bool()
        torch.testing.assert_close(result['OFF'][valid], explicit[valid], rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(any(p.grad is not None for p in m.parameters()))
        self.assertEqual(result['OFF'][~valid].count_nonzero().item(), 0)
        self.assertGreater(result['residual_norm'].item(), 0)


if __name__ == '__main__':
    unittest.main()
