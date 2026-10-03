"""The three exits use exactly the original task loss on the same batch."""
import unittest
from unittest.mock import patch

import torch

from gcnet_missing_m3 import train_gcnet as tr
from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
from tests.test_decision_correction import config


class DecisionTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_original_three_loss_average_and_gradient(self):
        self.assertTrue(hasattr(tr, '_decision_correction_loss'))
        for dataset, classes in (('CMUMOSI', 1), ('IEMOCAPSix', 6)):
            cfg = config(dataset=dataset)
            logits = {name: torch.randn(4, 2, classes, requires_grad=True) for name in ('local', 'base', 'full')}
            model = type('Model', (), {'last_decision_outputs': logits})()
            u = torch.tensor([[1., 1, 0, 0], [1., 1, 1, 0]])
            labels = torch.randn(2, 4) if classes == 1 else torch.randint(0, classes, (2, 4))
            view = {'labels': labels, 'umask': u}
            losses = {name: tr._task_loss(dataset, value, labels, u) for name, value in logits.items()}
            result, recorded = tr._decision_correction_loss(model, cfg, view, losses['full'])
            expected = (losses['local'] + losses['base'] + losses['full']) / 3
            self.assertTrue(torch.equal(result, expected))
            self.assertEqual(recorded.keys(), losses.keys())
            expected_grad = torch.autograd.grad(expected, tuple(logits.values()), retain_graph=True)
            result.backward()
            for value, grad in zip(logits.values(), expected_grad):
                self.assertTrue(torch.equal(value.grad, grad))

    def test_epoch_uses_single_forward_and_reports_three_exits(self):
        cfg = config()
        net = _build_model(cfg, (1, 1, 1))
        batch = [torch.randn(6, 2, 1) for _ in range(6)]
        batch += [torch.zeros(2, 6), torch.ones(2, 6), torch.tensor([[1., -1.] * 3, [-1., 1.] * 3]), ['a', 'b']]
        groups, _ = tr._optimizer_parameter_groups(net, cfg)
        optimizer = torch.optim.Adam(groups)
        with patch.object(net.osram, '_scan', wraps=net.osram._scan) as scan:
            result = tr.train_epoch(net, [batch, batch], optimizer, cfg, tr._schedules(cfg, 'train'),
                                    0, (1, 1, 1), torch.device('cpu'))
        self.assertEqual(scan.call_count, 2)
        self.assertEqual(result['model_forward_count'], 2)
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertEqual(result['jepa_loss'], 0)
        d = result['decision_correction']
        self.assertAlmostEqual(result['classification_loss'], sum(d[name + '_task_loss'] for name in ('local', 'base', 'full')) / 3, places=6)
        self.assertEqual(d['valid_tokens'], 24)
        self.assertIn('delta_base_norm', d)
        self.assertIn('delta_gap_norm', d)
        with patch.object(net.osram, '_scan', wraps=net.osram._scan) as scan:
            metrics, _ = tr.evaluate_rate(net, [batch], tr._schedules(cfg, 'test')[.5], cfg.dataset,
                                          (1, 1, 1), torch.device('cpu'), collect=False)
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(metrics['decision_correction']['valid_tokens'], 12)
        self.assertEqual(metrics['loss'], metrics['decision_correction']['full_task_loss'])


if __name__ == '__main__':
    unittest.main()
