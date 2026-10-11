import unittest
from types import SimpleNamespace

import torch

from gcnet_missing_m3.fwc_sampling import CreditPair
from gcnet_missing_m3.fwc_evaluation import evaluate_write_utility


class TinyModel(torch.nn.Module):
    def __init__(self, classification=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))
        self.osram = SimpleNamespace(last_diagnostics={'before': True})
        self.classification = classification
        self.calls = 0

    def forward(self, inputs, availability, qmask, umask, lengths, predict_missing=False, write_mask_override=None):
        self.calls += 1
        write = availability if write_mask_override is None else write_mask_override
        evidence = write[:, :, 0].float()
        history = evidence.cumsum(0) - evidence
        score = self.scale * (history - .5)
        logits = torch.stack((-score, score), -1) if self.classification else score[..., None]
        return logits, None


class FWCEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.view = dict(incomplete=torch.zeros(2, 2, 3),
                         availability=torch.tensor([[[1, 1, 0], [1, 1, 0]], [[0, 1, 0], [0, 1, 0]]]),
                         qmask=torch.zeros(2, 2, 1), umask=torch.ones(2, 2), lengths=[2, 2],
                         conversation_ids=['a', 'b'], labels=torch.tensor([[1., 1.], [-1., -1.]]))
        self.pairs = [CreditPair(b, c, 0, 0, 1, 0) for b, c in enumerate(['a', 'b'])]

    def test_regression_groups_and_no_gradient(self):
        model = TinyModel().eval()
        before = torch.random.get_rng_state().clone()
        report = evaluate_write_utility(model, self.view, self.pairs)
        self.assertEqual(model.calls, 2)
        self.assertIsNone(model.scale.grad)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        overall = report['summary']['overall']
        self.assertEqual(overall['pair_count'], 2)
        self.assertEqual(overall['corrections'], 1)
        self.assertEqual(overall['harms'], 1)
        self.assertEqual(overall['utility_positive_count'], 1)
        self.assertEqual(overall['utility_negative_count'], 1)
        self.assertAlmostEqual(overall['utility_mean'], 0.)
        self.assertEqual(report['summary']['by_source_modality']['A']['pair_count'], 2)
        self.assertEqual(report['summary']['by_lag']['1']['pair_count'], 2)

    def test_same_target_multiple_conditions_not_unique_samples(self):
        model = TinyModel().eval()
        pairs = self.pairs + [CreditPair(0, 'a', 0, 0, 1, 2)]
        report = evaluate_write_utility(model, self.view, pairs)
        self.assertEqual(model.calls, 3)
        self.assertEqual(report['summary']['overall']['pair_count'], 3)
        self.assertEqual(report['summary']['unique_target_count'], 2)
        self.assertEqual(report['summary']['repeated_target_pair_count'], 1)

    def test_neutral_excluded_and_classification(self):
        view = dict(self.view, labels=torch.zeros(2, 2))
        report = evaluate_write_utility(TinyModel().eval(), view, self.pairs)
        self.assertEqual(report['summary']['overall']['polarity_eligible_count'], 0)
        self.assertIsNone(report['rows'][0]['keep_correct'])
        view['labels'] = torch.tensor([[1, 1], [0, 0]])
        report = evaluate_write_utility(TinyModel(True).eval(), view, self.pairs, dataset='IEMOCAPFour')
        self.assertEqual(report['summary']['overall']['corrections'], 1)
        self.assertEqual(report['summary']['overall']['harms'], 1)
        self.assertTrue(all(x['loss_keep'] >= 0 for x in report['rows']))

    def test_empty_and_training_guard(self):
        model = TinyModel().eval()
        report = evaluate_write_utility(model, self.view, [])
        self.assertEqual(model.calls, 0)
        self.assertEqual(report['summary']['overall']['pair_count'], 0)
        with self.assertRaises(ValueError):
            evaluate_write_utility(model.train(), self.view, self.pairs)


if __name__ == '__main__':
    unittest.main()
