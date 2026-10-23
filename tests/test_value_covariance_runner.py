"""CPU-only protocol/report checks; no remote files or GPU required."""
import unittest

from experiments.osram_value_covariance_20261023.report import aggregate_rows, binary_metrics
from experiments.osram_value_covariance_20261023.run import validate_scope


class ValueCovarianceRunnerTests(unittest.TestCase):
    def test_locked_scope(self):
        validate_scope('CMUMOSI', ('.3', '.7'), 16, (.001, .01))
        with self.assertRaises(ValueError):
            validate_scope('MELD', ('.3', '.7'), 16, (.001, .01))
        with self.assertRaises(ValueError):
            validate_scope('CMUMOSEI', ('.3',), 16, (.001, .01))
        with self.assertRaises(ValueError):
            validate_scope('CMUMOSI', ('.3', '.7'), 1, (.001, .01))

    def test_zero_labels_not_binary_denominator(self):
        result = binary_metrics([1, -1, 0], [1, 1, -1])
        self.assertEqual(result['N_binary'], 2)
        self.assertEqual(result['ACC'], 50.)
        self.assertAlmostEqual(result['W_F1'], 100 / 3)

    def test_groups_use_matching_draws_and_samples(self):
        rows = [dict(dataset='CMUMOSI', rate='0.3', pattern='AT', sigma=.01,
                     label=y, clean_prediction=p, noisy_predictions=q,
                     clean_correct=(y != 0 and (p > 0) == (y > 0)),
                     history_count=2, kappa_mean=.2, read_noise_rms=.1,
                     task_variance=.04, value_rms=.5,
                     head_kappa=[.2, .3], head_task_variance=[.01, .03],
                     head_prediction_shift=[.001, .002])
                for y, p, q in [(1., -1., [1., -1.]), (-1., -1., [1., -1.])]]
        summary = aggregate_rows(rows)
        overall = next(r for r in summary['groups'] if r['group'] == 'overall')
        self.assertEqual(overall['N'], 2)
        self.assertEqual(overall['draws'], 2)
        self.assertEqual(overall['corrections_mean'], .5)
        self.assertEqual(overall['harms_mean'], .5)
        self.assertEqual(overall['clean']['ACC'], 50.)
        self.assertEqual(overall['N_clean_correct'], 1)
        self.assertEqual(overall['N_clean_wrong'], 1)
        self.assertEqual(overall['correction_rate_percent'], 50.)
        self.assertEqual(overall['harm_rate_percent'], 50.)
        self.assertEqual(len(summary['heads']), 2)


if __name__ == '__main__':
    unittest.main()
