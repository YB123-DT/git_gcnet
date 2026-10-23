import importlib.util
from pathlib import Path
import unittest


class CmercSentimentTests(unittest.TestCase):
    def module(self, suffix):
        name = 'experiments.osram_cmerc_sentiment_3seed_20261023.' + suffix
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, 'The approved five-job environment queue is missing')
        return __import__(name, fromlist=['main'])

    def test_plan_reuses_mosi66_and_launches_only_five_jobs(self):
        runner = self.module('run')
        jobs = runner.plan_jobs(Path('/fresh'), 'sha', 'GPU-good')
        self.assertEqual({(j['dataset'], j['seed']) for j in jobs}, {
            ('CMUMOSI', 67), ('CMUMOSI', 68), ('CMUMOSEI', 66),
            ('CMUMOSEI', 67), ('CMUMOSEI', 68)})
        self.assertEqual(len(jobs), 5)
        self.assertEqual(len({j['output'] for j in jobs}), 5)
        for job in jobs:
            self.assertIn('cmerc-repro/bin/python', job['command'][0])

    def test_learning_rate_override_only_applies_to_mosei(self):
        worker = self.module('worker')
        self.assertEqual(worker.learning_rate('CMUMOSI', 99), 3e-4)
        self.assertEqual(worker.learning_rate('CMUMOSI', 100), 3e-4 / 10)
        self.assertEqual(worker.learning_rate('CMUMOSEI', 0), 1e-4)
        self.assertEqual(worker.learning_rate('CMUMOSEI', 59), 1e-4)
        with self.assertRaises(ValueError):
            worker.learning_rate('CMUMOSEI', 60)

    def test_disabled_new_schema_fields_cannot_hide_enabled_modules(self):
        worker = self.module('worker')
        expected = {'seed': 66, 'learning_rate': 1e-4}
        worker.validate_reference_config(dict(expected, osram_source_role_v3=False), expected)
        with self.assertRaises(ValueError):
            worker.validate_reference_config(dict(expected, osram_source_role_v3=True), expected)
        with self.assertRaises(ValueError):
            worker.validate_reference_config(dict(expected, unknown_module=False), expected)


if __name__ == '__main__':
    unittest.main()
