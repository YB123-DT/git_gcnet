import importlib.util
import unittest


class EnvironmentControlTests(unittest.TestCase):
    def worker(self):
        name = 'experiments.osram_mosi_cmerc_20261023.worker'
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, 'The isolated environment-control runner is missing')
        return __import__(name, fromlist=['validate_environment'])

    def test_accepts_only_the_requested_environment(self):
        worker = self.worker()
        good = dict(torch='1.13.1+cu117', numpy='1.26.4', sklearn='1.3.2')
        worker.validate_environment(good)
        with self.assertRaises(ValueError):
            worker.validate_environment(dict(good, torch='2.2.2+cu121'))

    def test_training_parity_does_not_require_identical_cross_environment_loss(self):
        worker = self.worker()
        row = dict(epoch=1, mask_trace_sha256='abc', optimizer_steps=41,
                   learning_rates=[3e-4], task_MSE=1.0)
        worker.check_training_parity(dict(row, task_MSE=2.0), row)
        with self.assertRaises(ValueError):
            worker.check_training_parity(dict(row, mask_trace_sha256='wrong'), row)

    def test_baseline_source_changes_are_rejected(self):
        worker = self.worker()
        worker.validate_source_hashes({'model': 'original'}, {'model': 'original'})
        with self.assertRaises(ValueError):
            worker.validate_source_hashes({'model': 'changed'}, {'model': 'original'})

    def test_final_lr_uses_the_references_exact_division(self):
        self.assertEqual(self.worker().FINAL_LR, 3e-4 / 10)


if __name__ == '__main__':
    unittest.main()
