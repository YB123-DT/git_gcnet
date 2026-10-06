import importlib
import unittest


class ClassificationMetricTests(unittest.TestCase):
    def test_ua_is_mean_class_recall_not_accuracy(self):
        mod = importlib.import_module('experiments.osram_nested_cross_dataset_20261006.acc_ua')
        acc, ua, counts = mod.classification_scores([0, 0, 0, 1], [0, 0, 0, 0], classes=2)
        self.assertEqual((acc, ua, counts), (.75, .5, [3, 1]))

    def test_missing_class_and_invalid_prediction_rejected(self):
        mod = importlib.import_module('experiments.osram_nested_cross_dataset_20261006.acc_ua')
        with self.assertRaises(ValueError):
            mod.classification_scores([0], [0], classes=2)
        with self.assertRaises(ValueError):
            mod.classification_scores([0, 1], [0, 2], classes=2)


if __name__ == '__main__':
    unittest.main()
