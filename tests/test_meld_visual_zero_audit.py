import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "experiments/osram_meld_visual_zero_audit_20261017/audit.py"


class VisualZeroAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODULE.exists():
            raise AssertionError("visual-zero audit implementation is missing")
        spec = importlib.util.spec_from_file_location("visual_zero_audit", MODULE)
        cls.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.audit)

    def test_join_is_identity_based_and_rejects_label_mismatch(self):
        lookup = {("10", 0): {"label": 1, "speaker": 0, "visual_zero": True},
                  ("2", 0): {"label": 0, "speaker": 1, "visual_zero": False}}
        saved = dict(conversation_ids=np.array(["2", "10"]),
                     utterance_indices=np.array([0, 0]), labels=np.array([0, 1]),
                     speakers=np.array([1, 0]))
        np.testing.assert_array_equal(self.audit.align_zero_flags(saved, lookup), [False, True])
        saved["labels"] = np.array([1, 0])
        with self.assertRaises(ValueError):
            self.audit.align_zero_flags(saved, lookup)

    def test_groups_partition_zero_and_observed_separately(self):
        a = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]])
        groups = self.audit.visual_groups(np.array([True, False, True, False]), a)
        np.testing.assert_array_equal(np.stack(list(groups.values())).sum(0), np.ones(4))
        self.assertEqual([int(x.sum()) for x in groups.values()], [1, 1, 2])

    def test_metrics_distinguish_missing_class_support(self):
        result = self.audit.metrics(np.array([0, 0, 1]), np.array([[3., 0., 0., 0., 0., 0., 0.],
                                                                 [0., 3., 0., 0., 0., 0., 0.],
                                                                 [0., 3., 0., 0., 0., 0., 0.]]))
        self.assertAlmostEqual(result["UA_supported"], 75.)
        self.assertAlmostEqual(result["UA_7_zero_filled"], 150. / 7)
        self.assertEqual(result["support"], [2, 1, 0, 0, 0, 0, 0])
        self.assertIsNone(self.audit.metrics(np.array([], dtype=int), np.empty((0, 7)))["ACC"])


if __name__ == "__main__":
    unittest.main()
