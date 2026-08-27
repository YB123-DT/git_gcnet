import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.plot_plci_results import (
    ScoreRecord,
    build_matrix,
    load_original_records,
    load_plci_records,
    validate_grid,
    weighted_f1_score,
)


class PlotResultCollectionTests(unittest.TestCase):
    def test_weighted_f1_matches_hand_calculation(self):
        labels = np.array([0, 0, 1, 1, 2])
        predictions = np.array([0, 1, 1, 1, 0])
        self.assertAlmostEqual(
            weighted_f1_score(labels, predictions), 0.52, places=12
        )

    def test_loads_plci_json_as_percentage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "miss_0.5" / "seed_66"
            path.mkdir(parents=True)
            (path / "fold_metrics.json").write_text(
                json.dumps([{
                    "fold": 5,
                    "seed": 66,
                    "missing_rate": 0.5,
                    "weighted_f1": 0.63125,
                    "accuracy": 0.64,
                }]),
                encoding="utf-8",
            )
            records = load_plci_records(
                Path(tmp), "IEMOCAPSix", metric="weighted_f1"
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].seed, 66)
            self.assertAlmostEqual(records[0].value, 63.125)

    def test_recomputes_original_multiclass_weighted_f1_from_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = (
                Path(tmp) / "miss_0p5" / "seed_66" / "fold_5" / "saved"
            )
            saved.mkdir(parents=True)
            payload = {
                "test_labels": np.array([0, 0, 1, 1, 2]),
                "test_preds": np.array([
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]),
                "test_fmask": np.ones(5),
            }
            history = np.empty((1, 1), dtype=object)
            history[0, 0] = payload
            np.savez(saved / "result.npz", folder_savewhole=history)
            records = load_original_records(
                Path(tmp), "IEMOCAPSix", metric="weighted_f1"
            )
            self.assertEqual(len(records), 1)
            self.assertAlmostEqual(records[0].value, 52.0)

    def test_recomputes_original_mosi_nonzero_binary_weighted_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = (
                Path(tmp) / "miss_0p2" / "seed_67" / "official_split" / "saved"
            )
            saved.mkdir(parents=True)
            payload = {
                "test_labels": np.array([-1.0, 0.0, 2.0, -0.5]),
                "test_preds": np.array([-0.2, 4.0, 0.1, 0.3]),
                "test_fmask": np.ones(4),
            }
            history = np.empty((1, 1), dtype=object)
            history[0, 0] = payload
            np.savez(saved / "result.npz", folder_savewhole=history)
            records = load_original_records(
                Path(tmp), "CMUMOSI", metric="weighted_f1"
            )
            expected = weighted_f1_score(
                np.array([0, 1, 0]), np.array([0, 1, 1])
            )
            self.assertAlmostEqual(records[0].value, 100.0 * expected)

    def test_build_matrix_and_strict_grid_validation(self):
        records = [
            ScoreRecord("D", "PLCI", 0.0, 66, 60.0, "x"),
            ScoreRecord("D", "PLCI", 0.1, 66, 61.0, "y"),
        ]
        matrix = build_matrix(records, [66], [0.0, 0.1])
        np.testing.assert_allclose(matrix, [[60.0, 61.0]])
        validate_grid(records, [66], [0.0, 0.1], "PLCI")
        with self.assertRaisesRegex(ValueError, "seed=67"):
            validate_grid(records, [66, 67], [0.0, 0.1], "PLCI")


if __name__ == "__main__":
    unittest.main()
