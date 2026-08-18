from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_missing_sweep import build_jobs
from scripts.summarize_missing_sweep import classification_metrics_from_saved_fold


class ExperimentIsolationTest(unittest.TestCase):
    def test_recomputes_weighted_macro_and_accuracy_from_saved_predictions(self) -> None:
        saved_fold = {
            "test_labels": [np.array([0, 1, 1])],
            "test_preds": [
                np.array(
                    [
                        [4.0, 0.0],
                        [0.0, 4.0],
                        [4.0, 0.0],
                    ]
                )
            ],
        }

        metrics = classification_metrics_from_saved_fold(saved_fold)

        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["weighted_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["unweighted_accuracy"], 0.75)

    def test_builds_sixteen_isolated_jobs_on_two_plus_two_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            jobs = build_jobs(
                root=Path(temporary_directory),
                python=Path("/env/bin/python"),
                original_gpus=(1, 2),
                jepa_gpus=(3, 4),
                seed=66,
            )

        self.assertEqual(len(jobs), 16)
        self.assertEqual({job.gpu for job in jobs if job.method == "original"}, {1, 2})
        self.assertEqual({job.gpu for job in jobs if job.method == "jepa"}, {3, 4})
        self.assertEqual(len({job.output_dir for job in jobs}), 16)
        self.assertTrue(all(job.seed == 66 for job in jobs))
        self.assertTrue(all("constant-" in " ".join(job.command) for job in jobs))


if __name__ == "__main__":
    unittest.main()
