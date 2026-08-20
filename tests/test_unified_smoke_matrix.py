from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.run_unified_protocol_smoke import build_smoke_jobs


class UnifiedSmokeMatrixTest(unittest.TestCase):
    def test_matrix_has_four_paired_datasets_and_safe_gpu_density(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = build_smoke_jobs(
                Path(directory), "/official/python", gpus=(0, 1, 2)
            )

        self.assertEqual(len(jobs), 8)
        self.assertEqual(
            {(job.dataset, job.method) for job in jobs},
            {
                (dataset, method)
                for dataset in (
                    "IEMOCAPFour", "IEMOCAPSix", "CMUMOSI", "CMUMOSEI"
                )
                for method in ("baseline", "jepa")
            },
        )
        counts = Counter(job.gpu for job in jobs)
        self.assertNotIn(4, counts)
        self.assertLessEqual(max(counts.values()), 3)
        for job in jobs:
            self.assertIn("--allow-short-run", job.command)
            self.assertEqual(job.command[job.command.index("--epochs") + 1], "2")
            self.assertIn("--stability-recon-weight", job.command)
            if job.dataset.startswith("IEMOCAP"):
                self.assertIn("--fold", job.command)
            else:
                self.assertNotIn("--fold", job.command)


if __name__ == "__main__":
    unittest.main()
