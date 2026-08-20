from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import run_official_missing_sweep as sweep


class OfficialSweepTest(unittest.TestCase):
    def test_matrix_contains_exactly_640_unique_official_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = sweep.build_jobs(
                output_root=Path(directory),
                python="/official/python",
                gpus=(0, 1, 2, 3, 5),
                jobs_per_gpu=3,
            )

        self.assertEqual(len(jobs), 640)
        identities = {
            (job.dataset, job.method, job.missing_rate, job.seed)
            for job in jobs
        }
        self.assertEqual(len(identities), 640)
        self.assertEqual({job.seed for job in jobs}, set(range(66, 76)))
        self.assertEqual(
            {job.missing_rate for job in jobs},
            {round(index / 10, 1) for index in range(8)},
        )
        self.assertEqual({job.gpu for job in jobs}, {0, 1, 2, 3, 5})
        self.assertTrue(all(job.slot in (0, 1, 2) for job in jobs))
        self.assertTrue(
            all("--evaluation-protocol" in job.command for job in jobs)
        )
        self.assertTrue(
            all(
                job.command[job.command.index("--evaluation-protocol") + 1]
                == "official"
                for job in jobs
            )
        )
        for job in jobs:
            if job.dataset.startswith("IEMOCAP"):
                self.assertIn("--fold", job.command)
                self.assertEqual(job.command[job.command.index("--fold") + 1], "5")
            else:
                self.assertNotIn("--fold", job.command)

    def test_broken_gpu_four_and_invalid_density_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "GPU 4"):
                sweep.build_jobs(root, "/official/python", (0, 4), 2)
            with self.assertRaisesRegex(ValueError, "jobs_per_gpu"):
                sweep.build_jobs(root, "/official/python", (0,), 0)
            with self.assertRaisesRegex(ValueError, "jobs_per_gpu"):
                sweep.build_jobs(root, "/official/python", (0,), 4)

    def test_completed_job_requires_success_status_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            self.assertFalse(sweep.is_complete(output))

            (output / "status.json").write_text(
                '{"returncode": 0}\n', encoding="utf-8"
            )
            self.assertFalse(sweep.is_complete(output))

            records = output / "run_records" / "abc"
            records.mkdir(parents=True)
            (records / "run_manifest_fold_5.json").write_text(
                "{}\n", encoding="utf-8"
            )
            self.assertTrue(sweep.is_complete(output))


if __name__ == "__main__":
    unittest.main()
