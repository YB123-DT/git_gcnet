import unittest
from pathlib import Path

from experiments.mpfilm_iemocap6.run_locked_ab import build_command, build_jobs


class LockedRunnerTests(unittest.TestCase):
    def test_gate_contains_two_rates_two_seeds_and_two_arms(self):
        jobs = build_jobs("gate", Path("/tmp/results"))

        self.assertEqual(len(jobs), 8)
        self.assertEqual({job.missing_rate for job in jobs}, {0.0, 0.7})
        self.assertEqual({job.seed for job in jobs}, {66, 67})
        self.assertEqual({job.arm for job in jobs}, {"original", "full"})

    def test_formal_contains_eighty_unique_paired_jobs(self):
        jobs = build_jobs("formal", Path("/tmp/results"))
        keys = {(job.arm, job.missing_rate, job.seed) for job in jobs}

        self.assertEqual(len(jobs), 80)
        self.assertEqual(len(keys), 80)

    def test_command_locks_fold_and_omits_smoke_flag(self):
        job = build_jobs("gate", Path("/tmp/results"))[0]
        command = build_command(
            job,
            python=Path("/env/bin/python"),
            repository=Path("/repo"),
            data_root=Path("/data/IEMOCAP"),
            mask_bank_root=Path("/tmp/banks"),
        )

        joined = " ".join(command)
        self.assertIn("--fold-index 5", joined)
        self.assertIn("--epochs 100", joined)
        self.assertIn("--hidden 200", joined)
        self.assertIn("--num-threads 6", joined)
        self.assertNotIn("--allow-short-run", command)


if __name__ == "__main__":
    unittest.main()
