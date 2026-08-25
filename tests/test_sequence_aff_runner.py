import ast
import unittest
from pathlib import Path

from experiments.mpfilm_iemocap6.run_locked_ab import (
    ARM_TO_BRANCH_FUSION,
    ARM_TO_GRAPH_VARIANT,
    build_command,
    build_jobs,
)


class SequenceAFFRunnerTests(unittest.TestCase):
    def _command(self, arm):
        job = build_jobs(
            "formal",
            Path("/results"),
            arms=(arm,),
            rates=(0.3,),
            seeds=(68,),
        )[0]
        return build_command(
            job,
            Path("/python3.8"),
            Path("/repo"),
            Path("/data"),
            Path("/masks"),
        )

    def test_sequence_aff_command_changes_only_explicit_branch_fusion(self):
        original = self._command("original")
        candidate = self._command("sequence_aff")

        self.assertEqual(ARM_TO_GRAPH_VARIANT["sequence_aff"], "original")
        self.assertEqual(
            candidate[candidate.index("--branch-fusion") + 1],
            "mask_sequence_aff",
        )
        self.assertEqual(
            original[original.index("--branch-fusion") + 1], "addition"
        )
        normalized = list(candidate)
        normalized[normalized.index("--branch-fusion") + 1] = "addition"
        normalized[normalized.index("--output-dir") + 1] = original[
            original.index("--output-dir") + 1
        ]
        self.assertEqual(normalized, original)

    def test_explicit_sequence_aff_grid_has_eighty_unique_paired_jobs(self):
        jobs = build_jobs(
            "formal",
            Path("/results"),
            arms=("original", "sequence_aff"),
            rates=tuple(index / 10 for index in range(8)),
            seeds=(66, 67, 68, 69, 70),
        )
        keys = {(job.arm, job.missing_rate, job.seed) for job in jobs}

        self.assertEqual(len(jobs), 80)
        self.assertEqual(len(keys), 80)
        for rate in tuple(index / 10 for index in range(8)):
            for seed in range(66, 71):
                pair = [
                    job
                    for job in jobs
                    if job.missing_rate == rate and job.seed == seed
                ]
                self.assertEqual([job.arm for job in pair], ["original", "sequence_aff"])

    def test_every_preexisting_arm_passes_addition_explicitly(self):
        for arm in ARM_TO_GRAPH_VARIANT:
            if arm == "sequence_aff":
                continue
            with self.subTest(arm=arm):
                command = self._command(arm)
                self.assertEqual(ARM_TO_BRANCH_FUSION[arm], "addition")
                self.assertEqual(
                    command[command.index("--branch-fusion") + 1], "addition"
                )

    def test_runner_source_parses_with_python38_grammar(self):
        source = Path(
            "experiments/mpfilm_iemocap6/run_locked_ab.py"
        ).read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
