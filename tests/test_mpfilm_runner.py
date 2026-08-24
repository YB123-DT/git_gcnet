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

    def test_custom_ablation_grid_preserves_requested_labels(self):
        jobs = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("pattern_only", "parameter_matched"),
            rates=(0.3, 0.5, 0.7),
            seeds=(66, 67, 68, 69, 70),
        )

        self.assertEqual(len(jobs), 30)
        self.assertEqual(
            {job.arm for job in jobs},
            {"pattern_only", "parameter_matched"},
        )

    def test_parameter_matched_label_maps_to_content_control(self):
        job = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("parameter_matched",),
            rates=(0.5,),
            seeds=(66,),
        )[0]
        command = build_command(
            job,
            python=Path("/env/bin/python"),
            repository=Path("/repo"),
            data_root=Path("/data/IEMOCAP"),
            mask_bank_root=Path("/tmp/banks"),
        )

        variant_index = command.index("--graph-conv-variant") + 1
        self.assertEqual(command[variant_index], "content_film_control")

    def test_cp_lecc_label_maps_to_cp_lecc_graph_variant(self):
        job = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("cp_lecc",),
            rates=(0.5,),
            seeds=(66,),
        )[0]
        command = build_command(
            job,
            python=Path("/env/bin/python"),
            repository=Path("/repo"),
            data_root=Path("/data/IEMOCAP"),
            mask_bank_root=Path("/tmp/banks"),
        )

        self.assertEqual(
            command[command.index("--graph-conv-variant") + 1], "cp_lecc"
        )

    def test_cp_lecc_gate_grid_has_complete_and_ten_paired_jobs(self):
        complete = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("cp_lecc",),
            rates=(0.0,),
            seeds=(66,),
        )
        paired = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("cp_lecc",),
            rates=(0.5, 0.7),
            seeds=(66, 67, 68, 69, 70),
        )

        jobs = complete + paired
        self.assertEqual(len(jobs), 11)
        self.assertEqual(
            [(job.missing_rate, job.seed) for job in jobs],
            [(0.0, 66)]
            + [(rate, seed) for rate in (0.5, 0.7) for seed in range(66, 71)],
        )

    def test_film_ab_labels_map_to_the_only_changed_operation(self):
        jobs = build_jobs(
            "formal",
            Path("/tmp/results"),
            arms=("linearized_film", "faithful_edgewise_film"),
            rates=(0.5,),
            seeds=(66,),
        )
        variants = []
        for job in jobs:
            command = build_command(
                job,
                python=Path("/env/bin/python"),
                repository=Path("/repo"),
                data_root=Path("/data/IEMOCAP"),
                mask_bank_root=Path("/tmp/banks"),
            )
            variants.append(command[command.index("--graph-conv-variant") + 1])

        self.assertEqual(variants, ["full", "faithful_edgewise"])

    def test_child_environment_requires_deterministic_cublas(self):
        source = Path(
            "experiments/mpfilm_iemocap6/run_locked_ab.py"
        ).read_text(encoding="utf-8")

        self.assertIn('CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"', source)
        self.assertIn('PYTHONHASHSEED"] = "0"', source)

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
