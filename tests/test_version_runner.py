import sys
import unittest
from pathlib import Path

from run import build_command
from versions.registry import VERSION_NAMES, resolve


ROOT = Path(__file__).resolve().parents[1]


class VersionRunnerTest(unittest.TestCase):
    def test_registry_contains_only_completed_versions(self):
        self.assertEqual(
            VERSION_NAMES,
            (
                "original",
                "mpfilm",
                "cp_lecc",
                "sequence_aff",
                "full_fused_reconstruction",
            ),
        )

    def test_locked_arguments_for_each_version(self):
        self.assertEqual(resolve("original")["graph_conv_variant"], "original")
        self.assertEqual(resolve("mpfilm")["graph_conv_variant"], "full")
        self.assertEqual(resolve("cp_lecc")["graph_conv_variant"], "cp_lecc")
        self.assertEqual(
            resolve("sequence_aff")["branch_fusion"],
            "mask_sequence_aff",
        )
        self.assertEqual(
            resolve("full_fused_reconstruction")["reconstruction_target"],
            "full_fused",
        )
        for name in VERSION_NAMES[:-1]:
            self.assertEqual(resolve(name)["reconstruction_target"], "missing")

    def test_unknown_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown completed version"):
            resolve("remasker")

    def test_user_cannot_override_locked_method_switch(self):
        with self.assertRaisesRegex(ValueError, "locked argument"):
            build_command("mpfilm", ["--graph-conv-variant", "original"])
        with self.assertRaisesRegex(ValueError, "locked argument"):
            build_command("sequence_aff", ["--branch-fusion=addition"])
        with self.assertRaisesRegex(ValueError, "locked argument"):
            build_command(
                "full_fused_reconstruction",
                ["--reconstruction-target", "missing"],
            )

    def test_build_command_uses_shared_trainer_and_stable_flags(self):
        command = build_command("cp_lecc", ["--dataset", "IEMOCAPSix"])
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(
            Path(command[1]),
            ROOT / "common" / "gcnet" / "train_gcnet.py",
        )
        self.assertEqual(
            command[2:],
            [
                "--branch-fusion",
                "addition",
                "--graph-conv-variant",
                "cp_lecc",
                "--reconstruction-target",
                "missing",
                "--dataset",
                "IEMOCAPSix",
            ],
        )


if __name__ == "__main__":
    unittest.main()
