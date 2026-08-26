import json
import statistics
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "iemocap6" / "fold5"


class OrganizedLayoutTest(unittest.TestCase):
    def test_version_directories_are_self_describing(self):
        required = {"README.md", "config.json", "variant.py", "__init__.py"}
        for name in (
            "original",
            "mpfilm",
            "cp_lecc",
            "sequence_aff",
            "full_fused_reconstruction",
        ):
            actual = {
                path.name
                for path in (ROOT / "versions" / name).iterdir()
                if path.is_file()
            }
            self.assertTrue(required.issubset(actual), (name, actual))

    def test_only_completed_versions_are_published(self):
        version_root = ROOT / "versions"
        self.assertTrue(version_root.is_dir())
        names = {
            path.name
            for path in version_root.iterdir()
            if path.is_dir() and not path.name.startswith("__")
        }
        self.assertEqual(
            names,
            {
                "original",
                "mpfilm",
                "cp_lecc",
                "sequence_aff",
                "full_fused_reconstruction",
            },
        )

    def test_unfinished_surfaces_are_absent(self):
        forbidden = {
            "remasker",
            "genagg",
            "medoid",
            "g1u",
            "g1s",
            "dilation",
            "bilstm_ablation",
            "pconv",
        }
        paths = []
        for base in (ROOT / "versions", ROOT / "results"):
            self.assertTrue(base.is_dir())
            paths.extend(
                str(path.relative_to(ROOT)).lower()
                for path in base.rglob("*")
            )
        published = "\n".join(paths)
        for token in forbidden:
            self.assertNotIn(token, published)

    def test_result_manifests_declare_fixed_fold_five(self):
        for name in ("original", "mpfilm", "cp_lecc", "sequence_aff"):
            path = RESULTS / name / "provenance.json"
            self.assertTrue(path.is_file(), str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"], "IEMOCAPSix")
            self.assertEqual(payload["fold"], 5)
            self.assertEqual(payload["seeds"], [66, 67, 68, 69, 70])
            self.assertNotEqual(
                payload["protocol"], "five_fold_cross_validation"
            )

    def test_original_compact_summary_matches_task_level_source(self):
        paired = json.loads(
            (RESULTS / "sequence_aff" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        compact = json.loads(
            (RESULTS / "original" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(paired["tasks"]), compact["task_count"])
        for rate, expected in compact["rates"].items():
            values = [
                row["original"]["weighted_f1"]
                for row in paired["tasks"]
                if str(row["rate"]) == rate
            ]
            self.assertEqual(len(values), 5)
            self.assertAlmostEqual(statistics.mean(values), expected, places=15)
        self.assertAlmostEqual(
            paired["macro"]["original_mean"],
            compact["seed_macro_mean"],
            places=15,
        )

    def test_tracked_tree_excludes_legacy_bulk_and_large_artifacts(self):
        if not (ROOT / ".git").exists():
            self.skipTest("source export has no Git index")
        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=str(ROOT), text=True
        ).splitlines()
        forbidden_roots = (
            "baseline-",
            "dataset/",
            "experiments/",
            "feature_extraction/",
        )
        forbidden_suffixes = (".ckpt", ".pt", ".pth")
        for path in tracked:
            self.assertFalse(path.startswith(forbidden_roots), path)
            self.assertFalse(path.endswith(forbidden_suffixes), path)

    def test_original_reference_files_match_official_upstream(self):
        if not (ROOT / ".git").exists():
            self.skipTest("source export has no Git object database")
        names = (
            "dataloader_iemocap.py",
            "graph.py",
            "model.py",
            "train_gcnet.py",
        )
        for name in names:
            expected = subprocess.check_output(
                [
                    "git",
                    "show",
                    "f43d2788481fd5889148f08b688259c9fd712002:gcnet/" + name,
                ],
                cwd=str(ROOT),
            )
            actual = (ROOT / "versions" / "original" / "reference" / name).read_bytes()
            self.assertEqual(actual, expected, name)


if __name__ == "__main__":
    unittest.main()
