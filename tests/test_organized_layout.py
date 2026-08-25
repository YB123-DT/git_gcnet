import json
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "iemocap6" / "fold5"


class OrganizedLayoutTest(unittest.TestCase):
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
            {"original", "mpfilm", "cp_lecc", "sequence_aff"},
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


if __name__ == "__main__":
    unittest.main()
