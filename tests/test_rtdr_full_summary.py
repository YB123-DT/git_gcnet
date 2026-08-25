import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_rtdr_extension_summary import _extension_invocation
from tests.test_second_graph_aggregation_summary import (
    _fold,
    _write_invocation,
    _write_job,
)


RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SEEDS = (66, 67, 68, 69, 70)


def _full_invocation():
    return {
        "arms": ["rtdr"],
        "fold": 5,
        "gpus": ["5", "6", "7"],
        "job_count": 40,
        "parallel_arms": True,
        "rates": list(RATES),
        "seeds": list(SEEDS),
        "stage": "formal",
        "workers_per_gpu": 3,
    }


def _write_full_grid(root):
    original_formal = root / "historical" / "formal"
    phase_b = root / "phase_b" / "formal"
    for rate in RATES:
        for seed in SEEDS:
            mask = "paired-{}-{}".format(rate, seed)
            _write_job(
                _fold(original_formal, "original", rate, seed),
                "original",
                rate,
                seed,
                historical=True,
                mask_hash=mask,
            )
            _write_job(
                _fold(phase_b, "rtdr", rate, seed),
                "rtdr",
                rate,
                seed,
                mask_hash=mask,
            )
    _write_invocation(phase_b, _extension_invocation())
    _write_invocation(phase_b, _full_invocation())
    return original_formal / "original", phase_b


def _metric_rows(rate_deltas, seed_offsets=(0.01, 0.01, 0.01, -0.01, -0.01)):
    originals, candidates = [], []
    for rate, rate_delta in zip(RATES, rate_deltas):
        for seed, seed_offset in zip(SEEDS, seed_offsets):
            common = {
                "rate": rate,
                "seed": seed,
                "weighted_f1": 0.6,
                "accuracy": 0.6,
                "class_coverage": 6,
                "dominant_ratio": 0.3,
                "epoch": 7,
                "parameter_count": 1,
                "selected_path_parameter_count": 1,
                "runtime_seconds": 1.0,
                "mask_sha256": "m-{}-{}".format(rate, seed),
            }
            originals.append(dict(common, arm="original"))
            candidates.append(
                dict(
                    common,
                    arm="rtdr",
                    weighted_f1=0.6 + rate_delta + seed_offset,
                )
            )
    return originals, candidates


class FullInvocationTests(unittest.TestCase):
    def test_accepts_exact_base_extension_full_and_covers_full_only_cell(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import collect_job

        with tempfile.TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fold = _write_job(_fold(formal, "rtdr", 0.1, 69), "rtdr", 0.1, 69)
            _write_invocation(formal, _extension_invocation())
            _write_invocation(formal, _full_invocation())
            self.assertEqual(collect_job(fold, "rtdr", 0.1, 69)["rate"], 0.1)

    def test_rejects_base_plus_full_without_extension_and_extra_drift(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import collect_job

        for case in ("skip-extension", "extra-drift"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                formal = Path(tmp) / "formal"
                fold = _write_job(
                    _fold(formal, "rtdr", 0.1, 69), "rtdr", 0.1, 69
                )
                if case == "extra-drift":
                    _write_invocation(formal, _extension_invocation())
                _write_invocation(formal, _full_invocation())
                if case == "extra-drift":
                    _write_invocation(formal, dict(_full_invocation(), job_count=39))
                with self.assertRaisesRegex(ValueError, "invocation"):
                    collect_job(fold, "rtdr", 0.1, 69)


class FullCollectionTests(unittest.TestCase):
    def test_collects_40_pairs_after_relocation(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_rtdr_full import (
            collect_full_grid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_full_grid(root / "source")
            relocated = root / "relocated"
            shutil.move(str(root / "source" / "historical"), str(relocated / "historical"))
            shutil.move(str(root / "source" / "phase_b"), str(relocated / "phase_b"))
            rows = collect_full_grid(
                relocated / "historical" / "formal" / "original",
                relocated / "phase_b" / "formal",
            )
        self.assertEqual(set(rows), {"original", "rtdr"})
        self.assertEqual(len(rows["original"]), 40)
        self.assertEqual(len(rows["rtdr"]), 40)

    def test_rejects_a_missing_rate_task(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_rtdr_full import (
            collect_full_grid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_root, phase_b = _write_full_grid(root)
            shutil.rmtree(_fold(phase_b, "rtdr", 0.4, 70))
            with self.assertRaisesRegex(ValueError, "missing"):
                collect_full_grid(original_root, phase_b)


class FullSummaryTests(unittest.TestCase):
    def test_stable_positive_uses_six_of_eight_and_three_of_five(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_rtdr_full import (
            summarize_full,
        )

        originals, candidates = _metric_rows(
            (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.01, -0.01)
        )
        passed = summarize_full({"original": originals, "rtdr": candidates})
        self.assertTrue(passed["stable_positive"])
        self.assertEqual(passed["positive_rate_means"], 6)
        self.assertEqual(passed["positive_seed_macros"], 3)
        self.assertEqual(len(passed["tasks"]), 40)

        originals, candidates = _metric_rows(
            (0.01, 0.01, 0.01, 0.01, 0.01, -0.02, -0.02, -0.02)
        )
        failed = summarize_full({"original": originals, "rtdr": candidates})
        self.assertGreater(failed["overall_macro_delta"], 0.0)
        self.assertEqual(failed["positive_rate_means"], 5)
        self.assertFalse(failed["stable_positive"])

    def test_writes_atomic_json_and_detailed_trilingual_results(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_rtdr_full import (
            summarize_full,
            write_full_outputs,
        )

        originals, candidates = _metric_rows((0.01,) * 8)
        summary = summarize_full({"original": originals, "rtdr": candidates})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_full_outputs(output, summary)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            for name in ("RESULTS.md", "RESULTS.zh.md", "RESULTS.en.md"):
                text = (output / name).read_text(encoding="utf-8")
                self.assertIn("stable_positive", text)
                self.assertIn("0.4", text)
                self.assertIn("70", text)
            self.assertEqual(list(output.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
