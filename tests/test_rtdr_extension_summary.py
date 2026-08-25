import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_second_graph_aggregation_summary import (
    _fold,
    _write_invocation,
    _write_job,
)


RATES = (0.0, 0.5, 0.7)
SEEDS = (66, 67, 68, 69, 70)


def _extension_invocation():
    return {
        "arms": ["rtdr"],
        "fold": 5,
        "gpus": ["5", "6", "7"],
        "job_count": 15,
        "parallel_arms": True,
        "rates": [0.0, 0.5, 0.7],
        "seeds": [66, 67, 68, 69, 70],
        "stage": "formal",
        "workers_per_gpu": 3,
    }


def _write_extension_grid(root):
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
    return original_formal / "original", phase_b


def _metric_rows(seed_deltas):
    originals, candidates = [], []
    for rate in RATES:
        for seed, delta in zip(SEEDS, seed_deltas):
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
                dict(common, arm="rtdr", weighted_f1=0.6 + delta)
            )
    return originals, candidates


class InvocationContractTests(unittest.TestCase):
    def test_phase_b_base_only_accepts_a_base_cell(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        with tempfile.TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fold = _write_job(_fold(formal, "rtdr", 0.0, 66), "rtdr", 0.0, 66)
            self.assertEqual(collect_job(fold, "rtdr", 0.0, 66)["arm"], "rtdr")

    def test_phase_b_base_only_rejects_an_extension_only_cell(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        with tempfile.TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fold = _write_job(_fold(formal, "rtdr", 0.5, 69), "rtdr", 0.5, 69)
            with self.assertRaisesRegex(ValueError, "invocation.*cover"):
                collect_job(fold, "rtdr", 0.5, 69)

    def test_phase_b_base_plus_extension_accepts_an_extension_cell(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        with tempfile.TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fold = _write_job(_fold(formal, "rtdr", 0.5, 69), "rtdr", 0.5, 69)
            _write_invocation(formal, _extension_invocation())
            self.assertEqual(collect_job(fold, "rtdr", 0.5, 69)["arm"], "rtdr")

    def test_phase_b_rejects_extension_drift_and_superset(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        for mutation in ("drift", "superset"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                formal = Path(tmp) / "formal"
                fold = _write_job(
                    _fold(formal, "rtdr", 0.0, 66), "rtdr", 0.0, 66
                )
                extension = _extension_invocation()
                if mutation == "drift":
                    extension["gpus"] = ["4", "5", "6"]
                    _write_invocation(formal, extension)
                else:
                    _write_invocation(formal, extension)
                    extra = dict(extension, job_count=14)
                    _write_invocation(formal, extra)
                with self.assertRaisesRegex(ValueError, "invocation"):
                    collect_job(fold, "rtdr", 0.0, 66)


class ExtensionCollectionTests(unittest.TestCase):
    def test_collects_exact_15_pairs_from_relocated_artifacts(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_rtdr_extension import (
            collect_extension_grid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_root, phase_b = _write_extension_grid(root / "source")
            relocated = root / "relocated"
            shutil.move(str(root / "source" / "historical"), str(relocated / "historical"))
            shutil.move(str(root / "source" / "phase_b"), str(relocated / "phase_b"))
            rows = collect_extension_grid(
                relocated / "historical" / "formal" / "original",
                relocated / "phase_b" / "formal",
            )
        self.assertEqual(set(rows), {"original", "rtdr"})
        self.assertEqual(len(rows["original"]), 15)
        self.assertEqual(len(rows["rtdr"]), 15)

    def test_rejects_mask_drift_and_missing_task(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_rtdr_extension import (
            collect_extension_grid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_root, phase_b = _write_extension_grid(root)
            bad = _fold(phase_b, "rtdr", 0.5, 68) / "saved" / "run.npz"
            import numpy as np

            with np.load(str(bad), allow_pickle=True) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["mask_bank_manifest"] = np.array(
                {"sha256": "drift", "requested_missing_rate": 0.5, "seed": 68},
                dtype=object,
            )
            np.savez_compressed(str(bad), **payload)
            with self.assertRaisesRegex(ValueError, "mask.*mismatch"):
                collect_extension_grid(original_root, phase_b)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_root, phase_b = _write_extension_grid(root)
            shutil.rmtree(_fold(phase_b, "rtdr", 0.7, 70))
            with self.assertRaisesRegex(ValueError, "missing"):
                collect_extension_grid(original_root, phase_b)


class ExtensionGateTests(unittest.TestCase):
    def test_requires_at_least_three_of_five_positive_seed_macros(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            summarize_candidate,
        )

        originals, candidates = _metric_rows((0.03, 0.02, 0.01, -0.005, -0.005))
        passed = summarize_candidate(
            "rtdr", originals, candidates, rates=RATES, seeds=SEEDS
        )
        self.assertTrue(passed["gate"]["passed"])
        self.assertEqual(passed["gate"]["positive_seed_macros"], 3)
        self.assertTrue(passed["gate"]["all_rate_means_positive"])

        originals, candidates = _metric_rows((0.1, 0.1, -0.01, -0.01, -0.01))
        failed = summarize_candidate(
            "rtdr", originals, candidates, rates=RATES, seeds=SEEDS
        )
        self.assertGreater(failed["macro_delta"], 0.0)
        self.assertTrue(failed["gate"]["all_rate_means_positive"])
        self.assertEqual(failed["gate"]["positive_seed_macros"], 2)
        self.assertFalse(failed["gate"]["passed"])

    def test_outputs_atomic_detailed_json_and_markdown_mirrors(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_rtdr_extension import (
            summarize_extension,
            write_extension_outputs,
        )

        originals, candidates = _metric_rows((0.03, 0.02, 0.01, -0.005, -0.005))
        summary = summarize_extension({"original": originals, "rtdr": candidates})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_extension_outputs(output, summary)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(len(summary["candidate"]["tasks"]), 15)
            for name in ("RESULTS.md", "RESULTS.zh.md", "RESULTS.en.md"):
                text = (output / name).read_text(encoding="utf-8")
                self.assertIn("0.5", text)
                self.assertIn("66", text)
                self.assertIn("PASS", text)
                self.assertIn("post-gate extension criterion", text.lower())
                self.assertIn("initial advancement gate", text.lower())
            self.assertEqual(list(output.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
