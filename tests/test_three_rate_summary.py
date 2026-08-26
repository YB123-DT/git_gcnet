import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.test_rtdr_extension_summary import _extension_invocation
from tests.test_rtdr_full_summary import _full_invocation
from tests.test_second_graph_aggregation_summary import (
    _fold,
    _write_invocation,
    _write_job,
)


RATES = (0.0, 0.5, 0.7)
SEEDS = (66, 67, 68, 69, 70)
ARMS = ("genagg", "soft_medoid", "ssma", "rtdr")


def _phase_a_extension_invocation():
    return {
        "arms": ["genagg", "soft_medoid"],
        "fold": 5,
        "gpus": ["1", "2", "3", "4", "5", "6", "7"],
        "job_count": 30,
        "parallel_arms": True,
        "rates": [0.0, 0.5, 0.7],
        "seeds": [66, 67, 68, 69, 70],
        "stage": "formal",
        "workers_per_gpu": 3,
    }


def _ssma_extension_invocation():
    return {
        "arms": ["ssma"],
        "fold": 5,
        "gpus": ["5", "6", "7"],
        "job_count": 15,
        "parallel_arms": True,
        "rates": [0.0, 0.5, 0.7],
        "seeds": [66, 67, 68, 69, 70],
        "stage": "formal",
        "workers_per_gpu": 3,
    }


def _write_uniform_grid(root):
    original_formal = root / "historical" / "formal"
    phase_a = root / "phase_a" / "formal"
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
            for arm in ("genagg", "soft_medoid"):
                _write_job(
                    _fold(phase_a, arm, rate, seed),
                    arm,
                    rate,
                    seed,
                    mask_hash=mask,
                )
            for arm in ("ssma", "rtdr"):
                _write_job(
                    _fold(phase_b, arm, rate, seed),
                    arm,
                    rate,
                    seed,
                    mask_hash=mask,
                )
    _write_invocation(phase_a, _phase_a_extension_invocation())
    _write_invocation(phase_b, _extension_invocation())
    _write_invocation(phase_b, _full_invocation())
    _write_invocation(phase_b, _ssma_extension_invocation())
    return original_formal / "original", phase_a, phase_b


def _metric_rows(arm, seed_deltas):
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
            candidates.append(dict(common, arm=arm, weighted_f1=0.6 + delta))
    return originals, candidates


class UniformInvocationContractTests(unittest.TestCase):
    def test_phase_a_accepts_only_base_plus_exact_extension(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import collect_job

        with tempfile.TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fold = _write_job(
                _fold(formal, "genagg", 0.5, 70), "genagg", 0.5, 70
            )
            _write_invocation(formal, _phase_a_extension_invocation())
            self.assertEqual(collect_job(fold, "genagg", 0.5, 70)["seed"], 70)

        for mutation in ("drift", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                formal = Path(tmp) / "formal"
                fold = _write_job(
                    _fold(formal, "soft_medoid", 0.5, 70),
                    "soft_medoid",
                    0.5,
                    70,
                )
                extension = _phase_a_extension_invocation()
                if mutation == "drift":
                    extension["gpus"] = ["1", "2", "3", "4"]
                    _write_invocation(formal, extension)
                else:
                    _write_invocation(formal, extension)
                    _write_invocation(formal, dict(extension, job_count=29))
                with self.assertRaisesRegex(ValueError, "invocation"):
                    collect_job(fold, "soft_medoid", 0.5, 70)

    def test_phase_b_accepts_exact_existing_chain_plus_ssma_extension(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import collect_job

        with tempfile.TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fold = _write_job(_fold(formal, "ssma", 0.5, 70), "ssma", 0.5, 70)
            for invocation in (
                _extension_invocation(),
                _full_invocation(),
                _ssma_extension_invocation(),
            ):
                _write_invocation(formal, invocation)
            self.assertEqual(collect_job(fold, "ssma", 0.5, 70)["rate"], 0.5)

    def test_phase_b_rejects_ssma_extension_without_exact_existing_chain(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import collect_job

        with tempfile.TemporaryDirectory() as tmp:
            formal = Path(tmp) / "formal"
            fold = _write_job(_fold(formal, "ssma", 0.5, 70), "ssma", 0.5, 70)
            _write_invocation(formal, _ssma_extension_invocation())
            with self.assertRaisesRegex(ValueError, "invocation"):
                collect_job(fold, "ssma", 0.5, 70)


class UniformCollectionTests(unittest.TestCase):
    def test_collects_exact_15_pairs_for_each_of_four_arms_after_relocation(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_three_rate import (
            collect_uniform_grid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_uniform_grid(root / "source")
            relocated = root / "relocated"
            shutil.move(str(root / "source"), str(relocated))
            rows = collect_uniform_grid(
                relocated / "historical" / "formal" / "original",
                relocated / "phase_a" / "formal",
                relocated / "phase_b" / "formal",
            )
        self.assertEqual(set(rows), {"original", *ARMS})
        self.assertEqual(len(rows["original"]), 15)
        for arm in ARMS:
            self.assertEqual(len(rows[arm]), 15)

    def test_rejects_mask_drift(self):
        import numpy as np
        from experiments.second_graph_aggregation_iemocap6.summarize_three_rate import (
            collect_uniform_grid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            original_root, phase_a, phase_b = _write_uniform_grid(Path(tmp))
            bad = _fold(phase_a, "genagg", 0.5, 69) / "saved" / "run.npz"
            with np.load(str(bad), allow_pickle=True) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["mask_bank_manifest"] = np.array(
                {"sha256": "drift", "requested_missing_rate": 0.5, "seed": 69},
                dtype=object,
            )
            np.savez_compressed(str(bad), **payload)
            with self.assertRaisesRegex(ValueError, "mask.*mismatch"):
                collect_uniform_grid(original_root, phase_a, phase_b)


class UniformSummaryTests(unittest.TestCase):
    def test_uniform_stable_requires_all_rates_and_three_seeds_positive(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_three_rate import (
            summarize_uniform,
        )

        rows = {}
        originals, _ = _metric_rows("genagg", (0.0,) * 5)
        rows["original"] = originals
        for arm in ARMS:
            _, rows[arm] = _metric_rows(
                arm, (0.03, 0.02, 0.01, -0.005, -0.005)
            )
        summary = summarize_uniform(rows)
        for arm in ARMS:
            result = summary["candidates"][arm]
            self.assertTrue(result["uniform_stable"])
            self.assertEqual(result["positive_rate_means"], 3)
            self.assertEqual(result["positive_seed_macros"], 3)
            self.assertNotIn("status", result)

        _, rows["ssma"] = _metric_rows(
            "ssma", (0.1, 0.1, -0.01, -0.01, -0.01)
        )
        failed = summarize_uniform(rows)["candidates"]["ssma"]
        self.assertGreater(failed["overall_macro_delta"], 0.0)
        self.assertEqual(failed["positive_seed_macros"], 2)
        self.assertFalse(failed["uniform_stable"])

    def test_writes_atomic_detailed_json_and_trilingual_results(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_three_rate import (
            summarize_uniform,
            write_uniform_outputs,
        )

        rows = {}
        originals, _ = _metric_rows("genagg", (0.0,) * 5)
        rows["original"] = originals
        for arm in ARMS:
            _, rows[arm] = _metric_rows(arm, (0.01,) * 5)
        summary = summarize_uniform(rows)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_uniform_outputs(output, summary)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            for name in ("RESULTS.md", "RESULTS.zh.md", "RESULTS.en.md"):
                text = (output / name).read_text(encoding="utf-8")
                self.assertIn("uniform_stable", text)
                self.assertIn("soft_medoid", text)
                self.assertIn("0.5", text)
                self.assertIn("70", text)
                self.assertIn("initial", text.lower())
            self.assertEqual(list(output.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
