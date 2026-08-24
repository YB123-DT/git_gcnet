from argparse import Namespace
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.cp_lecc_iemocap6.summarize_gate import (
    archive_metrics,
    assert_complete_archive_equal,
    _collect,
    paired_gate,
)


def _snapshot(labels=None, predicted=None, hidden_shift=0):
    labels = np.asarray(labels if labels is not None else [0, 1, 2, 3, 4, 5])
    predicted = np.asarray(predicted if predicted is not None else labels)
    logits = np.full((len(labels), 6), -2.0)
    logits[np.arange(len(labels)), predicted] = 2.0
    return {
        "test_labels": [labels[:3], labels[3:]],
        "test_preds": [logits[:3], logits[3:]],
        "test_hiddens": [np.arange(6) + hidden_shift],
        "test_fmask": [np.ones((2, 3), dtype=np.int64)],
        "test_names": ["ignored"],
    }


PARAMETER_COUNTS = {
    "cp_lecc": 34200838,
    "original": 34140166,
    "full": 34712766,
}


def _write_archive(
    path,
    snapshot=None,
    mask_sha="same-mask",
    loss_value=1.0,
    variant="cp_lecc",
    seed=66,
    mask_seed=None,
    rate=0.0,
    manifest_rate=None,
    manifest_seed=None,
    fold_index=5,
    fold_numbers=(5,),
    smoke_only=False,
    parameter_count=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot() if snapshot is None else snapshot
    mask_seed = seed if mask_seed is None else mask_seed
    manifest_rate = rate if manifest_rate is None else manifest_rate
    manifest_seed = mask_seed if manifest_seed is None else manifest_seed
    parameter_count = (
        PARAMETER_COUNTS[variant] if parameter_count is None else parameter_count
    )
    args = Namespace(
        graph_conv_variant=variant,
        seed=seed,
        mask_seed=mask_seed,
        mask_type=f"constant-{rate:.1f}",
        fold_index=fold_index,
    )
    np.savez_compressed(
        path,
        args=np.array(args, dtype=object),
        fold_numbers=np.asarray(fold_numbers),
        folder_savewhole=np.array([[3, _snapshot(), snapshot]], dtype=object),
        folder_losswhole=np.array([[{"train_loss": np.array([loss_value])}]], dtype=object),
        mask_bank_manifest=np.array(
            {
                "sha256": mask_sha,
                "requested_missing_rate": manifest_rate,
                "seed": manifest_seed,
            },
            dtype=object,
        ),
        smoke_only=np.array(smoke_only),
        parameter_count=np.array(parameter_count),
    )


def _write_grid(root, variant, parameter_count):
    for rate in (0.5, 0.7):
        for seed in range(66, 71):
            _write_archive(
                root
                / f"miss_{str(rate).replace('.', 'p')}"
                / f"seed_{seed}"
                / "fold_5"
                / "saved"
                / "run.npz",
                variant=variant,
                seed=seed,
                rate=rate,
                mask_sha=f"mask-{rate}-{seed}",
                parameter_count=parameter_count,
            )


def _rows(kind):
    rows = []
    for rate in (0.5, 0.7):
        for seed in range(66, 71):
            if kind == "candidate":
                f1 = 0.706 if rate == 0.5 else 0.716
            elif kind == "original":
                f1 = 0.700 if rate == 0.5 else 0.710
            else:
                f1 = 0.705 if rate == 0.5 else 0.715
            rows.append(
                {
                    "rate": rate,
                    "seed": seed,
                    "weighted_f1": f1,
                    "accuracy": f1 - 0.01,
                    "class_coverage": 6,
                    "dominant_ratio": 0.25,
                    "manifest_hash": f"mask-{rate}-{seed}",
                    "parameter_count": 1000 + seed,
                    "epoch": seed - 60,
                }
            )
    return rows


class ArchiveMetricsTests(unittest.TestCase):
    def test_loads_single_archive_from_path_or_saved_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "saved"
            archive = saved / "run.npz"
            _write_archive(archive)

            by_file = archive_metrics(archive)
            by_directory = archive_metrics(saved)

        self.assertEqual(by_file, by_directory)
        self.assertEqual(by_file["weighted_f1"], 1.0)
        self.assertEqual(by_file["accuracy"], 1.0)
        self.assertEqual(by_file["class_coverage"], 6)
        self.assertAlmostEqual(by_file["dominant_ratio"], 1 / 6)
        self.assertEqual(by_file["manifest_hash"], "same-mask")
        self.assertEqual(by_file["parameter_count"], PARAMETER_COUNTS["cp_lecc"])
        self.assertEqual(by_file["epoch"], 3)
        self.assertEqual(by_file["graph_conv_variant"], "cp_lecc")
        self.assertEqual(by_file["seed"], 66)
        self.assertEqual(by_file["mask_seed"], 66)
        self.assertEqual(by_file["mask_type"], "constant-0.0")
        self.assertEqual(by_file["fold_index"], 5)
        self.assertEqual(by_file["fold_numbers"], [5])
        self.assertEqual(by_file["requested_missing_rate"], 0.0)
        self.assertEqual(by_file["manifest_seed"], 66)
        self.assertIs(by_file["smoke_only"], False)
        for value in by_file.values():
            self.assertNotIsInstance(value, np.generic)

    def test_rejects_directory_without_exactly_one_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                archive_metrics(saved)
            _write_archive(saved / "one.npz")
            _write_archive(saved / "two.npz")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                archive_metrics(saved)

    def test_complete_archive_equality_checks_all_locked_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.npz"
            second = Path(tmp) / "second.npz"
            _write_archive(first)
            _write_archive(second, variant="original")
            assert_complete_archive_equal(first, second)

    def test_complete_archive_rejects_original_used_as_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.npz"
            original = Path(tmp) / "original.npz"
            _write_archive(candidate, variant="original")
            _write_archive(original, variant="original")
            with self.assertRaisesRegex(ValueError, "graph_conv_variant"):
                assert_complete_archive_equal(candidate, original)

    def test_complete_archive_reports_each_mismatch_family(self):
        mutations = {
            "best epoch": lambda kw: kw.update(snapshot=_snapshot()),
            "folder_losswhole": lambda kw: kw.update(loss_value=2.0),
            "test_labels": lambda kw: kw.update(snapshot=_snapshot(labels=[1, 1, 2, 3, 4, 5])),
            "test_preds": lambda kw: kw.update(snapshot=_snapshot(predicted=[1, 1, 2, 3, 4, 5])),
            "test_hiddens": lambda kw: kw.update(snapshot=_snapshot(hidden_shift=1)),
            "test_fmask": lambda kw: kw.update(snapshot={**_snapshot(), "test_fmask": [np.zeros((2, 3), dtype=np.int64)]}),
            "mask sha": lambda kw: kw.update(mask_sha="different"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.npz"
            _write_archive(baseline, variant="original")
            for index, (message, mutate) in enumerate(mutations.items()):
                kwargs = {}
                mutate(kwargs)
                changed = Path(tmp) / f"changed-{index}.npz"
                _write_archive(changed, **kwargs)
                if message == "best epoch":
                    with np.load(changed, allow_pickle=True) as data:
                        values = {key: data[key] for key in data.files}
                    values["folder_savewhole"][0][0] = 4
                    np.savez_compressed(changed, **values)
                with self.subTest(message=message):
                    with self.assertRaisesRegex(AssertionError, message):
                        assert_complete_archive_equal(changed, baseline)

    def test_real_original_archive_uses_supported_schema_when_available(self):
        configured = os.environ.get("CP_LECC_REAL_ORIGINAL_ARCHIVE")
        if not configured or not Path(configured).exists():
            self.skipTest("CP_LECC_REAL_ORIGINAL_ARCHIVE is absent")
        metrics = archive_metrics(Path(configured))
        self.assertEqual(metrics["graph_conv_variant"], "original")
        self.assertIn("requested_missing_rate", metrics)
        self.assertIn("fold_numbers", metrics)


class ArchiveProvenanceTests(unittest.TestCase):
    def _assert_mutation_rejected(self, message, **mutation):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_grid(root, "cp_lecc", PARAMETER_COUNTS["cp_lecc"])
            target = root / "miss_0p5" / "seed_66" / "fold_5" / "saved" / "run.npz"
            arguments = {
                "variant": "cp_lecc",
                "seed": 66,
                "rate": 0.5,
                "parameter_count": PARAMETER_COUNTS["cp_lecc"],
            }
            arguments.update(mutation)
            _write_archive(target, **arguments)
            with self.assertRaisesRegex(ValueError, message):
                _collect(root, "cp_lecc", PARAMETER_COUNTS["cp_lecc"])

    def test_rejects_wrong_arm(self):
        self._assert_mutation_rejected(
            "graph_conv_variant", variant="original", parameter_count=PARAMETER_COUNTS["cp_lecc"]
        )

    def test_rejects_wrong_stored_seed(self):
        self._assert_mutation_rejected("seed", seed=67)

    def test_rejects_wrong_mask_type_rate(self):
        self._assert_mutation_rejected("mask_type", rate=0.7, manifest_rate=0.5)

    def test_rejects_wrong_manifest_rate(self):
        self._assert_mutation_rejected("requested_missing_rate", manifest_rate=0.7)

    def test_rejects_wrong_fold_index(self):
        self._assert_mutation_rejected("fold_index", fold_index=4)

    def test_rejects_wrong_fold_numbers(self):
        self._assert_mutation_rejected("fold_numbers", fold_numbers=(4,))

    def test_rejects_smoke_archive(self):
        self._assert_mutation_rejected("smoke_only", smoke_only=True)

    def test_rejects_wrong_parameter_count(self):
        self._assert_mutation_rejected("parameter_count", parameter_count=1)

    def test_rejects_wrong_mask_seed(self):
        self._assert_mutation_rejected("mask_seed", mask_seed=67, manifest_seed=66)

    def test_rejects_wrong_manifest_seed(self):
        self._assert_mutation_rejected("manifest seed", manifest_seed=67)


class SummaryCliTests(unittest.TestCase):
    def test_reject_decision_writes_atomic_json_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_root = root / "candidate"
            original_root = root / "original"
            full_root = root / "full"
            _write_grid(candidate_root, "cp_lecc", PARAMETER_COUNTS["cp_lecc"])
            _write_grid(original_root, "original", PARAMETER_COUNTS["original"])
            _write_grid(full_root, "full", PARAMETER_COUNTS["full"])
            complete_candidate = root / "candidate-complete.npz"
            _write_archive(complete_candidate, variant="cp_lecc")
            _write_archive(
                original_root
                / "miss_0p0"
                / "seed_66"
                / "fold_5"
                / "saved"
                / "run.npz",
                variant="original",
            )
            output = root / "summary" / "gate.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.cp_lecc_iemocap6.summarize_gate",
                    "--candidate-root",
                    str(candidate_root),
                    "--original-root",
                    str(original_root),
                    "--full-root",
                    str(full_root),
                    "--complete-candidate",
                    str(complete_candidate),
                    "--output-json",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "REJECT")
            self.assertFalse(json.loads(output.read_text())["promote"])
            self.assertEqual(list(output.parent.iterdir()), [output])


class PairedGateTests(unittest.TestCase):
    def setUp(self):
        self.candidate = _rows("candidate")
        self.original = _rows("original")
        self.full = _rows("full")

    def test_passing_fixture_returns_complete_evidence(self):
        evidence = paired_gate(self.candidate, self.original, self.full)

        self.assertTrue(evidence["promote"])
        self.assertEqual(len(evidence["task_rows"]), 10)
        self.assertEqual(set(evidence["rate_means"]), {"0.5", "0.7"})
        self.assertEqual(set(evidence["seed_deltas"]), {str(seed) for seed in range(66, 71)})
        self.assertEqual(evidence["wins"], 5)
        self.assertEqual(len(evidence["coverage_dominant"]), 10)
        self.assertTrue(all(evidence["conditions"].values()))

    def test_task_rows_retain_every_arm_audit_metric(self):
        evidence = paired_gate(self.candidate, self.original, self.full)
        task = evidence["task_rows"][0]
        required = {
            "rate",
            "seed",
            "weighted_f1",
            "accuracy",
            "class_coverage",
            "dominant_ratio",
            "epoch",
            "manifest_hash",
            "parameter_count",
        }

        self.assertEqual(
            set(task), {"rate", "seed", "candidate", "original", "full", "delta_original", "delta_full"}
        )
        for arm, source in (
            ("candidate", self.candidate[0]),
            ("original", self.original[0]),
            ("full", self.full[0]),
        ):
            self.assertEqual(set(task[arm]), required)
            self.assertEqual(task[arm], source)

    def test_positive_aggregate_full_delta_allows_one_losing_seed(self):
        candidate = copy.deepcopy(self.candidate)
        for row in candidate:
            full = next(
                item
                for item in self.full
                if item["rate"] == row["rate"] and item["seed"] == row["seed"]
            )
            row["weighted_f1"] = full["weighted_f1"] + (
                -0.001 if row["seed"] == 66 else 0.005
            )

        evidence = paired_gate(candidate, self.original, self.full)

        self.assertTrue(evidence["promote"])
        self.assertAlmostEqual(evidence["mean_delta_full"], 0.0038)
        self.assertTrue(
            evidence["conditions"]["candidate_seed_mean_strictly_greater_full"]
        )
        self.assertLess(evidence["seed_deltas"]["66"]["full"], 0)

    def test_nonpositive_aggregate_full_delta_fails_named_condition(self):
        candidate = copy.deepcopy(self.candidate)
        for row in candidate:
            full = next(
                item
                for item in self.full
                if item["rate"] == row["rate"] and item["seed"] == row["seed"]
            )
            row["weighted_f1"] = full["weighted_f1"]

        evidence = paired_gate(candidate, self.original, self.full)

        self.assertEqual(evidence["mean_delta_full"], 0.0)
        self.assertFalse(
            evidence["conditions"]["candidate_seed_mean_strictly_greater_full"]
        )
        self.assertFalse(evidence["promote"])

    def _assert_rejects_condition(self, condition, candidate=None, original=None, full=None):
        evidence = paired_gate(
            self.candidate if candidate is None else candidate,
            self.original if original is None else original,
            self.full if full is None else full,
        )
        self.assertFalse(evidence["promote"])
        self.assertFalse(evidence["conditions"][condition])

    def test_each_numeric_condition_can_fail_independently(self):
        cases = {}
        candidate = copy.deepcopy(self.candidate)
        for row in candidate:
            if row["rate"] == 0.5:
                row["weighted_f1"] = self.original[self.candidate.index(row)]["weighted_f1"] - 0.001
        cases["rate_0.5_nonnegative_vs_original"] = candidate

        candidate = copy.deepcopy(self.candidate)
        for row in candidate:
            if row["rate"] == 0.7:
                row["weighted_f1"] = 0.709
        cases["rate_0.7_nonnegative_vs_original"] = candidate

        candidate = copy.deepcopy(self.candidate)
        for row in candidate:
            row["weighted_f1"] = next(x["weighted_f1"] for x in self.original if x["rate"] == row["rate"] and x["seed"] == row["seed"]) + 0.0049
        cases["seed_mean_delta_original_at_least_0.005"] = candidate

        candidate = copy.deepcopy(self.candidate)
        for row in candidate:
            if row["seed"] in (66, 67):
                baseline = next(x for x in self.original if x["rate"] == row["rate"] and x["seed"] == row["seed"])
                row["weighted_f1"] = baseline["weighted_f1"] - 0.001
        cases["at_least_four_positive_seed_deltas"] = candidate

        candidate = copy.deepcopy(self.candidate)
        for row in candidate:
            row["weighted_f1"] = next(x["weighted_f1"] for x in self.full if x["rate"] == row["rate"] and x["seed"] == row["seed"])
        cases["candidate_seed_mean_strictly_greater_full"] = candidate

        candidate = copy.deepcopy(self.candidate)
        candidate[0]["class_coverage"] = 5
        cases["all_candidate_coverage_six"] = candidate

        for condition, candidate in cases.items():
            with self.subTest(condition=condition):
                self._assert_rejects_condition(condition, candidate=candidate)

    def test_mask_mismatch_is_a_failed_condition(self):
        candidate = copy.deepcopy(self.candidate)
        candidate[0]["manifest_hash"] = "wrong"
        self._assert_rejects_condition("all_pair_mask_hashes_match", candidate=candidate)

    def test_missing_or_duplicate_keys_raise(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            paired_gate(self.candidate[:-1], self.original, self.full)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_gate(self.candidate + [self.candidate[0]], self.original, self.full)

    def test_raw_precision_controls_threshold_not_rounded_display(self):
        candidate = copy.deepcopy(self.candidate)
        for row in candidate:
            original = next(x for x in self.original if x["rate"] == row["rate"] and x["seed"] == row["seed"])
            row["weighted_f1"] = original["weighted_f1"] + 0.0049999
        self._assert_rejects_condition(
            "seed_mean_delta_original_at_least_0.005", candidate=candidate
        )


if __name__ == "__main__":
    unittest.main()
