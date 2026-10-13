"""CPU checks for frozen P0 evidence, usable with unittest or pytest."""

import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.osram_history_effect_stratification_20261013.evaluate import (
    batch_metadata,
    evaluate_rate,
    history_bins,
    main,
    per_utterance_effects,
    summarize_strata,
    validate_source,
)
from experiments.osram_history_effect_stratification_20261013.dispatch import build_tasks
from experiments.osram_history_effect_stratification_20261013.summarize import aggregate, main as summarize_main, report_markdown, strata_rows


class HistoryEffectTests(unittest.TestCase):
    def view(self):
        return {
            "conversation_ids": ["dialogue_a", "dialogue_b"],
            "umask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
            "qmask": torch.tensor([[8, 3, 8, 8], [0, 1, 0, 0]]),
            "labels": torch.tensor([[0, 1, 1, 1], [1, 1, 9, 9]]),
            "availability": torch.tensor([
                [[1, 0, 0], [0, 1, 0]],
                [[0, 1, 0], [0, 0, 1]],
                [[0, 0, 1], [0, 0, 0]],
                [[1, 1, 1], [0, 0, 0]],
            ]),
        }

    def test_metadata_is_batch_major_strict_past_and_multispeaker(self):
        values = batch_metadata(self.view())
        np.testing.assert_array_equal(values["conversation_id"], ["dialogue_a"] * 4 + ["dialogue_b"] * 2)
        np.testing.assert_array_equal(values["position"], [0, 1, 2, 3, 0, 1])
        np.testing.assert_array_equal(values["speaker"], [8, 3, 8, 8, 0, 1])
        np.testing.assert_array_equal(values["prior_total"], [0, 1, 2, 3, 0, 1])
        np.testing.assert_array_equal(values["prior_same_speaker"], [0, 0, 1, 2, 0, 0])
        np.testing.assert_array_equal(values["speaker_switch"], [-1, 1, 1, 0, -1, 1])
        np.testing.assert_array_equal(values["history_modality_seen"], [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0], [0, 1, 0],
        ])
        np.testing.assert_array_equal(values["same_speaker_history_modality_seen"], [
            [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0],
        ])
        np.testing.assert_array_equal(values["adjacent_label_relation"], ["start", "different", "same", "same", "start", "same"])
        np.testing.assert_array_equal(values["same_speaker_label_relation"], ["first_for_speaker", "first_for_speaker", "different", "same", "first_for_speaker", "first_for_speaker"])

    def test_metadata_rejects_holes_and_noninteger_speakers(self):
        view = self.view()
        view["umask"][0] = torch.tensor([1, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            batch_metadata(view)
        view = self.view()
        view["qmask"] = view["qmask"].float()
        view["qmask"][0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "finite integers"):
            batch_metadata(view)

    def test_ce_sign_corrections_harms_and_probability_change(self):
        labels = np.array([0, 1, 0, 1])
        logits = {
            "L": np.array([[0., 2.], [0., 2.], [2., 0.], [2., 0.]]),
            "LB": np.array([[2., 0.], [2., 0.], [3., 0.], [3., 0.]]),
            "LBG": np.array([[3., 0.], [0., 2.], [0., 2.], [0., 2.]]),
        }
        effect = per_utterance_effects(labels, logits)
        np.testing.assert_array_equal(effect["correction_Base"], [1, 0, 0, 0])
        np.testing.assert_array_equal(effect["harm_Base"], [0, 1, 0, 0])
        np.testing.assert_array_equal(effect["correction_Gap"], [0, 1, 0, 1])
        np.testing.assert_array_equal(effect["harm_Gap"], [0, 0, 1, 0])
        np.testing.assert_allclose(effect["Delta_Base"][:2], [2, -2])
        np.testing.assert_allclose(effect["Delta_Total"], effect["Delta_Base"] + effect["Delta_Gap"])
        self.assertGreater(effect["probability_change_Base"][0], 0)
        self.assertLess(effect["probability_change_Base"][1], 0)

    def test_large_logits_are_stable_and_malformed_inputs_fail(self):
        values = {name: np.array([[10000., -10000.]]) for name in ("L", "LB", "LBG")}
        effect = per_utterance_effects(np.array([1]), values)
        self.assertEqual(effect["ce_L"][0], 20000)
        self.assertTrue(np.isfinite(effect["Delta_Base"]).all())
        with self.assertRaisesRegex(ValueError, "class range"):
            per_utterance_effects(np.array([2]), values)
        values["LB"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            per_utterance_effects(np.array([1]), values)

    def test_bins_have_exact_boundaries(self):
        np.testing.assert_array_equal(history_bins(np.array([0, 1, 2, 4, 5, 9, 10, 19, 20, 100])),
                                      ["0", "1", "2-4", "2-4", "5-9", "5-9", "10-19", "10-19", "20+", "20+"])

    def test_summary_covers_every_utterance_and_reconciles_accuracy(self):
        metadata = batch_metadata(self.view())
        labels = np.array([0, 1, 1, 1, 1, 1])
        local = np.tile([2., 0.], (6, 1))
        base = np.tile([0., 2.], (6, 1))
        effect = per_utterance_effects(labels, {"L": local, "LB": base, "LBG": base})
        summary = summarize_strata({"labels": labels, **metadata, **effect})
        for groups in summary["strata"].values():
            self.assertEqual(sum(group["count"] for group in groups.values()), 6)
        overall = summary["overall"]
        self.assertEqual(overall["effects"]["Base"]["corrections"], 5)
        self.assertEqual(overall["effects"]["Base"]["harms"], 1)
        self.assertAlmostEqual(overall["accuracy"]["LB"] - overall["accuracy"]["L"], 4 / 6)
        self.assertEqual(overall["effects"]["Gap"]["mean_delta_ce"], 0.)
        self.assertIsNone(summary["strata"]["adjacent_label_relation"]["same"]["effects"]["Base"]["harm_rate_among_before_correct"])

    def test_source_guard_accepts_only_requested_frozen_architectures(self):
        source = dict(dataset="IEMOCAPFour", backbone_type="osram", osram_readout_fusion="flat",
                      osram_meaningful_block=None, training_objective="emotion-only", osram_bidirectional=False)
        for dataset in ("IEMOCAPFour", "IEMOCAPSix", "MELD"):
            validate_source(dict(source, dataset=dataset))
        for key, value in (("osram_bidirectional", True), ("training_objective", "future-state"), ("osram_readout_fusion", "gate")):
            with self.assertRaises(ValueError):
                validate_source(dict(source, **{key: value}))

    def test_one_frozen_forward_exports_logits_and_checks_source_artifact(self):
        class FrozenFlat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scans = 0
                self.osram = torch.nn.Module()
                self.osram.context_dim = 1
                self.osram.local_skip = torch.nn.Linear(2, 4)
                self.osram.emotion_adapter = torch.nn.Linear(6, 4)
                self.osram.emotion_norm = torch.nn.LayerNorm(4)
                self.smax_fc = torch.nn.Linear(4, 2)

            def forward(self, inputs, availability, qmask, umask, lengths, predict_missing):
                self.scans += 1
                local = inputs[0]
                flat = torch.cat([local, local, local], dim=-1)
                hidden = self.osram.emotion_norm(
                    self.osram.local_skip(local) + self.osram.emotion_adapter(flat)
                )
                hidden = torch.where(umask.T.bool().unsqueeze(-1), hidden, torch.zeros_like(hidden))
                self.last_logits = self.smax_fc(hidden)
                return self.last_logits, None, None, None

        torch.manual_seed(19)
        model = FrozenFlat().eval().requires_grad_(False)
        view = self.view()
        view["incomplete"] = torch.randn(4, 2, 2)
        view["lengths"] = [4, 2]
        original_state = {name: value.clone() for name, value in model.state_dict().items()}
        config = SimpleNamespace(dataset="IEMOCAPFour", latent_dim=2)
        with patch("gcnet_missing_m3.train_gcnet._prepare_view", return_value=view):
            summary, artifacts = evaluate_rate(model, [[]], config, (1, 1, 1), None)
        self.assertEqual(model.scans, 1)
        self.assertEqual(summary["checks"]["memory_scan_count"], 1)
        self.assertLessEqual(summary["checks"]["full_logit_max_abs_error"], 2e-5)
        self.assertEqual(artifacts["logits_LBG"].shape, (6, 2))
        np.testing.assert_array_equal(artifacts["logits_LBG"], model.last_logits.transpose(0, 1)[view["umask"].bool()].numpy())
        np.testing.assert_array_equal(artifacts["labels"], [0, 1, 1, 1, 1, 1])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.npz"
            np.savez_compressed(source, labels=artifacts["labels"], predictions=artifacts["pred_LBG"],
                                availability=artifacts["current_availability"])
            with patch("gcnet_missing_m3.train_gcnet._prepare_view", return_value=view):
                summary, checked = evaluate_rate(model, [[]], config, (1, 1, 1), None, source)
            self.assertTrue(summary["checks"]["source_prediction_exact"])
            self.assertTrue(summary["checks"]["source_availability_exact"])
            output = Path(directory) / "utterances.npz"
            np.savez_compressed(output, **checked)
            with np.load(output, allow_pickle=False) as saved:
                for name in saved.files:
                    np.testing.assert_array_equal(saved[name], checked[name])

    def test_trainable_model_and_existing_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            evaluate_rate(torch.nn.Linear(2, 2), [], None, (), None)
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.argv", ["p0", "--source", "/missing/source", "--dataset-root", "/missing/data", "--output", directory]):
                with self.assertRaises(FileExistsError):
                    main()

    def test_dispatch_plan_has_30_iemocap_and_three_meld_sources(self):
        tasks = build_tasks(Path("/iemocap"), Path("/meld"), Path("/new/output"))
        self.assertEqual(len(tasks), 33)
        self.assertEqual(sum(task["dataset"] == "meld" for task in tasks), 3)
        self.assertEqual(len({task["output"] for task in tasks}), 33)
        meld = next(task for task in tasks if task["dataset"] == "meld")
        self.assertEqual(meld["source"], "/meld/flat_seed66")
        self.assertEqual(meld["fold"], 1)

    def fixture_run(self, root, dataset, seed, fold):
        directory = root / dataset / f"seed_{seed}" / f"fold_{fold}"
        directory.mkdir(parents=True)
        values = batch_metadata(self.view())
        values["conversation_id"] = np.asarray([f"fold{fold}_{conversation}" for conversation in values["conversation_id"]])
        labels = np.array([0, 1, 1, 1, 1, 1])
        logits = {"L": np.tile([2., 0.], (6, 1)), "LB": np.tile([0., 2.], (6, 1)), "LBG": np.tile([0., 2.], (6, 1))}
        values.update(labels=labels, **{f"logits_{name}": value for name, value in logits.items()},
                      **per_utterance_effects(labels, logits))
        np.savez_compressed(directory / "utterances_miss_0p0.npz", **values)
        summary = {"status": "complete", "training": False, "delta_definition": "test", "dataset": dataset,
                   "seed": seed, "fold": fold,
                   "rates": {"0.0": {"utterance_file": "utterances_miss_0p0.npz", "checkpoint_sha256": "fixture"}}}
        (directory / "SUMMARY.json").write_text(json.dumps(summary))
        return directory

    def test_aggregate_pools_folds_before_seeds_and_keeps_datasets_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [self.fixture_run(root, "IEMOCAPFour", seed, fold) for seed in (66, 67) for fold in range(1, 6)]
            inputs.append(self.fixture_run(root, "MELD", 66, 1))
            summary = aggregate(inputs)
            self.assertEqual(set(summary["datasets"]), {"IEMOCAPFour", "MELD"})
            rate = summary["datasets"]["IEMOCAPFour"]["rates"]["0.0"]
            self.assertEqual(rate["by_seed"]["66"]["overall"]["count"], 30)
            self.assertEqual(rate["by_seed"]["66"]["folds"], [1, 2, 3, 4, 5])
            self.assertEqual(rate["across_seed_overall"]["Base"]["mean_delta_ce"]["seed_count"], 2)
            self.assertEqual(rate["across_seed_strata"]["prior_total"]["0"]["counts_by_seed"], {"66": 10, "67": 10})
            self.assertEqual(summary["datasets"]["MELD"]["rates"]["0.0"]["by_seed"]["66"]["overall"]["count"], 6)
            self.assertIn("IEMOCAPFour", report_markdown(summary))
            self.assertIn("MELD", report_markdown(summary))
            rows = strata_rows(summary)
            self.assertTrue(rows)
            self.assertEqual(set(row["dataset"] for row in rows), {"IEMOCAPFour", "MELD"})
            output = root / "aggregate"
            with patch("sys.argv", ["p0-summary", "--input-root", str(root), "--output", str(output)]):
                summarize_main()
            for dataset in ("IEMOCAPFour", "MELD"):
                self.assertTrue((output / dataset / "STRATA.csv").is_file())
                dataset_summary = json.loads((output / dataset / "AGGREGATE.json").read_text())
                self.assertEqual(set(dataset_summary["datasets"]), {dataset})
            with patch("sys.argv", ["p0-summary", "--input-root", str(root), "--output", str(output)]):
                with self.assertRaises(FileExistsError):
                    summarize_main()

    def test_aggregate_rejects_incomplete_folds_duplicates_and_corrupted_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.fixture_run(root, "IEMOCAPSix", 66, 1)
            with self.assertRaisesRegex(ValueError, "expected"):
                aggregate([first])
            partial = aggregate([first], allow_partial=True)
            self.assertTrue(partial["allow_partial"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                aggregate([first, first], allow_partial=True)
            artifact = first / "utterances_miss_0p0.npz"
            with np.load(artifact, allow_pickle=False) as loaded:
                values = {name: loaded[name] for name in loaded.files}
            values["Delta_Base"][0] += 1
            np.savez_compressed(artifact, **values)
            with self.assertRaises(AssertionError):
                aggregate([first], allow_partial=True)


if __name__ == "__main__":
    unittest.main()
