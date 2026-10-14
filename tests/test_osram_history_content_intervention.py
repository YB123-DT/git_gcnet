"""CPU verification for P1 selection, frozen scans, coverage and aggregation."""

import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.osram_history_content_intervention_20261014.intervention import (
    ARMS, MODALITIES, evaluate_view, paired_effects, replacement_batch, select_replacements, summarize_rows,
)
from experiments.osram_history_content_intervention_20261014.evaluate import evaluate_rate, main, rows_to_arrays
from experiments.osram_history_content_intervention_20261014.dispatch import build_tasks, preflight, main as dispatch_main
from experiments.osram_history_content_intervention_20261014.summarize import aggregate, load_checked, main as summarize_main, report_markdown


class FrozenFlat(torch.nn.Module):
    def __init__(self, leak_local=False):
        super().__init__()
        self.scans = 0
        self.leak_local = leak_local
        self.osram = torch.nn.Module()
        self.osram.context_dim = 1
        self.osram.local_skip = torch.nn.Linear(6, 4)
        self.osram.emotion_adapter = torch.nn.Linear(10, 4)
        self.osram.emotion_norm = torch.nn.LayerNorm(4)
        self.smax_fc = torch.nn.Linear(4, 3)
        self.calls = []

    def forward(self, inputs, availability, qmask, umask, lengths, predict_missing):
        self.scans += 1
        current = inputs[0]
        summed = current.sum(-1, keepdim=True)
        past = summed.cumsum(0)-summed
        local = current+past if self.leak_local else current
        flat = torch.cat((local, past, past.expand(-1, -1, 3)), dim=-1)
        hidden = self.osram.emotion_norm(self.osram.local_skip(local)+self.osram.emotion_adapter(flat))
        hidden = torch.where(umask.T.bool().unsqueeze(-1), hidden, 0.)
        self.calls.append(dict(incomplete=current.detach().clone(), availability=availability.detach().clone(),
                               qmask=qmask.detach().clone(), umask=umask.detach().clone(), lengths=list(lengths)))
        return self.smax_fc(hidden), None, None, None


class HistoryContentTests(unittest.TestCase):
    def view(self, prefix=""):
        availability = torch.ones(7, 2, 3, dtype=torch.float64)
        availability[3:, 1] = 0
        availability[0, 0, 1] = 0
        availability[2, 0, 2] = 0
        incomplete = torch.arange(1, 85, dtype=torch.float64).reshape(7, 2, 6)/20
        incomplete *= torch.repeat_interleave(availability, torch.tensor([2, 3, 1]), dim=-1)
        return dict(incomplete=incomplete, availability=availability,
                    qmask=torch.tensor([[8, 3, 8, 1, 8, 3, 8], [2, 4, 2, 0, 0, 0, 0]]),
                    umask=torch.tensor([[1]*7, [1, 1, 1, 0, 0, 0, 0]]),
                    labels=torch.tensor([[0, 1, 2, 0, 1, 2, 0], [1, 2, 0, 0, 0, 0, 0]]),
                    lengths=[7, 3], conversation_ids=[prefix+"a", prefix+"b"])

    def model(self, leak=False):
        torch.manual_seed(19)
        return FrozenFlat(leak_local=leak).double().eval().requires_grad_(False)

    def test_selection_is_strict_past_observed_label_free_and_source_is_target_speaker(self):
        mask = np.ones((7, 3), dtype=int)
        mask[4, 1] = 0
        speakers = np.array([8, 3, 8, 1, 8, 3, 8])
        plans, skipped = select_replacements(mask, speakers, 6)
        self.assertFalse(skipped)
        self.assertEqual(set(inspect.signature(select_replacements).parameters), {"availability", "speakers", "target"})
        for plan in plans:
            modality = plan["modality_index"]
            source, donor = plan["source_position"], plan["donor_position"]
            self.assertLess(source, 6)
            self.assertLess(donor, 6)
            self.assertNotEqual(source, donor)
            self.assertEqual(mask[source, modality], 1)
            self.assertEqual(mask[donor, modality], 1)
            self.assertEqual(speakers[source], speakers[6])
            self.assertEqual(plan["donor_same_speaker"], speakers[donor] == speakers[6])
        self.assertEqual(next(row for row in plans if row["modality"] == "T")["source_position"], 2)
        self.assertEqual(next(row for row in plans if row["modality"] == "A" and row["arm"] == "other_speaker")["donor_position"], 5)
        same = next(row for row in plans if row["modality"] == "A" and row["arm"] == "same_speaker")
        self.assertEqual((same["source_position"], same["donor_position"]), (4, 2))

    def test_no_source_and_missing_donor_are_exhaustive_without_source_fallback(self):
        mask = np.ones((3, 3), dtype=int)
        plans, skipped = select_replacements(mask, [8, 3, 8], 0)
        self.assertFalse(plans)
        self.assertEqual(len(skipped), 6)
        self.assertEqual({row["reason"] for row in skipped}, {"no_same_speaker_source"})
        plans, skipped = select_replacements(mask, [8, 3, 8], 2)
        self.assertEqual(len(plans), 3)
        self.assertTrue(all(row["arm"] == "other_speaker" and row["source_position"] == 0 for row in plans))
        self.assertEqual(len(skipped), 3)
        self.assertEqual({row["reason"] for row in skipped}, {"no_observed_donor"})
        plans, skipped = select_replacements(mask, [8, 8, 8], 2)
        self.assertTrue(all(row["arm"] == "same_speaker" for row in plans))
        self.assertEqual(len(plans)+len(skipped), 6)

    def test_missing_current_modality_does_not_remove_history_test(self):
        mask = np.ones((4, 3), dtype=int)
        mask[3] = [1, 0, 0]
        plans, _ = select_replacements(mask, [8, 8, 3, 8], 3)
        self.assertEqual(len(plans), 6)

    def test_selection_rejects_invalid_speakers_mask_and_target(self):
        for mask, speakers, target in ((np.ones((2, 2)), [0, 1], 1),
                                      (np.ones((2, 3))*0.5, [0, 1], 1),
                                      (np.ones((2, 3)), [0], 1),
                                      (np.ones((2, 3)), [0, 0.1], 1),
                                      (np.ones((2, 3)), [0, np.nan], 1),
                                      (np.ones((2, 3)), [0, 1], 2)):
            with self.assertRaises(ValueError):
                select_replacements(mask, speakers, target)

    def test_replacement_edits_only_selected_slice_and_preserves_all_other_inputs(self):
        view = self.view()
        before = {name: value.clone() for name, value in view.items() if isinstance(value, torch.Tensor)}
        plans, _ = select_replacements(view["availability"][:, 0].numpy(), view["qmask"][0].numpy(), 6)
        jobs = [dict(plan, batch_index=0) for plan in plans]
        changed = replacement_batch(view, jobs, (2, 3, 1))
        offsets = [0, 2, 5, 6]
        for index, job in enumerate(jobs):
            expected = view["incomplete"][:, 0].clone()
            start, end = offsets[job["modality_index"]:job["modality_index"]+2]
            expected[job["source_position"], start:end] = view["incomplete"][job["donor_position"], 0, start:end]
            torch.testing.assert_close(changed["incomplete"][:, index], expected, rtol=0, atol=0)
            torch.testing.assert_close(changed["availability"][:, index], view["availability"][:, 0], rtol=0, atol=0)
            for name in ("qmask", "umask", "labels"):
                torch.testing.assert_close(changed[name][index], view[name][0], rtol=0, atol=0)
        for name, tensor in before.items():
            torch.testing.assert_close(view[name], tensor, rtol=0, atol=0)

    def test_replacement_rejects_unobserved_or_current_source_and_bad_dimensions(self):
        view = self.view()
        plan = dict(batch_index=0, target=6, source_position=4, donor_position=0, modality_index=1)
        with self.assertRaisesRegex(ValueError, "observed"):
            replacement_batch(view, [plan], (2, 3, 1))
        plan["source_position"] = 6
        with self.assertRaisesRegex(ValueError, "strict-past"):
            replacement_batch(view, [plan], (2, 3, 1))
        for dimensions in ((1, 1, 1), (2, 3, 0), (6,)):
            with self.assertRaises(ValueError):
                replacement_batch(view, [plan], dimensions)

    def test_ce_sign_probability_and_paired_correction_harm(self):
        labels = np.array([0, 1])
        effects = paired_effects(labels, np.array([[0., 2.], [0., 2.]]), np.array([[2., 0.], [2., 0.]]))
        np.testing.assert_allclose(effects["delta_ce"], [-2, 2])
        np.testing.assert_array_equal(effects["correction"], [True, False])
        np.testing.assert_array_equal(effects["harm"], [False, True])
        self.assertGreater(effects["true_probability_delta"][0], 0)
        self.assertLess(effects["true_probability_delta"][1], 0)

    def test_batched_and_unbatched_replacements_match_and_state_is_frozen(self):
        model, view = self.model(), self.view()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        rows, skipped, checks, original = evaluate_view(model, view, (2, 3, 1), chunk_size=16)
        serial_rows, serial_skipped, serial_checks, serial_original = evaluate_view(self.model(), view, (2, 3, 1), chunk_size=1)
        keyed = lambda row: (row["conversation_id"], row["position"], row["modality"], row["arm"])
        serial = {keyed(row): row for row in serial_rows}
        self.assertEqual(skipped, serial_skipped)
        np.testing.assert_array_equal(original, serial_original)
        self.assertGreater(checks["avoided_repeated_replacements"], 0)
        self.assertLess(checks["replacement_scans"], serial_checks["replacement_scans"])
        for row in rows:
            np.testing.assert_allclose(row["logits_replaced"], serial[keyed(row)]["logits_replaced"], rtol=1e-12, atol=1e-12)
            self.assertTrue(row["current_input_exact"] and row["current_availability_exact"] and row["availability_trajectory_exact"])
            self.assertTrue(row["current_local_exact"] and row["current_local_within_tolerance"])
        self.assertTrue(any(abs(row["delta_ce"]) > 1e-8 for row in rows))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        stats = summarize_rows(rows, skipped)
        for modality in MODALITIES:
            for arm in ARMS:
                self.assertEqual(stats[modality][arm]["targets"], 10)
                self.assertEqual(stats[modality][arm]["count"]+stats[modality][arm]["skipped"], 10)

    def test_local_history_leak_and_trainable_model_fail(self):
        with self.assertRaises(AssertionError):
            evaluate_view(self.model(leak=True), self.view(), (2, 3, 1))
        with self.assertRaisesRegex(ValueError, "frozen"):
            evaluate_view(torch.nn.Linear(2, 2), self.view(), (2, 3, 1))
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate_view(self.model(), self.view(), (2, 3, 1), chunk_size=0)

    def test_actual_causal_osram_preserves_local_and_model_state_under_content_replacement(self):
        from gcnet_missing_m3.model import MissingM3GraphModel

        torch.manual_seed(71)
        model = MissingM3GraphModel(
            base_model="LSTM", adim=2, tdim=3, vdim=1, D_e=4, graph_hidden_size=2,
            window_past=1, window_future=1,
            n_speakers=9, n_classes=3, dropout=0., no_cuda=True, latent_dim=8,
            num_experts=2, top_k=1, projector_dropout=0., predictor_dropout=0.,
            backbone_type="osram", osram_output_dim=10, osram_num_heads=2,
            osram_key_dim=3, osram_value_dim=4, osram_bidirectional=False,
            osram_readout_fusion="flat", completion_path="none",
        ).eval().requires_grad_(False)
        view = self.view()
        view["incomplete"] = view["incomplete"].float()
        view["availability"] = view["availability"].float()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        rows, _, checks, original = evaluate_view(model, view, (2, 3, 1), chunk_size=16)
        self.assertTrue(rows)
        self.assertEqual(original.shape, (10, 3))
        self.assertEqual(checks["original_scans"], 1)
        self.assertTrue(all(row["current_local_within_tolerance"] for row in rows))
        self.assertTrue(all(row["availability_trajectory_exact"] for row in rows))
        self.assertTrue(any(abs(row["delta_ce"]) > 1e-8 for row in rows))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_empty_intervention_arrays_have_stable_nonobject_schema(self):
        values = rows_to_arrays([], 3)
        self.assertEqual(values["logits_original"].shape, (0, 3))
        self.assertEqual(values["label"].dtype, np.int64)
        self.assertEqual(values["current_local_exact"].dtype, bool)
        self.assertNotEqual(values["conversation_id"].dtype, object)

    def run_rate(self, prefix=""):
        with patch("gcnet_missing_m3.train_gcnet._prepare_view", return_value=self.view(prefix)):
            return evaluate_rate(self.model(), [[]], SimpleNamespace(dataset="IEMOCAPFour", latent_dim=6), (2, 3, 1), None)

    def test_rate_reuses_p0_full_logits_and_exact_source_parity(self):
        summary, values, skipped, originals = self.run_rate()
        self.assertTrue(summary["checks"]["p0_original_logits_exact"])
        self.assertTrue(summary["checks"]["current_local_tolerance_all"])
        self.assertEqual(sum(len(originals["labels"]) for _ in [0]), 10)
        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory)/"source.npz"
            np.savez_compressed(prediction, labels=originals["labels"], predictions=originals["pred_LBG"], availability=originals["current_availability"])
            with patch("gcnet_missing_m3.train_gcnet._prepare_view", return_value=self.view()):
                checked, _, _, _ = evaluate_rate(self.model(), [[]], SimpleNamespace(dataset="IEMOCAPFour", latent_dim=6), (2, 3, 1), None, prediction)
            self.assertTrue(checked["original_p0"]["checks"]["source_prediction_exact"])

    def test_meld_legacy_path_materializes_same_fixed_masks_in_p0_and_p1(self):
        view = self.view()
        data = [None]*10
        data[7], data[-1] = view["umask"], view["conversation_ids"]
        generated = []

        def generate(conversation, length, valid_length, side, epoch):
            self.assertEqual(epoch, 0)
            generated.append((conversation, length, valid_length, side))
            mask = np.zeros((length, 3), dtype=np.uint8)
            mask[:valid_length] = 1
            return SimpleNamespace(availability=mask)

        schedule = SimpleNamespace(generate=generate)
        with patch("gcnet_missing_m3.train_gcnet._move_batch", return_value=data), patch("gcnet_missing_m3.train_gcnet._prepare_view_from_primary_masks", return_value=view) as prepare:
            result, _, _, _ = evaluate_rate(self.model(), [data], SimpleNamespace(dataset="MELD", latent_dim=6),
                (2, 3, 1), schedule, source_missing_mask_protocol="gcnet-legacy-fixed-v1")
        self.assertTrue(result["checks"]["p0_original_logits_exact"])
        self.assertEqual(len(generated), 8)
        self.assertEqual(generated[:4], generated[4:])
        self.assertEqual(prepare.call_count, 2)
        for index in (1, 2):
            torch.testing.assert_close(prepare.call_args_list[0].args[index], prepare.call_args_list[1].args[index], rtol=0, atol=0)

    def test_no_eligible_histories_are_valid_zero_coverage_outputs(self):
        view = self.view()
        for name in ("incomplete", "availability"):
            view[name] = view[name][:1, :1]
        for name in ("qmask", "umask", "labels"):
            view[name] = view[name][:1, :1]
        view.update(lengths=[1], conversation_ids=["start"])
        with patch("gcnet_missing_m3.train_gcnet._prepare_view", return_value=view):
            result, values, skipped, originals = evaluate_rate(self.model(), [[]],
                SimpleNamespace(dataset="IEMOCAPFour", latent_dim=6), (2, 3, 1), None)
        self.assertEqual(len(values["label"]), 0)
        self.assertEqual(len(skipped), 6)
        self.assertEqual(originals["labels"].shape, (1,))
        for modality in MODALITIES:
            for arm in ARMS:
                self.assertEqual(result["modalities"][modality][arm]["coverage"], 0)
                self.assertIsNone(result["modalities"][modality][arm]["mean_delta_ce"])

    def fixture_run(self, root, dataset, seed, fold, rates=("0.0",)):
        directory = root/dataset/f"seed_{seed}"/f"fold_{fold}"
        directory.mkdir(parents=True)
        result, values, skipped, originals = self.run_rate(prefix=f"fold{fold}_")
        np.savez_compressed(directory/"pairs.npz", **values)
        np.savez_compressed(directory/"original.npz", **originals)
        (directory/"skips.json").write_text(json.dumps(skipped))
        summary = dict(status="complete", training=False, diagnostic="P1_history_content", dataset=dataset, seed=seed, fold=fold,
            rates={rate: dict(intervention_file="pairs.npz", original_file="original.npz", skipped_file="skips.json", checkpoint_sha256="fixture", **result) for rate in rates})
        (directory/"SUMMARY.json").write_text(json.dumps(summary))
        return directory

    def test_aggregation_pools_folds_then_seeds_and_keeps_rates_datasets_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = [self.fixture_run(root, "IEMOCAPFour", seed, fold, ("0.0", "0.7")) for seed in (66, 67) for fold in range(1, 6)]
            inputs.append(self.fixture_run(root, "MELD", 66, 1))
            result = aggregate(inputs)
            self.assertEqual(set(result["datasets"]), {"IEMOCAPFour", "MELD"})
            dataset = result["datasets"]["IEMOCAPFour"]
            self.assertEqual(set(dataset["rates"]), {"0.0", "0.7"})
            self.assertEqual(dataset["rates"]["0.0"]["by_seed"]["66"]["targets"], 50)
            self.assertEqual(dataset["rates"]["0.0"]["across_seed"]["A"]["same_speaker"]["mean_delta_ce"]["seed_count"], 2)
            self.assertIn("MELD", report_markdown(result))
            output = root/"aggregate"
            with patch("sys.argv", ["summary", "--input-root", str(root), "--output", str(output)]):
                summarize_main()
            self.assertTrue((output/"MELD"/"SUMMARY.csv").is_file())
            with patch("sys.argv", ["summary", "--input-root", str(root), "--output", str(output)]):
                with self.assertRaises(FileExistsError):
                    summarize_main()

    def test_aggregation_rejects_missing_folds_duplicates_and_missing_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.fixture_run(root, "IEMOCAPSix", 66, 1)
            with self.assertRaisesRegex(ValueError, "expected"):
                aggregate([first])
            self.assertTrue(aggregate([first], allow_partial=True)["allow_partial"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                aggregate([first, first], allow_partial=True)
            second = self.fixture_run(root, "IEMOCAPSix", 67, 1, ("0.0", "0.7"))
            with self.assertRaisesRegex(ValueError, "expected"):
                aggregate([first, second])

    def test_aggregation_recomputes_effects_selection_and_exhaustive_skips(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.fixture_run(Path(temporary), "MELD", 66, 1)
            rate = json.loads((directory/"SUMMARY.json").read_text())["rates"]["0.0"]
            load_checked(directory, rate)
            with np.load(directory/"pairs.npz", allow_pickle=False) as loaded:
                original = {name: loaded[name] for name in loaded.files}
            for name in ("delta_ce", "donor_position", "label", "current_input_exact"):
                corrupted = {key: values.copy() for key, values in original.items()}
                corrupted[name][0] = False if name == "current_input_exact" else corrupted[name][0]+1
                np.savez_compressed(directory/"pairs.npz", **corrupted)
                with self.assertRaises((ValueError, AssertionError)):
                    load_checked(directory, rate)
            np.savez_compressed(directory/"pairs.npz", **original)
            skipped = json.loads((directory/"skips.json").read_text())
            (directory/"skips.json").write_text(json.dumps(skipped[:-1]))
            with self.assertRaisesRegex(ValueError, "missing skipped"):
                load_checked(directory, rate)

    def test_dispatch_is_33_sources_with_required_identity_and_prediction_preflight(self):
        tasks = build_tasks(Path("/iemocap"), Path("/meld"), Path("/new"))
        self.assertEqual(len(tasks), 33)
        self.assertEqual(sum(task["dataset"] == "meld" for task in tasks), 3)
        self.assertEqual(len({task["output"] for task in tasks}), 33)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/"source"
            source.mkdir()
            config = dict(dataset="IEMOCAPFour", seed=66, fold=1, backbone_type="osram", osram_readout_fusion="flat", training_objective="emotion-only")
            (source/"config.json").write_text(json.dumps(config))
            task = dict(dataset="iemocap4", seed=66, fold=1, source=str(source))
            with self.assertRaises(FileNotFoundError):
                preflight([task], [0.0])
            (source/"best_miss_0p0.pt").touch()
            with self.assertRaises(FileNotFoundError):
                preflight([task], [0.0])
            (source/"predictions_miss_0p0.npz").touch()
            preflight([task], [0.0])
            task["seed"] = 67
            with self.assertRaisesRegex(ValueError, "identity"):
                preflight([task], [0.0])

    def test_existing_outputs_and_invalid_rates_rejected_before_data_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            common = ["--source", "/absent", "--dataset-root", "/absent", "--output", temporary]
            with patch("sys.argv", ["evaluate", *common]):
                with self.assertRaises(FileExistsError):
                    main()
            common[-1] = str(Path(temporary)/"new")
            with patch("sys.argv", ["evaluate", *common, "--rates", "0.1", "0.1"]):
                with self.assertRaisesRegex(ValueError, "unique"):
                    main()
            with patch("sys.argv", ["dispatch", "--dataset-root", "/absent", "--output-root", temporary]):
                with self.assertRaises(FileExistsError):
                    dispatch_main()


if __name__ == "__main__":
    unittest.main()
