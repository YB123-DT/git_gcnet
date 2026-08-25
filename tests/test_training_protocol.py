import os
import pickle
import random
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_gcnet
from train_gcnet import seed_everything, select_epoch_mask_banks


class TrainingProtocolTests(unittest.TestCase):
    def test_relation_track_routing_cli_defaults_to_early_and_accepts_diagonal(self):
        parser = train_gcnet.create_argument_parser()

        self.assertEqual(parser.parse_args([]).relation_track_routing, "early")
        self.assertEqual(
            parser.parse_args(
                ["--relation-track-routing", "diagonal"]
            ).relation_track_routing,
            "diagonal",
        )
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--relation-track-routing", "dense"])

    def test_second_graph_aggregation_cli_defaults_accepts_candidates_and_rejects_unknown(self):
        parser = train_gcnet.create_argument_parser()

        self.assertEqual(parser.parse_args([]).second_graph_aggregation, "add")
        for mode in ("add", "genagg", "soft_medoid", "ssma"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    parser.parse_args(
                        ["--second-graph-aggregation", mode]
                    ).second_graph_aggregation,
                    mode,
                )
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    ["--second-graph-aggregation", "median"]
                )

    def test_context_cli_defaults_preserve_original_bilstm_protocol(self):
        args = train_gcnet.create_argument_parser().parse_args([])

        self.assertEqual(args.pre_graph_context, "bilstm")
        self.assertEqual(args.post_graph_context, "bilstm")
        self.assertEqual(args.branch_fusion, "addition")

    def test_branch_fusion_cli_accepts_only_registered_modes(self):
        parser = train_gcnet.create_argument_parser()
        for mode in ("addition", "mask_sequence_aff"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    parser.parse_args(["--branch-fusion", mode]).branch_fusion,
                    mode,
                )
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--branch-fusion", "attention"])

    def test_context_cli_accepts_each_linear_ablation_independently(self):
        args = train_gcnet.create_argument_parser().parse_args(
            [
                "--pre-graph-context",
                "linear",
                "--post-graph-context",
                "linear",
            ]
        )

        self.assertEqual(args.pre_graph_context, "linear")
        self.assertEqual(args.post_graph_context, "linear")

    def test_context_cli_rejects_values_outside_factorial_protocol(self):
        parser = train_gcnet.create_argument_parser()
        for option in ("--pre-graph-context", "--post-graph-context"):
            with self.subTest(option=option):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args([option, "gru"])

    def test_build_model_propagates_context_modes_to_graph_modules(self):
        for pre_context in ("bilstm", "linear"):
            for post_context in ("bilstm", "linear"):
                with self.subTest(pre=pre_context, post=post_context):
                    args = Namespace(
                        hidden=4,
                        base_model="LSTM",
                        n_speakers=2,
                        windowp=1,
                        windowf=1,
                        n_classes=6,
                        dropout=0.0,
                        time_attn=False,
                        no_cuda=True,
                        graph_conv_variant="original",
                        pre_graph_context=pre_context,
                        post_graph_context=post_context,
                        branch_fusion="mask_sequence_aff",
                    )

                    model = train_gcnet.build_model(args, adim=2, tdim=2, vdim=2)

                    self.assertEqual(model.pre_graph_context, pre_context)
                    self.assertEqual(model.branch_fusion, "mask_sequence_aff")
                    self.assertIsInstance(model.pre_graph_projection, nn.Linear)
                    for branch in (
                        model.graph_net_temporal,
                        model.graph_net_speaker,
                    ):
                        self.assertEqual(branch.post_graph_context, post_context)
                        self.assertIsInstance(branch.grufusion, nn.LSTM)

    def test_build_model_legacy_namespace_defaults_to_addition_identity(self):
        legacy_args = Namespace(
            hidden=4,
            base_model="LSTM",
            n_speakers=2,
            windowp=1,
            windowf=1,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
            graph_conv_variant="original",
            pre_graph_context="bilstm",
            post_graph_context="bilstm",
        )
        explicit_args = Namespace(
            **vars(legacy_args), branch_fusion="addition"
        )

        torch.manual_seed(181)
        legacy_model = train_gcnet.build_model(
            legacy_args, adim=2, tdim=2, vdim=2
        )
        legacy_rng = torch.get_rng_state().clone()
        torch.manual_seed(181)
        explicit_model = train_gcnet.build_model(
            explicit_args, adim=2, tdim=2, vdim=2
        )

        self.assertEqual(legacy_model.branch_fusion, "addition")
        self.assertTrue(torch.equal(legacy_rng, torch.get_rng_state()))
        for name, expected in legacy_model.state_dict().items():
            self.assertTrue(
                torch.equal(expected, explicit_model.state_dict()[name]), name
            )

    def test_build_model_legacy_namespace_defaults_to_add_second_graph_identity(self):
        legacy_args = Namespace(
            hidden=4,
            base_model="LSTM",
            n_speakers=2,
            windowp=1,
            windowf=1,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
            graph_conv_variant="original",
            pre_graph_context="bilstm",
            post_graph_context="bilstm",
            branch_fusion="addition",
        )
        explicit_args = Namespace(
            **vars(legacy_args), second_graph_aggregation="add"
        )

        torch.manual_seed(281)
        legacy_model = train_gcnet.build_model(
            legacy_args, adim=2, tdim=2, vdim=2
        )
        legacy_rng = torch.get_rng_state().clone()
        torch.manual_seed(281)
        explicit_model = train_gcnet.build_model(
            explicit_args, adim=2, tdim=2, vdim=2
        )

        self.assertEqual(legacy_model.second_graph_aggregation, "add")
        self.assertEqual(explicit_model.second_graph_aggregation, "add")
        self.assertTrue(torch.equal(legacy_rng, torch.get_rng_state()))
        for name, expected in legacy_model.state_dict().items():
            self.assertTrue(
                torch.equal(expected, explicit_model.state_dict()[name]), name
            )

    def test_build_model_legacy_namespace_defaults_to_early_relation_routing(self):
        legacy_args = Namespace(
            hidden=4,
            base_model="LSTM",
            n_speakers=2,
            windowp=1,
            windowf=1,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
            graph_conv_variant="original",
            pre_graph_context="bilstm",
            post_graph_context="bilstm",
            branch_fusion="addition",
            second_graph_aggregation="add",
        )
        explicit_args = Namespace(
            **vars(legacy_args), relation_track_routing="early"
        )

        torch.manual_seed(381)
        legacy_model = train_gcnet.build_model(
            legacy_args, adim=2, tdim=2, vdim=2
        )
        legacy_rng = torch.get_rng_state().clone()
        torch.manual_seed(381)
        explicit_model = train_gcnet.build_model(
            explicit_args, adim=2, tdim=2, vdim=2
        )

        self.assertEqual(legacy_model.relation_track_routing, "early")
        self.assertEqual(explicit_model.relation_track_routing, "early")
        self.assertTrue(torch.equal(legacy_rng, torch.get_rng_state()))
        self.assertEqual(
            list(legacy_model.state_dict()), list(explicit_model.state_dict())
        )
        for name, expected in legacy_model.state_dict().items():
            self.assertTrue(
                torch.equal(expected, explicit_model.state_dict()[name]), name
            )

    def test_build_model_propagates_diagonal_relation_routing_and_rejects_confounding(self):
        base_args = dict(
            hidden=4,
            base_model="LSTM",
            n_speakers=2,
            windowp=1,
            windowf=1,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
            graph_conv_variant="original",
            pre_graph_context="bilstm",
            post_graph_context="bilstm",
            branch_fusion="addition",
            second_graph_aggregation="add",
            relation_track_routing="diagonal",
        )

        model = train_gcnet.build_model(
            Namespace(**base_args), adim=2, tdim=2, vdim=2
        )
        self.assertEqual(model.relation_track_routing, "diagonal")
        self.assertEqual(
            model.graph_net_temporal.relation_track_routing, "diagonal"
        )
        self.assertEqual(
            model.graph_net_speaker.relation_track_routing, "diagonal"
        )

        with self.assertRaisesRegex(
            ValueError, "diagonal relation-track routing"
        ):
            train_gcnet.build_model(
                Namespace(**{
                    **base_args,
                    "second_graph_aggregation": "ssma",
                }),
                adim=2,
                tdim=2,
                vdim=2,
            )

    def test_build_model_propagates_second_graph_aggregation_candidates(self):
        base_args = dict(
            hidden=4,
            base_model="LSTM",
            n_speakers=2,
            windowp=1,
            windowf=1,
            n_classes=6,
            dropout=0.0,
            time_attn=False,
            no_cuda=True,
            graph_conv_variant="original",
            pre_graph_context="bilstm",
            post_graph_context="bilstm",
            branch_fusion="addition",
        )

        for selector in ("genagg", "soft_medoid", "ssma"):
            with self.subTest(selector=selector):
                model = train_gcnet.build_model(
                    Namespace(
                        **base_args, second_graph_aggregation=selector
                    ),
                    adim=2,
                    tdim=2,
                    vdim=2,
                )
                self.assertEqual(model.second_graph_aggregation, selector)

    def test_second_graph_auxiliary_loss_preserves_old_loss_object_unless_training_genagg(self):
        class AuxiliaryModel(nn.Module):
            def __init__(self, selector, relation_track_routing="early"):
                super().__init__()
                self.second_graph_aggregation = selector
                self.relation_track_routing = relation_track_routing
                self.temporal_auxiliary = nn.Parameter(torch.tensor(2.0))
                self.speaker_auxiliary = nn.Parameter(torch.tensor(3.0))
                self.auxiliary_calls = 0

            def second_graph_auxiliary_loss(self):
                self.auxiliary_calls += 1
                return self.temporal_auxiliary + self.speaker_auxiliary

        for selector, routing, train in (
            ("add", "early", True),
            ("add", "diagonal", True),
            ("soft_medoid", "early", True),
            ("ssma", "early", True),
            ("genagg", "early", False),
        ):
            with self.subTest(
                selector=selector, routing=routing, train=train
            ):
                model = AuxiliaryModel(selector, routing)
                base_loss = torch.tensor(7.0, requires_grad=True)
                result = train_gcnet.add_second_graph_auxiliary_loss(
                    base_loss, model, train=train
                )
                self.assertIs(result, base_loss)
                self.assertEqual(result.item(), 7.0)
                self.assertEqual(model.auxiliary_calls, 0)

        model = AuxiliaryModel("genagg")
        base_loss = torch.tensor(7.0, requires_grad=True)
        result = train_gcnet.add_second_graph_auxiliary_loss(
            base_loss, model, train=True
        )
        self.assertEqual(result.item(), 12.0)
        self.assertEqual(model.auxiliary_calls, 1)
        result.backward()
        self.assertEqual(base_loss.grad.item(), 1.0)
        self.assertEqual(model.temporal_auxiliary.grad.item(), 1.0)
        self.assertEqual(model.speaker_auxiliary.grad.item(), 1.0)

    def test_result_suffix_legacy_namespace_defaults_to_addition(self):
        legacy_args = Namespace(
            dataset="IEMOCAPSix",
            base_model="LSTM",
            graph_conv_variant="original",
            fold_index=1,
            seed=100,
            mask_type="constant-0.1",
            pre_graph_context="bilstm",
            post_graph_context="bilstm",
        )
        explicit_args = Namespace(
            **vars(legacy_args), branch_fusion="addition"
        )

        self.assertEqual(
            train_gcnet.build_result_suffix(legacy_args),
            train_gcnet.build_result_suffix(explicit_args),
        )
        self.assertIn(
            "_branchfusion:addition",
            train_gcnet.build_result_suffix(legacy_args),
        )

    def test_second_graph_result_identity_preserves_add_and_tags_candidates(self):
        legacy_args = Namespace(
            dataset="IEMOCAPSix",
            base_model="LSTM",
            graph_conv_variant="original",
            fold_index=1,
            seed=100,
            mask_type="constant-0.1",
            pre_graph_context="bilstm",
            post_graph_context="bilstm",
            branch_fusion="addition",
        )
        explicit_add = Namespace(
            **vars(legacy_args), second_graph_aggregation="add"
        )
        expected_suffix = (
            "iemocapsix_GraphLSTM_variant:original_fold:1_seed:100_mask:0.1"
            "_prectx:bilstm_postctx:bilstm_branchfusion:addition"
        )
        expected_filename = (
            "iemocapsix_original_addition_f1_s100_m0p1_123.5.npz"
        )

        self.assertEqual(
            train_gcnet.build_result_suffix(legacy_args), expected_suffix
        )
        self.assertEqual(
            train_gcnet.build_result_suffix(explicit_add), expected_suffix
        )
        self.assertEqual(
            train_gcnet.build_archive_filename(legacy_args, 123.5),
            expected_filename,
        )
        self.assertEqual(
            train_gcnet.build_archive_filename(explicit_add, 123.5),
            expected_filename,
        )

        candidate_suffixes = {
            selector: train_gcnet.build_result_suffix(
                Namespace(
                    **vars(legacy_args), second_graph_aggregation=selector
                )
            )
            for selector in ("genagg", "soft_medoid", "ssma")
        }
        candidate_filenames = {
            selector: train_gcnet.build_archive_filename(
                Namespace(
                    **vars(legacy_args), second_graph_aggregation=selector
                ),
                123.5,
            )
            for selector in ("genagg", "soft_medoid", "ssma")
        }
        self.assertEqual(len(set(candidate_suffixes.values())), 3)
        self.assertEqual(len(set(candidate_filenames.values())), 3)
        for selector in ("genagg", "soft_medoid", "ssma"):
            self.assertIn(selector, candidate_suffixes[selector])
            self.assertIn(selector, candidate_filenames[selector])
            self.assertNotEqual(candidate_suffixes[selector], expected_suffix)
            self.assertNotEqual(
                candidate_filenames[selector], expected_filename
            )

    def test_relation_track_result_identity_preserves_early_and_tags_diagonal(self):
        legacy_args = Namespace(
            dataset="IEMOCAPSix",
            base_model="LSTM",
            graph_conv_variant="original",
            fold_index=5,
            seed=66,
            mask_type="constant-0.7",
            pre_graph_context="bilstm",
            post_graph_context="bilstm",
            branch_fusion="addition",
            second_graph_aggregation="add",
        )
        explicit_early = Namespace(
            **vars(legacy_args), relation_track_routing="early"
        )
        diagonal = Namespace(
            **vars(legacy_args), relation_track_routing="diagonal"
        )

        legacy_suffix = train_gcnet.build_result_suffix(legacy_args)
        legacy_filename = train_gcnet.build_archive_filename(legacy_args, 123.5)
        self.assertEqual(
            train_gcnet.build_result_suffix(explicit_early), legacy_suffix
        )
        self.assertEqual(
            train_gcnet.build_archive_filename(explicit_early, 123.5),
            legacy_filename,
        )
        self.assertIn(
            "relationtrack:diagonal",
            train_gcnet.build_result_suffix(diagonal),
        )
        self.assertIn(
            "relationtrack_diagonal",
            train_gcnet.build_archive_filename(diagonal, 123.5),
        )
        self.assertNotEqual(
            train_gcnet.build_archive_filename(diagonal, 123.5),
            legacy_filename,
        )

    def test_archive_filename_keeps_only_compact_run_identity(self):
        args = Namespace(
            dataset="IEMOCAPSix",
            graph_conv_variant="original",
            branch_fusion="mask_sequence_aff",
            fold_index=5,
            seed=66,
            mask_type="constant-0.7",
        )

        name = train_gcnet.build_archive_filename(args, timestamp=123.5)

        self.assertEqual(
            name,
            "iemocapsix_original_mask_sequence_aff_f5_s66_m0p7_123.5.npz",
        )
        self.assertNotIn("prectx", name)
        self.assertNotIn("features", name)

    def test_short_mask_sequence_aff_run_records_full_provenance(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "results"
            feature_root = data_root / "features"
            feature_names = ("a", "t", "v")
            for feature_name in feature_names:
                (feature_root / feature_name).mkdir(parents=True)

            video_ids = {}
            video_labels = {}
            video_speakers = {}
            video_sentences = {}
            videos = []
            for session in range(1, 6):
                video = "Ses0{}_fixture".format(session)
                utterances = [
                    "{}_utt0".format(video),
                    "{}_utt1".format(video),
                ]
                videos.append(video)
                video_ids[video] = utterances
                video_labels[video] = [session % 6, (session + 1) % 6]
                video_speakers[video] = ["F", "M"]
                video_sentences[video] = ["first", "second"]
                for feature_index, feature_name in enumerate(feature_names):
                    for utterance_index, utterance in enumerate(utterances):
                        values = np.array(
                            [session, feature_index + 1, utterance_index + 1],
                            dtype=np.float32,
                        )
                        np.save(feature_root / feature_name / utterance, values)

            label_path = data_root / "IEMOCAP_features_raw_6way.pkl"
            with label_path.open("wb") as handle:
                pickle.dump(
                    (
                        video_ids,
                        video_labels,
                        video_speakers,
                        video_sentences,
                        set(videos[:4]),
                        set(videos[4:]),
                    ),
                    handle,
                )

            environment = os.environ.copy()
            environment["GCNET_CACHE_ROOT"] = str(root / "cache")
            completed = subprocess.run(
                [
                    sys.executable,
                    "train_gcnet.py",
                    "--dataset",
                    "IEMOCAPSix",
                    "--data-root",
                    str(data_root),
                    "--audio-feature",
                    "a",
                    "--text-feature",
                    "t",
                    "--video-feature",
                    "v",
                    "--base-model",
                    "LSTM",
                    "--graph-conv-variant",
                    "original",
                    "--pre-graph-context",
                    "linear",
                    "--post-graph-context",
                    "linear",
                    "--branch-fusion",
                    "mask_sequence_aff",
                    "--hidden",
                    "2",
                    "--windowp",
                    "1",
                    "--windowf",
                    "1",
                    "--dropout",
                    "0",
                    "--batch-size",
                    "4",
                    "--epochs",
                    "1",
                    "--fold-index",
                    "1",
                    "--num-threads",
                    "1",
                    "--mask-type",
                    "constant-0.1",
                    "--mask-bank-root",
                    str(root / "mask-banks"),
                    "--output-dir",
                    str(output_root),
                    "--allow-short-run",
                    "--no-cuda",
                ],
                cwd=project_root / "gcnet",
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            archives = list(output_root.glob("*.npz"))
            self.assertEqual(len(archives), 1)
            archive_path = archives[0]
            self.assertTrue(
                archive_path.name.startswith(
                    "iemocapsix_original_mask_sequence_aff_f1_s100_m0p1_"
                ),
                archive_path.name,
            )
            model = train_gcnet.GraphModel(
                "LSTM",
                3,
                3,
                3,
                2,
                1,
                n_speakers=2,
                window_past=1,
                window_future=1,
                n_classes=6,
                dropout=0.0,
                time_attn=False,
                no_cuda=True,
                graph_conv_variant="original",
                pre_graph_context="linear",
                post_graph_context="linear",
                branch_fusion="mask_sequence_aff",
            )
            with np.load(archive_path, allow_pickle=True) as archive:
                stored_args = archive["args"].item()
                self.assertTrue(bool(archive["smoke_only"]))
                self.assertEqual(stored_args.pre_graph_context, "linear")
                self.assertEqual(stored_args.post_graph_context, "linear")
                self.assertEqual(stored_args.branch_fusion, "mask_sequence_aff")
                self.assertEqual(
                    int(archive["parameter_count"]),
                    sum(parameter.numel() for parameter in model.parameters()),
                )
                self.assertEqual(
                    int(archive["selected_path_parameter_count"]),
                    model.selected_path_parameter_count(),
                )

    def test_epoch_mask_selection_uses_distinct_explicit_stages(self):
        bundle = {
            "train": ("train-zero", "train-one"),
            "validation": "fixed-validation",
            "test": "fixed-test",
        }

        self.assertEqual(
            select_epoch_mask_banks(bundle, 0),
            ("train-zero", "fixed-validation", "fixed-test"),
        )
        self.assertEqual(
            select_epoch_mask_banks(bundle, 1),
            ("train-one", "fixed-validation", "fixed-test"),
        )

    def test_cli_accepts_cp_lecc_graph_convolution_variant(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "train_gcnet.py", "--help"],
            cwd=project_root / "gcnet",
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("cp_lecc", completed.stdout)

    def test_seed_controls_python_numpy_and_torch(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark
        try:
            seed_everything(2025)
            first = (random.random(), np.random.rand(), torch.rand(3))
            seed_everything(2025)
            second = (random.random(), np.random.rand(), torch.rand(3))

            self.assertEqual(first[0], second[0])
            self.assertEqual(first[1], second[1])
            torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark

    def test_seed_uses_official_cuda_reproducibility_settings(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark
        try:
            torch.use_deterministic_algorithms(False)
            seed_everything(66)

            self.assertFalse(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark


if __name__ == "__main__":
    unittest.main()
