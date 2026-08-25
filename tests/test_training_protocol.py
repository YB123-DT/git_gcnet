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
    def test_context_cli_defaults_preserve_original_bilstm_protocol(self):
        args = train_gcnet.create_argument_parser().parse_args([])

        self.assertEqual(args.pre_graph_context, "bilstm")
        self.assertEqual(args.post_graph_context, "bilstm")

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
                    )

                    model = train_gcnet.build_model(args, adim=2, tdim=2, vdim=2)

                    self.assertEqual(model.pre_graph_context, pre_context)
                    self.assertIsInstance(model.pre_graph_projection, nn.Linear)
                    for branch in (
                        model.graph_net_temporal,
                        model.graph_net_speaker,
                    ):
                        self.assertEqual(branch.post_graph_context, post_context)
                        self.assertIsInstance(branch.grufusion, nn.LSTM)

    def test_short_linear_linear_run_records_context_and_parameter_provenance(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "results"
            feature_root = data_root / "features"
            feature_names = ("audio", "text", "video")
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
                    "audio",
                    "--text-feature",
                    "text",
                    "--video-feature",
                    "video",
                    "--base-model",
                    "LSTM",
                    "--graph-conv-variant",
                    "original",
                    "--pre-graph-context",
                    "linear",
                    "--post-graph-context",
                    "linear",
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
            self.assertIn(
                "_prectx:linear_postctx:linear_",
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
            )
            with np.load(archive_path, allow_pickle=True) as archive:
                stored_args = archive["args"].item()
                self.assertTrue(bool(archive["smoke_only"]))
                self.assertEqual(stored_args.pre_graph_context, "linear")
                self.assertEqual(stored_args.post_graph_context, "linear")
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
