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

    def test_result_archive_records_context_identity_and_parameter_counts(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.selected = nn.Linear(3, 2)
                self.bypassed = nn.Linear(5, 4)

            def selected_path_parameter_count(self):
                return sum(parameter.numel() for parameter in self.selected.parameters())

        args = Namespace(
            dataset="IEMOCAPSix",
            base_model="LSTM",
            graph_conv_variant="original",
            fold_index=3,
            seed=66,
            mask_type="constant-0.5",
            pre_graph_context="linear",
            post_graph_context="linear",
        )
        model = TinyModel()
        suffix = train_gcnet.build_result_suffix(args)
        self.assertIn("_prectx:linear_postctx:linear", suffix)

        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / (suffix + ".npz")
            train_gcnet.save_result_archive(
                str(archive_path),
                args=args,
                fold_numbers=[3],
                mask_bank_manifest={"sha256": "trusted"},
                model=model,
                smoke_only=True,
                folder_losswhole=[],
                folder_savewhole=[],
            )
            with np.load(archive_path, allow_pickle=True) as archive:
                stored_args = archive["args"].item()
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
