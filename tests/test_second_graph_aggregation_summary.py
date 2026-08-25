import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from experiments.mpfilm_iemocap6.run_locked_ab import Job, build_command


CANDIDATES = ("genagg", "soft_medoid", "ssma", "rtdr")
STORED_COUNTS = {
    # The inherited pre-refactor Original archive stored its selected path only.
    "original": 34_140_166,
    "genagg": 36_419_934,
    "soft_medoid": 36_419_816,
    "ssma": 37_015_216,
    "rtdr": 36_419_816,
}
SELECTED_COUNTS = {
    "original": 34_140_166,
    "genagg": 34_140_284,
    "soft_medoid": 34_140_166,
    "ssma": 34_735_566,
    "rtdr": 34_140_166,
}

OFFICIAL_PYTHON = Path("/data2/yb/reproduction_envs/gcnet-official/bin/python")
DATA_ROOT = Path(
    "/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset/IEMOCAP"
)
MASK_ROOT = Path("/data2/yb/paper/experiments/mpfilm_iemocap6_20260824/mask_banks")
HISTORICAL_REPOSITORY = Path("/data2/yb/paper/GCNet_cp_lecc_20260824")
CANDIDATE_REPOSITORY = Path("/data2/yb/paper/GCNet_second_graph_candidates")


def _locked_training():
    return {
        "dataset": "IEMOCAPSix",
        "audio_feature": "wav2vec-large-c-UTT",
        "text_feature": "deberta-large-4-UTT",
        "video_feature": "manet_UTT",
        "base_model": "LSTM",
        "windowp": 2,
        "windowf": 2,
        "hidden": 200,
        "lr": 0.001,
        "l2": 0.00001,
        "dropout": 0.5,
        "batch_size": 32,
        "num_threads": 6,
        "epochs": 100,
        "loss_recon": True,
        "reccls_flag": False,
        "lower_bound": False,
        "time_attn": False,
        "fold": 5,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_invocation(formal, payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    name = hashlib.sha256(canonical).hexdigest() + ".json"
    _write_json(formal / "invocations" / name, payload)


def _ensure_provenance(formal, historical, arm):
    repository = HISTORICAL_REPOSITORY if historical else CANDIDATE_REPOSITORY
    manifest = {
        "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"},
        "fold": 5,
        "git": {
            "clean": True,
            "head": (
                "d64fa9b6003d9a37fef5f135ce194fd206baac2a"
                if historical
                else "a" * 40
            ),
        },
        "gpu_names": ["Tesla V100-SXM2-32GB"] * 4,
        "locked_training": _locked_training(),
        "python": {
            "executable": str(OFFICIAL_PYTHON),
            "requested": str(OFFICIAL_PYTHON),
            "version": "3.8.20",
        },
        "roots": {
            "data": str(DATA_ROOT),
            "mask_bank": str(MASK_ROOT),
            "repository": str(repository),
        },
        "stage": "formal",
        "versions": {
            "cuda": "10.2",
            "cudnn": 7605,
            "torch": "1.8.0",
            "torch_geometric": "2.0.1",
        },
    }
    _write_json(formal / "run_manifest.json", manifest)
    if historical:
        invocations = (
            {"arms": ["original"], "rates": [0.5, 0.7], "job_count": 10},
            {"arms": ["original", "full"], "rates": [0.0, 0.1, 0.3], "job_count": 30},
            {"arms": ["original", "full"], "rates": [0.2, 0.4, 0.6], "job_count": 30},
            {"arms": ["full", "cp_lecc"], "rates": [0.5, 0.7], "job_count": 20},
        )
        for item in invocations:
            _write_invocation(
                formal,
                {
                    **item,
                    "fold": 5,
                    "gpus": ["0", "1", "2", "3"],
                    "seeds": [66, 67, 68, 69, 70],
                    "stage": "formal",
                    "workers_per_gpu": 3,
                },
            )
    else:
        phase_arms = ["genagg", "soft_medoid"] if arm in ("genagg", "soft_medoid") else ["ssma", "rtdr"]
        _write_invocation(
            formal,
            {
                "arms": phase_arms,
                "fold": 5,
                "gpus": ["0", "1", "2", "3"],
                "job_count": 12,
                "parallel_arms": True,
                "rates": [0.0, 0.7],
                "seeds": [66, 67, 68],
                "stage": "formal",
                "workers_per_gpu": 3,
            },
        )


def _snapshot(predicted=(0, 1, 2, 3, 4, 5)):
    labels = np.arange(6)
    scores = np.zeros((6, 6), dtype=np.float64)
    scores[np.arange(6), np.asarray(predicted)] = 1.0
    return {"test_labels": [labels], "test_preds": [scores]}


def _args(arm, rate, seed, historical=False, overrides=None):
    values = dict(
        dataset="IEMOCAPSix",
        audio_feature="wav2vec-large-c-UTT",
        text_feature="deberta-large-4-UTT",
        video_feature="manet_UTT",
        base_model="LSTM",
        windowp=2,
        windowf=2,
        hidden=200,
        lr=0.001,
        l2=0.00001,
        dropout=0.5,
        batch_size=32,
        num_threads=6,
        epochs=100,
        loss_recon=True,
        reccls_flag=False,
        lower_bound=False,
        time_attn=False,
        graph_conv_variant="original",
        seed=seed,
        mask_seed=seed,
        mask_type="constant-{:.1f}".format(rate),
        fold_index=5,
    )
    if not historical:
        values.update(
            pre_graph_context="bilstm",
            post_graph_context="bilstm",
            branch_fusion="addition",
            second_graph_aggregation=(arm if arm != "rtdr" else "add"),
            relation_track_routing=("diagonal" if arm == "rtdr" else "early"),
        )
    values.update(overrides or {})
    return Namespace(**values)


def _strip_flag(command, flag):
    command = list(command)
    if flag in command:
        index = command.index(flag)
        del command[index : index + 2]
    return command


def _write_job(
    fold,
    arm,
    rate,
    seed,
    historical=False,
    mask_hash=None,
    predicted=(0, 1, 2, 3, 4, 5),
    arg_overrides=None,
    archive_overrides=None,
):
    formal = fold.parents[3]
    _ensure_provenance(formal, historical, arm)
    saved = fold / "saved"
    saved.mkdir(parents=True)
    values = dict(
        args=np.array(
            _args(arm, rate, seed, historical=historical, overrides=arg_overrides),
            dtype=object,
        ),
        fold_numbers=np.array([5]),
        folder_savewhole=np.array([[7, _snapshot(predicted)]], dtype=object),
        folder_losswhole=np.array(
            [[{"train_loss": [1.0], "val_loss": [1.0], "test_loss": [1.0]} for _ in range(100)]],
            dtype=object,
        ),
        mask_bank_manifest=np.array(
            {
                "sha256": mask_hash or "mask-{}-{}".format(rate, seed),
                "requested_missing_rate": rate,
                "seed": seed,
            },
            dtype=object,
        ),
        smoke_only=np.array(False),
        parameter_count=np.array(STORED_COUNTS[arm]),
        selected_path_parameter_count=np.array(SELECTED_COUNTS[arm]),
    )
    if historical:
        values.pop("selected_path_parameter_count")
    values.update(archive_overrides or {})
    np.savez_compressed(saved / "run.npz", **values)

    job = Job("formal", arm, rate, seed, fold)
    repository = HISTORICAL_REPOSITORY if historical else CANDIDATE_REPOSITORY
    command_data_root = (
        HISTORICAL_REPOSITORY / "dataset" / "IEMOCAP"
        if historical and rate in (0.5, 0.7)
        else DATA_ROOT
    )
    command = build_command(
        job, OFFICIAL_PYTHON, repository, command_data_root, MASK_ROOT
    )
    if historical:
        command = _strip_flag(command, "--branch-fusion")
    payload = {
        "stage": "formal",
        "arm": arm,
        "missing_rate": rate,
        "seed": seed,
        "fold": 5,
        "gpu": "0",
        "command": command,
    }
    (fold / "command.json").write_text(json.dumps(payload), encoding="utf-8")
    (fold / "status.json").write_text(
        json.dumps(
            {"status": "success", "return_code": 0, "elapsed_seconds": 12.5}
        ),
        encoding="utf-8",
    )
    (fold / "train.log").write_text(
        "\n".join(
            ["epoch:{}; train_fscore:0".format(index) for index in range(1, 101)]
            + ["SMOKE_ONLY=False", "Finish fold 5", "save results in run.npz"]
        ),
        encoding="utf-8",
    )
    return fold


def _fold(root, arm, rate, seed):
    return (
        root
        / arm
        / "miss_{}".format("{:.1f}".format(rate).replace(".", "p"))
        / "seed_{}".format(seed)
        / "fold_5"
    )


class TrustedJobTests(unittest.TestCase):
    def test_collects_each_current_candidate_with_exact_counts_and_runtime(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        for arm in CANDIDATES:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                fold = _write_job(_fold(Path(tmp) / "formal", arm, 0.7, 68), arm, 0.7, 68)
                row = collect_job(fold, arm, 0.7, 68)
                self.assertEqual(row["weighted_f1"], 1.0)
                self.assertEqual(row["class_coverage"], 6)
                self.assertEqual(row["parameter_count"], STORED_COUNTS[arm])
                self.assertEqual(
                    row["selected_path_parameter_count"], SELECTED_COUNTS[arm]
                )
                self.assertEqual(row["runtime_seconds"], 12.5)

    def test_historical_original_allows_only_absent_new_defaults(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        with tempfile.TemporaryDirectory() as tmp:
            fold = _write_job(
                _fold(Path(tmp) / "formal", "original", 0.0, 66), "original", 0.0, 66, historical=True
            )
            row = collect_job(fold, "original", 0.0, 66, historical_original=True)
        self.assertEqual(row["selected_path_parameter_count"], 34_140_166)
        self.assertEqual(row["branch_fusion"], "addition")
        self.assertEqual(row["second_graph_aggregation"], "add")
        self.assertEqual(row["relation_track_routing"], "early")

    def test_historical_original_rejects_present_nondefault_new_field(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        for field, bad in (
            ("pre_graph_context", "linear"),
            ("post_graph_context", "linear"),
            ("branch_fusion", "mask_sequence_aff"),
            ("second_graph_aggregation", "genagg"),
            ("relation_track_routing", "diagonal"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                fold = _write_job(
                    _fold(Path(tmp) / "formal", "original", 0.0, 66),
                    "original",
                    0.0,
                    66,
                    historical=True,
                    arg_overrides={field: bad},
                )
                with self.assertRaisesRegex(ValueError, field):
                    collect_job(
                        fold, "original", 0.0, 66, historical_original=True
                    )

    def test_rejects_each_locked_archive_provenance_family(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        expected_locked = {
            "dataset": "IEMOCAPSix",
            "audio_feature": "wav2vec-large-c-UTT",
            "text_feature": "deberta-large-4-UTT",
            "video_feature": "manet_UTT",
            "base_model": "LSTM",
            "windowp": 2,
            "windowf": 2,
            "hidden": 200,
            "lr": 0.001,
            "l2": 0.00001,
            "dropout": 0.5,
            "batch_size": 32,
            "num_threads": 6,
            "epochs": 100,
            "loss_recon": True,
            "reccls_flag": False,
            "lower_bound": False,
            "time_attn": False,
        }
        mutations = {
            "graph_conv_variant": {
                "arg_overrides": {"graph_conv_variant": "full"}
            },
            "pre_graph_context": {"arg_overrides": {"pre_graph_context": "linear"}},
            "post_graph_context": {"arg_overrides": {"post_graph_context": "linear"}},
            "branch_fusion": {"arg_overrides": {"branch_fusion": "mask_sequence_aff"}},
            "second_graph_aggregation": {
                "arg_overrides": {"second_graph_aggregation": "add"}
            },
            "relation_track_routing": {
                "arg_overrides": {"relation_track_routing": "diagonal"}
            },
            "seed": {"arg_overrides": {"seed": 67}},
            "mask_seed": {"arg_overrides": {"mask_seed": 67}},
            "mask_type": {"arg_overrides": {"mask_type": "constant-0.1"}},
            "fold_index": {"arg_overrides": {"fold_index": 4}},
            "fold_numbers": {"archive_overrides": {"fold_numbers": np.array([4])}},
            "smoke_only": {"archive_overrides": {"smoke_only": np.array(True)}},
            "parameter_count": {
                "archive_overrides": {"parameter_count": np.array(1)}
            },
            "selected_path_parameter_count": {
                "archive_overrides": {
                    "selected_path_parameter_count": np.array(1)
                }
            },
            "requested_missing_rate": {
                "archive_overrides": {
                    "mask_bank_manifest": np.array(
                        {"sha256": "x", "requested_missing_rate": 0.1, "seed": 66},
                        dtype=object,
                    )
                }
            },
            "manifest_seed": {
                "archive_overrides": {
                    "mask_bank_manifest": np.array(
                        {"sha256": "x", "requested_missing_rate": 0.0, "seed": 67},
                        dtype=object,
                    )
                }
            },
        }
        for field, expected in expected_locked.items():
            bad = (not expected) if isinstance(expected, bool) else (
                expected + 1 if isinstance(expected, (int, float)) else "wrong"
            )
            mutations[field] = {"arg_overrides": {field: bad}}
        for message, options in mutations.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                fold = _write_job(
                    _fold(Path(tmp) / "formal", "genagg", 0.0, 66), "genagg", 0.0, 66, **options
                )
                with self.assertRaisesRegex(ValueError, message):
                    collect_job(fold, "genagg", 0.0, 66)

    def test_rejects_command_status_log_lock_and_unsafe_archive_drift(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        mutations = {
            "command.json mismatch": lambda fold: (
                (fold / "command.json").write_text(
                    (fold / "command.json").read_text().replace("constant-0.0", "constant-0.1")
                )
            ),
            "return_code": lambda fold: (fold / "status.json").write_text(
                json.dumps({"status": "success", "return_code": 1})
            ),
            "100 epoch": lambda fold: (fold / "train.log").write_text("epoch:1"),
            "active or stale lock": lambda fold: (fold / ".active.lock").write_text("x"),
            "exactly one": lambda fold: (fold / "saved" / "extra.npz").write_bytes(b"x"),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                fold = _write_job(_fold(Path(tmp) / "formal", "genagg", 0.0, 66), "genagg", 0.0, 66)
                mutate(fold)
                with self.assertRaisesRegex(ValueError, message):
                    collect_job(fold, "genagg", 0.0, 66)

        with tempfile.TemporaryDirectory() as tmp:
            fold = _write_job(_fold(Path(tmp) / "formal", "genagg", 0.0, 66), "genagg", 0.0, 66)
            archive = fold / "saved" / "run.npz"
            archive.unlink()
            archive.symlink_to(Path(tmp) / "outside.npz")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                collect_job(fold, "genagg", 0.0, 66)

    def test_historical_command_rejects_nondefault_new_flag(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        with tempfile.TemporaryDirectory() as tmp:
            fold = _write_job(
                _fold(Path(tmp) / "formal", "original", 0.0, 66), "original", 0.0, 66, historical=True
            )
            payload = json.loads((fold / "command.json").read_text())
            payload["command"].extend(("--branch-fusion", "mask_sequence_aff"))
            (fold / "command.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "branch-fusion"):
                collect_job(fold, "original", 0.0, 66, historical_original=True)

    def test_rejects_manifest_and_invocation_drift_before_opening_npz(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        mutations = {
            "git head": lambda manifest: manifest["git"].update(head="b" * 40),
            "git clean": lambda manifest: manifest["git"].update(clean=False),
            "python executable": lambda manifest: manifest["python"].update(executable="/wrong/python"),
            "torch": lambda manifest: manifest["versions"].update(torch="2.0.0"),
            "torch_geometric": lambda manifest: manifest["versions"].update(torch_geometric="2.5.0"),
            "locked_training": lambda manifest: manifest["locked_training"].update(hidden=201),
            "roots.repository": lambda manifest: manifest["roots"].update(repository="/wrong/repo"),
            "roots.data": lambda manifest: manifest["roots"].update(data="/wrong/data"),
            "roots.mask_bank": lambda manifest: manifest["roots"].update(mask_bank="/wrong/masks"),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                fold = _write_job(
                    _fold(Path(tmp) / "formal", "original", 0.0, 66),
                    "original",
                    0.0,
                    66,
                    historical=True,
                )
                manifest_path = Path(tmp) / "formal" / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                mutate(manifest)
                _write_json(manifest_path, manifest)
                (fold / "saved" / "run.npz").write_bytes(b"not-an-npz")
                with self.assertRaisesRegex(ValueError, message):
                    collect_job(fold, "original", 0.0, 66, historical_original=True)

        with tempfile.TemporaryDirectory() as tmp:
            fold = _write_job(
                _fold(Path(tmp) / "formal", "genagg", 0.0, 66),
                "genagg",
                0.0,
                66,
            )
            invocation = next((Path(tmp) / "formal" / "invocations").glob("*.json"))
            payload = json.loads(invocation.read_text())
            payload["job_count"] = 11
            _write_json(invocation, payload)
            with self.assertRaisesRegex(ValueError, "invocation"):
                collect_job(fold, "genagg", 0.0, 66)

    def test_rejects_invalid_loss_history_scores_and_labels(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_job,
        )

        loss_99 = np.array(
            [[{"train_loss": [1.0]} for _ in range(99)]], dtype=object
        )
        nan_scores = np.zeros((6, 6), dtype=np.float64)
        nan_scores[0, 0] = np.nan
        cases = {
            "folder_losswhole": {"archive_overrides": {"folder_losswhole": loss_99}},
            "finite": {
                "archive_overrides": {
                    "folder_savewhole": np.array(
                        [[7, {"test_labels": [np.arange(6)], "test_preds": [nan_scores]}]],
                        dtype=object,
                    )
                }
            },
            "six columns": {
                "archive_overrides": {
                    "folder_savewhole": np.array(
                        [[7, {"test_labels": [np.arange(6)], "test_preds": [np.zeros((6, 5))]}]],
                        dtype=object,
                    )
                }
            },
            "label range": {
                "archive_overrides": {
                    "folder_savewhole": np.array(
                        [[7, {"test_labels": [np.array([0, 1, 2, 3, 4, 6])], "test_preds": [np.zeros((6, 6))]}]],
                        dtype=object,
                    )
                }
            },
        }
        for message, options in cases.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                fold = _write_job(
                    _fold(Path(tmp) / "formal", "genagg", 0.0, 66),
                    "genagg",
                    0.0,
                    66,
                    **options
                )
                with self.assertRaisesRegex(ValueError, message):
                    collect_job(fold, "genagg", 0.0, 66)


class GridAndGateTests(unittest.TestCase):
    def test_collect_grid_joins_split_phases_to_inherited_original_by_mask(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_grid,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_formal = root / "historical" / "formal"
            original_root = original_formal / "original"
            phase_a = root / "phase_a" / "formal"
            phase_b = root / "phase_b" / "formal"
            for rate in (0.0, 0.7):
                for seed in (66, 67, 68):
                    _write_job(
                        _fold(original_formal, "original", rate, seed),
                        "original",
                        rate,
                        seed,
                        historical=True,
                    )
                    for arm in ("genagg", "soft_medoid"):
                        _write_job(_fold(phase_a, arm, rate, seed), arm, rate, seed)
                    for arm in ("ssma", "rtdr"):
                        _write_job(_fold(phase_b, arm, rate, seed), arm, rate, seed)
            rows = collect_grid(original_root, phase_a, phase_b)
        self.assertEqual(set(rows), {"original", *CANDIDATES})
        self.assertTrue(all(len(value) == 6 for value in rows.values()))

    def test_grid_rejects_mask_mismatch_missing_duplicate_and_ambiguous_candidate(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            collect_grid,
            index_rows,
        )

        original = [{"rate": 0.0, "seed": 66}]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            index_rows(original + original, rates=(0.0,), seeds=(66,))
        with self.assertRaisesRegex(ValueError, "grid mismatch"):
            index_rows([], rates=(0.0,), seeds=(66,))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_formal = root / "historical" / "formal"
            original_root = original_formal / "original"
            phase_a, phase_b = root / "a" / "formal", root / "b" / "formal"
            _write_job(
                _fold(original_formal, "original", 0.0, 66),
                "original",
                0.0,
                66,
                historical=True,
                mask_hash="original-mask",
            )
            for arm in CANDIDATES:
                target = phase_a if arm in ("genagg", "soft_medoid") else phase_b
                _write_job(
                    _fold(target, arm, 0.0, 66),
                    arm,
                    0.0,
                    66,
                    mask_hash=("wrong" if arm == "genagg" else "original-mask"),
                )
            with self.assertRaisesRegex(ValueError, "mask.*mismatch"):
                collect_grid(
                    original_root, phase_a, phase_b, rates=(0.0,), seeds=(66,)
                )

    def test_phase_roots_reject_arm_path_swap(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import collect_grid

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_formal = root / "historical" / "formal"
            original_root = original_formal / "original"
            phase_a, phase_b = root / "phase_a" / "formal", root / "phase_b" / "formal"
            _write_job(
                _fold(original_formal, "original", 0.0, 66),
                "original", 0.0, 66, historical=True,
            )
            for arm in ("genagg", "soft_medoid"):
                _write_job(_fold(phase_b, arm, 0.0, 66), arm, 0.0, 66)
            for arm in ("ssma", "rtdr"):
                _write_job(_fold(phase_a, arm, 0.0, 66), arm, 0.0, 66)
            with self.assertRaisesRegex(ValueError, "Phase A|phase A"):
                collect_grid(original_root, phase_a, phase_b, rates=(0.0,), seeds=(66,))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_formal = root / "historical" / "formal"
            original_root = original_formal / "original"
            phase_a, phase_b = root / "a" / "formal", root / "b" / "formal"
            _write_job(
                _fold(original_formal, "original", 0.0, 66),
                "original",
                0.0,
                66,
                historical=True,
            )
            for phase in (phase_a, phase_b):
                _write_job(_fold(phase, "genagg", 0.0, 66), "genagg", 0.0, 66)
            with self.assertRaisesRegex(ValueError, "Phase A"):
                collect_grid(
                    original_root, phase_a, phase_b, rates=(0.0,), seeds=(66,)
                )

    def test_gate_uses_unrounded_rate_and_seed_macro_rules(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            summarize_candidate,
        )

        originals, candidate = [], []
        deltas = {0.0: (0.03, 0.02, -0.01), 0.7: (0.01, 0.01, -0.005)}
        for rate in (0.0, 0.7):
            for seed, delta in zip((66, 67, 68), deltas[rate]):
                common = dict(
                    rate=rate,
                    seed=seed,
                    weighted_f1=0.6,
                    accuracy=0.6,
                    class_coverage=6,
                    dominant_ratio=0.3,
                    epoch=7,
                    parameter_count=1,
                    selected_path_parameter_count=1,
                    runtime_seconds=1.0,
                    mask_sha256="m-{}-{}".format(rate, seed),
                )
                originals.append(dict(common, arm="original"))
                candidate.append(
                    dict(common, arm="genagg", weighted_f1=0.6 + delta)
                )
        passed = summarize_candidate(
            "genagg", originals, candidate, rates=(0.0, 0.7), seeds=(66, 67, 68)
        )
        self.assertTrue(passed["gate"]["passed"])
        self.assertEqual(passed["gate"]["positive_seed_macros"], 2)
        self.assertEqual(len(passed["tasks"]), 6)

        for index in (3, 4, 5):
            candidate[index]["weighted_f1"] = 0.6 - 1e-15
        failed = summarize_candidate(
            "genagg", originals, candidate, rates=(0.0, 0.7), seeds=(66, 67, 68)
        )
        self.assertFalse(failed["gate"]["passed"])
        self.assertLess(failed["rate_means"]["0.7"]["mean_delta"], 0.0)

    def test_gate_rejects_nonfinite_and_collapsed_run(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            summarize_candidate,
        )

        def rows(coverage=6, f1=0.7):
            result = []
            for rate in (0.0, 0.7):
                for seed in (66, 67, 68):
                    result.append(
                        dict(
                            rate=rate,
                            seed=seed,
                            arm="x",
                            weighted_f1=f1,
                            accuracy=0.7,
                            class_coverage=coverage,
                            dominant_ratio=0.3,
                            epoch=1,
                            parameter_count=1,
                            selected_path_parameter_count=1,
                            runtime_seconds=1.0,
                            mask_sha256="m-{}-{}".format(rate, seed),
                        )
                    )
            return result

        original = rows(f1=0.6)
        self.assertFalse(summarize_candidate("x", original, rows(coverage=1))["gate"]["passed"])
        nonfinite = rows()
        nonfinite[0]["weighted_f1"] = float("nan")
        self.assertFalse(summarize_candidate("x", original, nonfinite)["gate"]["passed"])


class OutputTests(unittest.TestCase):
    def test_writes_atomic_json_and_three_markdown_mirrors(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import (
            write_outputs,
        )

        summary = {"candidates": {"genagg": {"gate": {"passed": True}}}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_outputs(output, summary)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            for name in ("RESULTS.md", "RESULTS.zh.md", "RESULTS.en.md"):
                text = (output / name).read_text(encoding="utf-8")
                self.assertIn("genagg", text)
            self.assertEqual(list(output.glob("*.tmp")), [])

    def test_chinese_report_has_chinese_headers_and_status_explanation(self):
        from experiments.second_graph_aggregation_iemocap6.summarize_gate import write_outputs

        summary = {"candidates": {"genagg": {"macro_delta": 0.1, "gate": {"passed": True}}}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_outputs(output, summary)
            chinese = (output / "RESULTS.zh.md").read_text(encoding="utf-8")
        self.assertIn("候选模块", chinese)
        self.assertIn("是否晋级", chinese)
        self.assertIn("PASS 表示", chinese)


if __name__ == "__main__":
    unittest.main()
