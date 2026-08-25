import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from versions.mpfilm.run_locked_ab import Job, build_command


def _snapshot(predicted=(0, 1, 2, 3, 4, 5)):
    labels = np.arange(6)
    scores = np.zeros((6, 6), dtype=np.float64)
    scores[np.arange(6), np.asarray(predicted)] = 1.0
    return {"test_labels": [labels], "test_preds": [scores]}


def _write_job(
    root,
    arm,
    rate,
    seed,
    mask_hash=None,
    predicted=(0, 1, 2, 3, 4, 5),
    arg_overrides=None,
):
    from versions.sequence_aff.summarize import (
        SELECTED_PARAMETER_COUNTS,
        STORED_PARAMETER_COUNT,
    )

    fold = root / arm / f"miss_{rate:.1f}".replace(".", "p") / f"seed_{seed}" / "fold_5"
    saved = fold / "saved"
    saved.mkdir(parents=True)
    fusion = "mask_sequence_aff" if arm == "sequence_aff" else "addition"
    args = Namespace(
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
        pre_graph_context="bilstm",
        post_graph_context="bilstm",
        graph_conv_variant="original",
        branch_fusion=fusion,
        seed=seed,
        mask_seed=seed,
        mask_type=f"constant-{rate:.1f}",
        fold_index=5,
    )
    for field, value in (arg_overrides or {}).items():
        setattr(args, field, value)
    np.savez_compressed(
        saved / "run.npz",
        args=np.array(args, dtype=object),
        fold_numbers=np.array([5]),
        folder_savewhole=np.array([[7, _snapshot(predicted)]], dtype=object),
        mask_bank_manifest=np.array(
            {
                "sha256": mask_hash or f"mask-{rate}-{seed}",
                "requested_missing_rate": rate,
                "seed": seed,
            },
            dtype=object,
        ),
        smoke_only=np.array(False),
        parameter_count=np.array(STORED_PARAMETER_COUNT),
        selected_path_parameter_count=np.array(SELECTED_PARAMETER_COUNTS[arm]),
    )
    job = Job("formal", arm, rate, seed, fold)
    command = build_command(job, Path("/env/python"), Path("/repo"), Path("/data"), Path("/masks"))
    (fold / "command.json").write_text(json.dumps({
        "stage": "formal", "arm": arm, "missing_rate": rate, "seed": seed,
        "fold": 5, "gpu": "0", "command": command,
    }), encoding="utf-8")
    (fold / "status.json").write_text(
        json.dumps({"status": "success", "return_code": 0}), encoding="utf-8"
    )
    (fold / "train.log").write_text("\n".join(
        [f"epoch:{index}; train_fscore:0" for index in range(1, 101)]
        + ["SMOKE_ONLY=False", "Finish fold 5", "save results in run.npz"]
    ), encoding="utf-8")
    return fold


class SequenceAFFArchiveTests(unittest.TestCase):
    def test_collects_trusted_complete_job_and_metrics(self):
        from versions.sequence_aff.summarize import collect_job

        with tempfile.TemporaryDirectory() as tmp:
            fold = _write_job(Path(tmp), "sequence_aff", 0.3, 68)
            row = collect_job(fold, "sequence_aff", 0.3, 68)

        self.assertEqual(row["weighted_f1"], 1.0)
        self.assertEqual(row["accuracy"], 1.0)
        self.assertEqual(row["class_coverage"], 6)
        self.assertAlmostEqual(row["dominant_ratio"], 1 / 6)
        self.assertEqual(row["epoch"], 7)
        self.assertEqual(row["branch_fusion"], "mask_sequence_aff")
        self.assertEqual(row["parameter_count"], 36_419_816)
        self.assertEqual(row["selected_path_parameter_count"], 34_393_416)

    def test_rejects_artifact_and_provenance_drift(self):
        from versions.sequence_aff.summarize import collect_job

        mutations = {
            "return_code": lambda fold: (fold / "status.json").write_text(
                json.dumps({"status": "success", "return_code": 1})
            ),
            "100 epoch": lambda fold: (fold / "train.log").write_text("epoch:1"),
            "exactly one": lambda fold: (fold / "saved" / "extra.npz").write_bytes(b"x"),
            "lock": lambda fold: (fold / ".active.lock").write_text("active"),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                fold = _write_job(Path(tmp), "original", 0.2, 66)
                mutate(fold)
                with self.assertRaisesRegex(ValueError, message):
                    collect_job(fold, "original", 0.2, 66)

        with tempfile.TemporaryDirectory() as tmp:
            fold = _write_job(Path(tmp), "sequence_aff", 0.2, 66)
            archive = fold / "saved" / "run.npz"
            with np.load(archive, allow_pickle=True) as data:
                values = {key: data[key] for key in data.files}
            values["parameter_count"] = np.array(1)
            np.savez_compressed(archive, **values)
            with self.assertRaisesRegex(ValueError, "parameter_count"):
                collect_job(fold, "sequence_aff", 0.2, 66)

        for field in ("pre_graph_context", "post_graph_context"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                fold = _write_job(
                    Path(tmp), "sequence_aff", 0.4, 69,
                    arg_overrides={field: "linear"},
                )
                with self.assertRaisesRegex(ValueError, field):
                    collect_job(fold, "sequence_aff", 0.4, 69)

    def test_pair_hash_drift_is_rejected(self):
        from versions.sequence_aff.summarize import collect_grid

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm in ("original", "sequence_aff"):
                _write_job(root, arm, 0.0, 66, mask_hash=arm)
            with self.assertRaisesRegex(ValueError, "mask.*hash"):
                collect_grid(root, rates=(0.0,), seeds=(66,))

class SequenceAFFStatisticsTests(unittest.TestCase):
    def test_paired_summary_uses_sample_sd_and_seed_macro(self):
        from versions.sequence_aff.summarize import (
            paired_summary,
            render_markdown,
        )

        original = []
        candidate = []
        for rate in (0.0, 0.1):
            for offset, seed in enumerate((66, 67, 68)):
                base = 0.60 + rate + offset * 0.01
                delta = (offset + 1) * 0.01 + rate
                common = {
                    "rate": rate, "seed": seed, "accuracy": base,
                    "class_coverage": 6, "dominant_ratio": 0.2,
                    "epoch": 7, "manifest_hash": f"m-{rate}-{seed}",
                }
                original.append({**common, "arm": "original", "weighted_f1": base})
                candidate.append({**common, "arm": "sequence_aff", "weighted_f1": base + delta})

        result = paired_summary(original, candidate, rates=(0.0, 0.1), seeds=(66, 67, 68))

        self.assertAlmostEqual(result["rates"]["0.0"]["mean_delta"], 0.02)
        self.assertAlmostEqual(result["rates"]["0.0"]["sd_delta"], 0.01)
        self.assertEqual(result["rates"]["0.0"]["wins"], 3)
        self.assertAlmostEqual(result["macro"]["mean_delta"], 0.07)
        self.assertAlmostEqual(result["macro"]["sd_delta"], 0.01)
        self.assertIn("paired_t_test", result["macro"])
        self.assertIn("wilcoxon", result["macro"])
        json.dumps(result)
        markdown = render_markdown(result)
        self.assertIn("Sequence AFF 配对实验汇总", markdown)
        self.assertIn("八档宏平均", markdown)


if __name__ == "__main__":
    unittest.main()
