"""Frozen-pretrained target provenance and unchanged joint-loss lifecycle."""

from dataclasses import asdict
import json

import pytest
import torch

from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.mixed_rate import MISSING_RATES
from test_missing_m3 import _LifecycleModel, _CountingOptimizer, _install_train_epoch_lifecycle_fakes


def config(**overrides):
    values = dict(teacher_mode="pretrained-frozen", teacher_checkpoint="source.pt",
                  backbone_type="osram", osram_bidirectional=False, osram_write_step=.6)
    values.update(overrides)
    return tr.TrainConfig(**values)


def test_teacher_default_and_cli_roundtrip():
    assert tr.TrainConfig().teacher_mode == "ema"
    assert tr.TrainConfig().teacher_checkpoint is None
    cfg = config()
    assert tr.TrainConfig(**asdict(cfg)) == cfg
    args = tr.build_parser().parse_args([
        "--audio-feature", "a", "--text-feature", "t", "--video-feature", "v",
        "--output-dir", "out", "--teacher-mode", "pretrained-frozen",
        "--teacher-checkpoint", "source.pt",
    ])
    assert args.teacher_mode == "pretrained-frozen"
    assert args.teacher_checkpoint == "source.pt"


@pytest.mark.parametrize("override", [
    {"teacher_checkpoint": None}, {"teacher_checkpoint": ""},
    {"training_objective": "complete-state"}, {"training_objective": "write-state"},
    {"training_objective": "future-state"}, {"training_objective": "emotion-only"},
    {"backbone_type": "gcnet"}, {"osram_bidirectional": True}, {"osram_write_step": 1.0},
    {"osram_forward_slot_reuse": True}, {"osram_predictor_mode": "legacy-hidden"},
    {"fusion_type": "slot"}, {"osram_readout_fusion": "local-gated"},
    {"classification_completion": True}, {"completion_path": "pre_osram_b2"},
    {"initial_backbone_checkpoint": "old.pt"}, {"osram_emotion_ablation": "local-only"},
    {"osram_ablation": "local-only"}, {"local_context_residual": True},
    {"node_interaction_residual": True}, {"readout_type": "pattern"},
])
def test_pretrained_teacher_rejects_unapproved_architectures(override):
    with pytest.raises(ValueError, match="pretrained-frozen"):
        config(**override)


def test_ema_mode_does_not_silently_ignore_checkpoint():
    with pytest.raises(ValueError, match="teacher_checkpoint"):
        config(teacher_mode="ema")
    with pytest.raises(ValueError, match="teacher_mode"):
        config(teacher_mode="unknown")


@pytest.mark.parametrize("teacher_mode,ema_calls", [("ema", 1), ("pretrained-frozen", 0)])
def test_joint_loss_stays_active_but_frozen_teacher_skips_ema(monkeypatch, teacher_mode, ema_calls):
    _install_train_epoch_lifecycle_fakes(monkeypatch)
    model = _LifecycleModel()
    optimizer = _CountingOptimizer(model.weight)
    cfg = config(teacher_mode=teacher_mode,
                 teacher_checkpoint="source.pt" if teacher_mode != "ema" else None,
                 train_rate_mode="fixed", fixed_missing_rate=.5)
    result = tr.train_epoch(model, [["batch"]], optimizer, cfg,
                           {r: r for r in MISSING_RATES}, 0, (1, 1, 1), torch.device("cpu"))
    assert model.predict_missing_flags == [True]
    assert model.teacher_calls == 1
    assert model.ema_calls == ema_calls
    assert result["jepa_target_count"] == 1
    assert len(optimizer.step_gradients) == 1


@pytest.mark.parametrize("mutate_teacher", [False, True])
@pytest.mark.parametrize("mismatch", [None, "dataset", "fold", "seed", "latent_dim", "absent"])
def test_runner_records_teacher_provenance_and_detects_mutation(monkeypatch, tmp_path, mutate_teacher, mismatch):
    class TinyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            assert kwargs["teacher_mode"] == "pretrained-frozen"
            assert kwargs["teacher_checkpoint"] == "source.pt"
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.osram = torch.nn.Identity()
            self.teacher_hash = "fixed-target-space"
            source = dict(dataset="CMUMOSI", fold=1, seed=66, latent_dim=256)
            if mismatch is not None and mismatch != "absent":
                source[mismatch] = "wrong"
            self.teacher_provenance = {
                "checkpoint_sha256": "checkpoint-hash", "source_prefix": "observed_set.projectors.",
                "source_config": source if mismatch != "absent" else {},
            }
            self.ema_step = 0
            self.readout_type, self.readout_rank = "shared", 8

        def teacher_integrity(self):
            return self.teacher_hash

    monkeypatch.setattr(tr, "MissingM3GraphModel", TinyModel)
    monkeypatch.setattr(tr, "get_loaders", lambda **kwargs: ([[]], [[]], [[]], 1, 1, 1))
    monkeypatch.setattr(tr, "_schedules", lambda *args: {r: r for r in MISSING_RATES})

    def train(model, *args):
        if mutate_teacher:
            model.teacher_hash = "unexpected-change"
        return dict(weighted_f1=.5, classification_loss=1., jepa_loss=.2)

    monkeypatch.setattr(tr, "train_epoch", train)
    monkeypatch.setattr(tr, "evaluate_rate", lambda *args, **kwargs: ({"weighted_f1": .6}, None))
    cfg = config(dataset="CMUMOSI", fold=1, epochs=1, device="cpu", evaluate_test=False,
                 train_rate_mode="fixed", fixed_missing_rate=.5)
    if mismatch is not None:
        with pytest.raises(ValueError, match="teacher.*(dataset|fold|seed|latent_dim)"):
            tr.run_experiment(cfg, "a", "t", "v", tmp_path)
    elif mutate_teacher:
        with pytest.raises(RuntimeError, match="teacher.*changed"):
            tr.run_experiment(cfg, "a", "t", "v", tmp_path)
    else:
        result = tr.run_experiment(cfg, "a", "t", "v", tmp_path)
        assert result["teacher_mode"] == "pretrained-frozen"
        assert result["teacher_provenance"]["checkpoint_sha256"] == "checkpoint-hash"
        assert result["teacher_integrity"]["before"] == result["teacher_integrity"]["after"] == "fixed-target-space"
        saved = json.loads((tmp_path / "metrics.json").read_text())
        assert saved["teacher_integrity"]["unchanged"] is True
