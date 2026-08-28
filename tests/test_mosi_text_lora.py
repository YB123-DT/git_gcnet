from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_package_import_does_not_eagerly_load_graph_dependencies():
    command = (
        "import sys; import gcnet_missing_m3; "
        "assert 'gcnet_missing_m3.model' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def _modules():
    try:
        core = importlib.import_module("gcnet_missing_m3.text_lora")
        train = importlib.import_module("gcnet_missing_m3.train_text_lora")
    except ModuleNotFoundError as error:
        pytest.fail(f"MOSI text-LoRA module is not implemented: {error}")
    return core, train


def test_label_pickle_loader_preserves_alignment_and_disjoint_splits(tmp_path: Path):
    core, _ = _modules()
    payload = [
        {"video_b": ["b_1", "b_2"], "video_a": ["a_1"], "video_c": ["c_1"]},
        {"video_b": [1.5, -0.5], "video_a": [0.25], "video_c": [-2.0]},
        {"video_b": ["x", "x"], "video_a": ["x"], "video_c": ["x"]},
        {
            "video_b": ["text b1", "text b2"],
            "video_a": ["text a1"],
            "video_c": ["text c1"],
        },
        {"video_b"},
        {"video_a"},
        {"video_c"},
    ]
    label_pickle = tmp_path / "labels.pkl"
    with label_pickle.open("wb") as handle:
        pickle.dump(payload, handle)

    splits = core.load_mosi_records(label_pickle)

    assert list(splits) == ["train", "validation", "test"]
    assert [(item.uid, item.text, item.label) for item in splits["train"]] == [
        ("b_1", "text b1", 1.5),
        ("b_2", "text b2", -0.5),
    ]
    assert splits["validation"][0].uid == "a_1"
    assert splits["test"][0].uid == "c_1"
    uid_sets = [{item.uid for item in splits[name]} for name in splits]
    assert not (uid_sets[0] & uid_sets[1] | uid_sets[0] & uid_sets[2] | uid_sets[1] & uid_sets[2])
    assert set().union(*uid_sets) == {"a_1", "b_1", "b_2", "c_1"}


def test_masked_mean_pool_excludes_bos_eos_pad_and_attention_padding():
    core, _ = _modules()
    hidden = torch.tensor(
        [[[100.0, 100.0], [1.0, 3.0], [3.0, 5.0], [200.0, 200.0], [9.0, 9.0]]]
    )
    input_ids = torch.tensor([[0, 10, 11, 2, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])

    pooled = core.masked_mean_pool(hidden, input_ids, attention_mask, {0, 1, 2})

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))


def test_lora_builder_freezes_base_and_targets_48_query_value_modules():
    core, _ = _modules()

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Linear(2, 2)
            self.value = torch.nn.Linear(2, 2)

    class FakeBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Attention() for _ in range(24)])
            self.config = SimpleNamespace(hidden_size=1024)
            self.checkpointing_enabled = False

        def gradient_checkpointing_enable(self):
            self.checkpointing_enabled = True

    base = FakeBase()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == "/local/roberta"
            assert kwargs == {"local_files_only": True}
            return base

    captured = {}

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_get_peft_model(model, config):
        model.register_parameter("lora_adapter", torch.nn.Parameter(torch.ones(())))
        return model

    model = core.build_lora_encoder(
        "/local/roberta",
        auto_model_cls=FakeAutoModel,
        lora_config_cls=FakeLoraConfig,
        get_peft_model_fn=fake_get_peft_model,
    )

    assert model.checkpointing_enabled
    assert captured == {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["query", "value"],
    }
    assert core.count_lora_targets(model) == 48
    assert [name for name, parameter in model.named_parameters() if parameter.requires_grad] == [
        "lora_adapter"
    ]


def test_training_loaders_are_seeded_and_never_touch_test_records():
    core, train = _modules()
    records = [core.MosiRecord(f"u{i}", f"text {i}", float(i), "train") for i in range(7)]
    validation = [core.MosiRecord("v0", "validation", 0.0, "validation")]

    class GuardedSplits(Mapping):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("test data must not be accessed while constructing training loaders")
            return {"train": records, "validation": validation}[key]

        def __iter__(self):
            return iter(("train", "validation", "test"))

        def __len__(self):
            return 3

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.arange(len(texts)).unsqueeze(1),
                "attention_mask": torch.ones(len(texts), 1, dtype=torch.long),
            }

    def order(seed):
        loader, validation_loader = train.build_training_loaders(
            GuardedSplits(), Tokenizer(), batch_size=3, seed=seed, max_length=192
        )
        assert next(iter(validation_loader))["uids"] == ["v0"]
        return [uid for batch in loader for uid in batch["uids"]]

    assert order(66) == order(66)
    assert order(66) != order(67)
    assert set(order(66)) == {item.uid for item in records}


def test_training_uses_val_mae_patience_and_keeps_trainable_only_best_state(monkeypatch):
    _, train = _modules()

    class TinyRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.frozen = torch.nn.Parameter(torch.tensor(7.0), requires_grad=False)

        def forward(self, input_ids, attention_mask):
            return input_ids.float().mean(dim=1) * self.weight

    model = TinyRegressor()
    batch = {
        "input_ids": torch.tensor([[1], [2]]),
        "attention_mask": torch.ones(2, 1, dtype=torch.long),
        "labels": torch.tensor([0.0, 1.0]),
    }
    scores = iter([3.0, 2.0, 2.5, 2.6, 2.7, 0.1])
    clipped = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters, max_norm):
        clipped.append(max_norm)
        return original_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)

    result = train.fit_model(
        model,
        [batch],
        [batch],
        train.TrainingConfig(max_epochs=20, patience=3, amp=False),
        device=torch.device("cpu"),
        evaluate_fn=lambda *_: {
            "mae": next(scores),
            "correlation": 0.0,
            "weighted_f1": 0.0,
        },
    )

    assert result.best_epoch == 1
    assert result.best_metrics["mae"] == 2.0
    assert len(result.history) == 5
    assert set(result.best_state) == {"weight"}
    assert clipped and set(clipped) == {1.0}


def test_label_free_export_writes_finite_float32_features_and_recomputable_hashes(
    tmp_path: Path,
):
    core, train = _modules()

    class LabelForbidden:
        def __init__(self, uid, text):
            self.uid = uid
            self.text = text

        @property
        def label(self):
            raise AssertionError("export must not access labels")

    class Tokenizer:
        bos_token_id, eos_token_id, pad_token_id = 0, 2, 1

        def __call__(self, texts, **kwargs):
            count = len(texts)
            return {
                "input_ids": torch.tensor([[0, 10, 2]] * count),
                "attention_mask": torch.ones(count, 3, dtype=torch.long),
            }

    class Encoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            batch = input_ids.shape[0]
            hidden = torch.tensor([[[9.0, 9.0, 9.0], [1.0, 2.0, 3.0], [8.0, 8.0, 8.0]]])
            return SimpleNamespace(last_hidden_state=hidden.repeat(batch, 1, 1))

    records = [LabelForbidden(f"uid_{index}", f"text {index}") for index in range(4)]
    hashes = train.export_feature_bank(
        Encoder(),
        Tokenizer(),
        records,
        tmp_path,
        batch_size=2,
        max_length=192,
        device=torch.device("cpu"),
        expected_count=4,
        hidden_size=3,
    )

    assert list(hashes) == [f"uid_{index}" for index in range(4)]
    for uid, expected_hash in hashes.items():
        path = tmp_path / f"{uid}.npy"
        value = np.load(path)
        assert value.shape == (3,)
        assert value.dtype == np.float32
        assert np.isfinite(value).all()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    aggregate = core.canonical_feature_hash(hashes)
    assert aggregate == core.canonical_feature_hash(dict(reversed(list(hashes.items()))))
    manifest = {"feature_sha256": hashes, "feature_aggregate_sha256": aggregate}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert json.loads((tmp_path / "manifest.json").read_text())["feature_aggregate_sha256"] == aggregate
