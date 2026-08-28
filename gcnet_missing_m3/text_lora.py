"""Leakage-safe data and model primitives for MOSI text-LoRA adaptation."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Dict, Mapping, Optional, Sequence

import torch
from torch import nn


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class MosiRecord:
    uid: str
    text: str
    label: Optional[float]
    split: str


def load_mosi_records(
    path: str | Path,
    *,
    label_splits: Collection[str] = SPLIT_NAMES,
) -> Dict[str, tuple[MosiRecord, ...]]:
    """Load the canonical MOSI pickle without reordering its utterances.

    ``label_splits`` lets the training and export phases make label access
    explicit. In particular, training can request only train/validation labels
    and export can request none.
    """
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, (list, tuple)) or len(payload) != 7:
        raise ValueError("MOSI label pickle must contain seven entries")
    video_ids, video_labels, _, video_texts, train_videos, val_videos, test_videos = payload
    split_videos = {
        "train": set(train_videos),
        "validation": set(val_videos),
        "test": set(test_videos),
    }
    requested_labels = set(label_splits)
    unknown = requested_labels.difference(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unknown label splits: {sorted(unknown)}")
    if any(
        split_videos[left] & split_videos[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise ValueError("MOSI conversation splits overlap")

    records: Dict[str, list[MosiRecord]] = {name: [] for name in SPLIT_NAMES}
    seen_uids: set[str] = set()
    assigned_videos: set[str] = set()
    for video_id, uids in video_ids.items():
        memberships = [name for name in SPLIT_NAMES if video_id in split_videos[name]]
        if len(memberships) != 1:
            raise ValueError(f"video {video_id!r} must belong to exactly one split")
        split = memberships[0]
        assigned_videos.add(video_id)
        texts = video_texts[video_id]
        if len(uids) != len(texts):
            raise ValueError(f"UID/text length mismatch for video {video_id!r}")
        labels = None
        if split in requested_labels:
            labels = video_labels[video_id]
            if len(uids) != len(labels):
                raise ValueError(f"UID/label length mismatch for video {video_id!r}")
        for index, (uid, text) in enumerate(zip(uids, texts)):
            uid = str(uid)
            if uid in seen_uids:
                raise ValueError(f"duplicate MOSI UID: {uid}")
            seen_uids.add(uid)
            label = float(labels[index]) if labels is not None else None
            records[split].append(MosiRecord(uid, str(text), label, split))

    declared_videos = set().union(*split_videos.values())
    if assigned_videos != declared_videos:
        missing = sorted(declared_videos - assigned_videos)
        raise ValueError(f"split references unknown videos: {missing}")
    return {name: tuple(records[name]) for name in SPLIT_NAMES}


def masked_mean_pool(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids: Collection[int],
) -> torch.Tensor:
    """Mean-pool valid tokens while excluding BOS, EOS, and PAD."""
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, tokens, hidden]")
    if input_ids.shape != attention_mask.shape or input_ids.shape != last_hidden_state.shape[:2]:
        raise ValueError("token tensors must share batch and sequence dimensions")
    valid = attention_mask.bool()
    for token_id in special_token_ids:
        valid &= input_ids.ne(int(token_id))
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("every sequence must contain a non-special valid token")
    weighted = last_hidden_state * valid.unsqueeze(-1).to(last_hidden_state.dtype)
    return weighted.sum(dim=1) / counts.unsqueeze(-1).to(last_hidden_state.dtype)


def count_lora_targets(model: nn.Module) -> int:
    """Count attention modules whose leaf name is query or value."""
    return sum(
        1
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in {"query", "value"}
        and hasattr(module, "weight")
    )


def build_lora_encoder(
    base_model_path: str | Path,
    *,
    auto_model_cls=None,
    lora_config_cls=None,
    get_peft_model_fn=None,
) -> nn.Module:
    """Build local RoBERTa-large with query/value LoRA adapters.

    Dependency injection keeps unit tests independent of transformers/PEFT;
    real imports occur only when this function is called without fakes.
    """
    if auto_model_cls is None:
        from transformers import AutoModel

        auto_model_cls = AutoModel
    if lora_config_cls is None or get_peft_model_fn is None:
        from peft import LoraConfig, get_peft_model

        lora_config_cls = lora_config_cls or LoraConfig
        get_peft_model_fn = get_peft_model_fn or get_peft_model

    base = auto_model_cls.from_pretrained(
        str(base_model_path), local_files_only=True
    )
    hidden_size = int(getattr(base.config, "hidden_size", -1))
    if hidden_size != 1024:
        raise ValueError(f"expected RoBERTa-large hidden size 1024, got {hidden_size}")
    target_count = count_lora_targets(base)
    if target_count != 48:
        raise ValueError(f"expected 48 query/value LoRA targets, got {target_count}")
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.gradient_checkpointing_enable()
    config = lora_config_cls(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "value"],
    )
    model = get_peft_model_fn(base, config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise ValueError(f"non-LoRA encoder parameters are trainable: {unexpected}")
    return model


class SentimentRegressor(nn.Module):
    """Temporary regression head used only to supervise LoRA adaptation."""

    def __init__(
        self,
        encoder: nn.Module,
        special_token_ids: Collection[int],
        hidden_size: int = 1024,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.special_token_ids = tuple(int(value) for value in special_token_ids)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(0.1), nn.Linear(hidden_size, 1)
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        return masked_mean_pool(
            hidden, input_ids, attention_mask, self.special_token_ids
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(input_ids, attention_mask)).squeeze(-1)


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def canonical_uid_hash(uids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for uid in sorted(str(uid) for uid in uids):
        digest.update(uid.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_feature_hash(feature_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for uid in sorted(feature_hashes):
        digest.update(str(uid).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(feature_hashes[uid]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash one file or a directory tree in canonical relative-path order."""
    root = Path(path)
    digest = hashlib.sha256()
    if root.is_file():
        with root.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(child.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()
