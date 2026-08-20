"""Deterministic random-stream helpers for the unified experiment protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Iterator, List

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class SeedBundle:
    """Derive stable component seeds from one experiment seed."""

    master_seed: int

    def derive(self, component: str) -> int:
        payload = "{}:{}".format(self.master_seed, component).encode("utf-8")
        digest_prefix = hashlib.sha256(payload).digest()[:4]
        return int.from_bytes(digest_prefix, byteorder="big") & 0x7FFFFFFF


class EpochSeededSubsetSampler(Sampler):
    """Shuffle a subset reproducibly without consuming global RNG state."""

    def __init__(self, indices: Iterable[int], seed: int) -> None:
        self.indices: List[int] = list(indices)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        positions = torch.randperm(len(self.indices), generator=generator).tolist()
        return (self.indices[position] for position in positions)

    def __len__(self) -> int:
        return len(self.indices)
