"""Inference-only Gap-query identity swaps for causal OSRAM diagnostics."""

from __future__ import annotations

import torch


def active_gap_swap_mask(availability: torch.Tensor) -> torch.Tensor:
    """Return the valid-pattern mask where exactly two Gap slots are active.

    ``availability`` is ``[L, B, 3]`` with one column per A/T/V modality.
    A swap is defined only when exactly one modality is observed, so that the
    two missing modalities are both active Gap slots.  Padding rows contain no
    observed modality and therefore are not selected.
    """

    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError("availability must have shape [L, B, 3]")
    if not bool(((availability == 0) | (availability == 1)).all()):
        raise ValueError("availability must be binary")
    return availability.sum(dim=-1).eq(1)


def swap_active_gap_queries(
    queries: torch.Tensor,
    availability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange the two currently active Gap queries, leaving Base untouched.

    ``queries`` has shape ``[L, B, 4, H, K]``.  Query slot 0 is Base and
    slots 1/2/3 are Audio/Text/Visual.  For a row with exactly one observed
    modality, the two missing-modality query vectors are exchanged.  All
    observed/complete/padding rows are copied unchanged.

    The returned mask is ``[L, B]`` and records which rows were changed.
    """

    if queries.ndim != 5 or queries.shape[2] != 4:
        raise ValueError("queries must have shape [L, B, 4, H, K]")
    if availability.shape != (queries.shape[0], queries.shape[1], 3):
        raise ValueError("availability must match queries as [L, B, 3]")
    swap_mask = active_gap_swap_mask(availability)
    swapped = queries.clone()
    # Exactly one observed modality determines the two missing slots.  Use
    # pairwise masks so every row is handled without sequential writes.
    for first in range(3):
        for second in range(first + 1, 3):
            pair = swap_mask & availability[..., first].eq(0) & availability[..., second].eq(0)
            first_slot = queries[..., first + 1, :, :]
            second_slot = queries[..., second + 1, :, :]
            swapped[..., first + 1, :, :] = torch.where(
                pair.unsqueeze(-1).unsqueeze(-1), second_slot, swapped[..., first + 1, :, :]
            )
            swapped[..., second + 1, :, :] = torch.where(
                pair.unsqueeze(-1).unsqueeze(-1), first_slot, swapped[..., second + 1, :, :]
            )
    return swapped, swap_mask


class GapQuerySwap:
    """Temporarily swap active Gap queries and capture post-write memory.

    This context manager patches only the evaluation instance.  The query
    projection is changed after keys/values are computed, while the scan's
    write inputs remain untouched.  The optional snapshots make the causal
    invariant directly testable: reference and swapped runs must have the
    same persistent memory tensor after every write.
    """

    def __init__(self, backbone, *, swap_queries: bool, capture_memory: bool = True):
        if getattr(backbone, "bidirectional", False):
            raise ValueError("GapQuerySwap requires a causal OSRAM scan")
        self.backbone = backbone
        self.swap_queries = bool(swap_queries)
        self.capture_memory = bool(capture_memory)
        self.memory_snapshots: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self.swap_masks: list[tuple[int, torch.Tensor]] = []
        self._had_project = "_project_sequence" in backbone.__dict__
        self._had_scan = "_scan" in backbone.__dict__

    def __enter__(self):
        original_project = self.backbone._project_sequence
        original_scan = self.backbone._scan

        def project_wrapper(node, latents, availability, qmask, **kwargs):
            keys, values, queries = original_project(
                node, latents, availability, qmask, **kwargs
            )
            if self.swap_queries:
                queries, mask = swap_active_gap_queries(queries, availability)
                self.swap_masks.append((len(self.swap_masks), mask.detach().clone()))
            return keys, values, queries

        def scan_wrapper(*args, **kwargs):
            user_observer = kwargs.get("post_write_observer")

            def observer(time_index, memory, active):
                if self.capture_memory:
                    self.memory_snapshots.append(
                        (int(time_index), memory.detach().clone(), active.detach().clone())
                    )
                if user_observer is not None:
                    user_observer(time_index, memory, active)

            kwargs["post_write_observer"] = observer
            return original_scan(*args, **kwargs)

        self.backbone._project_sequence = project_wrapper
        self.backbone._scan = scan_wrapper
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._had_project:
            raise RuntimeError("GapQuerySwap cannot restore a pre-existing project patch")
        if self._had_scan:
            raise RuntimeError("GapQuerySwap cannot restore a pre-existing scan patch")
        del self.backbone._project_sequence
        del self.backbone._scan
        return False
