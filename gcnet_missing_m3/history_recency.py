"""Strict-past modality coverage and recency metadata."""

from __future__ import annotations

import torch


def strict_past_history_recency(
    availability: torch.Tensor,
    umask: torch.Tensor,
) -> torch.Tensor:
    """Return ``[seen_A/T/V, distance_A/T/V]`` from indices before ``t``.

    The current availability is applied only after the current row is emitted.
    Invalid utterances produce zeros and never update the history state.
    """
    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError('availability must have shape [length, batch, 3]')
    length, batch, _ = availability.shape
    if umask.shape != (batch, length):
        raise ValueError('umask must have shape [batch, length]')

    device = availability.device
    dtype = availability.dtype if availability.is_floating_point() else torch.float32
    valid = umask.T.bool()
    observed = availability.bool()
    seen = torch.zeros(batch, 3, dtype=torch.bool, device=device)
    last_seen = torch.zeros(batch, 3, dtype=torch.long, device=device)
    rows = []
    for index in range(length):
        current_valid = valid[index]
        coverage = seen & current_valid[:, None]
        distance = torch.where(
            coverage,
            torch.full_like(last_seen, index) - last_seen,
            torch.zeros_like(last_seen),
        )
        rows.append(torch.cat((coverage.to(dtype), distance.to(dtype)), dim=-1))

        current_observed = observed[index] & current_valid[:, None]
        last_seen = torch.where(
            current_observed, torch.full_like(last_seen, index), last_seen)
        seen = seen | current_observed
    return torch.stack(rows, dim=0)

