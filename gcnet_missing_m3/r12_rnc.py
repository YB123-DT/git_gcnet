"""Independent implementation of Rank-N-Contrast, arXiv:2210.01189 Eq. (3).

Inputs are single-view, valid utterances only (Appendix G.3); filtering/capping
belongs to the caller. OSRAM's MSE + 0.1 * RNC integration is a one-stage
adaptation, whereas the paper's original framework trains in two stages.
No author source is copied (the audited repository supplies no license).
"""

import math

import torch


def rnc_loss(features, labels, temperature=2.0):
    """Average label-distance-ranked contrastive loss for scalar regression.

    ``features`` is [N,D], ``labels`` is [N] or [N,1]. Negative Euclidean
    distance is used without normalization. Every nonself target contrasts
    against all nonself candidates with greater *or equal* label distance.
    Sorting followed by cumulative logsumexp avoids cubic pair masks/work;
    ties share a denominator containing the complete equal-distance group.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError('features must be a floating [N,D] tensor')
    count = features.shape[0]
    if labels.shape not in {(count,), (count, 1)}:
        raise ValueError('labels must have shape [N] or [N,1]')
    if count < 2:
        return features.sum() * 0.0

    # Keep distance/logsumexp arithmetic safe under mixed precision.
    values = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
    targets = labels.detach().to(device=features.device, dtype=values.dtype).reshape(count)
    distances = (targets[:, None] - targets[None, :]).abs()
    similarities = -torch.cdist(
        values, values, p=2, compute_mode='donot_use_mm_for_euclid_dist'
    ) / temperature
    nonself = ~torch.eye(count, dtype=torch.bool, device=features.device)
    distances = distances[nonself].reshape(count, count - 1)
    similarities = similarities[nonself].reshape(count, count - 1)

    distances, order = distances.sort(dim=1, descending=True)
    similarities = similarities.gather(1, order)
    cumulative = similarities.logcumsumexp(dim=1)

    # In descending order, the denominator ends at the last element of a tie.
    group_end = torch.ones_like(distances, dtype=torch.bool)
    group_end[:, :-1] = distances[:, :-1] != distances[:, 1:]
    positions = torch.arange(count - 1, device=features.device).expand_as(order)
    ends = torch.where(group_end, positions, count - 2)
    ends = ends.flip(1).cummin(dim=1).values.flip(1)
    return (cumulative.gather(1, ends) - similarities).mean()
