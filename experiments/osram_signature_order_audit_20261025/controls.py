"""Read-only path controls for the *existing* frozen SignatureBlock.

No CHM input, availability, parent prediction, or learned weight is modified.
The controls isolate within-path latent order or the antisymmetric second-level
area. They do not claim CHM itself is invariant to history order.
"""
import torch
from torch.nn import functional as F


MODES = ("normal", "interior_shuffle", "degree2_no_area")


def _window_paths(block, frame):
    """Mirror SignatureBlock.descriptor's strict-past valid-utterance window."""
    projected = torch.stack([layer(frame.latents[..., m, :])
                             for m, layer in enumerate(block.slots)], -2)
    t, b = frame.valid.shape
    window = block.window
    if t == 0:
        return (projected.new_empty((b, 0, 3, window, 9)),
                frame.valid.new_empty((b, 0, 3, window)))
    valid = frame.valid.T
    positions = torch.arange(t, device=valid.device).expand(b, t)
    ordered = positions.masked_fill(~valid, t).sort(-1).values
    prior_count = valid.long().cumsum(-1) - valid.long()
    ranks = prior_count[..., None] + torch.arange(-window, 0, device=valid.device)
    indices = ordered.gather(1, ranks.clamp(0, t - 1).reshape(b, -1)).reshape(b, t, window)
    indices = indices.clamp_max(t - 1)
    batches = torch.arange(b, device=valid.device)[:, None, None]
    points = projected.transpose(0, 1)[batches, indices]
    observed = frame.availability.transpose(0, 1)[batches, indices] & (ranks >= 0)[..., None]
    times = torch.arange(1, window + 1, device=valid.device, dtype=points.dtype) / window
    times = times.view(1, 1, window, 1, 1).expand(b, t, window, 3, 1)
    return torch.cat([points, times], -1).transpose(2, 3), observed.transpose(2, 3)


def _levels(points, observed):
    """Use the original vectorized arithmetic and zero basepoint exactly."""
    slots = torch.arange(1, points.shape[-2] + 1, device=points.device)
    latest = torch.where(observed, slots, 0).cummax(-1).values
    previous = F.pad(latest[..., :-1], (1, 0))
    padded = F.pad(points, (0, 0, 1, 0))
    previous_points = padded.gather(-2, previous[..., None].expand(*previous.shape, 9))
    dx = (points - previous_points) * observed[..., None]
    prefix = dx.cumsum(-2) - dx
    second = ((prefix[..., :, :, None] * dx[..., :, None, :])
              + .5 * dx[..., :, :, None] * dx[..., :, None, :]).sum(-3)
    return dx.sum(-2), second


def _interior_permutation(observed, seed):
    """Permute >=2 observed interior slots with a private CPU generator.

    Identity draws are replaced by swapping the first two interior slots.
    This defines a reproducible intervention, not a uniform distribution over
    all nonidentity permutations. Time coordinates never follow this mapping.
    """
    b, t, modalities, window = observed.shape
    slots = torch.arange(window, device=observed.device).expand_as(observed)
    if window < 4:
        return slots
    count = observed.long().sum(-1, keepdim=True)
    ordinal = observed.long().cumsum(-1)
    interior = observed & (ordinal > 1) & (ordinal < count)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    # Time-major generation keeps prior-target draws unchanged if T is extended.
    keys = torch.rand((t, b, modalities, window), generator=generator,
                      device="cpu").permute(1, 0, 2, 3).to(observed.device)
    order = torch.where(interior, keys, 2.0 + slots).argsort(-1)
    rank = (interior.long().cumsum(-1) - 1).clamp_min(0)
    permutation = torch.where(interior, order.gather(-1, rank), slots)
    eligible = interior.sum(-1) >= 2
    identity = (permutation == slots).all(-1) & eligible
    natural_order = torch.where(interior, slots, window).sort(-1).values
    first, second = natural_order[..., :1], natural_order[..., 1:2]
    swapped = torch.where(slots == first, second, torch.where(slots == second, first, slots))
    return torch.where(identity[..., None], swapped, permutation)


def descriptor_control(block, frame, mode, shuffle_seed=66):
    """Return ``(descriptor, eligible, diagnostics)`` without changing weights.

    descriptor: [T,B,532], ready for ``block.out``. ``normal`` is obtained
      directly from the original ``block.descriptor(frame)`` (bit-exact).
    eligible: [T,B] bool, valid targets with >=4 observed points in at least
      one modal path, allowing a nonidentity endpoint-preserving shuffle.
      The same structural eligibility is returned for all controls; fewer
      points may still carry nonzero area in ``degree2_no_area``.
    diagnostics: counts/path_eligible [T,B,3], observed/permutation
      [T,B,3,W], original_points/controlled_points [T,B,3,W,9],
      has_history [T,B], original_area_norm/controlled_area_norm [T,B,3].
      Point tensors include unused slots, excluded by ``observed``.

    Interior shuffling changes only the eight projected latent coordinates of
    observed non-endpoint points. First/last observations, all time coordinates,
    missing slots, current Local/availability, and historical counts stay fixed.
    ``degree2_no_area`` retains S1 and replaces S2 by .5*S1 outer S1, removing
    oriented antisymmetric area rather than removing the entire second level.
    A structural permutation need not change a degenerate/collinear path.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown descriptor control mode: {mode!r}")
    points, observed = _window_paths(block, frame)
    b, t, _, window, _ = points.shape
    identity = torch.arange(window, device=points.device).expand(b, t, 3, window)
    permutation = identity
    controlled = points
    if mode == "interior_shuffle":
        permutation = _interior_permutation(observed, shuffle_seed)
        reordered_latents = points[..., :8].gather(-2, permutation[..., None].expand(b, t, 3, window, 8))
        controlled = torch.cat([reordered_latents, points[..., 8:]], -1)
    first, second = _levels(points, observed)
    changed_first, changed_second = _levels(controlled, observed)
    if mode == "degree2_no_area":
        changed_second = .5 * first[..., :, None] * first[..., None, :]
    counts = observed.sum(-1).transpose(0, 1)
    path_eligible = (counts >= 4) & frame.valid[..., None]
    signatures = torch.cat([changed_first, changed_second.flatten(-2)], -1).flatten(-2)
    descriptor = (block.descriptor(frame) if mode == "normal" else
                  torch.cat([signatures.transpose(0, 1), frame.local,
                             frame.availability.to(frame.local.dtype),
                             counts.to(frame.local.dtype)], -1))
    diagnostics = {
        "counts": counts,
        "path_eligible": path_eligible,
        "observed": observed.transpose(0, 1),
        "permutation": permutation.transpose(0, 1),
        "original_points": points.transpose(0, 1),
        "controlled_points": controlled.transpose(0, 1),
        "has_history": (frame.valid.long().cumsum(0) - frame.valid.long()) > 0,
        "original_area_norm": (.5 * (second - second.transpose(-1, -2))).norm(dim=(-2, -1)).transpose(0, 1),
        "controlled_area_norm": (.5 * (changed_second - changed_second.transpose(-1, -2))).norm(dim=(-2, -1)).transpose(0, 1),
    }
    return descriptor, path_eligible.any(-1), diagnostics
