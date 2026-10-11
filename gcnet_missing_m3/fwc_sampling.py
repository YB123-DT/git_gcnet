"""Label-free, private-RNG trajectory and future write-credit sampling.

Ratios are ordered random/transition/persistent. A trajectory remasks raw
observations; it is NOT the write-only subset override used by Omit.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


@dataclass(frozen=True)
class CreditPair:
    batch: int
    conversation_id: str
    source_t: int
    source_modality: int
    future_j: int
    missing_modality: int
    regime: str = 'random'


_REGIMES = ('random', 'transition', 'persistent')
_PATTERNS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1))
_NAMES = ('A', 'T', 'V')


def _inputs(availability, umask, conversation_ids):
    if availability.ndim != 3 or availability.shape[-1] != 3:
        raise ValueError('availability must be [T,B,3]')
    if tuple(umask.shape) != (availability.shape[1], availability.shape[0]):
        raise ValueError('umask must be [B,T]')
    if len(conversation_ids) != availability.shape[1]:
        raise ValueError('one conversation ID per batch entry is required')
    u = umask.detach().cpu()
    if not torch.all((u == 0) | (u == 1)):
        raise ValueError('umask must be finite binary')
    u = u.bool()
    if u.shape[1] > 1 and torch.any(u[:, 1:] & ~u[:, :-1]):
        raise ValueError('valid utterances must be a contiguous prefix')
    a = availability.detach().cpu()
    a = torch.where(u.T[..., None], a, torch.zeros_like(a))
    if not torch.all((a == 0) | (a == 1)) or torch.any(u.T & ~a.bool().any(-1)):
        raise ValueError('valid availability must be nonempty finite binary')
    return a.bool(), u, [str(x) for x in conversation_ids]


def _generator(seed):
    return torch.Generator(device='cpu').manual_seed(int(seed))


def make_trajectories(base_availability, umask, conversation_ids, *, seed,
                      ratios=(1 / 3, 1 / 3, 1 / 3)):
    """Draw one regime per conversation, without consuming global RNG.

    Transition has a full prefix and a proper nonempty subset suffix; a
    conversation shorter than two turns falls back to its random base mask.
    Padding is always zero, including nonfinite values in padded input slots.
    """
    a, u, _ = _inputs(base_availability, umask, conversation_ids)
    if len(ratios) != 3 or any(not math.isfinite(float(x)) or x < 0 for x in ratios) or sum(ratios) <= 0:
        raise ValueError('ratios must be three finite nonnegative weights with positive sum')
    gen = _generator(seed)
    weights = torch.tensor(ratios, dtype=torch.float64)
    regimes, changes, fallbacks = [], [], 0
    for b in range(a.shape[1]):
        n = int(u[b].sum())
        regime = _REGIMES[int(torch.multinomial(weights, 1, generator=gen))]
        change = None
        if regime == 'transition' and n < 2:
            regime = 'random'
            fallbacks += 1
        if regime != 'random' and n:
            pattern = torch.tensor(_PATTERNS[int(torch.randint(6, (), generator=gen))], dtype=torch.bool)
            if regime == 'persistent':
                a[:n, b] = pattern
            else:
                change = int(torch.randint(1, n, (), generator=gen))
                a[:change, b] = True
                a[change:n, b] = pattern
        regimes.append(regime)
        changes.append(change)
    valid_count = int(u.sum())
    missing = ((~a) & u.T[..., None]).sum((0, 1))
    pattern_counts, by_regime = {}, {}
    for regime in _REGIMES:
        indices = [b for b, r in enumerate(regimes) if r == regime]
        rows = [a[:int(u[b].sum()), b] for b in indices]
        rows = torch.cat(rows) if rows else torch.empty((0, 3), dtype=torch.bool)
        counts = {}
        for row in rows.tolist():
            name = ''.join(m for m, active in zip(_NAMES, row) if active)
            counts[name] = counts.get(name, 0) + 1
            pattern_counts[name] = pattern_counts.get(name, 0) + 1
        by_regime[regime] = {
            'conversations': len(indices), 'valid_utterances': len(rows),
            'pattern_counts': counts,
            'missing_ratio_by_modality': {m: float((~rows[:, i]).float().mean()) if len(rows) else 0.0
                                          for i, m in enumerate(_NAMES)},
        }
    diagnostics = {
        'valid_utterances': valid_count, 'pattern_counts': pattern_counts,
        'missing_ratio': float(missing.sum()) / max(1, 3 * valid_count),
        'missing_ratio_by_modality': {m: int(missing[i]) / max(1, valid_count) for i, m in enumerate(_NAMES)},
        'by_regime': by_regime, 'transition_points': changes,
        'transition_short_fallbacks': fallbacks,
    }
    return a.to(device=base_availability.device, dtype=base_availability.dtype), regimes, diagnostics


def sample_pairs(availability, umask, conversation_ids, *, seed, horizon=4,
                 pairs_per_conversation=1, strategy='uniform', regimes=None):
    """Sample without replacement; each conversation has an independent cap.

    Uniform is uniform over legal tuples. Source-balanced round-robins available
    source modalities with a randomized initial order; within each modality all
    legal tuples are uniformly shuffled. No labels are accepted by this API.
    """
    a, u, ids = _inputs(availability, umask, conversation_ids)
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    if not isinstance(pairs_per_conversation, int) or pairs_per_conversation < 0:
        raise ValueError('pairs_per_conversation must be a nonnegative integer')
    if strategy not in ('uniform', 'source-balanced', 'source_balanced'):
        raise ValueError('unknown pair sampling strategy')
    regimes = ['random'] * len(ids) if regimes is None else list(regimes)
    if len(regimes) != len(ids) or any(r not in _REGIMES for r in regimes):
        raise ValueError('regimes must match the batch and be valid regime names')
    gen = _generator(seed)
    selected = []
    for b, cid in enumerate(ids):
        n = int(u[b].sum())
        candidates = []
        for t in range(n):
            if int(a[t, b].sum()) < 2:
                continue
            for j in range(t + 1, min(n, t + horizon + 1)):
                for s in range(3):
                    for m in range(3):
                        if a[t, b, s] and not a[j, b, m]:
                            candidates.append(CreditPair(b, cid, t, s, j, m, regimes[b]))
        order = torch.randperm(len(candidates), generator=gen).tolist()
        shuffled = [candidates[i] for i in order]
        if strategy == 'uniform':
            selected.extend(shuffled[:pairs_per_conversation])
        else:
            buckets = [[p for p in shuffled if p.source_modality == s] for s in range(3)]
            sources = torch.randperm(3, generator=gen).tolist()
            count = 0
            while count < pairs_per_conversation and any(buckets):
                for s in sources:
                    if buckets[s] and count < pairs_per_conversation:
                        selected.append(buckets[s].pop())
                        count += 1
    return selected


def validate_pairs(pairs, availability, umask, conversation_ids, *, horizon=4):
    """Raise on illegal tuples; useful for recording replay/manifest checks."""
    a, u, ids = _inputs(availability, umask, conversation_ids)
    for p in pairs:
        valid = (0 <= p.batch < len(ids) and 0 <= p.source_modality < 3
                 and 0 <= p.missing_modality < 3 and 0 <= p.source_t < p.future_j < a.shape[0]
                 and p.future_j - p.source_t <= horizon)
        if not valid:
            raise ValueError('invalid pair indices or horizon')
        b, t, j = p.batch, p.source_t, p.future_j
        if (p.conversation_id != ids[b] or not u[b, t] or not u[b, j]
                or not a[t, b, p.source_modality] or a[t, b].sum() < 2
                or a[j, b, p.missing_modality] or p.regime not in _REGIMES):
            raise ValueError('pair violates conversation, observation or padding constraints')
    return True


def pair_hash(pairs):
    """Stable ordered manifest digest for E2/E3/E4 pairing audits."""
    encoded = json.dumps([asdict(p) for p in pairs], sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()
