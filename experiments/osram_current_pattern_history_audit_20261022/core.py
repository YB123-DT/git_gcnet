"""Read-only seven-current-pattern CHM diagnostics; no optimizer or updates.

Writer gradients exclude the target's direct Key -> Gap-addressing derivative.
They measure only paths through strictly earlier joint writes, not a new loss.
Fixed-state replay is a separate no-gradient memory-content intervention.
"""
from contextlib import contextmanager, nullcontext
import math

import torch


PATTERNS = ('A', 'T', 'V', 'AT', 'AV', 'TV', 'ATV')
MODALITIES = ('audio', 'text', 'visual')


def pattern_bits(name):
    if name not in PATTERNS:
        raise ValueError('pattern must be one of ' + ', '.join(PATTERNS))
    return tuple(int(letter in name) for letter in 'ATV')


def prefix_for_pattern(background, b, t, dimensions, name):
    """Keep one conversation's complete past, changing only its target mask.

    ``complete`` must be the original input features, not previously masked
    ones: all seven current combinations are interventions on the same target.
    Masks are pre-existing backgrounds; this function does not sample any RNG.
    """
    bits = pattern_bits(name)
    a = background['availability']
    umask = background['umask']
    complete = background['complete']
    if (a.ndim != 3 or a.shape[-1] != 3
            or umask.shape != (a.shape[1], a.shape[0])
            or complete.shape[:2] != a.shape[:2]):
        raise ValueError('inconsistent availability/umask/feature shapes')
    if (len(dimensions) != 3 or any(not isinstance(d, int) or d <= 0 for d in dimensions)
            or sum(dimensions) != complete.shape[-1]):
        raise ValueError('dimensions must match the complete A/T/V input')
    if (isinstance(b, bool) or isinstance(t, bool) or not isinstance(b, int)
            or not isinstance(t, int) or not (0 <= b < a.shape[1] and 0 <= t < a.shape[0])):
        raise ValueError('target indices are out of bounds')
    for label, value in (('availability', a), ('umask', umask)):
        if not bool(torch.isfinite(value).all()) or not bool(((value == 0) | (value == 1)).all()):
            raise ValueError(label + ' must be finite and binary')
    if not bool(umask[b, t]):
        raise ValueError('target must be a valid utterance')
    valid = umask.bool().T
    if bool(a[~valid].any()) or bool((a.sum(-1)[valid] < 1).any()):
        raise ValueError('availability must be nonempty on valid utterances and zero on padding')
    selected = a[:t+1, b:b+1].detach().clone()
    selected[-1, 0] = selected.new_tensor(bits)
    features = complete[:t+1, b:b+1].detach().clone()
    expanded = torch.repeat_interleave(selected.bool(),
        torch.tensor(dimensions, device=selected.device), dim=-1)
    incomplete = torch.where(expanded, features, torch.zeros_like(features))
    # Preserve already-materialized past features byte-for-byte, including the
    # exact original masking representation and source tensor dtype/layout.
    if 'incomplete' in background and t:
        incomplete[:-1] = background['incomplete'][:t, b:b+1].detach()
    mask = umask[b:b+1, :t+1].detach().clone()
    return dict(complete=features, incomplete=incomplete, availability=selected,
        qmask=background['qmask'][b:b+1, :t+1].detach().clone(), umask=mask,
        labels=background['labels'][b:b+1, :t+1].detach().clone(),
        lengths=[int(mask.sum())], conversation_ids=[background['conversation_ids'][b]])


def write_parameters(model):
    """Unique existing Writer parameters only; no Query/encoder/Flat parameters."""
    named = [(name, p) for name, p in model.named_parameters()
             if name.startswith(('osram.key_projectors.', 'osram.value_projectors.'))]
    if not named or len({id(p) for _, p in named}) != len(named):
        raise ValueError('expected a unique nonempty original Key/Value projector parameter list')
    return named


@contextmanager
def historical_writer_only(model):
    """Detach only the current Key output, retaining its identical numeric value.

    Original read-before-write makes current Values irrelevant to the target.
    Historical Keys still enter their original joint writes and intermediate
    Gap addressing. Shared projector parameters are not changed or re-created.
    """
    if any(module.training for module in model.modules()):
        raise ValueError('historical writer diagnostics require eval mode')
    osram = model.osram
    if osram.bidirectional:
        raise ValueError('historical writer diagnostics require a causal CHM')
    original = osram._project_sequence
    had_override = '_project_sequence' in osram.__dict__
    def projected(*args, **kwargs):
        keys, values, queries = original(*args, **kwargs)
        return {name: torch.cat((key[:-1], key[-1:].detach()), dim=0)
                for name, key in keys.items()}, values, queries
    osram._project_sequence = projected
    try:
        yield
    finally:
        if had_override:
            osram._project_sequence = original
        else:
            del osram._project_sequence


def target_gradient(model, view, named=None):
    """Target MSE and historical-writer derivatives without touching ``.grad``.

    Returned gradients are detached CPU tensors. ``memory`` is the original
    target pre-write, already-alpha-decayed state, also detached on CPU.
    All seven callers must compare ``memory`` and ``past_prediction`` exactly.
    """
    from experiments.osram_tasr_20261018.runner_helpers import forward
    from experiments.osram_text_condition_optimization_audit_20261022.run import (
        isolated_current_encoding, prewrite_capture)
    named = write_parameters(model) if named is None else list(named)
    allowed = dict(write_parameters(model))
    if (not named or len({id(p) for _, p in named}) != len(named)
            or any(name not in allowed or allowed[name] is not p for name, p in named)):
        raise ValueError('named must contain unique original Writer parameters only')
    if any(not p.requires_grad for _, p in named):
        raise ValueError('Writer parameters must require gradients for this diagnostic')
    target = view['availability'].shape[0] - 1
    if view['availability'].shape[1] != 1 or target < 0 or not bool(view['umask'][0, target]):
        raise ValueError('expected one valid target-prefix conversation')
    segmented = isolated_current_encoding(model) if target else nullcontext()
    with torch.enable_grad(), segmented, historical_writer_only(model):
        with prewrite_capture(model.osram, target) as capture:
            prediction = forward(model, view)
        if prediction.shape[-1] != 1:
            raise ValueError('this diagnostic defines scalar sentiment MSE, not multiclass CE')
        residual = prediction[target, 0, 0] - view['labels'][0, target]
        loss = residual.square()
        grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    if capture['calls'] != target + 1 or 'memory' not in capture:
        raise RuntimeError('expected one original causal CHM scan and target pre-write capture')
    if not bool(torch.isfinite(loss)) or any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
        raise RuntimeError('nonfinite target loss or historical Writer gradient')
    return dict(loss=float(loss.detach()), prediction=float(prediction[target, 0, 0].detach()),
        residual=float(residual.detach()), gradients=tuple(None if g is None else g.detach().cpu() for g in grads),
        memory=capture['memory'], past_prediction=prediction[:target].detach().cpu().clone(),
        calls=capture['calls'])


def geometry(named, grads_a, grads_b):
    """CPU float64 cosine/dot/norm, disjoint Key/Value groups and overall.

    Undefined/zero gradients produce a null cosine, not a conflict. ``visible``
    and ``missing`` field names retain the existing audit report schema; they
    here denote the first and second of any two of the seven current patterns.
    """
    named, grads_a, grads_b = list(named), list(grads_a), list(grads_b)
    if len(named) != len(grads_a) or len(named) != len(grads_b):
        raise ValueError('gradient lengths must match named parameters')
    if len({name for name, _ in named}) != len(named) or len({id(p) for _, p in named}) != len(named):
        raise ValueError('named parameters must be unique')
    partition = {'WriteKey': [], 'WriteValue': []}
    for index, (name, p) in enumerate(named):
        if name.startswith('osram.key_projectors.'):
            group = 'WriteKey'
        elif name.startswith('osram.value_projectors.'):
            group = 'WriteValue'
        else:
            raise ValueError('geometry expects Writer parameters only')
        partition[group].append(index)
        for gradient in (grads_a[index], grads_b[index]):
            if gradient is not None and (gradient.shape != p.shape or not bool(torch.isfinite(gradient).all())):
                raise ValueError('invalid gradient for ' + name)
    def summarize(indices):
        aa = bb = dot = 0.
        defined_a = defined_b = 0
        for index in indices:
            a, b = grads_a[index], grads_b[index]
            if a is not None:
                a = a.detach().to(device='cpu', dtype=torch.float64)
                aa += float(a.square().sum()); defined_a += 1
            if b is not None:
                b = b.detach().to(device='cpu', dtype=torch.float64)
                bb += float(b.square().sum()); defined_b += 1
            if a is not None and b is not None:
                dot += float((a*b).sum())
        norm_a, norm_b = math.sqrt(aa), math.sqrt(bb)
        return dict(dot=dot, visible_norm=norm_a, missing_norm=norm_b,
            cosine=max(-1., min(1., dot/norm_a/norm_b)) if norm_a and norm_b else None,
            parameter_count=len(indices), numel=sum(named[i][1].numel() for i in indices),
            visible_defined=defined_a, missing_defined=defined_b,
            visible_unused=len(indices)-defined_a, missing_unused=len(indices)-defined_b)
    return dict(overall=summarize(list(range(len(named)))),
        groups={name: summarize(indices) for name, indices in partition.items()},
        partition={name: [named[i][0] for i in indices] for name, indices in partition.items()},
        zero_norm_note='A zero norm has null cosine and is not evidence of gradient conflict.')


@contextmanager
def fixed_history_reads(osram, decayed_states):
    """Read fixed *already decayed* [T,B,H,V,K] states; never write/decay again.

    Current original Query/Key/availability still go through original_read.
    This is no-grad content replay, not the first experiment's gradient path.
    States are cloned once so an external caller cannot alter them mid-replay.
    """
    from experiments.osram_meld_write_retention_20261014.task_replay import original_read
    if osram.training or osram.bidirectional:
        raise ValueError('fixed history replay requires eval and causal CHM')
    states = torch.stack(list(decayed_states)) if not torch.is_tensor(decayed_states) else decayed_states
    if (states.ndim != 5 or tuple(states.shape[-3:]) != (osram.num_heads, osram.value_dim, osram.key_dim)
            or states.shape[0] < 1 or states.shape[1] < 1):
        raise ValueError('fixed states shape must be [T,B,heads,value_dim,key_dim]')
    if not bool(torch.isfinite(states).all()):
        raise ValueError('fixed states must be finite')
    states = states.detach().clone()
    original = osram._scan
    had_override = '_scan' in osram.__dict__
    def scan(keys, values, queries, availability, valid, reverse=False, **kwargs):
        if reverse or any(value is not None for value in kwargs.values()):
            raise ValueError('fixed history reads require one plain causal scan without interventions')
        if states.shape[:2] != queries.shape[:2] or states.device != queries.device or states.dtype != queries.dtype:
            raise ValueError('fixed state shape/device/dtype must match the current Query sequence')
        bases, gaps = [], []
        for t in range(queries.shape[0]):
            observed = torch.stack([keys[name][t] for name in MODALITIES], -1)
            base, gap = original_read(osram, states[t], queries[t], observed, availability[t], valid[t])
            bases.append(base); gaps.append(gap)
        diagnostics = {name: {metric: [] for metric in ('rho', 'eta', 'cosine')} for name in MODALITIES}
        return torch.stack(bases), torch.stack(gaps), diagnostics
    osram._scan = scan
    try:
        yield
    finally:
        if had_override:
            osram._scan = original
        else:
            del osram._scan
