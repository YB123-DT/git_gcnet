"""Replay only Flat/head; the original Local Skip and Memory are retained."""
import math

import torch


def disable_correction_slot(raw, corrected, slot):
    """Remove one external Nested correction, never the original evidence."""
    if '+' in slot:
        for item in slot.split('+'):
            corrected = disable_correction_slot(raw, corrected, item)
        return corrected
    assert slot in ('Local', 'Base', 'Gap-A', 'Gap-T', 'Gap-V')
    local, base, gap = corrected
    if slot == 'Local':
        local = raw[0]
    elif slot == 'Base':
        base = raw[1]
    else:
        gap = gap.clone()
        index = ('Gap-A', 'Gap-T', 'Gap-V').index(slot)
        gap[:, :, index] = raw[2][:, :, index]
    return local, base, gap


def scale_correction_slot(raw, corrected, slot, scale):
    """Scale exactly one external Nested correction around its raw anchor.

    The other four correction slots are returned byte-for-byte unchanged.  This
    is an inference intervention only: it never changes Memory, availability,
    or the original Local Skip input.
    """
    names = ('Local', 'Base', 'Gap-A', 'Gap-T', 'Gap-V')
    if slot not in names or not math.isfinite(float(scale)):
        raise ValueError(f'Invalid correction scale: slot={slot!r}, scale={scale!r}')
    local, base, gap = corrected
    if slot == 'Local':
        local = raw[0] + float(scale) * (local - raw[0])
    elif slot == 'Base':
        base = raw[1] + float(scale) * (base - raw[1])
    else:
        gap = gap.clone()
        index = names[2:].index(slot)
        gap[:, :, index] = (raw[2][:, :, index]
                            + float(scale) * (gap[:, :, index] - raw[2][:, :, index]))
    return local, base, gap


def flat_prediction(model, local, base, gap, availability, umask, readout=None):
    valid = umask.T.bool().unsqueeze(-1)
    original_local = torch.where(valid, local, 0.)
    rl, rb, rg = (local, base, gap) if readout is None else readout
    rl = torch.where(valid, rl, 0.)
    rb = torch.where(valid, rb, 0.)
    active_gap = valid.unsqueeze(-1) & (availability < .5).unsqueeze(-1)
    rg = torch.where(active_gap, rg, 0.)
    inputs = torch.cat((rl, rb, rg.flatten(2)), -1)
    hidden = model.osram.emotion_norm(
        model.osram.local_skip(original_local) + model.osram.emotion_adapter(inputs))
    pred = model.smax_fc(hidden).squeeze(-1)
    return torch.where(valid.squeeze(-1), pred, 0.)
