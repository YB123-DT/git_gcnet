"""Replay only Flat/head; the original Local Skip and Memory are retained."""
import torch


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
