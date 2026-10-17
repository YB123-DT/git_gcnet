"""Isolated adapter for the existing ComP implementation (two stages intact)."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.model import ComP
from .vendor.loss import DisentangledLoss, MaskedCELoss


def _get(config, *keys, default=None):
    for key in keys:
        value = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
        if value is not None:
            return value
    return default


class ComPAdapter(nn.Module):
    def __init__(self, config, dims, nclasses, nspeakers):
        super().__init__()
        if isinstance(dims, dict):
            dims = tuple(dims[k] for k in ('audio', 'text', 'visual'))
        self.seq_len = int(_get(config, 'seq_len'))
        self.hidden = int(_get(config, 'hidden', default=256))
        self.nclasses = nclasses
        self.temporal_causal = bool(_get(config, 'temporal_causal', default=True))
        args = SimpleNamespace(
            seq_len=self.seq_len,
            drop_rate=float(_get(config, 'dropout', 'drop_rate', default=0.5)),
            temporal_causal=self.temporal_causal,
            device=torch.device('cpu'),
        )
        self.model = ComP(
            args, *dims, self.hidden, nclasses,
            depth=int(_get(config, 'depth', default=4)),
            num_heads=int(_get(config, 'heads', 'num_heads', default=2)),
            mlp_ratio=float(_get(config, 'mlp_ratio', default=1)),
            drop_rate=args.drop_rate,
            attn_drop_rate=float(_get(config, 'attention_dropout', 'attn_drop_rate', default=0)),
            no_cuda=True,
            lbd=float(_get(config, 'lambda', 'lbd', default=0.3)),
        )

    def forward(self, a, t, v, availability, padding, first_stage=False):
        batch, length, _ = a.shape
        if length > self.seq_len:
            raise ValueError('ComP seq_len must be the dataset maximum, not the minibatch maximum')
        if availability.shape != (batch, length, 3) or padding.shape != (batch, length):
            raise ValueError('Expected availability BL3 and padding BL (True means valid)')
        valid = availability.bool() & padding.bool().unsqueeze(-1)
        if not valid.any(-1)[padding.bool()].all():
            raise ValueError('Every valid utterance needs at least one visible modality')
        features = [torch.where(valid[..., i:i+1], x, torch.zeros_like(x))
                    for i, x in enumerate((a, t, v))]
        extra = self.seq_len - length
        features = [F.pad(x, (0, 0, 0, extra)) for x in features]
        full_valid = F.pad(valid, (0, 0, 0, extra)).float()
        full_padding = F.pad(padding, (0, extra)).float()
        # The implementation accepts label but never reads it; never pass targets.
        dummy_labels = torch.zeros(batch, self.seq_len, dtype=torch.long, device=a.device)
        raw = self.model(torch.cat(features, -1).transpose(0, 1), dummy_labels,
                         full_valid.transpose(0, 1), full_padding, first_stage)
        return {'logits': raw[1][:, :length],
                'aux': raw,
                'full_padding': full_padding,
                'length': length,
                'first_stage': first_stage}

    def stage_loss(self, output, labels, padding, first_stage):
        return stage_loss(output, labels, padding, first_stage)

    def after_backward(self, output, labels, padding, first_stage=False):
        return after_backward(self, output, labels, padding, first_stage)


def build_model(config, dims, nclasses, nspeakers):
    """dims is (audio, text, visual); speaker count is unused by original ComP."""
    return ComPAdapter(config, dims, nclasses, nspeakers)


def stage_loss(output, labels, padding, first_stage):
    """Reproduce train_mr classification objectives, including duplicated stage-2 CE."""
    raw = output['aux']
    _, logits, out_a, out_t, out_v, _, projected, reconstructed, shared, private, private_logits = raw
    full_padding = output['full_padding']
    labels = F.pad(labels.long(), (0, full_padding.shape[1] - labels.shape[1]))
    # Ignore labels of padding; labels enter losses only, never model prediction.
    labels = labels.masked_fill(~full_padding.bool(), 0).reshape(-1)
    cls = MaskedCELoss()
    modality = [x.reshape(-1, x.shape[-1]) for x in (out_a, out_t, out_v)]
    modality_loss = sum(cls(x, labels, full_padding) for x in modality)
    if first_stage:
        for x in private_logits.chunk(3, -1):
            x = x.reshape(-1, x.shape[-1])
            modality_loss = modality_loss + cls(torch.ones_like(x) - x, labels, full_padding)
    disentangle = logits.new_zeros(())
    private_list, shared_list = private.chunk(3, -1), shared.chunk(3, -1)
    for i in range(3):
        for j in range(i + 1, 3):
            disentangle = disentangle + DisentangledLoss().foward_2(
                private_list[i], private_list[j], shared_list[i], shared_list[j], full_padding)
    if first_stage:
        return modality_loss + disentangle
    fusion = cls(logits.reshape(-1, logits.shape[-1]), labels, full_padding)
    return fusion + 2 * modality_loss + F.mse_loss(reconstructed, projected) + disentangle


def after_backward(model, output, labels, padding, first_stage=False):
    """Preserve upstream modulation's actual no-op; its parameter names do not match.

    train_mr looks for prop_former_a.proj_n.fc1 whereas the model defines
    prop_former_a_1.proj_n.fc1 (and likewise for t/v). Correcting that would change
    both the prior causal implementation and the upstream original algorithm.
    """
    return {'matched_parameters': 0, 'reason': 'upstream_parameter_pattern_matches_none'}
