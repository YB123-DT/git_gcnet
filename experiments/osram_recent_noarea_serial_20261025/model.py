"""Recent-history branches on exactly one original Encoder/CHM trajectory."""
from types import SimpleNamespace

import torch
from torch import nn

from .blocks import NoAreaSignatureBlock, SerialNoAreaBlock


class RecentNoAreaModel(nn.Module):
    def __init__(self, parent, mode='parallel', stage='joint'):
        super().__init__()
        if mode not in ('parallel', 'serial'):
            raise ValueError('unknown recent NoArea mode: ' + mode)
        if stage not in ('joint', 'frozen'):
            raise ValueError('unknown recent NoArea stage: ' + stage)
        if not isinstance(parent.smax_fc, nn.Linear):
            raise ValueError('requires original deterministic Flat linear task head')
        self.parent = parent.requires_grad_(stage == 'joint')
        self.mode, self.stage = mode, stage
        self.block = NoAreaSignatureBlock() if mode == 'parallel' else SerialNoAreaBlock()
        if stage == 'frozen':
            self.parent.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.stage == 'frozen':
            self.parent.eval()
        return self

    def _decode(self, frame, base=None, gap=None):
        base = frame.base if base is None else base
        gap = frame.gap if gap is None else gap
        value = torch.cat((frame.local, base, gap.flatten(-2)), -1)
        anchor = self.parent.osram.local_skip(frame.local) + self.parent.osram.emotion_adapter(value)
        hidden = self.parent.osram.emotion_norm(anchor) * frame.valid[..., None]
        return self.parent.smax_fc(hidden)

    def predict_frame(self, frame):
        """Decode an original frozen FlatFrame without another memory scan."""
        if self.training or self.parent.training:
            raise ValueError('cached original FlatFrame prediction requires eval mode')
        if self.mode == 'serial':
            return self._decode(frame, *self.block.updated_reads(frame))
        hidden = self.parent.osram.emotion_norm(frame.anchor + self.block(frame))
        return self.parent.smax_fc(hidden * frame.valid[..., None])

    def forward(self, view, generator=None, query_keys=None):
        captured = {}
        valid, availability = view['umask'].T.bool(), view['availability'].bool()

        def encoded(module, args, output):
            if 'latents' in captured:
                raise RuntimeError('requires exactly one observed-set encoding')
            captured['latents'] = torch.stack([output[1][name]
                for name in ('audio', 'text', 'visual')], -2)

        def adapter(module, args):
            value = args[0]
            if value.shape[-1] != 4352:
                raise ValueError('requires original Large Flat 4352-dimensional adapter input')
            if 'frame' in captured:
                raise RuntimeError('requires exactly one original Flat adapter')
            frame = SimpleNamespace(latents=captured['latents'], local=value[..., :256],
                base=value[..., 256:1280], gap=value[..., 1280:].reshape(*value.shape[:2], 3, 1024),
                availability=availability, valid=valid)
            frame.context = torch.cat((frame.local, frame.base[..., :512],
                                       frame.gap[..., :512].flatten(-2)), -1)
            captured['frame'] = frame
            if self.mode == 'serial':
                delta = self.block(frame)
                frame.delta = delta
                base, gap = self.block.updated_reads(frame, delta)
                return (torch.cat((frame.local, base, gap.flatten(-2)), -1),)

        def anchor(module, args):
            frame = captured['frame']
            if hasattr(frame, 'anchor'):
                raise RuntimeError('requires exactly one original Flat normalization')
            frame.anchor = args[0]
            if self.mode == 'parallel':
                return (args[0] + self.block(frame),)

        handles = [self.parent.observed_set.register_forward_hook(encoded),
                   self.parent.osram.emotion_adapter.register_forward_pre_hook(adapter),
                   self.parent.osram.emotion_norm.register_forward_pre_hook(anchor)]
        try:
            prediction, hidden, _, completion = self.parent(
                [view['incomplete']], view['availability'], view['qmask'], view['umask'],
                view['lengths'], predict_missing=False)
        finally:
            for handle in handles:
                handle.remove()
        if completion is not None or 'frame' not in captured or not hasattr(captured['frame'], 'anchor'):
            raise RuntimeError('requires one original emotion-only Flat trajectory')
        frame = captured['frame']
        base_prediction = prediction
        if not self.training:
            if self.mode == 'serial':
                base_prediction = self._decode(frame)
            else:
                base_prediction = self.parent.smax_fc(
                    self.parent.osram.emotion_norm(frame.anchor) * valid[..., None])
        frame.hidden, frame.prediction = hidden, prediction
        return dict(prediction=prediction, base_prediction=base_prediction,
                    aux_loss=prediction.sum() * 0, frame=frame)
