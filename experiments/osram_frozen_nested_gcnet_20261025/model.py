"""Original Nested input corrections on a single frozen original Flat scan."""
from types import SimpleNamespace

import torch
from torch import nn

from gcnet_missing_m3.meaningful_input import MeaningfulInputAdapter


class FrozenNestedModel(nn.Module):
    def __init__(self, parent):
        super().__init__()
        if not isinstance(parent.smax_fc, nn.Linear):
            raise ValueError('requires original deterministic Flat linear task head')
        self.parent = parent.requires_grad_(False).eval()
        self.block = MeaningfulInputAdapter(
            256, 1024, 1600, 'nested_gnn_rooted_evidence', 8, 64)

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def _decode(self, frame, reads):
        local, base, gap = reads
        value = torch.cat((local, base, gap.flatten(-2)), -1)
        # Corrected Local belongs only to the Adapter, never to Local Skip.
        # The captured concat slice has stride 4352; the original raw Local is
        # contiguous. Preserve its Linear GEMM layout for exact CUDA replay.
        anchor = (self.parent.osram.local_skip(frame.local.contiguous())
                  + self.parent.osram.emotion_adapter(value))
        hidden = self.parent.osram.emotion_norm(anchor) * frame.valid[..., None]
        return self.parent.smax_fc(hidden)

    def predict_frame(self, frame):
        """Decode cached original reads; keep gradients through the frozen head."""
        self.parent.eval()
        reads = self.block(frame.local, frame.base, frame.gap,
                           frame.availability, frame.valid.T)
        return self._decode(frame, reads)

    def forward(self, view):
        self.parent.eval()
        captured = {}
        valid = view['umask'].T.bool()
        availability = view['availability'].bool()

        def adapter(module, args):
            value = args[0]
            if value.ndim != 3 or value.shape[-1] != 4352:
                raise ValueError('requires original Large Flat 4352-dimensional adapter input')
            if 'frame' in captured:
                raise RuntimeError('requires exactly one original Flat adapter')
            frame = SimpleNamespace(
                local=value[..., :256], base=value[..., 256:1280],
                gap=value[..., 1280:].reshape(*value.shape[:2], 3, 1024),
                availability=availability, valid=valid)
            captured['frame'] = frame
            local, base, gap = self.block(frame.local, frame.base, frame.gap,
                                          availability, view['umask'])
            return (torch.cat((local, base, gap.flatten(-2)), -1),)

        handle = self.parent.osram.emotion_adapter.register_forward_pre_hook(adapter)
        try:
            prediction, hidden, _, completion = self.parent(
                [view['incomplete']], view['availability'], view['qmask'], view['umask'],
                view['lengths'], predict_missing=False)
        finally:
            handle.remove()
        if completion is not None or 'frame' not in captured:
            raise RuntimeError('requires one original emotion-only Flat trajectory')
        frame = captured['frame']
        base_prediction = prediction
        if not self.training:
            base_prediction = self._decode(frame, (frame.local, frame.base, frame.gap))
        frame.hidden, frame.prediction = hidden, prediction
        return dict(prediction=prediction, base_prediction=base_prediction,
                    aux_loss=prediction.sum() * 0, frame=frame)
