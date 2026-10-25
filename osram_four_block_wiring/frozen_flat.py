"""Read one frozen causal Flat forward; decode with frozen original weights."""
from dataclasses import dataclass, fields, replace
import torch


@dataclass
class FlatFrame:
    latents: torch.Tensor
    local: torch.Tensor
    base: torch.Tensor
    gap: torch.Tensor
    context: torch.Tensor
    anchor: torch.Tensor
    hidden: torch.Tensor
    prediction: torch.Tensor
    availability: torch.Tensor
    valid: torch.Tensor

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})


class FrozenFlatReader:
    def __init__(self, parent):
        self.parent = parent.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, incomplete, availability, qmask, umask, lengths):
        self.parent.eval()
        anchor = []
        scans = []
        scan_handle = (self.parent.osram.register_forward_hook(lambda *args: scans.append(1))
                       if hasattr(self.parent.osram, '_scan') else None)
        handle = self.parent.osram.emotion_norm.register_forward_pre_hook(
            lambda module, args: anchor.append(args[0]))
        try:
            prediction, hidden, latents, completion = self.parent(
                [incomplete], availability, qmask, umask, lengths, predict_missing=False)
        finally:
            handle.remove()
            if scan_handle is not None:
                scan_handle.remove()
        if len(anchor) != 1 or completion is not None:
            raise RuntimeError('Require one original Flat normalization, no completion')
        if scan_handle is not None and len(scans) != 1:
            raise RuntimeError('Require exactly one original OSRAM forward')
        c = self.parent.last_osram_context
        valid, a = umask.T.bool(), availability.bool()
        base = c['base']
        gap = c['gap'].masked_fill(a[..., None], 0)
        if base.shape[-1] != 1024 or hidden.shape[-1] != 1600:
            raise ValueError('Original Large Flat dimensions required')
        frame = FlatFrame(torch.stack([latents[n] for n in ('audio','text','visual')], -2),
            c['local'], base, gap,
            torch.cat((c['local'], base[..., :512], gap[..., :512].flatten(-2)), -1),
            anchor[0], hidden, prediction, a, valid)
        return frame

    def decode_delta(self, frame, delta):
        # The frozen weights must remain differentiable with respect to delta.
        anchor = frame.anchor if delta.ndim == 3 else frame.anchor.unsqueeze(2)
        hidden = self.parent.osram.emotion_norm(anchor + delta)
        valid = frame.valid[..., None] if delta.ndim == 3 else frame.valid[..., None, None]
        return self.parent.smax_fc(hidden * valid)
