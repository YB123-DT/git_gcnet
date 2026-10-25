"""One-stage corrections attached to one live original Flat trajectory."""
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from torch import nn

from osram_four_block_wiring.blocks import NeuMissBlock, SignatureBlock
from osram_four_block_wiring.csdi_latent import HypothesisBridge
from .diffusion import DifferentiableLatentCSDI


class JointBlockModel(nn.Module):
    def __init__(self, parent, mode, projection=None):
        super().__init__()
        if mode not in ('neumiss', 'signature', 'csdi'):
            raise ValueError('unknown joint block: ' + mode)
        if not isinstance(parent.smax_fc, nn.Linear):
            raise ValueError('requires original deterministic Flat linear task head')
        self.parent = parent.requires_grad_(True)
        self.mode = mode
        if mode == 'csdi':
            if projection is None:
                raise ValueError('CSDI requires TRAIN-fitted fixed projection')
            self.projection = projection.requires_grad_(False)
            self.denoiser = DifferentiableLatentCSDI(steps=20)
            self.bridge = HypothesisBridge()
        else:
            self.block = NeuMissBlock() if mode == 'neumiss' else SignatureBlock()

    @property
    def branch(self):
        return self.bridge if self.mode == 'csdi' else self.block

    def _target(self, complete, frame):
        # Targets use the current projector weights, without stochastic dropout
        # or a second observed-set fusion / OSRAM memory trajectory.
        projectors = self.parent.observed_set.projectors
        modes = [(m, m.training) for m in projectors.modules()]
        try:
            projectors.eval()
            devices = [complete.device.index] if complete.is_cuda else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                targets = []
                parts = torch.split(complete, self.parent.observed_set.dimensions, -1)
                for name, part in zip(('audio', 'text', 'visual'), parts):
                    latent = part.new_zeros(*part.shape[:2], 256)
                    if frame.valid.any():
                        latent[frame.valid] = projectors[name](part[frame.valid])
                    targets.append(latent)
                return self.projection(torch.stack(targets, -2))
        finally:
            for module, training in modes:
                module.training = training

    def forward(self, view, generator=None, query_keys=None):
        captured = {}
        valid = view['umask'].T.bool()
        availability = view['availability'].bool()

        def encoded(module, args, output):
            captured['latents'] = torch.stack([output[1][n] for n in ('audio', 'text', 'visual')], -2)

        def adapter(module, args):
            value = args[0]
            if value.shape[-1] != 4352:
                raise ValueError('requires original Large Flat 4352-dimensional adapter input')
            captured['local'] = value[..., :256]
            captured['base'] = value[..., 256:1280]
            captured['gap'] = value[..., 1280:].reshape(*value.shape[:2], 3, 1024)

        def anchor(module, args):
            if 'frame' in captured:
                raise RuntimeError('requires exactly one original Flat normalization')
            frame = SimpleNamespace(**captured, anchor=args[0], availability=availability, valid=valid)
            frame.context = torch.cat((frame.local, frame.base[..., :512],
                                       frame.gap[..., :512].flatten(-2)), -1)
            captured['frame'] = frame
            if self.mode != 'csdi':
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
        if completion is not None or 'frame' not in captured:
            raise RuntimeError('requires one original emotion-only Flat trajectory')
        frame = captured['frame']
        if self.mode == 'csdi':
            base_prediction = prediction
        else:
            base_prediction = self.parent.smax_fc(
                self.parent.osram.emotion_norm(frame.anchor) * valid[..., None])
        aux_loss = prediction.sum() * 0
        if self.mode == 'csdi':
            if generator is None:
                generator = torch.Generator(device=prediction.device).manual_seed(torch.initial_seed())
            observed = self.projection(frame.latents, availability)
            with nullcontext() if self.training else torch.no_grad():
                samples = self.denoiser.rsample(observed, availability, frame.context, valid,
                    n_samples=4, generator=generator, query_keys=query_keys)
            delta = self.bridge(frame.local, samples, availability, valid)
            hypothesis_hidden = self.parent.osram.emotion_norm(frame.anchor.unsqueeze(2) + delta)
            hypothesis_predictions = self.parent.smax_fc(hypothesis_hidden * valid[..., None, None])
            prediction = hypothesis_predictions.mean(2)
            if self.training:
                if 'complete' not in view:
                    raise ValueError('TRAIN CSDI requires complete features for noise targets')
                target = self._target(view['complete'], frame)
                aux_loss = self.denoiser.noise_loss(observed.detach(), target,
                    availability, frame.context.detach(), valid, generator)
        frame.hidden, frame.prediction = hidden, prediction
        return dict(prediction=prediction, base_prediction=base_prediction,
                    aux_loss=aux_loss, frame=frame)
