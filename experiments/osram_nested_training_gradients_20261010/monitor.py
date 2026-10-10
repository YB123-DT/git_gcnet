"""Passive training-only gradient observer; never changes tensors or gradients."""
import hashlib
import math

import torch


class GradientMonitor:
    def __init__(self, kind, latent_dim=256):
        if kind not in ('flat', 'nested'):
            raise ValueError(kind)
        self.kind, self.latent_dim = kind, latent_dim
        self.active = False
        self.handles = []
        self.rows = []

    def start_epoch(self, model, epoch, config):
        self.model, self.epoch, self.config = model, epoch, config
        self.rows, self.batch = [], 0
        self.active = True
        if not self.handles:
            backbone = model.osram
            self.handles.append(backbone.register_forward_pre_hook(self._forward))
            self.handles.append(backbone.local_skip.register_forward_pre_hook(self._local))
            if self.kind == 'nested':
                self.handles.append(backbone.meaningful_block.register_forward_pre_hook(self._nested_input))
                self.handles.append(backbone.meaningful_block.register_forward_hook(self._nested_output))
            else:
                self.handles.append(backbone.emotion_adapter.register_forward_pre_hook(self._flat_input))

    def _forward(self, module, args):
        if not self.active:
            return
        av, umask = args[2], args[4]
        self.valid = umask.T.bool()
        self.history = self.valid & (self.valid.long().cumsum(0) > 1)
        self.av = av.bool()
        rate = float(self.config.train_missing_rates[(self.epoch+self.batch) % len(self.config.train_missing_rates)])
        self.row = dict(epoch=self.epoch+1, batch=self.batch+1, rate=rate,
                        valid_count=int(self.valid.sum()), inputs={}, residual={})
        self.row['availability_sha256'] = hashlib.sha256(av.detach().cpu().numpy().tobytes()).hexdigest()
        self.rows.append(self.row)
        self.batch += 1

    @staticmethod
    def _stats(values):
        return dict(count=int(values.numel()), mean=float(values.mean()) if values.numel() else None,
                    maximum=float(values.max()) if values.numel() else None)

    def _watch(self, tensor, slot, mask, select=None):
        if not tensor.requires_grad:
            raise ValueError('Observed training readout input must require gradients')
        row = self.row
        def hook(gradient):
            with torch.no_grad():
                value = select(gradient) if select else gradient
                norms = value[mask].float().norm(dim=-1)
                if not bool(torch.isfinite(norms).all()):
                    raise ValueError('Nonfinite input gradient')
                row['inputs'][slot] = self._stats(norms)
                # Native sample-mean MSE includes1/N. Also save N-scaled norms
                # for batch-denominator-independent readout comparisons.
                row['inputs'][slot]['sample_sum_equivalent_mean'] = (
                    float(norms.mean()) * row['valid_count'] if norms.numel() else None)
        tensor.register_hook(hook)

    def _local(self, module, args):
        if self.active:
            self._watch(args[0], 'Local', self.valid)

    def _nested_input(self, module, args):
        if not self.active:
            return
        local, base, gap, av, umask = args
        self._watch(base, 'Base', self.history, lambda g:g[..., :512])
        for i, slot in enumerate(('Gap-A', 'Gap-T', 'Gap-V')):
            self._watch(gap, slot, self.history & ~self.av[..., i], lambda g,i=i:g[..., i, :512])

    def _flat_input(self, module, args):
        if not self.active:
            return
        x = args[0]
        def memory(g):
            return g[..., self.latent_dim:].reshape(*g.shape[:2], 4, 1024)
        self._watch(x, 'Base', self.history, lambda g:memory(g)[..., 0, :512])
        for i, slot in enumerate(('Gap-A', 'Gap-T', 'Gap-V')):
            self._watch(x, slot, self.history & ~self.av[..., i], lambda g,i=i:memory(g)[..., i+1, :512])

    def _nested_output(self, module, args, output):
        if not self.active:
            return
        with torch.no_grad():
            old = (args[0], args[1][..., :512], *(args[2][..., i, :512] for i in range(3)))
            new = (output[0], output[1][..., :512], *(output[2][..., i, :512] for i in range(3)))
            masks = (self.valid, self.history, *(self.history & ~self.av[..., i] for i in range(3)))
            for slot, before, after, mask in zip(('Local', 'Base', 'Gap-A', 'Gap-T', 'Gap-V'), old, new, masks):
                delta = (after.detach()-before.detach())[mask].float().norm(dim=-1)
                magnitude = before.detach()[mask].float().norm(dim=-1)
                self.row['residual'][slot] = dict(count=int(delta.numel()),
                    norm_mean=float(delta.mean()) if delta.numel() else None,
                    ratio_mean=float((delta/(magnitude+1e-8)).mean()) if delta.numel() else None)

    def before_clip(self, model, max_norm):
        if not self.active:
            return
        groups = {'nested':'osram.meaningful_block.', 'flat_adapter':'osram.emotion_adapter.',
                  'local_skip':'osram.local_skip.', 'local_path':'osram.local_path.',
                  'observed_encoder':'observed_set.'}
        squares, counts = {}, {}
        total = 0.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                value = float(p.grad.detach().double().square().sum())
                if not math.isfinite(value):
                    raise ValueError('Nonfinite parameter gradient')
                total += value
                for group, prefix in groups.items():
                    if name.startswith(prefix):
                        squares[group] = squares.get(group, 0.) + value
                        counts[group] = counts.get(group, 0) + p.numel()
        norm = math.sqrt(total)
        self.row['preclip_global_norm'] = norm
        self.row['expected_clip_coefficient'] = min(1., float(max_norm)/(norm+1e-6))
        self.row['parameter_groups'] = {group: dict(norm=math.sqrt(squares.get(group, 0.)),
            gradient_parameter_count=counts.get(group, 0),
            rms=math.sqrt(squares.get(group, 0.)/max(1, counts.get(group, 0)))) for group in groups}

    def finish_epoch(self, expected_steps):
        self.active = False
        if len(self.rows) != expected_steps or any('parameter_groups' not in r for r in self.rows):
            raise ValueError('Observer step count mismatch')
        if any(set(r['inputs']) != {'Local','Base','Gap-A','Gap-T','Gap-V'} for r in self.rows):
            raise ValueError('Missing input gradient observation')
        return self.rows
