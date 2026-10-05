"""Observed-only coalition encoding and exact Boolean-lattice Möbius dividends.

The supplied mean ObservedSetEncoder owns all feature/fusion parameters. Its
projectors run once per observed modality; the resulting real modality latents
are also returned unchanged for OSRAM values. F(empty) is fixed to zero.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


MODALITIES = ('audio', 'text', 'visual')
COALITIONS = {'A': (0,), 'T': (1,), 'V': (2,), 'AT': (0, 1),
              'AV': (0, 2), 'TV': (1, 2), 'ATV': (0, 1, 2)}


def mobius_transform(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return exact nonempty-set dividends, with F(empty)=0."""
    dividends = {}
    for name, members in COALITIONS.items():
        value = values[name]
        for subset, indices in COALITIONS.items():
            if len(indices) < len(members) and set(indices).issubset(members):
                value = value - dividends[subset]
        dividends[name] = value
    return dividends


class CoalitionMobiusBlock(nn.Module):
    """Three separately projected interaction orders followed by node fusion."""
    def __init__(self, latent_dim: int):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError('latent_dim must be positive')
        self.latent_dim = int(latent_dim)
        def projection(width):
            return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent_dim), nn.GELU())
        self.p1 = projection(latent_dim)
        self.p2 = projection(latent_dim)
        self.p3 = projection(latent_dim)
        self.pout = projection(3 * latent_dim + 3)
        self.diagnostics = {}
        self.last_diagnostics = self.diagnostics
        self.last_order_slots = None

    def decompose(self, encoder, features, availability, umask):
        """Encode every eligible S⊆O and return masked exact dividends.

        All coalition tensors have [L,B,D] shape and are zero outside their
        eligibility mask. The full observed-set encoding is evaluated first,
        exactly as in the original encoder, including its dropout/RNG advance.
        Every proper subset reuses that row's same fusion dropout mask.
        """
        if encoder.fusion_type != 'mean' or encoder.latent_dim != self.latent_dim:
            raise ValueError('CED requires a mean encoder of matching latent dimension')
        if len(encoder.fusion) != 4 or not isinstance(encoder.fusion[-1], nn.Dropout):
            raise ValueError('CED requires LN/Linear/GELU/Dropout mean fusion')
        valid = encoder._validate(features, availability, umask)
        shape = (*features.shape[:2], self.latent_dim)
        evidence = features.new_zeros(shape)
        latents, start = {}, 0
        for index, (name, width) in enumerate(zip(MODALITIES, encoder.dimensions)):
            selected = valid & availability[..., index].bool()
            latent = features.new_zeros(shape)
            if bool(selected.any()):
                projected = encoder.projectors[name](features[..., start:start + width][selected])
                latent[selected] = projected
                evidence[selected] += projected + encoder.modality_embedding.weight[index]
            latents[name] = latent
            start += width
        counts = availability.sum(-1, keepdim=True).clamp_min(1).to(features.dtype)
        pattern_id = (availability.long() * availability.new_tensor([4, 2, 1], dtype=torch.long)).sum(-1)
        full_input = evidence / counts + encoder.pattern_embedding(pattern_id)
        full_node, shared_mask = features.new_zeros(shape), features.new_zeros(shape)
        if bool(valid.any()):
            cpu_state = torch.get_rng_state()
            devices = [features.device.index if features.device.index is not None
                       else torch.cuda.current_device()] if features.is_cuda else []
            cuda_state = torch.cuda.get_rng_state(devices[0]) if devices else None
            full_node[valid] = encoder.fusion(full_input[valid])
            # Only the final fusion Dropout consumes RNG. Replay on ones within
            # a fork so mask extraction does not advance the caller's stream.
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, devices[0])
                shared_mask[valid] = encoder.fusion[-1](torch.ones_like(full_node[valid]))
        values, eligible = {}, {}
        for name, indices in COALITIONS.items():
            active = valid & availability[..., list(indices)].bool().all(-1)
            eligible[name] = active
            value = features.new_zeros(shape)
            full = active & (availability.sum(-1) == len(indices))
            value[full] = full_node[full]
            proper = active & ~full
            if bool(proper.any()):
                subset_evidence = sum(
                    (latents[MODALITIES[i]][proper] + encoder.modality_embedding.weight[i]
                     for i in indices), features.new_zeros((int(proper.sum()), self.latent_dim)))
                subset_pattern = sum((4, 2, 1)[i] for i in indices)
                subset_input = subset_evidence / len(indices) + encoder.pattern_embedding.weight[subset_pattern]
                value[proper] = encoder.fusion[:3](subset_input) * shared_mask[proper]
            values[name] = value
        dividends = mobius_transform(values)
        dividends = {name: value.masked_fill(~eligible[name][..., None], 0)
                     for name, value in dividends.items()}
        return {'coalitions': values, 'dividends': dividends, 'eligible': eligible,
                'latents': latents, 'valid': valid, 'availability': availability,
                'full_node': full_node}

    def forward(self, encoder, features, availability, umask):
        parts = self.decompose(encoder, features, availability, umask)
        valid = parts['valid']
        slots, order_counts = [], []
        for order, projector in enumerate((self.p1, self.p2, self.p3), start=1):
            slot = features.new_zeros((*features.shape[:2], self.latent_dim))
            count = features.new_zeros((*features.shape[:2], 1))
            for name, indices in COALITIONS.items():
                if len(indices) != order:
                    continue
                active = parts['eligible'][name]
                if bool(active.any()):
                    slot[active] += projector(parts['dividends'][name][active])
                count += active[..., None].to(features.dtype)
            slots.append(slot / count.clamp_min(1))
            order_counts.append(count.squeeze(-1))
        node = features.new_zeros((*features.shape[:2], self.latent_dim))
        if bool(valid.any()):
            output_input = torch.cat([*slots, availability.to(features.dtype)], dim=-1)
            node[valid] = self.pout(output_input[valid])
        self.last_order_slots = torch.stack(slots, dim=-2).detach()
        with torch.no_grad():
            reconstructed = sum(parts['dividends'].values())
            coalition_count = sum(parts['eligible'].values())
            self.diagnostics = {
                'coalition_count_mean': float(coalition_count[valid].float().mean()) if bool(valid.any()) else 0.0,
                'order_active_counts': [int((count > 0).sum()) for count in order_counts],
                'order_norms': [float(slot[valid].norm(dim=-1).mean()) if bool(valid.any()) else 0.0 for slot in slots],
                'reconstruction_max_error': float((reconstructed - parts['full_node']).abs().max()) if node.numel() else 0.0,
                'output_norm': float(node[valid].norm(dim=-1).mean()) if bool(valid.any()) else 0.0,
            }
            self.last_diagnostics = self.diagnostics
        return node, parts['latents']
