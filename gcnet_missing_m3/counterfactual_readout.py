"""Evidence-effect readouts that preserve the original Flat anchor."""

from __future__ import annotations

import torch
from torch import nn


PAIR_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _validate_adapter(adapter: nn.Module) -> None:
    expected = (nn.LayerNorm, nn.Linear, nn.GELU, nn.Dropout, nn.Linear)
    if not isinstance(adapter, nn.Sequential) or len(adapter) != len(expected):
        raise ValueError('counterfactual readout requires the original five-layer Flat adapter')
    if any(not isinstance(layer, kind) for layer, kind in zip(adapter, expected)):
        raise ValueError('counterfactual readout requires LN/Linear/GELU/Dropout/Linear')


def evaluate_flat_views(adapter: nn.Module, views: torch.Tensor) -> torch.Tensor:
    """Evaluate [L,B,V,I] views with one dropout mask shared across V.

    The mask contains the same number and ordering of random draws as one
    ordinary [L,B,I] Flat call. Deterministic layers still process all views.
    """
    _validate_adapter(adapter)
    hidden = adapter[2](adapter[1](adapter[0](views)))
    if adapter.training and adapter[3].p:
        mask_shape = (*hidden.shape[:-2], 1, hidden.shape[-1])
        mask = adapter[3](torch.ones(mask_shape, dtype=hidden.dtype, device=hidden.device))
        hidden = hidden * mask
    return adapter[4](hidden)


class CounterfactualEvidenceReadout(nn.Module):
    """Encode Flat response changes under Base/Gap evidence interventions."""

    def __init__(self, mode: str, context_dim: int, output_dim: int,
                 *, encoder_dropout: float = 0.5):
        super().__init__()
        if mode not in ('marginal', 'coalition'):
            raise ValueError('counterfactual readout must be marginal or coalition')
        if context_dim <= 0 or output_dim <= 0:
            raise ValueError('counterfactual readout dimensions must be positive')
        self.mode = mode
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.token_count = 4 if mode == 'marginal' else 10
        self.effect_projection = nn.Linear(output_dim, 128)
        self.type_embedding = nn.Embedding(self.token_count, 128)
        self.availability_projection = nn.Linear(3, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=encoder_dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.residual = nn.Linear(128, output_dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.last_diagnostics: dict[str, float | int | str] = {}
        self.last_effects: torch.Tensor | None = None
        self.last_active: torch.Tensor | None = None
        self.last_view_outputs: torch.Tensor | None = None
        self.last_view_count = 5 if mode == 'marginal' else 12

    @staticmethod
    def _masks(availability: torch.Tensor, umask: torch.Tensor):
        valid = umask.T.bool()
        history = valid & ((valid.long().cumsum(0) - valid.long()) > 0)
        gap_active = history[..., None] & ~availability.bool()
        evidence_active = torch.cat((history[..., None], gap_active), -1)
        return valid, history, gap_active, evidence_active

    def _evidence(self, base: torch.Tensor, gap: torch.Tensor,
                  history: torch.Tensor, gap_active: torch.Tensor) -> torch.Tensor:
        safe_base = torch.where(history[..., None], base, torch.zeros_like(base))
        safe_gap = torch.where(gap_active[..., None], gap, torch.zeros_like(gap))
        return torch.cat((safe_base.unsqueeze(-2), safe_gap), -2)

    @staticmethod
    def _flat_input(local: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return torch.cat((local, evidence.flatten(-2)), -1)

    def _marginal_views(self, local: torch.Tensor, evidence: torch.Tensor):
        full = self._flat_input(local, evidence)
        views = [full]
        for index in range(4):
            omitted = evidence.clone()
            omitted[..., index, :] = 0
            views.append(self._flat_input(local, omitted))
        return torch.stack(views, -2)

    def _coalition_views(self, local: torch.Tensor, evidence: torch.Tensor):
        empty = torch.zeros_like(evidence)
        views = [self._flat_input(local, empty)]
        for index in range(4):
            selected = torch.zeros_like(evidence)
            selected[..., index, :] = evidence[..., index, :]
            views.append(self._flat_input(local, selected))
        for left, right in PAIR_INDICES:
            selected = torch.zeros_like(evidence)
            selected[..., left, :] = evidence[..., left, :]
            selected[..., right, :] = evidence[..., right, :]
            views.append(self._flat_input(local, selected))
        views.append(self._flat_input(local, evidence))
        return torch.stack(views, -2)

    def _effects(self, outputs: torch.Tensor, evidence_active: torch.Tensor):
        if self.mode == 'marginal':
            effects = outputs[..., :1, :] - outputs[..., 1:, :]
            active = evidence_active
        else:
            empty = outputs[..., 0, :]
            singles = outputs[..., 1:5, :]
            pair_values = outputs[..., 5:11, :]
            singleton_effects = singles - empty.unsqueeze(-2)
            pair_effects = torch.stack([
                pair_values[..., index, :] - singles[..., left, :]
                - singles[..., right, :] + empty
                for index, (left, right) in enumerate(PAIR_INDICES)
            ], -2)
            pair_active = torch.stack(
                [evidence_active[..., left] & evidence_active[..., right]
                 for left, right in PAIR_INDICES], -1)
            effects = torch.cat((singleton_effects, pair_effects), -2)
            active = torch.cat((evidence_active, pair_active), -1)
        effects = torch.where(active[..., None], effects, torch.zeros_like(effects))
        return effects, active

    def _encode(self, effects: torch.Tensor, active: torch.Tensor,
                availability: torch.Tensor) -> torch.Tensor:
        shape = effects.shape
        flat_effects = effects.reshape(-1, self.token_count, self.output_dim)
        flat_active = active.reshape(-1, self.token_count)
        flat_availability = availability.to(effects.dtype).reshape(-1, 3)
        pooled = effects.new_zeros((flat_effects.shape[0], 128))
        rows = flat_active.any(-1)
        if bool(rows.any()):
            tokens = self.effect_projection(flat_effects[rows])
            types = self.type_embedding.weight.unsqueeze(0)
            conditions = self.availability_projection(flat_availability[rows]).unsqueeze(1)
            tokens = tokens + types + conditions
            selected_active = flat_active[rows]
            tokens = torch.where(selected_active[..., None], tokens, torch.zeros_like(tokens))
            encoded = self.encoder(tokens, src_key_padding_mask=~selected_active)
            encoded = torch.where(selected_active[..., None], encoded, torch.zeros_like(encoded))
            pooled[rows] = encoded.sum(1) / selected_active.sum(1, keepdim=True).to(encoded.dtype)
        projected = self.residual(pooled).reshape(*shape[:2], self.output_dim)
        return torch.where(active.any(-1)[..., None], projected, torch.zeros_like(projected))

    def forward(self, local: torch.Tensor, base: torch.Tensor, gap: torch.Tensor,
                availability: torch.Tensor, umask: torch.Tensor, *,
                local_skip: nn.Module, adapter: nn.Module):
        length, batch, _ = local.shape
        if (base.shape != (length, batch, self.context_dim)
                or gap.shape != (length, batch, 3, self.context_dim)
                or availability.shape != (length, batch, 3)
                or umask.shape != (batch, length)):
            raise ValueError('counterfactual readout received incompatible evidence shapes')
        _validate_adapter(adapter)
        valid, history, gap_active, evidence_active = self._masks(availability, umask)
        safe_local = torch.where(valid[..., None], local, torch.zeros_like(local))
        evidence = self._evidence(base, gap, history, gap_active)
        views = (self._marginal_views(safe_local, evidence) if self.mode == 'marginal'
                 else self._coalition_views(safe_local, evidence))
        outputs = evaluate_flat_views(adapter, views)
        full_index = 0 if self.mode == 'marginal' else 11
        anchor = local_skip(safe_local) + outputs[..., full_index, :]
        effects, active = self._effects(outputs, evidence_active)
        residual = self._encode(effects, active, availability)
        anchor = torch.where(valid[..., None], anchor, torch.zeros_like(anchor))
        residual = torch.where(valid[..., None], residual, torch.zeros_like(residual))
        self.last_effects = effects.detach()
        self.last_active = active.detach()
        self.last_view_outputs = outputs.detach()
        with torch.no_grad():
            count = active.sum(-1)
            self.last_diagnostics = {
                'mode': self.mode,
                'view_count': self.last_view_count,
                'active_evidence_count_mean': float(evidence_active[valid].sum(-1).float().mean())
                if bool(valid.any()) else 0.0,
                'effect_norm': float(effects[active].norm(dim=-1).mean())
                if bool(active.any()) else 0.0,
                'residual_norm': float(residual[valid].norm(dim=-1).mean())
                if bool(valid.any()) else 0.0,
                'residual_anchor_ratio': float(
                    residual[valid].norm(dim=-1).mean()
                    / anchor[valid].norm(dim=-1).mean().clamp_min(1e-8))
                if bool(valid.any()) else 0.0,
                'active_token_count_mean': float(count[valid].float().mean())
                if bool(valid.any()) else 0.0,
            }
        return anchor, residual
