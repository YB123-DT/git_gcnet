"""Base-first readout with a strictly history-supported Gap correction."""

import math

import torch
from torch import nn


def validate_base_first_gap(options):
    """Reject combinations that would change the independent fusion experiment."""
    forbidden = (
        'bidirectional', 'osram_bidirectional', 'forward_slot_reuse',
        'osram_forward_slot_reuse', 'history_query_adapter', 'osram_history_query_adapter',
        'osram_post_grn', 'osram_local_skip_gate', 'osram_memory_only_adapter',
        'osram_history_input_gate', 'osram_local_evidence_gate',
        'osram_local_evidence_gate_gap_only', 'osram_hierarchical_evidence_gate',
        'osram_hierarchical_feature_only', 'osram_relation_block', 'osram_relation_dual_readout',
        'osram_gap_increment_filter', 'osram_decision_correction', 'osram_adapter_hidden_dim',
        'classification_completion', 'completion_write_to_memory', 'complete_state_jepa',
        'write_state_completion', 'future_state_jepa', 'text_core', 'local_context_residual',
        'node_interaction_residual', 'paired_history_views', 'osram_ced_block',
        'simple_regression_predictor',
    )
    expected = dict(backbone_type='osram', osram_readout_fusion='flat',
                    osram_ablation='full', osram_emotion_ablation='full',
                    osram_readout_candidate='none', osram_meaningful_block='none',
                    core20_method='none', completion_path='none',
                    training_objective='emotion-only', readout_type='shared')
    if (any(options.get(k, False) for k in forbidden)
            or any(k in options and options[k] != v for k, v in expected.items())):
        raise ValueError('Base-first Gap requires independent causal Flat no-JEPA without completion or other readout adaptations')


class BaseFirstGapFusion(nn.Module):
    def __init__(self, local_dim, context_dim, output_dim, dropout=0.5):
        super().__init__()
        self.local_dim, self.context_dim = local_dim, context_dim
        a, b, d = local_dim + context_dim, output_dim + 3 * context_dim, output_dim
        old_input = local_dim + 4 * context_dim
        budget = 2 * old_input + old_input * d + d + d * d + d
        constant, slope = 2 * (a + b) + 4 * d, a + b + 2 * d + 2
        ideal = (budget - constant) / slope
        candidates = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
        self.hidden_dim = min(candidates, key=lambda h: (abs(constant + slope*h - budget), h))
        def mlp(width):
            result = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.hidden_dim),
                                   nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim, d))
            nn.init.zeros_(result[-1].weight)
            nn.init.zeros_(result[-1].bias)
            return result
        self.base_mlp = mlp(a)
        self.condition_norm = nn.LayerNorm(d)
        self.gap_mlp = mlp(b)

    def forward(self, local, base, gap, availability, umask, *, local_skip, emotion_norm):
        length, batch, _ = local.shape
        if (local.shape[-1] != self.local_dim or base.shape != (length, batch, self.context_dim)
                or gap.shape != (length, batch, 3, self.context_dim)
                or availability.shape != (length, batch, 3) or umask.shape != (batch, length)):
            raise ValueError('Base-first Gap expects full-width Base and three fixed Gap slots')
        valid = umask.T.bool()
        history = valid & ((valid.long().cumsum(0) - valid.long()) > 0)
        active = history[..., None] & ~availability.bool()
        safe_local = torch.where(valid[..., None], local, 0.)
        safe_base = torch.where(history[..., None], base, 0.)
        safe_gap = torch.where(active[..., None], gap, 0.)
        u = local_skip(safe_local) + self.base_mlp(torch.cat((safe_local, safe_base), -1))
        delta = self.gap_mlp(torch.cat((self.condition_norm(u), safe_gap.flatten(2)), -1))
        delta = torch.where(active.any(-1)[..., None], delta, 0.)
        return torch.where(valid[..., None], emotion_norm(u + delta), 0.)
