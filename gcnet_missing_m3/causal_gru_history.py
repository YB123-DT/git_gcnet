"""Ordinary strictly past recurrent context for the P2 training control."""

import torch
from torch import nn


class CausalGRUHistory(nn.Module):
    """A fresh single-layer GRU per batch, reading h[t-1] before updating t.

    No current query, modality address, or CHM write is used. Each batch column
    is an independent conversation. Invalid positions neither read nor update.
    """

    def __init__(self, input_dim: int, forward_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, 256, num_layers=1, bidirectional=False)
        # No bias: an empty history must produce the exact all-zero Base slot.
        self.project = nn.Linear(256, forward_dim, bias=False)

    def forward(self, node: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        state = node.new_zeros(1, node.shape[1], 256)
        contexts = []
        for index in range(node.shape[0]):
            active = valid[index, :, None]
            contexts.append(torch.where(active, self.project(state[0]), 0.))
            current = torch.where(active, node[index], 0.).unsqueeze(0)
            _, updated = self.gru(current, state)
            state = torch.where(active.unsqueeze(0), updated, state)
        return torch.stack(contexts)


def validate_causal_gru_history(options: dict) -> None:
    """Keep this control on the ordinary mean-node Local+Flat task path."""
    aliases = {"bidirectional": "osram_bidirectional",
               "forward_slot_reuse": "osram_forward_slot_reuse",
               "history_query_adapter": "osram_history_query_adapter"}
    def value(name, default=None):
        return options.get(name, options.get(aliases.get(name, name), default))
    incompatible = ("osram_base_first_gap", "osram_history_recency", "paired_history_views",
                    "osram_relation_block", "osram_relation_dual_readout",
                    "osram_decision_correction", "osram_gap_increment_filter", "osram_post_grn",
                    "history_query_adapter", "osram_local_skip_gate", "osram_memory_only_adapter",
                    "osram_history_input_gate", "osram_local_evidence_gate",
                    "osram_hierarchical_evidence_gate", "osram_hierarchical_feature_only",
                    "osram_local_evidence_gate_gap_only", "classification_completion",
                    "completion_write_to_memory", "local_context_residual", "node_interaction_residual",
                    "text_core", "osram_ced_block", "complete_state_jepa", "write_state_completion",
                    "future_state_jepa", "simple_regression_predictor")
    pretrained_sources = ("initial_backbone_checkpoint", "joint_pretrain_checkpoint",
                          "b2_base_checkpoint", "b2_pretrain_checkpoint", "teacher_checkpoint",
                          "text_subspace_checkpoint", "pretrained_learning_rate")
    if (any(value(name, False) for name in incompatible)
            or any(value(name) is not None for name in pretrained_sources)
            or value("bidirectional", False) or value("forward_slot_reuse", False)
            or value("osram_readout_fusion", "flat") != "flat"
            or value("osram_ablation", "full") != "full"
            or value("osram_emotion_ablation", "full") != "full"
            or value("osram_meaningful_block", "none") != "none"
            or value("osram_readout_candidate", "none") != "none"
            or value("osram_counterfactual_readout", "none") != "none"
            or value("osram_adapter_hidden_dim", 0)
            or value("backbone_type", "osram") != "osram"
            or value("fusion_type", "mean") != "mean"
            or value("representation_type", "slot") != "slot"
            or value("training_objective", "emotion-only") != "emotion-only"
            or value("completion_path", "none") != "none"
            or value("readout_type", "shared") != "shared"
            or value("emotion_loss_mode", "sample-mean") != "sample-mean"
            or value("core20_method", "none") != "none"
            or value("teacher_mode", "ema") != "ema"
            or not value("disable_unused_aux_modules", True)
            ):
        raise ValueError("causal GRU history requires ordinary causal mean-node Local+Flat emotion-only training")
