"""SDR-style conversation graph backbone for Missing-M3."""

from .layers import (
    SPEAKER_RELATION_TABLE,
    TEMPORAL_RELATION_TABLE,
    FrequencyAwareConv,
    GraphData,
    SDRHypergraphConv,
    SDRRelationBranch,
    conversation_to_nodes,
    graphify,
    nodes_to_conversation,
)
from .model import SDRConversationBackbone

__all__ = [
    "FrequencyAwareConv",
    "GraphData",
    "SDRHypergraphConv",
    "SDRRelationBranch",
    "SDRConversationBackbone",
    "SPEAKER_RELATION_TABLE",
    "TEMPORAL_RELATION_TABLE",
    "conversation_to_nodes",
    "graphify",
    "nodes_to_conversation",
]
