"""SDR-style conversation graph backbone for Missing-M3."""

from .layers import (
    SPEAKER_RELATION_TABLE,
    TEMPORAL_RELATION_TABLE,
    FrequencyAwareConv,
    GraphData,
    SDRHypergraphConv,
    conversation_to_nodes,
    graphify,
    nodes_to_conversation,
)

__all__ = [
    "FrequencyAwareConv",
    "GraphData",
    "SDRHypergraphConv",
    "SPEAKER_RELATION_TABLE",
    "TEMPORAL_RELATION_TABLE",
    "conversation_to_nodes",
    "graphify",
    "nodes_to_conversation",
]
