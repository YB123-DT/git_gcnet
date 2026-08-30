"""Deterministic graph utilities and SDR message-passing layers."""

from numbers import Real
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import RGCNConv


TEMPORAL_RELATION_TABLE = {"past": 0, "now": 1, "future": 2}

# GCNet's public graph code names a speaker relation as target speaker followed
# by source speaker.  Keep the lookup literal so label identities cannot depend
# on set or dictionary construction order.
SPEAKER_RELATION_TABLE = {
    1: {(0, 0): 0},
    2: {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3},
}


class GraphData(NamedTuple):
    """Flattened nodes and their typed, directed conversation-local edges."""

    node_features: Tensor
    edge_index: Tensor
    edge_type: Tensor


def _validated_lengths(lengths, batch_size=None, max_length=None):
    try:
        values = torch.as_tensor(lengths)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("lengths must be a one-dimensional integer sequence") from error
    if values.dim() != 1:
        raise ValueError("lengths must be one-dimensional")
    if batch_size is not None and values.numel() != batch_size:
        raise ValueError("lengths must contain one value per conversation")
    if values.dtype == torch.bool:
        raise ValueError("lengths must contain integers")
    if values.is_floating_point():
        if not torch.isfinite(values).all().item() or not torch.equal(
            values, values.round()
        ):
            raise ValueError("lengths must contain finite integers")
    normalized = values.to(dtype=torch.long, device="cpu")
    if torch.any(normalized < 0).item():
        raise ValueError("lengths cannot be negative")
    if max_length is not None and torch.any(normalized > max_length).item():
        raise ValueError("lengths cannot exceed the padded sequence length")
    return [int(value) for value in normalized.tolist()]


def _validated_dropout(dropout):
    if isinstance(dropout, bool):
        raise TypeError("dropout must be a real probability, not bool")
    if not isinstance(dropout, Real):
        raise TypeError("dropout must be a real probability")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("dropout must be between 0 and 1")
    return float(dropout)


def _run_packed_bigru(recurrent, values, lengths):
    """Run a BiGRU over non-empty valid prefixes and restore batch padding."""

    normalized_lengths = _validated_lengths(
        lengths,
        batch_size=values.size(1),
        max_length=values.size(0),
    )
    active_batches = [
        batch_index
        for batch_index, length in enumerate(normalized_lengths)
        if length > 0
    ]
    output_width = recurrent.hidden_size * (2 if recurrent.bidirectional else 1)
    output = values.new_zeros((values.size(0), values.size(1), output_width))
    if not active_batches:
        dependency = values.sum() * 0.0
        for parameter in recurrent.parameters():
            dependency = dependency + parameter.sum() * 0.0
        return output + dependency

    active_index = torch.tensor(
        active_batches,
        dtype=torch.long,
        device=values.device,
    )
    active_values = values.index_select(1, active_index)
    active_lengths = torch.tensor(
        [normalized_lengths[index] for index in active_batches],
        dtype=torch.long,
        device="cpu",
    )
    packed = pack_padded_sequence(
        active_values,
        active_lengths,
        enforce_sorted=False,
    )
    packed_output, _ = recurrent(packed)
    active_output, _ = pad_packed_sequence(
        packed_output,
        total_length=values.size(0),
    )
    return output.index_copy(1, active_index, active_output)


def conversation_to_nodes(values, lengths):
    """Flatten valid ``[L, B, D]`` prefixes in conversation-major order."""

    if not isinstance(values, Tensor) or values.dim() != 3:
        raise ValueError("values must be a tensor with shape [L, B, D]")
    sequence_length, batch_size, feature_dim = values.shape
    normalized_lengths = _validated_lengths(
        lengths,
        batch_size=batch_size,
        max_length=sequence_length,
    )
    valid_conversations = [
        values[:length, batch_index]
        for batch_index, length in enumerate(normalized_lengths)
        if length > 0
    ]
    if not valid_conversations:
        return values.reshape(-1, feature_dim)[:0]
    return torch.cat(valid_conversations, dim=0)


def nodes_to_conversation(nodes, lengths, max_length=None):
    """Restore conversation-major nodes to zero-padded ``[L, B, D]`` form."""

    if not isinstance(nodes, Tensor) or nodes.dim() != 2:
        raise ValueError("nodes must be a tensor with shape [N, D]")
    normalized_lengths = _validated_lengths(lengths)
    inferred_length = max(normalized_lengths, default=0)
    if max_length is None:
        max_length = inferred_length
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < inferred_length
    ):
        raise ValueError("max_length must be an integer covering every conversation")
    if sum(normalized_lengths) != nodes.size(0):
        raise ValueError("the node count must equal the sum of lengths")

    restored = nodes.new_zeros(
        (max_length, len(normalized_lengths), nodes.size(-1))
    )
    offset = 0
    for batch_index, length in enumerate(normalized_lengths):
        if length:
            restored[:length, batch_index] = nodes[offset : offset + length]
        offset += length
    if nodes.size(0) == 0:
        restored = restored + nodes.sum() * 0.0
    return restored


def _validate_window(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise ValueError("{} must be -1 or a non-negative integer".format(name))


def _speaker_id(speakers, batch_index, time_index, n_speakers):
    value = speakers[batch_index, time_index]
    if value.dtype == torch.bool:
        raise ValueError("valid speaker ids must be integers")
    if value.is_floating_point():
        if not torch.isfinite(value).item() or value.item() != value.round().item():
            raise ValueError("valid speaker ids must be finite integers")
    speaker_id = int(value.item())
    if speaker_id < 0 or speaker_id >= n_speakers:
        raise ValueError("valid speaker ids must be in [0, n_speakers - 1]")
    return speaker_id


def graphify(
    values,
    speakers,
    lengths,
    n_speakers=2,
    window_past=-1,
    window_future=-1,
    relation="temporal",
):
    """Build a deterministic typed graph for padded conversations.

    Edges are ordered by conversation, then source time, then target time.  The
    direction and relation names intentionally match GCNet's public graph code:
    ``target > source`` is ``past`` and ``target < source`` is ``future``.
    """

    if not isinstance(values, Tensor) or values.dim() != 3:
        raise ValueError("values must be a tensor with shape [L, B, D]")
    if not isinstance(speakers, Tensor) or speakers.dim() != 2:
        raise ValueError("speakers must be a tensor with shape [B, L]")
    sequence_length, batch_size, _ = values.shape
    if tuple(speakers.shape) != (batch_size, sequence_length):
        raise ValueError("speakers must have shape [B, L] matching values")
    if speakers.device != values.device:
        raise ValueError("values and speakers must be on the same device")
    if n_speakers not in SPEAKER_RELATION_TABLE:
        raise ValueError("n_speakers must be 1 or 2")
    if relation not in ("temporal", "speaker"):
        raise ValueError("relation must be 'temporal' or 'speaker'")
    _validate_window("window_past", window_past)
    _validate_window("window_future", window_future)

    normalized_lengths = _validated_lengths(
        lengths,
        batch_size=batch_size,
        max_length=sequence_length,
    )
    node_features = conversation_to_nodes(values, normalized_lengths)
    sources = []
    targets = []
    edge_types = []
    node_offset = 0

    for batch_index, length in enumerate(normalized_lengths):
        for source_time in range(length):
            first_target = (
                0
                if window_past == -1
                else max(0, source_time - window_past)
            )
            final_target = (
                length
                if window_future == -1
                else min(length, source_time + window_future + 1)
            )
            for target_time in range(first_target, final_target):
                sources.append(node_offset + source_time)
                targets.append(node_offset + target_time)
                if relation == "temporal":
                    if target_time > source_time:
                        relation_name = "past"
                    elif target_time == source_time:
                        relation_name = "now"
                    else:
                        relation_name = "future"
                    edge_types.append(TEMPORAL_RELATION_TABLE[relation_name])
                else:
                    source_speaker = _speaker_id(
                        speakers, batch_index, source_time, n_speakers
                    )
                    target_speaker = _speaker_id(
                        speakers, batch_index, target_time, n_speakers
                    )
                    edge_types.append(
                        SPEAKER_RELATION_TABLE[n_speakers][
                            (target_speaker, source_speaker)
                        ]
                    )
        node_offset += length

    edge_index = torch.tensor(
        [sources, targets],
        dtype=torch.long,
        device=values.device,
    )
    edge_type = torch.tensor(
        edge_types,
        dtype=torch.long,
        device=values.device,
    )
    return GraphData(node_features, edge_index, edge_type)


def _validate_index(name, index, rows):
    if not isinstance(index, Tensor) or index.dim() != 2 or index.size(0) != rows:
        raise ValueError("{} must have shape [{}, E]".format(name, rows))
    if index.dtype not in (torch.int32, torch.int64):
        raise ValueError("{} must contain integer indices".format(name))


class SDRHypergraphConv(nn.Module):
    """Two-stage mean-normalized node-hyperedge-node propagation."""

    def __init__(self, feature_dim, negative_slope=0.01, bias=True):
        super().__init__()
        if (
            isinstance(feature_dim, bool)
            or not isinstance(feature_dim, int)
            or feature_dim <= 0
        ):
            raise ValueError("feature_dim must be a positive integer")
        if (
            isinstance(negative_slope, bool)
            or not isinstance(negative_slope, (int, float))
            or negative_slope < 0
        ):
            raise ValueError("negative_slope must be non-negative")
        if not isinstance(bias, bool):
            raise ValueError("bias must be a boolean")
        self.feature_dim = feature_dim
        self.negative_slope = float(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.zeros(feature_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, features, incidence_index):
        if (
            not isinstance(features, Tensor)
            or features.dim() != 2
            or features.size(-1) != self.feature_dim
        ):
            raise ValueError(
                "features must have shape [N, {}]".format(self.feature_dim)
            )
        _validate_index("incidence_index", incidence_index, 2)
        if incidence_index.device != features.device:
            raise ValueError("features and incidence_index must share a device")
        node_index = incidence_index[0].long()
        hyperedge_index = incidence_index[1].long()
        if node_index.numel() == 0:
            output = features * 0.0
            if self.bias is not None:
                output = output + self.bias
            return F.leaky_relu(output, negative_slope=self.negative_slope)
        if (
            torch.any(node_index < 0).item()
            or torch.any(node_index >= features.size(0)).item()
            or torch.any(hyperedge_index < 0).item()
        ):
            raise ValueError("incidence_index contains an out-of-range index")

        num_hyperedges = int(hyperedge_index.max().item()) + 1
        incidence_weight = features.new_ones(node_index.numel())

        hyperedge_degree = features.new_zeros(num_hyperedges)
        hyperedge_degree.index_add_(0, hyperedge_index, incidence_weight)
        hyperedge_features = features.new_zeros(
            (num_hyperedges, features.size(-1))
        )
        hyperedge_features.index_add_(0, hyperedge_index, features[node_index])
        hyperedge_features = hyperedge_features / hyperedge_degree.clamp_min(
            1
        ).unsqueeze(-1)

        node_degree = features.new_zeros(features.size(0))
        node_degree.index_add_(0, node_index, incidence_weight)
        output = features.new_zeros(features.shape)
        output.index_add_(0, node_index, hyperedge_features[hyperedge_index])
        output = output / node_degree.clamp_min(1).unsqueeze(-1)
        if self.bias is not None:
            output = output + self.bias
        return F.leaky_relu(output, negative_slope=self.negative_slope)


class FrequencyAwareConv(nn.Module):
    """Aggregate degree-normalized, scalar-gated source messages."""

    def __init__(self, feature_dim):
        super().__init__()
        if (
            isinstance(feature_dim, bool)
            or not isinstance(feature_dim, int)
            or feature_dim <= 0
        ):
            raise ValueError("feature_dim must be a positive integer")
        self.feature_dim = feature_dim
        self.gate = nn.Linear(2 * feature_dim, 1)

    def forward(self, features, edge_index, degree_norm=None):
        if (
            not isinstance(features, Tensor)
            or features.dim() != 2
            or features.size(-1) != self.feature_dim
        ):
            raise ValueError(
                "features must have shape [N, {}]".format(self.feature_dim)
            )
        _validate_index("edge_index", edge_index, 2)
        if edge_index.device != features.device:
            raise ValueError("features and edge_index must share a device")
        source = edge_index[0].long()
        target = edge_index[1].long()
        if (
            torch.any(source < 0).item()
            or torch.any(source >= features.size(0)).item()
            or torch.any(target < 0).item()
            or torch.any(target >= features.size(0)).item()
        ):
            raise ValueError("edge_index contains an out-of-range node index")

        if degree_norm is None:
            degree = features.new_zeros(features.size(0))
            degree.index_add_(
                0,
                source,
                features.new_ones(source.numel()),
            )
            inverse_sqrt_degree = degree.pow(-0.5)
            inverse_sqrt_degree.masked_fill_(
                torch.isinf(inverse_sqrt_degree),
                0.0,
            )
            degree_norm = (
                inverse_sqrt_degree[source] * inverse_sqrt_degree[target]
            )
        elif (
            not isinstance(degree_norm, Tensor)
            or degree_norm.dim() != 1
            or degree_norm.numel() != source.numel()
        ):
            raise ValueError("degree_norm must have one scalar per edge")
        if degree_norm.device != features.device:
            raise ValueError("degree_norm and features must share a device")
        degree_norm = degree_norm.to(dtype=features.dtype)

        gate_input = torch.cat((features[target], features[source]), dim=-1)
        gate = torch.tanh(self.gate(gate_input))
        message = degree_norm.unsqueeze(-1) * features[source] * gate
        output = features.new_zeros(features.shape)
        output.index_add_(0, target, message)
        return output


class SDRRelationBranch(nn.Module):
    """One temporal or speaker relation path from recurrent states to SDR hidden."""

    def __init__(
        self,
        recurrent_dim=400,
        graph_hidden=100,
        num_relations=3,
        relation="temporal",
        n_speakers=1,
        window_past=2,
        window_future=2,
        dropout=0.5,
    ):
        super().__init__()
        dimensions = {
            "recurrent_dim": recurrent_dim,
            "graph_hidden": graph_hidden,
            "num_relations": num_relations,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if relation not in ("temporal", "speaker"):
            raise ValueError("relation must be 'temporal' or 'speaker'")
        if isinstance(n_speakers, bool) or n_speakers not in SPEAKER_RELATION_TABLE:
            raise ValueError("n_speakers must be 1 or 2")
        expected_relations = 3 if relation == "temporal" else n_speakers ** 2
        if num_relations != expected_relations:
            raise ValueError(
                "num_relations must be {} for the {} branch".format(
                    expected_relations,
                    relation,
                )
            )
        _validate_window("window_past", window_past)
        _validate_window("window_future", window_future)
        dropout = _validated_dropout(dropout)

        self.recurrent_dim = recurrent_dim
        self.graph_hidden = graph_hidden
        self.output_dim = recurrent_dim + graph_hidden
        self.num_relations = num_relations
        self.relation = relation
        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future

        self.rgcn = RGCNConv(recurrent_dim, graph_hidden, num_relations)
        self.hypergraph = SDRHypergraphConv(graph_hidden)
        self.high_conv = FrequencyAwareConv(graph_hidden)
        self.post_graph_bigru = nn.GRU(
            input_size=self.output_dim,
            hidden_size=self.output_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.output_linear = nn.Linear(2 * self.output_dim, self.output_dim)

    def forward(self, recurrent, qmask, umask, lengths):
        if (
            not isinstance(recurrent, Tensor)
            or recurrent.dim() != 3
            or recurrent.size(-1) != self.recurrent_dim
        ):
            raise ValueError(
                "recurrent must have shape [L, B, {}]".format(
                    self.recurrent_dim
                )
            )
        sequence_length, batch_size, _ = recurrent.shape
        expected_mask_shape = (batch_size, sequence_length)
        if not isinstance(qmask, Tensor) or tuple(qmask.shape) != expected_mask_shape:
            raise ValueError("qmask must have shape [B, L]")
        if not isinstance(umask, Tensor) or tuple(umask.shape) != expected_mask_shape:
            raise ValueError("umask must have shape [B, L]")
        if qmask.device != recurrent.device or umask.device != recurrent.device:
            raise ValueError("recurrent, qmask, and umask must share a device")
        normalized_lengths = _validated_lengths(
            lengths,
            batch_size=batch_size,
            max_length=sequence_length,
        )

        graph = graphify(
            recurrent,
            qmask,
            normalized_lengths,
            n_speakers=self.n_speakers,
            window_past=self.window_past,
            window_future=self.window_future,
            relation=self.relation,
        )
        if graph.node_features.size(0):
            graph_nodes = self.rgcn(
                graph.node_features,
                graph.edge_index,
                graph.edge_type,
            )
        else:
            dependency = graph.node_features.sum() * 0.0
            for parameter in self.rgcn.parameters():
                dependency = dependency + parameter.sum() * 0.0
            graph_nodes = graph.node_features.new_zeros((0, self.graph_hidden))
            graph_nodes = graph_nodes + dependency
        graph_nodes = self.hypergraph(graph_nodes, graph.edge_index)
        graph_nodes = self.high_conv(graph_nodes, graph.edge_index)
        nodes = torch.cat((graph.node_features, graph_nodes), dim=-1)
        conversation = nodes_to_conversation(
            nodes,
            normalized_lengths,
            max_length=sequence_length,
        )
        fused = _run_packed_bigru(
            self.post_graph_bigru,
            conversation,
            normalized_lengths,
        )
        hidden = F.relu(self.output_linear(fused))
        valid = umask.transpose(0, 1).bool().unsqueeze(-1)
        return hidden.masked_fill(~valid, 0.0)


__all__ = [
    "FrequencyAwareConv",
    "GraphData",
    "SDRRelationBranch",
    "SDRHypergraphConv",
    "SPEAKER_RELATION_TABLE",
    "TEMPORAL_RELATION_TABLE",
    "conversation_to_nodes",
    "graphify",
    "nodes_to_conversation",
]
