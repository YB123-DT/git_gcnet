import pytest
import torch

from gcnet_missing_m3_sdr_backbone.layers import (
    SPEAKER_RELATION_TABLE,
    FrequencyAwareConv,
    SDRHypergraphConv,
    conversation_to_nodes,
    graphify,
    nodes_to_conversation,
)


def _conversation_batch():
    values = torch.tensor(
        [
            [[1.0, 10.0], [4.0, 40.0]],
            [[2.0, 20.0], [5.0, 50.0]],
            [[3.0, 30.0], [999.0, 999.0]],
            [[888.0, 888.0], [777.0, 777.0]],
        ]
    )
    speakers = torch.tensor(
        [
            [0, 1, 0, 9],
            [1, 0, 9, 9],
        ]
    )
    lengths = torch.tensor([3, 2])
    return values, speakers, lengths


def test_conversation_to_nodes_flattens_valid_prefixes_in_batch_order():
    values, _, lengths = _conversation_batch()

    nodes = conversation_to_nodes(values, lengths)

    expected = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]
    )
    assert torch.equal(nodes, expected)


def test_nodes_to_conversation_restores_values_and_zero_padding():
    values, _, lengths = _conversation_batch()
    nodes = conversation_to_nodes(values, lengths)

    restored = nodes_to_conversation(nodes, lengths, max_length=values.size(0))

    valid = torch.arange(values.size(0)).unsqueeze(1) < lengths.unsqueeze(0)
    assert restored.shape == values.shape
    assert torch.equal(restored[valid], values[valid])
    assert torch.count_nonzero(restored[~valid]).item() == 0


def test_all_zero_lengths_round_trip_preserves_zero_gradient_paths():
    values = torch.randn(3, 2, 4, requires_grad=True)
    lengths = torch.tensor([0, 0])

    nodes = conversation_to_nodes(values, lengths)

    assert nodes.shape == (0, 4)
    assert nodes.requires_grad
    nodes.retain_grad()
    restored = nodes_to_conversation(nodes, lengths, max_length=3)
    assert restored.shape == values.shape
    assert torch.count_nonzero(restored).item() == 0
    restored.sum().backward()
    assert nodes.grad is not None
    assert torch.equal(nodes.grad, torch.zeros_like(nodes))
    assert values.grad is not None
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_empty_nodes_restore_keeps_zero_gradient_dependency():
    nodes = torch.empty((0, 4), requires_grad=True)

    restored = nodes_to_conversation(nodes, [0, 0], max_length=3)

    restored.sum().backward()
    assert nodes.grad is not None
    assert torch.equal(nodes.grad, torch.zeros_like(nodes))


def test_graphify_has_deterministic_conversation_source_target_order():
    values, speakers, lengths = _conversation_batch()

    first = graphify(values, speakers, lengths, relation="temporal")
    second = graphify(values, speakers, lengths, relation="temporal")

    expected_edges = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4],
        ]
    )
    assert torch.equal(first.node_features, second.node_features)
    assert torch.equal(first.edge_index, expected_edges)
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.edge_type, second.edge_type)


def test_graphify_excludes_padding_values_and_padding_speaker_ids():
    values, speakers, lengths = _conversation_batch()
    poisoned_values = values.clone()
    poisoned_values[3, 0] = float("nan")
    poisoned_values[2:, 1] = float("nan")
    poisoned_speakers = speakers.clone()
    poisoned_speakers[0, 3] = -100
    poisoned_speakers[1, 2:] = 100

    clean = graphify(values, speakers, lengths, relation="speaker")
    poisoned = graphify(
        poisoned_values,
        poisoned_speakers,
        lengths,
        relation="speaker",
    )

    assert torch.isfinite(poisoned.node_features).all().item()
    assert torch.equal(clean.node_features, poisoned.node_features)
    assert torch.equal(clean.edge_index, poisoned.edge_index)
    assert torch.equal(clean.edge_type, poisoned.edge_type)


def test_graphify_uses_fixed_temporal_relation_ids():
    values, speakers, lengths = _conversation_batch()

    graph = graphify(values, speakers, lengths, relation="temporal")

    expected = torch.tensor([1, 0, 0, 2, 1, 0, 2, 2, 1, 1, 0, 2, 1])
    assert torch.equal(graph.edge_type, expected)


def test_graphify_uses_explicit_target_then_source_speaker_table():
    values, speakers, lengths = _conversation_batch()

    graph = graphify(
        values,
        speakers,
        lengths,
        relation="speaker",
        n_speakers=2,
    )

    assert SPEAKER_RELATION_TABLE == {
        1: {(0, 0): 0},
        2: {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3},
    }
    expected = torch.tensor([0, 2, 0, 1, 3, 1, 0, 2, 0, 3, 1, 2, 0])
    assert torch.equal(graph.edge_type, expected)


def test_graphify_never_connects_different_conversations():
    values, speakers, lengths = _conversation_batch()

    graph = graphify(values, speakers, lengths, relation="temporal")

    source, target = graph.edge_index
    crosses_boundary = ((source < 3) & (target >= 3)) | (
        (source >= 3) & (target < 3)
    )
    assert not crosses_boundary.any().item()


@pytest.mark.parametrize(
    ("window_past", "window_future", "expected_edges"),
    [
        (
            0,
            0,
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        ),
        (
            -1,
            0,
            [
                [0, 1, 1, 2, 3, 3, 4, 4, 4],
                [0, 0, 1, 2, 2, 3, 2, 3, 4],
            ],
        ),
        (
            0,
            -1,
            [
                [0, 0, 1, 2, 2, 2, 3, 3, 4],
                [0, 1, 1, 2, 3, 4, 3, 4, 4],
            ],
        ),
    ],
)
def test_graphify_windows_have_exact_edges_with_zero_length_conversation(
    window_past,
    window_future,
    expected_edges,
):
    values = torch.arange(9.0).reshape(3, 3, 1)
    speakers = torch.zeros((3, 3), dtype=torch.long)
    lengths = torch.tensor([2, 0, 3])

    graph = graphify(
        values,
        speakers,
        lengths,
        n_speakers=1,
        window_past=window_past,
        window_future=window_future,
        relation="temporal",
    )

    assert graph.node_features.shape == (5, 1)
    assert torch.equal(graph.edge_index, torch.tensor(expected_edges))


def test_hypergraph_conv_matches_two_stage_incidence_normalization():
    features = torch.tensor(
        [[1.0, 3.0], [3.0, 1.0], [5.0, -1.0]],
        requires_grad=True,
    )
    incidence = torch.tensor(
        [[0, 1, 1, 2], [0, 0, 1, 1]],
    )
    layer = SDRHypergraphConv(feature_dim=2)
    with torch.no_grad():
        layer.bias.copy_(torch.tensor([-1.0, 0.5]))

    output = layer(features, incidence)

    expected = torch.tensor([[1.0, 2.5], [2.0, 1.5], [3.0, 0.5]])
    assert torch.allclose(output, expected)
    output.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all().item()
    assert layer.bias.grad is not None
    assert torch.isfinite(layer.bias.grad).all().item()


def test_hypergraph_conv_single_node_self_incidence_is_finite():
    features = torch.tensor([[-2.0, 3.0]], requires_grad=True)
    incidence = torch.tensor([[0], [0]])

    output = SDRHypergraphConv(feature_dim=2)(features, incidence)

    assert torch.allclose(output, torch.tensor([[-0.02, 3.0]]))
    assert torch.isfinite(output).all().item()
    output.sum().backward()
    assert torch.isfinite(features.grad).all().item()


def test_frequency_aware_conv_uses_only_the_required_gated_message():
    features = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        requires_grad=True,
    )
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    )
    layer = FrequencyAwareConv(feature_dim=2)
    with torch.no_grad():
        layer.gate.weight.copy_(
            torch.tensor([[0.1, -0.2, 0.3, -0.4]])
        )
        layer.gate.bias.copy_(torch.tensor([0.2]))

    output = layer(features, edge_index)

    source, target = edge_index
    degree = torch.zeros(features.size(0))
    degree.index_add_(0, source, torch.ones(source.numel()))
    inverse_sqrt_degree = degree.pow(-0.5)
    inverse_sqrt_degree[torch.isinf(inverse_sqrt_degree)] = 0.0
    degree_norm = inverse_sqrt_degree[source] * inverse_sqrt_degree[target]
    gates = torch.tanh(
        layer.gate(torch.cat((features[target], features[source]), dim=-1))
    )
    messages = degree_norm.unsqueeze(-1) * features[source] * gates
    expected = torch.zeros_like(features)
    expected.index_add_(0, target, messages)
    assert torch.allclose(output, expected)
    output.sum().backward()
    assert torch.isfinite(features.grad).all().item()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in layer.parameters()
    )


def test_frequency_aware_conv_single_node_self_loop_is_finite():
    features = torch.tensor([[1.5, -2.5]], requires_grad=True)
    edge_index = torch.tensor([[0], [0]])
    layer = FrequencyAwareConv(feature_dim=2)

    output = layer(features, edge_index)

    assert output.shape == features.shape
    assert torch.isfinite(output).all().item()
    output.sum().backward()
    assert torch.isfinite(features.grad).all().item()


def test_frequency_aware_conv_empty_edges_keep_gate_gradient_paths():
    features = torch.randn(2, 3, requires_grad=True)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    degree_norm = torch.empty(0)
    layer = FrequencyAwareConv(feature_dim=3)

    output = layer(features, edge_index, degree_norm=degree_norm)

    assert torch.equal(output, torch.zeros_like(features))
    output.sum().backward()
    assert features.grad is not None
    assert torch.equal(features.grad, torch.zeros_like(features))
    assert layer.gate.weight.grad is not None
    assert torch.equal(layer.gate.weight.grad, torch.zeros_like(layer.gate.weight))
    assert layer.gate.bias.grad is not None
    assert torch.equal(layer.gate.bias.grad, torch.zeros_like(layer.gate.bias))


def test_frequency_aware_conv_empty_edges_reject_nonempty_degree_norm():
    features = torch.randn(2, 3)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    layer = FrequencyAwareConv(feature_dim=3)

    with pytest.raises(ValueError, match="one scalar per edge"):
        layer(features, edge_index, degree_norm=torch.ones(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_frequency_aware_conv_empty_edges_reject_cross_device_degree_norm():
    features = torch.randn(2, 3)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    degree_norm = torch.empty(0, device="cuda")
    layer = FrequencyAwareConv(feature_dim=3)

    with pytest.raises(ValueError, match="share a device"):
        layer(features, edge_index, degree_norm=degree_norm)
