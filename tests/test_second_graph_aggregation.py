import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GraphConv

from second_graph_aggregation import (
    GenAggGraphConv,
    SoftMedoidGraphConv,
    build_second_graph_conv,
)


def _copy_graph_conv_parameters(source, target):
    target_neighbor = getattr(target, "lin_l", getattr(target, "lin_rel", None))
    target_root = getattr(target, "lin_r", getattr(target, "lin_root", None))
    with torch.no_grad():
        target_neighbor.weight.copy_(source.lin_l.weight)
        if source.lin_l.bias is not None:
            target_neighbor.bias.copy_(source.lin_l.bias)
        target_root.weight.copy_(source.lin_r.weight)


class GenAggGraphConvTest(unittest.TestCase):
    def test_identity_maps_recover_sum_when_alpha_is_one(self):
        conv = GenAggGraphConv(2, 2)
        conv.forward_map = nn.Identity()
        conv.inverse_map = nn.Identity()
        with torch.no_grad():
            conv.alpha.fill_(1.0)
            conv.beta.zero_()
            conv.lin_l.weight.copy_(torch.eye(2))
            conv.lin_l.bias.zero_()
            conv.lin_r.weight.zero_()

        x = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
        edge_index = torch.tensor([[0, 1], [2, 2]])

        actual = conv(x, edge_index)
        expected = torch.tensor([[0.0, 0.0], [0.0, 0.0], [4.0, 7.0]])
        torch.testing.assert_close(actual, expected)

    def test_augmented_f_mean_matches_hand_formula(self):
        conv = GenAggGraphConv(1, 1)
        conv.forward_map = nn.Identity()
        conv.inverse_map = nn.Identity()
        with torch.no_grad():
            conv.alpha.fill_(0.5)
            conv.beta.fill_(0.25)
            conv.lin_l.weight.fill_(2.0)
            conv.lin_l.bias.fill_(0.3)
            conv.lin_r.weight.zero_()

        x = torch.tensor([[1.0], [3.0], [9.0]])
        edge_index = torch.tensor([[0, 1], [2, 2]])
        actual = conv(x, edge_index)[2]
        mean = x[:2].mean(0)
        aggregate = 2.0 ** 0.5 * (x[:2] - 0.25 * mean).mean(0)
        expected = 2.0 * aggregate + 0.3
        torch.testing.assert_close(actual, expected)

    def test_real_maps_match_indexed_formula_and_cache_inverse_loss(self):
        torch.manual_seed(29)
        conv = GenAggGraphConv(2, 3).eval()
        with torch.no_grad():
            conv.alpha.fill_(0.35)
            conv.beta.fill_(-0.2)
        x = torch.tensor(
            [[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25], [2.0, 1.0]]
        )
        edge_index = torch.tensor([[0, 1, 2, 0, 3], [2, 2, 2, 3, 3]])

        with torch.no_grad():
            actual = conv(x, edge_index)
            expected = conv.lin_l(torch.zeros(x.size(0), x.size(1)))
            expected = expected + conv.lin_r(x)
            centered_edges = []
            encoded_edges = []
            for target in range(x.size(0)):
                edge_mask = edge_index[1].eq(target)
                if not edge_mask.any():
                    continue
                incoming = x.index_select(0, edge_index[0, edge_mask])
                centered = incoming - conv.beta * incoming.mean(0)
                encoded = conv.forward_map(centered.unsqueeze(-1))
                scaled_mean = encoded.mean(0) * (
                    float(incoming.size(0)) ** conv.alpha
                )
                aggregate = conv.inverse_map(scaled_mean).squeeze(-1)
                expected[target] = conv.lin_l(aggregate) + conv.lin_r(x[target])
                centered_edges.append(centered)
                encoded_edges.append(encoded)

            centered = torch.cat(centered_edges, dim=0)
            encoded = torch.cat(encoded_edges, dim=0)
            reconstructed = conv.inverse_map(encoded).squeeze(-1)
            expected_inverse_loss = (
                reconstructed.abs() - centered.abs()
            ).square().mean()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            conv.inverse_consistency_loss(), expected_inverse_loss
        )

    def test_empty_targets_have_zero_neighbor_aggregate(self):
        conv = GenAggGraphConv(3, 2)
        inputs = torch.empty((0, 3))
        index = torch.empty((0,), dtype=torch.long)
        actual = conv.aggregate(inputs, index, dim_size=4)
        torch.testing.assert_close(actual, torch.zeros(4, 3))

    def test_inverse_consistency_empty_input_is_zero(self):
        conv = GenAggGraphConv(3, 2)
        loss = conv.inverse_consistency_loss(torch.empty(0, 3))
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.item(), 0.0)

    def test_inverse_consistency_uses_squared_absolute_reconstruction_error(self):
        conv = GenAggGraphConv(1, 1)
        conv.forward_map = nn.Identity()

        class Negate(nn.Module):
            def forward(self, values):
                return -values

        conv.inverse_map = Negate()
        centered = torch.tensor([[-2.0], [3.0]])
        self.assertEqual(conv.inverse_consistency_loss(centered).item(), 0.0)

    def test_forward_reuses_encoded_values_for_inverse_loss(self):
        conv = GenAggGraphConv(2, 2)
        x = torch.randn(4, 2)
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])
        tracked = conv.forward_map.layers[1].num_batches_tracked.item()
        conv(x, edge_index)
        self.assertEqual(
            conv.forward_map.layers[1].num_batches_tracked.item(),
            tracked + 1,
        )

    def test_adds_exactly_fifty_nine_trainable_parameters(self):
        baseline = GraphConv(3, 5)
        candidate = GenAggGraphConv(3, 5)
        baseline_count = sum(
            parameter.numel() for parameter in baseline.parameters()
        )
        candidate_count = sum(
            parameter.numel() for parameter in candidate.parameters()
        )
        self.assertEqual(candidate_count - baseline_count, 59)
        self.assertEqual(
            set(candidate.state_dict()),
            {
                "alpha",
                "beta",
                "lin_l.weight",
                "lin_l.bias",
                "lin_r.weight",
                "forward_map.layers.0.weight",
                "forward_map.layers.0.bias",
                "forward_map.layers.1.weight",
                "forward_map.layers.1.bias",
                "forward_map.layers.1.running_mean",
                "forward_map.layers.1.running_var",
                "forward_map.layers.1.num_batches_tracked",
                "forward_map.layers.3.weight",
                "forward_map.layers.3.bias",
                "forward_map.layers.4.weight",
                "forward_map.layers.4.bias",
                "forward_map.layers.4.running_mean",
                "forward_map.layers.4.running_var",
                "forward_map.layers.4.num_batches_tracked",
                "forward_map.layers.6.weight",
                "forward_map.layers.6.bias",
                "inverse_map.layers.0.weight",
                "inverse_map.layers.0.bias",
                "inverse_map.layers.1.weight",
                "inverse_map.layers.1.bias",
                "inverse_map.layers.1.running_mean",
                "inverse_map.layers.1.running_var",
                "inverse_map.layers.1.num_batches_tracked",
                "inverse_map.layers.3.weight",
                "inverse_map.layers.3.bias",
                "inverse_map.layers.4.weight",
                "inverse_map.layers.4.bias",
                "inverse_map.layers.4.running_mean",
                "inverse_map.layers.4.running_var",
                "inverse_map.layers.4.num_batches_tracked",
                "inverse_map.layers.6.weight",
                "inverse_map.layers.6.bias",
            },
        )

    def test_extra_parameter_initialization_preserves_cpu_rng(self):
        torch.manual_seed(1234)
        GraphConv(3, 5)
        expected_next = torch.rand(8)

        torch.manual_seed(1234)
        GenAggGraphConv(3, 5)
        actual_next = torch.rand(8)

        torch.testing.assert_close(actual_next, expected_next)

    def test_forward_and_inverse_paths_have_finite_gradients(self):
        torch.manual_seed(7)
        conv = GenAggGraphConv(3, 2)
        x = torch.randn(5, 3, requires_grad=True)
        edge_index = torch.tensor(
            [[0, 1, 2, 1, 2, 3, 4], [3, 3, 3, 4, 4, 4, 4]]
        )
        loss = conv(x, edge_index).square().mean()
        loss = loss + conv.inverse_consistency_loss(x)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in conv.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class SoftMedoidGraphConvTest(unittest.TestCase):
    def test_three_message_neighborhood_matches_hand_formula(self):
        temperature = 0.75
        conv = SoftMedoidGraphConv(2, 2, temperature=temperature)
        with torch.no_grad():
            conv.lin_l.weight.copy_(torch.eye(2))
            conv.lin_l.bias.zero_()
            conv.lin_r.weight.zero_()
        messages = torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 3.0]])
        x = torch.cat((messages, torch.zeros(1, 2)), dim=0)
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])

        actual = conv(x, edge_index)[3]
        pairwise = torch.cdist(messages, messages)
        expected_weights = torch.softmax(-pairwise.sum(-1) / temperature, dim=0)
        expected = messages.size(0) * (expected_weights[:, None] * messages).sum(0)
        torch.testing.assert_close(actual, expected)

    def test_edge_permutation_does_not_change_packed_result(self):
        conv = SoftMedoidGraphConv(2, 2)
        x = torch.randn(6, 2)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 5, 4, 5, 4]])
        permutation = torch.tensor([4, 1, 3, 0, 2])
        torch.testing.assert_close(
            conv(x, edge_index), conv(x, edge_index[:, permutation])
        )

    def test_unequal_degrees_match_ragged_formula_after_edge_permutation(self):
        torch.manual_seed(31)
        conv = SoftMedoidGraphConv(3, 2, temperature=0.6)
        x = torch.randn(6, 3)
        edge_index = torch.tensor([[0, 1, 2, 2, 3], [4, 4, 4, 5, 5]])

        def ragged_expected(index):
            source, target = index
            messages = F.linear(
                x.index_select(0, source), conv.lin_l.weight
            )
            neighbor = messages.new_zeros((x.size(0), conv.out_channels))
            for target_node in range(x.size(0)):
                incoming = messages[target.eq(target_node)]
                if incoming.size(0) == 0:
                    continue
                distance_sum = torch.cdist(incoming, incoming).sum(-1)
                weights = torch.softmax(
                    -distance_sum / conv.temperature, dim=0
                )
                neighbor[target_node] = incoming.size(0) * (
                    weights.unsqueeze(1) * incoming
                ).sum(0)
            return neighbor + conv.lin_l.bias + conv.lin_r(x)

        expected = ragged_expected(edge_index)
        torch.testing.assert_close(conv(x, edge_index), expected)

        permutation = torch.tensor([4, 0, 3, 2, 1])
        permuted = edge_index[:, permutation]
        torch.testing.assert_close(conv(x, permuted), ragged_expected(permuted))
        torch.testing.assert_close(conv(x, permuted), conv(x, edge_index))

    def test_single_neighbor_matches_original_graph_conv(self):
        self._assert_matches_original(
            torch.tensor([[0, 1, 2], [3, 3, 3]]), homogeneous=False
        )

    def test_homogeneous_neighborhood_matches_original_graph_conv(self):
        self._assert_matches_original(
            torch.tensor([[0, 1, 2], [3, 3, 3]]), homogeneous=True
        )

    def _assert_matches_original(self, edge_index, homogeneous):
        torch.manual_seed(11)
        candidate = SoftMedoidGraphConv(2, 3)
        baseline = GraphConv(2, 3)
        _copy_graph_conv_parameters(candidate, baseline)
        if homogeneous:
            neighbor = torch.tensor([[2.0, -1.0]])
            x = torch.cat((neighbor.expand(3, -1), torch.randn(1, 2)), dim=0)
        else:
            edge_index = torch.tensor([[0], [3]])
            x = torch.randn(4, 2)
        torch.testing.assert_close(
            candidate(x, edge_index), baseline(x, edge_index)
        )

    def test_zero_degree_keeps_only_bias_and_root_path(self):
        conv = SoftMedoidGraphConv(2, 2)
        with torch.no_grad():
            conv.lin_l.bias.copy_(torch.tensor([0.5, -0.25]))
            conv.lin_r.weight.copy_(torch.eye(2))
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        edge_index = torch.empty((2, 0), dtype=torch.long)
        expected = x + torch.tensor([0.5, -0.25])
        torch.testing.assert_close(conv(x, edge_index), expected)

    def test_heterogeneous_neighborhood_backward_is_finite(self):
        conv = SoftMedoidGraphConv(3, 2)
        x = torch.randn(5, 3, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 1, 3], [3, 3, 3, 4, 4]])
        conv(x, edge_index).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in conv.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_adds_no_parameters(self):
        baseline = GraphConv(3, 5)
        candidate = SoftMedoidGraphConv(3, 5)
        self.assertEqual(
            sum(parameter.numel() for parameter in candidate.parameters()),
            sum(parameter.numel() for parameter in baseline.parameters()),
        )

    def test_temperature_must_be_positive(self):
        for invalid in (0.0, -1.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    SoftMedoidGraphConv(2, 2, temperature=invalid)


class SecondGraphConvFactoryTest(unittest.TestCase):
    def test_builds_each_supported_selector(self):
        self.assertIs(type(build_second_graph_conv("add", 2, 3)), GraphConv)
        self.assertIsInstance(
            build_second_graph_conv("genagg", 2, 3), GenAggGraphConv
        )
        self.assertIsInstance(
            build_second_graph_conv("soft_medoid", 2, 3), SoftMedoidGraphConv
        )

    def test_rejects_unknown_selector(self):
        with self.assertRaisesRegex(ValueError, "unknown second graph aggregation"):
            build_second_graph_conv("median", 2, 3)


if __name__ == "__main__":
    unittest.main()
