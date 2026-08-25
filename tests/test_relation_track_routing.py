import unittest

import torch
from torch_geometric.nn import GraphConv, RGCNConv

from relation_track_routing import (
    decompose_rgcn_relation_tracks,
    relation_track_graph_conv,
)


class RelationTrackRoutingTests(unittest.TestCase):
    @staticmethod
    def _graph():
        x = torch.tensor(
            [[0.2, -0.5, 0.7], [0.8, 0.1, -0.4], [-0.3, 0.6, 0.9]],
            dtype=torch.float64,
        )
        edge_index = torch.tensor(
            [[0, 1, 2, 0, 2], [2, 2, 2, 0, 1]], dtype=torch.long
        )
        edge_type = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
        return x, edge_index, edge_type

    @staticmethod
    def _linear_modules(conv):
        return (
            getattr(conv, "lin_l", getattr(conv, "lin_rel", None)),
            getattr(conv, "lin_r", getattr(conv, "lin_root", None)),
        )

    def test_relation_decomposition_matches_native_rgcn_forward(self):
        torch.manual_seed(401)
        conv = RGCNConv(3, 4, 3, aggr="mean").double()
        x, edge_index, edge_type = self._graph()

        common, tracks, fused = decompose_rgcn_relation_tracks(
            conv, x, edge_index, edge_type
        )

        self.assertEqual(tracks.shape, (3, 3, 4))
        self.assertTrue(torch.equal(tracks[2], torch.zeros_like(tracks[2])))
        self.assertTrue(torch.allclose(fused, conv(x, edge_index, edge_type), atol=1e-12, rtol=0))
        self.assertTrue(torch.allclose(fused, common + tracks.sum(0), atol=1e-12, rtol=0))

    def test_relation_decomposition_matches_native_rgcn_backward(self):
        torch.manual_seed(409)
        native = RGCNConv(3, 4, 2, aggr="mean").double()
        routed = RGCNConv(3, 4, 2, aggr="mean").double()
        routed.load_state_dict(native.state_dict())
        x, edge_index, edge_type = self._graph()
        edge_type = edge_type.remainder(2)
        x_native = x.clone().requires_grad_(True)
        x_routed = x.clone().requires_grad_(True)

        native(x_native, edge_index, edge_type).square().sum().backward()
        decompose_rgcn_relation_tracks(
            routed, x_routed, edge_index, edge_type
        )[2].square().sum().backward()

        self.assertTrue(torch.allclose(x_native.grad, x_routed.grad, atol=1e-11, rtol=0))
        for native_parameter, routed_parameter in zip(
            native.parameters(), routed.parameters()
        ):
            self.assertTrue(
                torch.allclose(
                    native_parameter.grad, routed_parameter.grad, atol=1e-11, rtol=0
                )
            )

    def test_full_transition_helper_matches_native_graph_conv_forward_and_backward(self):
        torch.manual_seed(419)
        native_conv1 = RGCNConv(3, 4, 2, aggr="mean").double()
        routed_conv1 = RGCNConv(3, 4, 2, aggr="mean").double()
        routed_conv1.load_state_dict(native_conv1.state_dict())
        native = GraphConv(4, 5, aggr="add").double()
        routed = GraphConv(4, 5, aggr="add").double()
        routed.load_state_dict(native.state_dict())
        x, edge_index, edge_type = self._graph()
        edge_type = edge_type.remainder(2)
        x_native = x.clone().requires_grad_(True)
        x_routed = x.clone().requires_grad_(True)

        z_native = native_conv1(x_native, edge_index, edge_type)
        expected = native(z_native, edge_index)
        common, tracks, z_routed = decompose_rgcn_relation_tracks(
            routed_conv1, x_routed, edge_index, edge_type
        )
        actual = relation_track_graph_conv(
            routed, common, tracks, z_routed, edge_index, edge_type,
            diagonal=False,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-11, rtol=0))

        expected.square().sum().backward()
        native_x_grad = x_native.grad.clone()
        native_conv1_grads = [
            parameter.grad.clone() for parameter in native_conv1.parameters()
        ]
        native_grads = [parameter.grad.clone() for parameter in native.parameters()]
        actual.square().sum().backward()
        self.assertTrue(torch.allclose(x_routed.grad, native_x_grad, atol=1e-10, rtol=0))
        for expected_grad, parameter in zip(
            native_conv1_grads, routed_conv1.parameters()
        ):
            self.assertTrue(torch.allclose(parameter.grad, expected_grad, atol=1e-10, rtol=0))
        for expected_grad, parameter in zip(native_grads, routed.parameters()):
            self.assertTrue(torch.allclose(parameter.grad, expected_grad, atol=1e-10, rtol=0))

    def test_diagonal_routing_removes_only_cross_relation_transitions(self):
        torch.manual_seed(421)
        conv1 = RGCNConv(3, 4, 2, aggr="mean").double()
        conv2 = GraphConv(4, 5, aggr="add").double()
        x, edge_index, edge_type = self._graph()
        edge_type = edge_type.remainder(2)
        common, tracks, z = decompose_rgcn_relation_tracks(
            conv1, x, edge_index, edge_type
        )

        diagonal = relation_track_graph_conv(
            conv2, common, tracks, z, edge_index, edge_type, diagonal=True
        )
        full = relation_track_graph_conv(
            conv2, common, tracks, z, edge_index, edge_type, diagonal=False
        )

        cross_neighbor = torch.zeros_like(z)
        for relation_in in range(tracks.size(0)):
            for relation_out in range(tracks.size(0)):
                if relation_in == relation_out:
                    continue
                selected = edge_index[:, edge_type == relation_out]
                cross_neighbor = cross_neighbor + conv2.propagate(
                    selected,
                    x=(tracks[relation_in], tracks[relation_in]),
                    edge_weight=None,
                    size=(z.size(0), z.size(0)),
                )
        lin_l, _ = self._linear_modules(conv2)
        expected_difference = torch.nn.functional.linear(
            cross_neighbor, lin_l.weight, bias=None
        )
        self.assertTrue(torch.allclose(full - diagonal, expected_difference, atol=1e-11, rtol=0))
        self.assertFalse(torch.allclose(full, diagonal))

    def test_single_relation_degenerates_to_original(self):
        torch.manual_seed(431)
        conv1 = RGCNConv(3, 4, 1, aggr="mean").double()
        conv2 = GraphConv(4, 5, aggr="add").double()
        x, edge_index, _ = self._graph()
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
        common, tracks, z = decompose_rgcn_relation_tracks(
            conv1, x, edge_index, edge_type
        )
        actual = relation_track_graph_conv(
            conv2, common, tracks, z, edge_index, edge_type, diagonal=True
        )
        self.assertTrue(torch.allclose(actual, conv2(z, edge_index), atol=1e-11, rtol=0))

    def test_root_bias_only_and_empty_relations_are_finite(self):
        torch.manual_seed(433)
        conv1 = RGCNConv(3, 4, 3, aggr="mean").double()
        conv2 = GraphConv(4, 5, aggr="add").double()
        x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        common, tracks, z = decompose_rgcn_relation_tracks(
            conv1, x, edge_index, edge_type
        )
        actual = relation_track_graph_conv(
            conv2, common, tracks, z, edge_index, edge_type, diagonal=True
        )
        expected = conv2(z, edge_index)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=0))
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_rejects_non_native_rgcn_contract(self):
        x, edge_index, edge_type = self._graph()
        for conv in (
            RGCNConv(3, 4, 2, num_bases=1),
            RGCNConv(3, 4, 2, num_blocks=1),
            RGCNConv(3, 4, 2, aggr="add"),
        ):
            with self.subTest(conv=conv):
                with self.assertRaises(ValueError):
                    decompose_rgcn_relation_tracks(
                        conv.double(), x, edge_index, edge_type.remainder(2)
                    )


if __name__ == "__main__":
    unittest.main()
