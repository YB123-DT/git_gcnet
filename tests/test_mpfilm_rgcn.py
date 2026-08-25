import unittest

import torch
from torch_geometric.nn import RGCNConv

from versions.mpfilm.variant import MissingPatternFiLMRGCNConv


assert_close = getattr(torch.testing, "assert_close", torch.testing.assert_allclose)


def _graph(device):
    edge_index = torch.tensor(
        [[0, 2, 1, 3, 0, 3], [1, 1, 2, 2, 3, 0]],
        dtype=torch.long,
        device=device,
    )
    edge_type = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    return edge_index, edge_type


def _copy_rgcn_parameters(source, target):
    with torch.no_grad():
        target.weight.copy_(source.weight)
        target.root.copy_(source.root)
        target.bias.copy_(source.bias)


class MPFiLMConvolutionTests(unittest.TestCase):
    def _assert_complete_parity(self, device, variant="full"):
        torch.manual_seed(17)
        x_ref = torch.randn(4, 5, device=device, requires_grad=True)
        x_new = x_ref.detach().clone().requires_grad_(True)
        edge_index, edge_type = _graph(device)
        node_mask = torch.ones(4, 3, device=device)
        reference = RGCNConv(5, 3, 3).to(device)
        candidate = MissingPatternFiLMRGCNConv(5, 3, 3, variant=variant).to(
            device
        )
        _copy_rgcn_parameters(reference, candidate)

        expected = reference(x_ref, edge_index, edge_type)
        actual = candidate(x_new, edge_index, edge_type, node_mask)

        self.assertTrue(
            torch.equal(actual, expected),
            msg=f"complete forward differs by {(actual - expected).abs().max().item()}",
        )
        gradient = torch.randn_like(expected)
        expected.backward(gradient)
        actual.backward(gradient)
        pairs = (
            (x_ref.grad, x_new.grad),
            (reference.weight.grad, candidate.weight.grad),
            (reference.root.grad, candidate.root.grad),
            (reference.bias.grad, candidate.bias.grad),
        )
        for reference_gradient, candidate_gradient in pairs:
            self.assertTrue(
                torch.equal(reference_gradient, candidate_gradient),
                msg=(
                    "complete backward differs by "
                    f"{(reference_gradient - candidate_gradient).abs().max().item()}"
                ),
            )
        for parameter in (candidate.pattern_weight, candidate.film_weight):
            if parameter.grad is not None:
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_complete_forward_and_backward_match_pyg_rgcn_on_cpu(self):
        self._assert_complete_parity(torch.device("cpu"))

    def test_faithful_edgewise_complete_forward_matches_pyg_rgcn(self):
        self._assert_complete_parity(
            torch.device("cpu"), variant="faithful_edgewise"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_complete_forward_and_backward_match_pyg_rgcn_on_gpu(self):
        self._assert_complete_parity(torch.device("cuda"))

    def test_severe_missing_patterns_remain_distinguishable(self):
        layer = MissingPatternFiLMRGCNConv(2, 2, 1, variant="full")
        with torch.no_grad():
            layer.weight.zero_()
            layer.root.zero_()
            layer.bias.zero_()
            layer.pattern_weight.zero_()
            layer.pattern_weight[0, 0] = torch.tensor([1.0, 0.0])
            layer.pattern_weight[0, 1] = torch.tensor([0.0, 1.0])
            layer.pattern_weight[0, 2] = torch.tensor([2.0, 2.0])
        x = torch.zeros(3, 2)
        edges = torch.tensor([[0, 1, 2], [0, 1, 2]])
        relations = torch.zeros(3, dtype=torch.long)
        masks = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        output = layer(x, edges, relations, masks)

        self.assertFalse(torch.equal(output[0], output[1]))
        self.assertFalse(torch.equal(output[1], output[2]))
        self.assertFalse(torch.equal(output[0], output[2]))

    def test_full_modulates_a_only_homogeneous_neighborhood_by_content(self):
        full = MissingPatternFiLMRGCNConv(2, 2, 1, variant="full")
        reference = RGCNConv(2, 2, 1)
        _copy_rgcn_parameters(reference, full)
        with torch.no_grad():
            full.pattern_weight[0, 0].fill_(0.25)
            full.film_weight.zero_()
            full.film_weight[0, 0, 0] = 0.5
        edge_index = torch.tensor([[0, 1], [2, 2]])
        edge_type = torch.zeros(2, dtype=torch.long)
        masks = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        x_first = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        x_second = x_first.clone()
        x_second[0, 0] = 3.0

        original = reference(x_first, edge_index, edge_type)
        first = full(x_first, edge_index, edge_type, masks)
        second = full(x_second, edge_index, edge_type, masks)

        self.assertFalse(torch.allclose(first, original))
        self.assertFalse(torch.allclose(first[2], second[2]))

    def test_single_neighbor_relation_includes_weight_root_and_bias(self):
        layer = MissingPatternFiLMRGCNConv(2, 2, 2, variant="full")
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[1].copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
            layer.root.copy_(torch.eye(2))
            layer.bias.copy_(torch.tensor([0.5, -0.5]))
        x = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
        edge_index = torch.tensor([[0], [1]])
        edge_type = torch.tensor([1])
        masks = torch.tensor([[1, 0, 0], [0, 1, 0]])

        output = layer(x, edge_index, edge_type, masks)

        expected = torch.stack(
            (x[0] + layer.bias, x[0] @ layer.weight[1] + x[1] + layer.bias)
        )
        assert_close(output, expected)

    def test_faithful_activation_is_applied_per_edge_before_mean(self):
        linearized = MissingPatternFiLMRGCNConv(1, 1, 1, variant="full")
        faithful = MissingPatternFiLMRGCNConv(
            1, 1, 1, variant="faithful_edgewise"
        )
        faithful.load_state_dict(linearized.state_dict())
        with torch.no_grad():
            for layer in (linearized, faithful):
                layer.weight.fill_(1.0)
                layer.root.zero_()
                layer.bias.zero_()
                layer.pattern_weight.zero_()
                layer.film_weight.zero_()
        x = torch.tensor([[-2.0], [1.0], [0.0]])
        edge_index = torch.tensor([[0, 1], [2, 2]])
        edge_type = torch.zeros(2, dtype=torch.long)
        node_mask = torch.tensor([[1, 0, 0]] * 3)

        linearized_output = linearized(x, edge_index, edge_type, node_mask)
        faithful_output = faithful(x, edge_index, edge_type, node_mask)

        assert_close(linearized_output[2], torch.tensor([-0.5]))
        assert_close(faithful_output[2], torch.tensor([0.5]))

    def test_faithful_and_linearized_have_identical_parameter_count(self):
        linearized = MissingPatternFiLMRGCNConv(5, 3, 3, variant="full")
        faithful = MissingPatternFiLMRGCNConv(
            5, 3, 3, variant="faithful_edgewise"
        )

        self.assertEqual(
            sum(parameter.numel() for parameter in linearized.parameters()),
            sum(parameter.numel() for parameter in faithful.parameters()),
        )

    def test_variant_parameter_counts_are_reportable(self):
        full = MissingPatternFiLMRGCNConv(5, 3, 3, variant="full")
        pattern = MissingPatternFiLMRGCNConv(5, 3, 3, variant="pattern_only")
        control = MissingPatternFiLMRGCNConv(
            5, 3, 3, variant="content_film_control"
        )
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in (
                ("pattern_only", pattern),
                ("full", full),
                ("content_film_control", control),
            )
        }

        self.assertGreater(counts["full"], counts["pattern_only"])
        self.assertGreater(counts["content_film_control"], 0)


if __name__ == "__main__":
    unittest.main()
