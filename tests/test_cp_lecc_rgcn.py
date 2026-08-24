import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import RGCNConv

from cp_lecc_rgcn import CompletePreservingLowRankECCConv


def _graph(device=torch.device("cpu")):
    return (
        torch.tensor([[0, 2, 1, 3, 0, 3], [1, 1, 2, 2, 3, 0]], device=device),
        torch.tensor([0, 0, 1, 1, 2, 2], device=device),
    )


def _copy_base(source, target):
    with torch.no_grad():
        target.weight.copy_(source.weight)
        target.root.copy_(source.root)
        target.bias.copy_(source.bias)


def _zero_layer(layer):
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()


def _constant_correction_layer(num_relations=1):
    layer = CompletePreservingLowRankECCConv(
        1,
        1,
        num_relations,
        content_dim=1,
        relation_dim=1,
        generator_hidden=1,
        num_bases=1,
        basis_rank=1,
    )
    _zero_layer(layer)
    with torch.no_grad():
        layer.generator_output_bias.fill_(math.atanh(0.5))
        layer.basis_left.fill_(1.0)
        layer.basis_right.fill_(2.0)
    return layer


class CompletePreservingLowRankECCTests(unittest.TestCase):
    def test_graph_convs_do_not_require_pyg_utils_scatter_export(self):
        gcnet_dir = Path(__file__).resolve().parents[1] / "gcnet"
        script = f"""
import builtins
import sys
from torch_geometric.nn import RGCNConv

real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'torch_geometric.utils' and 'scatter' in fromlist:
        raise ImportError('simulated PyG 2.0 without utils.scatter')
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
sys.path.insert(0, {str(gcnet_dir)!r})
import cp_lecc_rgcn
import mpfilm_rgcn
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_complete_forward_backward_is_bitwise_pyg(self):
        torch.manual_seed(17)
        reference = RGCNConv(5, 3, 3)
        candidate = CompletePreservingLowRankECCConv(5, 3, 3)
        _copy_base(reference, candidate)
        with torch.no_grad():
            for name, parameter in candidate.named_parameters():
                if name not in {"weight", "root", "bias"}:
                    parameter.copy_(
                        torch.arange(parameter.numel(), dtype=parameter.dtype)
                        .reshape_as(parameter)
                        .remainder(7)
                        .sub_(3)
                    )

        x_reference = torch.randn(4, 5, requires_grad=True)
        x_candidate = x_reference.detach().clone().requires_grad_(True)
        edge_index, edge_type = _graph()
        mask = torch.ones(4, 3)
        expected = reference(x_reference, edge_index, edge_type)
        actual = candidate(x_candidate, edge_index, edge_type, mask)

        self.assertTrue(torch.equal(actual, expected))
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        actual.backward(upstream)
        for expected_gradient, actual_gradient in (
            (x_reference.grad, x_candidate.grad),
            (reference.weight.grad, candidate.weight.grad),
            (reference.root.grad, candidate.root.grad),
            (reference.bias.grad, candidate.bias.grad),
        ):
            self.assertTrue(torch.equal(actual_gradient, expected_gradient))
        for name, parameter in candidate.named_parameters():
            if name not in {"weight", "root", "bias"}:
                self.assertTrue(
                    parameter.grad is None
                    or torch.count_nonzero(parameter.grad).item() == 0
                )

    def test_construction_preserves_rgcn_rng_progression(self):
        torch.manual_seed(131)
        reference = RGCNConv(5, 3, 3)
        downstream_reference = nn.Linear(7, 4)

        torch.manual_seed(131)
        candidate = CompletePreservingLowRankECCConv(5, 3, 3)
        downstream_candidate = nn.Linear(7, 4)

        for reference_parameter, candidate_parameter in zip(
            reference.parameters(),
            (candidate.weight, candidate.root, candidate.bias),
        ):
            self.assertTrue(torch.equal(reference_parameter, candidate_parameter))
        self.assertTrue(
            torch.equal(downstream_reference.weight, downstream_candidate.weight)
        )
        self.assertTrue(
            torch.equal(downstream_reference.bias, downstream_candidate.bias)
        )

    def test_locked_extra_parameter_counts(self):
        temporal = CompletePreservingLowRankECCConv(400, 100, 3)
        speaker = CompletePreservingLowRankECCConv(400, 100, 4)
        temporal_base = RGCNConv(400, 100, 3)
        speaker_base = RGCNConv(400, 100, 4)

        temporal_extra = sum(p.numel() for p in temporal.parameters()) - sum(
            p.numel() for p in temporal_base.parameters()
        )
        speaker_extra = sum(p.numel() for p in speaker.parameters()) - sum(
            p.numel() for p in speaker_base.parameters()
        )
        self.assertEqual(temporal_extra, 30332)
        self.assertEqual(speaker_extra, 30340)
        self.assertEqual(temporal_extra + speaker_extra, 60672)

    def test_zero_initialized_generator_matches_pyg_and_still_learns(self):
        torch.manual_seed(23)
        reference = RGCNConv(2, 2, 1)
        candidate = CompletePreservingLowRankECCConv(
            2,
            2,
            1,
            content_dim=2,
            relation_dim=1,
            generator_hidden=3,
            num_bases=2,
            basis_rank=2,
        )
        _copy_base(reference, candidate)
        self.assertEqual(
            torch.count_nonzero(candidate.generator_hidden_bias).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(candidate.generator_output_weight).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(candidate.generator_output_bias).item(), 0
        )
        x = torch.tensor([[1.0, 2.0], [3.0, -1.0], [2.0, 1.0]])
        edge_index = torch.tensor([[0, 1], [2, 2]])
        edge_type = torch.zeros(2, dtype=torch.long)
        mask = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 1]])

        expected = reference(x, edge_index, edge_type)
        actual = candidate(x, edge_index, edge_type, mask)

        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        gradient = candidate.generator_output_weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(torch.count_nonzero(gradient).item(), 0)

    def test_descriptor_uses_ordered_patterns_content_and_relation(self):
        layer = CompletePreservingLowRankECCConv(
            2,
            1,
            2,
            content_dim=1,
            relation_dim=1,
            generator_hidden=1,
            num_bases=1,
            basis_rank=1,
        )
        _zero_layer(layer)
        with torch.no_grad():
            layer.target_content.copy_(torch.tensor([[1.0], [0.0]]))
            layer.source_content.copy_(torch.tensor([[0.0], [1.0]]))
            layer.relation_embedding.copy_(torch.tensor([[1.0], [3.0]]))
            layer.generator_output_weight.fill_(1.0)
            layer.basis_left[0, 0, 0] = 1.0
            layer.basis_right.fill_(1.0)
        x = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
        masks = torch.tensor([[1, 0, 0], [0, 1, 0]])

        def correction(weight_column, source=0, target=1, relation=0, scale=1.0):
            with torch.no_grad():
                layer.generator_hidden_weight.zero_()
                layer.generator_hidden_weight[weight_column, 0] = scale
            edge_index = torch.tensor([[source], [target]])
            edge_type = torch.tensor([relation])
            return layer(x, edge_index, edge_type, masks)[target, 0].item()

        with torch.no_grad():
            layer.generator_hidden_weight.zero_()
            layer.generator_hidden_weight[1, 0] = 1.0
            layer.generator_hidden_weight[6, 0] = 2.0
        forward = layer(
            x, torch.tensor([[0], [1]]), torch.tensor([0]), masks
        )[1, 0].item()
        reverse = layer(
            x, torch.tensor([[1], [0]]), torch.tensor([0]), masks
        )[0, 0].item()
        product_pattern = correction(12, source=0, target=0)
        relation_zero = correction(18, relation=0)
        relation_one = correction(18, relation=1)
        content_forward = correction(19, source=0, target=1)
        content_reverse = correction(19, source=1, target=0)

        self.assertNotEqual(forward, reverse)
        self.assertNotEqual(product_pattern, 0.0)
        self.assertNotEqual(relation_zero, relation_one)
        self.assertNotEqual(content_forward, content_reverse)

    def test_complete_complete_edge_is_inactive_but_missing_endpoint_is_active(self):
        layer = _constant_correction_layer()
        x = torch.tensor([[1.0], [2.0], [3.0]])
        edge_index = torch.tensor([[0, 0], [1, 2]])
        edge_type = torch.zeros(2, dtype=torch.long)
        mask = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 0, 0]])

        base = RGCNConv.forward(layer, x, edge_index, edge_type)
        actual = layer(x, edge_index, edge_type, mask)

        self.assertEqual((actual - base)[1, 0].item(), 0.0)
        self.assertEqual((actual - base)[2, 0].item(), 1.0)

    def test_correction_means_within_relation_then_sums_relations(self):
        layer = _constant_correction_layer(num_relations=2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[[2.0]], [[3.0]]]))
            layer.root.fill_(4.0)
            layer.bias.fill_(5.0)
        x = torch.tensor([[1.0], [3.0], [2.0], [7.0]])
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])
        edge_type = torch.tensor([0, 0, 1])
        mask = torch.tensor([[1, 0, 0]] * 4)

        actual = layer(x, edge_index, edge_type, mask)

        base_target = ((1.0 + 3.0) / 2.0) * 2.0 + 2.0 * 3.0 + 7.0 * 4.0 + 5.0
        correction = ((1.0 + 3.0) / 2.0) + 2.0
        self.assertEqual(actual[3, 0].item(), base_target + correction)
        self.assertEqual(actual[0, 0].item(), 1.0 * 4.0 + 5.0)

    def test_homogeneous_a_only_neighborhood_uses_source_content(self):
        layer = CompletePreservingLowRankECCConv(
            2,
            1,
            1,
            content_dim=1,
            relation_dim=1,
            generator_hidden=1,
            num_bases=1,
            basis_rank=1,
        )
        _zero_layer(layer)
        with torch.no_grad():
            layer.target_content.copy_(torch.tensor([[1.0], [0.0]]))
            layer.source_content.copy_(torch.tensor([[0.0], [1.0]]))
            layer.generator_hidden_weight[19, 0] = 1.0
            layer.generator_output_weight.fill_(1.0)
            layer.basis_left[0, 0, 0] = 1.0
            layer.basis_right.fill_(1.0)
        masks = torch.tensor([[1, 0, 0]] * 3)
        edge_index = torch.tensor([[0], [2]])
        edge_type = torch.zeros(1, dtype=torch.long)
        x_low = torch.tensor([[1.0, 1.0], [1.0, 4.0], [1.0, 0.0]])
        x_high = x_low.clone()
        x_high[0, 1] = 4.0

        low = layer(x_low, edge_index, edge_type, masks)
        high = layer(x_high, edge_index, edge_type, masks)

        self.assertNotEqual(low[2, 0].item(), high[2, 0].item())

    def test_fixed_edge_uses_target_content(self):
        layer = CompletePreservingLowRankECCConv(
            2,
            1,
            1,
            content_dim=1,
            relation_dim=1,
            generator_hidden=1,
            num_bases=1,
            basis_rank=1,
        )
        _zero_layer(layer)
        with torch.no_grad():
            layer.target_content.copy_(torch.tensor([[0.0], [1.0]]))
            layer.source_content.copy_(torch.tensor([[1.0], [0.0]]))
            layer.generator_hidden_weight[19, 0] = 1.0
            layer.generator_output_weight.fill_(1.0)
            layer.basis_left[0, 0, 0] = 1.0
            layer.basis_right.fill_(1.0)
        masks = torch.tensor([[1, 0, 0], [1, 0, 0]])
        edge_index = torch.tensor([[0], [1]])
        edge_type = torch.zeros(1, dtype=torch.long)
        x_low = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        x_high = x_low.clone()
        x_high[1, 1] = 4.0

        low = layer(x_low, edge_index, edge_type, masks)
        high = layer(x_high, edge_index, edge_type, masks)

        self.assertNotEqual(low[1, 0].item(), high[1, 0].item())

    def test_rejects_invalid_inputs(self):
        layer = CompletePreservingLowRankECCConv(2, 2, 2)
        x = torch.ones(3, 2)
        edges = torch.tensor([[0, 1], [1, 2]])
        relations = torch.tensor([0, 1])
        mask = torch.ones(3, 3)
        cases = (
            (torch.ones(3), edges, relations, mask),
            (x, torch.tensor([[0, 1]]), relations, mask),
            (x, edges, torch.tensor([0]), mask),
            (x, edges, relations, torch.ones(2, 3)),
            (x, edges, relations, torch.ones(3, 2)),
            (x, torch.tensor([[0], [3]]), torch.tensor([0]), mask),
            (x, torch.tensor([[-1], [2]]), torch.tensor([0]), mask),
            (x, torch.tensor([[0], [2]]), torch.tensor([2]), mask),
            (x, torch.tensor([[0], [2]]), torch.tensor([-1]), mask),
        )
        for bad_x, bad_edges, bad_relations, bad_mask in cases:
            with self.subTest(
                x_shape=tuple(bad_x.shape),
                edge_shape=tuple(bad_edges.shape),
                relation=bad_relations.tolist(),
                mask_shape=tuple(bad_mask.shape),
            ):
                with self.assertRaises(ValueError):
                    layer(bad_x, bad_edges, bad_relations, bad_mask)

    def test_rejects_non_long_graph_indices(self):
        layer = CompletePreservingLowRankECCConv(2, 2, 2)
        x = torch.ones(3, 2)
        edges = torch.tensor([[0, 1], [1, 2]])
        relations = torch.tensor([0, 1])
        mask = torch.ones(3, 3)

        with self.assertRaises(ValueError):
            layer(x, edges.to(torch.float32), relations, mask)
        with self.assertRaises(ValueError):
            layer(x, edges, relations.to(torch.float32), mask)

    def test_rejects_invalid_masks_through_convolution_forward(self):
        layer = CompletePreservingLowRankECCConv(2, 2, 2)
        x = torch.ones(3, 2)
        edges = torch.tensor([[0, 1], [1, 2]])
        relations = torch.tensor([0, 1])
        invalid_masks = (
            torch.tensor([[0, 0, 0], [1, 1, 1], [1, 1, 1]]),
            torch.tensor([[0.5, 1, 0], [1, 1, 1], [1, 1, 1]]),
        )

        for mask in invalid_masks:
            with self.subTest(mask=mask.tolist()):
                with self.assertRaises(ValueError):
                    layer(x, edges, relations, mask)

    def _assert_finite_forward_backward(self, device):
        torch.manual_seed(37)
        layer = CompletePreservingLowRankECCConv(4, 3, 2).to(device)
        with torch.no_grad():
            layer.generator_output_weight.normal_()
        x = torch.randn(5, 4, device=device, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device)
        edge_type = torch.tensor([0, 1, 0, 1], device=device)
        mask = torch.tensor(
            [[1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]],
            dtype=torch.float32,
            device=device,
        )

        output = layer(x, edge_index, edge_type, mask)
        output.square().mean().backward()

        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        for parameter in layer.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_cpu_fp32_forward_backward_is_finite(self):
        self._assert_finite_forward_backward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fp32_forward_backward_is_finite(self):
        self._assert_finite_forward_backward(torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
