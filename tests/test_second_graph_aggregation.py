import math
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GraphConv

from second_graph_aggregation import (
    GenAggGraphConv,
    SSMAGraphConv,
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


class SSMAGraphConvTest(unittest.TestCase):
    @staticmethod
    def _explicit_dft2(signal):
        height, width = signal.shape
        real = signal.new_zeros((height, width))
        imaginary = signal.new_zeros((height, width))
        for frequency_row in range(height):
            for frequency_column in range(width):
                for row in range(height):
                    for column in range(width):
                        angle = 2.0 * math.pi * (
                            float(frequency_row * row) / height
                            + float(frequency_column * column) / width
                        )
                        real[frequency_row, frequency_column] += (
                            signal[row, column] * math.cos(angle)
                        )
                        imaginary[frequency_row, frequency_column] -= (
                            signal[row, column] * math.sin(angle)
                        )
        return real, imaginary

    @staticmethod
    def _explicit_idft2_real(real, imaginary):
        height, width = real.shape
        output = real.new_zeros((height, width))
        for row in range(height):
            for column in range(width):
                for frequency_row in range(height):
                    for frequency_column in range(width):
                        angle = 2.0 * math.pi * (
                            float(frequency_row * row) / height
                            + float(frequency_column * column) / width
                        )
                        output[row, column] += (
                            real[frequency_row, frequency_column]
                            * math.cos(angle)
                            - imaginary[frequency_row, frequency_column]
                            * math.sin(angle)
                        )
        return output / float(height * width)

    @staticmethod
    def _reference_aggregate(inputs, target, target_count, kappa, epsilon):
        dimension = inputs.size(-1)
        signal_height = kappa + 1
        signal_width = kappa * (dimension - 1) + 1
        output = inputs.new_zeros(
            (target_count, signal_height * signal_width)
        )
        for node in range(target_count):
            incoming = inputs[target.eq(node)]
            if incoming.size(0) == 0:
                continue
            signals = inputs.new_zeros(
                (incoming.size(0), signal_height, signal_width)
            )
            signals[:, 0, :dimension] = -incoming
            signals[:, 1, 0] = 1.0
            spectra = torch.fft.fft2(signals)
            magnitude = torch.exp(
                torch.log(spectra.abs() + epsilon).mean(dim=0)
            )
            phase = torch.angle(spectra).sum(dim=0)
            mixed = torch.complex(
                magnitude * torch.cos(phase),
                magnitude * torch.sin(phase),
            )
            output[node] = torch.fft.ifft2(mixed).real.reshape(-1)
        return output

    def test_tiny_signal_matches_dft_loop_oracle_without_fft_reconstruction(self):
        conv = SSMAGraphConv(2, 2, kappa=2, epsilon=1e-6).double()
        self.assertEqual(conv.signal_height, 3)
        self.assertEqual(conv.signal_width, 3)
        self.assertEqual(conv.signal_size, 9)
        with torch.no_grad():
            conv.compressor.weight.copy_(
                torch.arange(1, 19, dtype=torch.double).reshape(2, 9)
                / 19.0
            )
            conv.compressor.bias.copy_(torch.tensor([0.15, -0.35]))
            conv.lin_l.weight.copy_(
                torch.tensor([[1.25, -0.4], [0.3, 0.85]])
            )
            conv.lin_l.bias.copy_(torch.tensor([-0.2, 0.45]))
            conv.lin_r.weight.copy_(
                torch.tensor([[0.75, 0.2], [-0.6, 1.1]])
            )
        x = torch.tensor(
            [[0.4, -0.7], [-1.1, 0.9], [0.3, 0.6]], dtype=torch.double
        )
        edge_index = torch.tensor([[0, 1], [2, 2]])

        spectra = []
        for value in x[:2]:
            signal = x.new_zeros((3, 3))
            signal[0, :2] = -value
            signal[1, 0] = 1.0
            spectra.append(self._explicit_dft2(signal))
        real_parts = torch.stack([spectrum[0] for spectrum in spectra])
        imaginary_parts = torch.stack([spectrum[1] for spectrum in spectra])
        magnitudes = torch.sqrt(real_parts.square() + imaginary_parts.square())
        magnitude = torch.exp(torch.log(magnitudes + conv.epsilon).mean(0))
        phase = torch.atan2(imaginary_parts, real_parts).sum(0)
        mixed_real = magnitude * torch.cos(phase)
        mixed_imaginary = magnitude * torch.sin(phase)
        decoded = self._explicit_idft2_real(mixed_real, mixed_imaginary).reshape(-1)
        expected_neighbor = conv.compressor(decoded)
        expected = conv.lin_l(expected_neighbor) + conv.lin_r(x[2])

        torch.testing.assert_close(
            conv(x, edge_index)[2], expected, rtol=1e-9, atol=1e-9
        )

    def test_signal_height_and_parameter_count_follow_kappa(self):
        baseline = GraphConv(2, 2)
        candidate = SSMAGraphConv(2, 2, kappa=2)
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
        self.assertEqual(candidate.signal_height, 3)
        self.assertEqual(candidate.signal_width, 3)
        self.assertEqual(candidate.signal_size, 9)
        self.assertEqual(candidate_count - baseline_count, 20)

    def test_backward_does_not_call_torch_polar(self):
        conv = SSMAGraphConv(2, 2)
        x = torch.randn(3, 2, requires_grad=True)
        edge_index = torch.tensor([[0, 1], [2, 2]])
        with mock.patch(
            "torch.polar",
            side_effect=RuntimeError("Torch 1.8 complex backward unsupported"),
        ):
            conv(x, edge_index).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_matches_explicit_per_neighborhood_fft_formula(self):
        conv = SSMAGraphConv(2, 3, kappa=5, epsilon=1e-6).double()
        x = torch.tensor(
            [[0.2, -0.7], [1.3, 0.4], [-0.8, 1.1], [0.5, -0.2]],
            dtype=torch.double,
        )
        edge_index = torch.tensor([[0, 1, 2, 0], [3, 3, 3, 2]])
        edge_weight = torch.tensor([1.0, 0.25, -0.5, 1.5], dtype=torch.double)

        actual = conv(x, edge_index, edge_weight=edge_weight)
        messages = x.index_select(0, edge_index[0]) * edge_weight[:, None]
        encoded = self._reference_aggregate(
            messages, edge_index[1], x.size(0), conv.kappa, conv.epsilon
        )
        degree = torch.bincount(edge_index[1], minlength=x.size(0)).gt(0)
        compressed = conv.compressor(encoded) * degree[:, None]
        expected = conv.lin_l(compressed) + conv.lin_r(x)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_edge_permutation_is_invariant(self):
        torch.manual_seed(41)
        conv = SSMAGraphConv(3, 2)
        x = torch.randn(6, 3)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 5, 4, 5, 4]])
        permutation = torch.tensor([4, 1, 3, 0, 2])
        torch.testing.assert_close(
            conv(x, edge_index), conv(x, edge_index[:, permutation])
        )

    def test_mixed_neighbor_derivative_is_nonzero_but_add_is_zero(self):
        torch.manual_seed(43)
        candidate = SSMAGraphConv(2, 1).double()
        baseline = GraphConv(2, 1).double()
        _copy_graph_conv_parameters(candidate, baseline)
        with torch.no_grad():
            candidate.compressor.weight.normal_(mean=0.2, std=0.1)
            candidate.compressor.bias.zero_()
        edge_index = torch.tensor([[0, 1], [2, 2]])

        def mixed_derivative(module):
            values = torch.tensor(
                [[0.3, 0.8], [1.1, -0.4], [0.2, 0.5]],
                dtype=torch.double,
                requires_grad=True,
            )
            output = module(values, edge_index)[2, 0]
            first = torch.autograd.grad(output, values, create_graph=True)[0][0, 0]
            if not first.requires_grad:
                return first.new_zeros(())
            second = torch.autograd.grad(
                first, values, allow_unused=True
            )[0]
            if second is None:
                return first.new_zeros(())
            return second[1, 0]

        self.assertGreater(abs(mixed_derivative(candidate).item()), 1e-8)
        self.assertEqual(mixed_derivative(baseline).item(), 0.0)

    def test_rejects_neighborhood_larger_than_kappa(self):
        conv = SSMAGraphConv(2, 2, kappa=5)
        x = torch.randn(7, 2)
        edge_index = torch.stack(
            (torch.arange(6, dtype=torch.long), torch.full((6,), 6))
        )
        with self.assertRaisesRegex(ValueError, "degree.*kappa"):
            conv(x, edge_index)

    def test_zero_degree_keeps_only_neighbor_bias_and_root(self):
        conv = SSMAGraphConv(2, 2)
        with torch.no_grad():
            conv.lin_l.bias.copy_(torch.tensor([0.5, -0.25]))
            conv.lin_r.weight.copy_(torch.eye(2))
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        edge_index = torch.empty((2, 0), dtype=torch.long)
        torch.testing.assert_close(
            conv(x, edge_index), x + torch.tensor([0.5, -0.25])
        )

    def test_formal_width_adds_297700_parameters_per_branch(self):
        baseline = GraphConv(100, 100)
        candidate = SSMAGraphConv(100, 100, kappa=5)
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
        candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
        self.assertEqual(candidate.signal_width, 496)
        self.assertEqual(candidate.signal_size, 2976)
        self.assertEqual(candidate_count - baseline_count, 297700)

    def test_extra_parameter_initialization_preserves_cpu_rng(self):
        torch.manual_seed(1234)
        GraphConv(3, 5)
        expected_next = torch.rand(8)

        torch.manual_seed(1234)
        SSMAGraphConv(3, 5)
        actual_next = torch.rand(8)
        torch.testing.assert_close(actual_next, expected_next)

    def test_forward_and_complex_fft_path_have_finite_gradients(self):
        torch.manual_seed(47)
        conv = SSMAGraphConv(3, 2)
        x = torch.randn(5, 3, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 1, 3], [3, 3, 3, 4, 4]])
        conv(x, edge_index).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in conv.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class SecondGraphConvFactoryTest(unittest.TestCase):
    def test_builds_each_supported_selector(self):
        self.assertIs(type(build_second_graph_conv("add", 2, 3)), GraphConv)
        self.assertIsInstance(
            build_second_graph_conv("genagg", 2, 3), GenAggGraphConv
        )
        self.assertIsInstance(
            build_second_graph_conv("soft_medoid", 2, 3), SoftMedoidGraphConv
        )
        self.assertIsInstance(
            build_second_graph_conv("ssma", 2, 3), SSMAGraphConv
        )

    def test_rejects_unknown_selector(self):
        with self.assertRaisesRegex(ValueError, "unknown second graph aggregation"):
            build_second_graph_conv("median", 2, 3)


if __name__ == "__main__":
    unittest.main()
