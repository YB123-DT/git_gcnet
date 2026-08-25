import inspect
import unittest

import torch

from versions.sequence_aff import variant as sequence_aff
from missing_patterns import encode_missing_patterns
from versions.sequence_aff.variant import MaskConditionedSequenceAFF


assert_close = getattr(torch.testing, "assert_close", torch.testing.assert_allclose)


PATTERNS = torch.tensor(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.float32,
)


class SequenceAFFTests(unittest.TestCase):
    def test_vectorized_pattern_encoding_matches_shared_encoder(self):
        module = MaskConditionedSequenceAFF(8)
        valid = torch.ones(7, 1, dtype=torch.bool)

        actual, incomplete = module._encode_patterns(PATTERNS[:, None, :], valid)
        expected, complete = encode_missing_patterns(PATTERNS)

        assert_close(actual[:, 0], expected)
        assert_close(incomplete[:, 0], ~complete)

    def test_forward_source_has_no_host_pattern_conversion(self):
        source = inspect.getsource(sequence_aff)

        self.assertNotIn(".cpu(", source)
        self.assertNotIn(".tolist(", source)
        self.assertNotIn("encode_missing_patterns", source)

    def test_encodes_seven_patterns_in_time_batch_order_and_ignores_padding(self):
        module = MaskConditionedSequenceAFF(8)
        modality_mask = torch.zeros(4, 2, 3)
        modality_mask[0, 0] = PATTERNS[0]
        modality_mask[0, 1] = PATTERNS[1]
        modality_mask[1, 0] = PATTERNS[2]
        modality_mask[1, 1] = PATTERNS[3]
        modality_mask[2, 0] = PATTERNS[4]
        modality_mask[2, 1] = PATTERNS[5]
        modality_mask[3, 0] = PATTERNS[6]
        umask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])

        encoded, incomplete = module._encode_patterns(modality_mask, umask.t().bool())

        expected = torch.zeros(4, 2, 6)
        expected[0, 0, 0] = 1
        expected[0, 1, 1] = 1
        expected[1, 0, 2] = 1
        expected[1, 1, 3] = 1
        expected[2, 0, 4] = 1
        expected[2, 1, 5] = 1
        assert_close(encoded, expected)
        assert_close(
            incomplete,
            torch.tensor(
                [[True, True], [True, True], [True, True], [False, False]]
            ),
        )

    def test_forward_shape_and_input_validation(self):
        module = MaskConditionedSequenceAFF(4)
        x = torch.randn(3, 2, 4)
        y = torch.randn_like(x)
        mask = torch.ones(3, 2, 3)
        umask = torch.ones(2, 3)

        self.assertEqual(module(x, y, mask, umask).shape, x.shape)

        invalid_calls = (
            (x[:, :, :3], y[:, :, :3], mask, umask),
            (x, y[:, :1], mask, umask),
            (x.long(), y.long(), mask, umask),
            (x, y, mask[:, :, :2], umask),
            (x, y, mask, umask[:, :2]),
            (x, y, mask.clone().fill_(0.5), umask),
            (x, y, mask, umask.clone().fill_(0.5)),
        )
        for arguments in invalid_calls:
            with self.subTest(shapes=[tuple(value.shape) for value in arguments]):
                with self.assertRaises(ValueError):
                    module(*arguments)

        zero_valid = mask.clone()
        zero_valid[1, 0] = 0
        with self.assertRaisesRegex(ValueError, "valid.*modality"):
            module(x, y, zero_valid, umask)

        empty_conversation = umask.clone()
        empty_conversation[1] = 0
        with self.assertRaisesRegex(ValueError, "conversation"):
            module(x, y, mask, empty_conversation)

    def test_all_atv_is_exact_addition_with_exact_direct_gradients(self):
        torch.manual_seed(3)
        module = MaskConditionedSequenceAFF(8)
        for parameter in module.parameters():
            torch.nn.init.uniform_(parameter, -1.0, 1.0)
        x = torch.randn(3, 2, 8, requires_grad=True)
        y = torch.randn(3, 2, 8, requires_grad=True)
        mask = torch.ones(3, 2, 3)
        umask = torch.tensor([[1, 1, 1], [1, 1, 0]])

        output = module(x, y, mask, umask)

        assert_close(output, x + y, rtol=0, atol=0)
        output.sum().backward()
        assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
        assert_close(y.grad, torch.ones_like(y), rtol=0, atol=0)

    def test_zero_initialized_outputs_reproduce_addition_for_all_patterns(self):
        module = MaskConditionedSequenceAFF(8)
        x = torch.randn(7, 1, 8)
        y = torch.randn_like(x)
        umask = torch.ones(1, 7)

        output = module(x, y, PATTERNS[:, None, :], umask)

        assert_close(output, x + y, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(module.local_context[-1].weight), 0)
        self.assertEqual(torch.count_nonzero(module.local_context[-1].bias), 0)
        self.assertEqual(torch.count_nonzero(module.global_context[-1].weight), 0)
        self.assertEqual(torch.count_nonzero(module.global_context[-1].bias), 0)

    def test_manual_parameters_make_selection_context_dependent(self):
        module = MaskConditionedSequenceAFF(8)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
            module.local_context[0].weight[0, 2] = 1
            module.local_context[0].weight[0, 8] = 2
            module.local_context[0].weight[1] = -module.local_context[0].weight[0]
            module.local_context[1].weight.fill_(1)
            module.local_context[-1].weight[0, 0] = 2
            module.global_context[0].weight[0, 3] = 1
            module.global_context[0].weight[1, 3] = -1
            module.global_context[1].weight.fill_(1)
            module.global_context[-1].weight[0, 0] = 2

        x = torch.zeros(2, 4, 8)
        y = torch.zeros_like(x)
        x[:, :, 0] = 1
        y[:, :, 0] = -1
        x[0, 0, 2] = 1
        x[0, 1, 2] = -1
        x[0, 2, 2] = -3
        x[0, 3, 2] = -3
        x[1, 2, 3] = 4
        x[1, 3, 3] = -4
        mask = PATTERNS[1].repeat(2, 4, 1)
        umask = torch.ones(4, 2)

        output = module(x, y, mask, umask)

        self.assertNotEqual(output[0, 0, 0].item(), output[0, 1, 0].item())
        self.assertNotEqual(output[0, 2, 0].item(), output[0, 3, 0].item())
        mask_with_pattern_change = mask.clone()
        mask_with_pattern_change[0, 1] = PATTERNS[0]
        output_with_pattern_change = module(x, y, mask_with_pattern_change, umask)
        self.assertNotEqual(
            output[0, 1, 0].item(), output_with_pattern_change[0, 1, 0].item()
        )

    def test_padded_content_does_not_affect_valid_global_context(self):
        module = MaskConditionedSequenceAFF(8)
        with torch.no_grad():
            module.global_context[-1].weight.fill_(0.5)
        x = torch.randn(3, 2, 8)
        y = torch.randn_like(x)
        mask = PATTERNS[0].repeat(3, 2, 1)
        umask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        changed_x = x.clone()
        changed_y = y.clone()
        changed_x[2, 1] = float("nan")
        changed_y[2, 1] = float("inf")

        original = module(x, y, mask, umask)
        changed = module(changed_x, changed_y, mask, umask)

        assert_close(original[:2, 1], changed[:2, 1], rtol=0, atol=0)

    def test_cpu_float32_backward_is_finite(self):
        module = MaskConditionedSequenceAFF(7, reduction=4)
        x = torch.randn(4, 2, 7, requires_grad=True)
        y = torch.randn_like(x, requires_grad=True)
        mask = PATTERNS[:4, None, :].repeat(1, 2, 1)
        umask = torch.ones(2, 4)

        module(x, y, mask, umask).square().mean().backward()

        gradients = [x.grad, y.grad] + [
            parameter.grad for parameter in module.parameters()
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_small_channel_module_retains_content_dependent_gate(self):
        module = MaskConditionedSequenceAFF(7, reduction=4)
        self.assertEqual(module.local_context[0].out_features, 2)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
            module.local_context[0].weight[0, 2] = 1
            module.local_context[0].weight[1, 2] = -1
            module.local_context[1].weight.fill_(1)
            module.local_context[-1].weight[0, 0] = 2

        x = torch.zeros(2, 1, 7)
        y = torch.zeros_like(x)
        x[:, :, 0] = 1
        y[:, :, 0] = -1
        x[0, 0, 2] = 1
        x[1, 0, 2] = -1
        mask = PATTERNS[0].repeat(2, 1, 1)

        output = module(x, y, mask, torch.ones(1, 2))

        self.assertNotEqual(output[0, 0, 0].item(), output[1, 0, 0].item())

    def test_complete_bypass_ignores_nonfinite_aff_candidate(self):
        module = MaskConditionedSequenceAFF(8)
        x = torch.zeros(2, 1, 8)
        y = torch.ones_like(x)
        x[1, 0] = float("inf")
        mask = torch.stack((PATTERNS[6], PATTERNS[0]))[:, None, :]

        output = module(x, y, mask, torch.ones(1, 2))

        assert_close(output[0], (x + y)[0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output[0]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_forward_backward_is_finite(self):
        module = MaskConditionedSequenceAFF(8).cuda()
        x = torch.randn(4, 2, 8, device="cuda", requires_grad=True)
        y = torch.randn(4, 2, 8, device="cuda", requires_grad=True)
        mask = PATTERNS[:4, None, :].repeat(1, 2, 1).cuda()
        umask = torch.ones(2, 4, device="cuda")

        output = module(x, y, mask, umask)
        output.square().mean().backward()

        self.assertEqual(output.device.type, "cuda")
        gradients = [x.grad, y.grad] + [
            parameter.grad for parameter in module.parameters()
        ]
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_parameter_count_matches_two_independent_context_mlps(self):
        channels = 8
        bottleneck = 2
        module = MaskConditionedSequenceAFF(channels, reduction=4)
        expected = 2 * (bottleneck * (2 * channels + 9) + channels)

        self.assertEqual(
            sum(parameter.numel() for parameter in module.parameters()), expected
        )
        self.assertIsNot(module.local_context[0], module.global_context[0])


if __name__ == "__main__":
    unittest.main()
