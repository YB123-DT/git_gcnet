"""Locked contracts for the six-run MOSEI Nested capacity screen."""
import unittest


EXPECTED_METHODS = {
    "mosei_nested_dim32": {"dim": 32},
    "mosei_nested_depth2": {"depth": 2},
    "mosei_nested_dim32_depth2": {"dim": 32, "depth": 2},
    "mosei_nested_mlp128": {"mlp_hidden": 128},
    "mosei_nested_groups4": {"groups": 4},
    "mosei_nested_dim32_depth2_mlp64": {
        "dim": 32,
        "depth": 2,
        "mlp_hidden": 64,
    },
}


class MoseiNestedTuningRegistryTests(unittest.TestCase):
    def test_locked_methods_are_registered_exactly(self):
        from gcnet_missing_m3.meaningful_new40_registry import NESTED_SWEEP

        for method, expected in EXPECTED_METHODS.items():
            with self.subTest(method=method):
                self.assertEqual(NESTED_SWEEP[method], expected)

    def test_locked_methods_keep_residual_zero_decoder_defaults(self):
        from gcnet_missing_m3.meaningful_input_new40 import build_new40

        for method in EXPECTED_METHODS:
            with self.subTest(method=method):
                module = build_new40(method, 256, 8, 64)
                self.assertTrue(module.residual)
                self.assertEqual(module.local_decoder.weight.count_nonzero(), 0)
                self.assertTrue(all(layer.weight.count_nonzero() == 0
                                    for layer in module.memory_decoders))


if __name__ == "__main__":
    unittest.main()
