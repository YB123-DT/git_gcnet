"""Finite MELD Block2 registry, protocol and real adapter contracts."""
import copy
import importlib
import importlib.util
import json
import unittest


MODULE = "experiments.osram_meld_block2_auto_20261014.methods"
BLOCKS = {
    "Flat": "none",
    "B1": "graph_matching_base_gap_pairs",
    "B2": "dual_attention_symbolic_relations",
    "B3": "mbt_role_bottleneck",
    "B4": "conditional_new_07_neural_production",
    "B5": "tokenlearner_fuser_v11",
    "B6": "learned_robust_pca_evidence_decomposition",
}


def baseline():
    return dict(dataset="MELD", seed=66, epochs=50, batch_size=32,
                learning_rate=0.001, weight_decay=0.00001, dropout=0.5,
                backbone_type="osram", training_objective="emotion-only", jepa_weight=0.,
                missing_mask_protocol="gcnet-random-mask-v1", train_rate_mode="cyclic",
                train_missing_rates=[i / 10 for i in range(8)],
                checkpoint_selection="test-oracle-per-rate", checkpoint_selection_metric="accuracy",
                osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
                osram_write_step=0.6, osram_output_dim=1600, osram_adapter_hidden_dim=0,
                osram_bidirectional=False, osram_forward_slot_reuse=False,
                osram_readout_fusion="flat", osram_disable_local_skip=False,
                osram_meaningful_block="none", feature_roots=["real/A", "real/T", "real/V"])


class RegistryTests(unittest.TestCase):
    def test_registry_module_exists(self):
        try:
            spec = importlib.util.find_spec(MODULE)
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, "Missing finite MELD Block2 registry")

    def test_exact_seven_existing_mechanisms_and_local_update_metadata(self):
        methods = importlib.import_module(MODULE).METHODS
        self.assertEqual({key: value["block"] for key, value in methods.items()}, BLOCKS)
        self.assertEqual([key for key, value in methods.items() if value["updates_local"]],
                         ["B2", "B3", "B5"])
        for record in methods.values():
            self.assertTrue(record["name"])
            self.assertTrue(record["sources"])
            self.assertTrue(record["initnotes"])

    def test_config_preserves_actual_baseline_and_copies_nested_values(self):
        build = importlib.import_module(MODULE).build_config
        original = baseline()
        saved = copy.deepcopy(original)
        config = build(original, "B3", "/tmp/block2-test")
        self.assertEqual(original, saved)
        for key, value in saved.items():
            if key != "osram_meaningful_block":
                self.assertEqual(config[key], value, key)
        self.assertEqual(config["osram_meaningful_block"], "mbt_role_bottleneck")
        self.assertTrue(config["preserve_selection_artifacts"])
        self.assertIs(type(config), dict)
        json.dumps(config)
        config["feature_roots"][0] = "changed"
        self.assertEqual(original, saved)

    def test_hpo_changes_only_three_training_knobs_and_budget(self):
        build = importlib.import_module(MODULE).build_config
        original = baseline()
        config = build(original, "Flat", "/tmp/block2-test",
                       overrides={"learning_rate": 0.002, "weight_decay": 0., "dropout": 0.2},
                       epochs=50)
        self.assertEqual(config["epochs"], 50)
        self.assertEqual(config["learning_rate"], 0.002)
        self.assertEqual(config["weight_decay"], 0.)
        self.assertEqual(config["dropout"], 0.2)
        self.assertEqual(config["osram_output_dim"], 1600)
        for override in ({"osram_num_heads": 4}, {"seed": 67}, {"epochs": 20},
                         {"dropout": 1.}, {"learning_rate": -1}, {"weight_decay": float("nan")}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                build(original, "Flat", "/tmp/block2-test", overrides=override)
        for epochs in (0, 20, 100, True):
            with self.assertRaises(ValueError):
                build(original, "Flat", "/tmp/block2-test", epochs=epochs)

    def test_protocol_mismatch_or_nonregistry_method_is_rejected(self):
        build = importlib.import_module(MODULE).build_config
        for key, value in (("dataset", "MOSI"), ("seed", 67),
                           ("missing_mask_protocol", "conversation-mask-v2"),
                           ("checkpoint_selection", "validation"),
                           ("osram_bidirectional", True), ("osram_write_step", 1.),
                           ("osram_disable_local_skip", True), ("osram_output_dim", 256)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                build(dict(baseline(), **{key: value}), "Flat", "/tmp/block2-test")
        with self.assertRaises(ValueError):
            build(baseline(), "neural_production_local", "/tmp/block2-test")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch CPU environment required")
class MechanismTests(unittest.TestCase):
    def test_real_wrapper_masks_identity_and_upstream_learning(self):
        import torch
        from gcnet_missing_m3.meaningful_input import INPUT_METHODS, MeaningfulInputAdapter

        torch.set_num_threads(1)
        methods = importlib.import_module(MODULE).METHODS
        for name, record in methods.items():
            if name == "Flat":
                continue
            with self.subTest(method=name):
                torch.manual_seed(66)
                self.assertIn(record["block"], INPUT_METHODS)
                adapter = MeaningfulInputAdapter(256, 1024, 1600, record["block"], 8, 64)
                availability = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                             [1, 1, 0], [1, 0, 1], [0, 1, 1]])
                availability = availability[None].expand(4, -1, -1).float()
                umask = torch.tensor([[1, 1, 1, 0]] * 7).float()
                valid = umask.T.bool()
                active_gap = valid[..., None] & ~availability.bool()
                local = torch.randn(4, 7, 256)
                base = torch.randn(4, 7, 1024)
                gap = torch.randn(4, 7, 3, 1024)
                base[..., 512:] = 0.
                gap[..., 512:] = 0.
                base[0] = 0.
                gap[0] = 0.
                clean = (torch.where(valid[..., None], local, 0.),
                         torch.where(valid[..., None], base, 0.),
                         torch.where(active_gap[..., None], gap, 0.))
                poison = (local.masked_fill(~valid[..., None], float("nan")),
                          base.masked_fill(~valid[..., None], float("inf")),
                          gap.masked_fill(~active_gap[..., None], float("nan")))
                rng = torch.get_rng_state().clone()
                initial = adapter(*poison, availability, umask)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for left, right in zip(initial, clean):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertEqual(adapter.last_diagnostics["history_count"], 14)
                probes = tuple(torch.randn_like(value) for value in clean)
                optimizer = torch.optim.SGD(adapter.parameters(), lr=1e-3)
                upstream_live = False
                for step in range(3):
                    optimizer.zero_grad()
                    result = adapter(*poison, availability, umask)
                    sum((value * probe).sum() / value.numel()
                        for value, probe in zip(result, probes)).backward()
                    for parameter in adapter.parameters():
                        self.assertTrue(parameter.grad is None or torch.isfinite(parameter.grad).all())
                    if step:
                        upstream_live |= any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                             for key, parameter in adapter.named_parameters()
                                             if not any(part in key for part in
                                                        ("decoder", "output", "bridge", "gamma")))
                    optimizer.step()
                self.assertTrue(upstream_live, "Core must learn beyond its identity bridge")
                learned = adapter(*poison, availability, umask)
                for a, b in zip(learned, adapter(*clean, availability, umask)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(a).all())
                    self.assertEqual(a[~valid].count_nonzero(), 0)
                    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
                torch.testing.assert_close(learned[0][0], clean[0][0], rtol=0, atol=0)
                self.assertEqual(learned[1][0].count_nonzero(), 0)
                self.assertEqual(learned[2][0].count_nonzero(), 0)
                self.assertEqual(learned[2][~active_gap].count_nonzero(), 0)
                self.assertEqual(learned[1][..., 512:].count_nonzero(), 0)
                self.assertEqual(learned[2][..., 512:].count_nonzero(), 0)
                local_changed = not torch.equal(learned[0][valid], clean[0][valid])
                self.assertEqual(local_changed, record["updates_local"])
                self.assertFalse(torch.equal(learned[1], clean[1]) and torch.equal(learned[2], clean[2]))
                # Even a trained adapter must bypass an ablated empty history.
                no_history = adapter(local, torch.zeros_like(base), torch.zeros_like(gap), availability, umask)
                torch.testing.assert_close(no_history[0], clean[0], rtol=0, atol=0)
                self.assertEqual(no_history[1].count_nonzero(), 0)
                self.assertEqual(no_history[2].count_nonzero(), 0)
                invalid_availability = availability.clone()
                invalid_availability[1, 0] = 0.
                with self.assertRaises(ValueError):
                    adapter(*clean, invalid_availability, umask)


if __name__ == "__main__":
    unittest.main()
