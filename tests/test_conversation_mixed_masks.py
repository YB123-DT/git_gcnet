import unittest
import torch

from gcnet_missing_m3 import train_gcnet as trainer


class ConversationMixedMaskTests(unittest.TestCase):
    def config(self):
        return trainer.TrainConfig(dataset="CMUMOSI", train_rate_mode="conversation-mixed",
                                   training_objective="emotion-only", backbone_type="osram",
                                   osram_bidirectional=False)

    def helper(self):
        self.assertTrue(hasattr(trainer, "_conversation_mixed_mask_tensors"),
                        "conversation-level mixed masks must be implemented")
        return trainer._conversation_mixed_mask_tensors

    def test_masks_are_deterministic_order_invariant_and_keep_random_schedule(self):
        helper = self.helper()
        config = self.config()
        schedule = trainer._build_schedule(config, "train", .3)
        ids = ["c" + str(i) for i in range(200)]
        umask = torch.tensor([[1, 1, 1, 0]] * len(ids))
        host, guest, audit = helper(config, schedule, ids, umask, 3)
        reverse = helper(config, schedule, ids[::-1], umask, 3)
        self.assertTrue(torch.equal(host, reverse[0].flip(1)))
        self.assertEqual(audit["assignments"], reverse[2]["assignments"][::-1])
        self.assertTrue(torch.equal(host, helper(config, schedule, ids, umask, 3)[0]))
        self.assertFalse(torch.equal(host, helper(config, schedule, ids, umask, 4)[0]))
        self.assertTrue(torch.all(host[3] == 0))
        for i, row in enumerate(audit["assignments"]):
            if row["regime"] == "persistent":
                self.assertTrue(torch.equal(host[:, i], guest[:, i]))
                self.assertTrue(torch.equal(host[0, i], host[2, i]))
                self.assertIn(int(host[0, i].sum()), (1, 2))
            else:
                for side, tensor in (("host", host), ("guest", guest)):
                    expected = schedule.generate(ids[i], 4, side, epoch=3, valid_length=3)
                    self.assertTrue(torch.equal(tensor[:, i], torch.as_tensor(expected.availability)))

    def test_bernoulli_half_and_uniform_six(self):
        helper = self.helper()
        config = self.config()
        ids = [str(i) for i in range(12000)]
        _, _, audit = helper(config, trainer._build_schedule(config, "train", .7),
                             ids, torch.ones(len(ids), 1), 1)
        counts = {name: 0 for name in ("A", "T", "V", "AT", "AV", "TV")}
        for row in audit["assignments"]:
            if row["regime"] == "persistent":
                counts[row["pattern"]] += 1
        self.assertLess(abs(sum(counts.values()) / len(ids) - .5), .025)
        for count in counts.values():
            self.assertLess(abs(count / sum(counts.values()) - 1 / 6), .025)

    def test_protocol_rates(self):
        self.assertEqual(trainer._protocol_rates(self.config()), trainer.MISSING_RATES)
        self.assertEqual(trainer.TrainConfig().train_rate_mode, "cyclic")
        parsed = trainer.build_parser().parse_args([
            "--train-rate-mode", "conversation-mixed", "--audio-feature", "a",
            "--text-feature", "t", "--video-feature", "v", "--output-dir", "unused",
        ])
        self.assertEqual(parsed.train_rate_mode, "conversation-mixed")

    def test_single_forward_step_and_regime_audit(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(3, 1))
                self.calls = 0
            def forward(self, inputs, availability, qmask, umask, lengths, predict_missing):
                self.calls += 1
                output = inputs[0] @ self.weight
                return output, output, None, None
        config = self.config()
        size, length = 32, 3
        batch = [torch.randn(length, size, 1) for _ in range(6)]
        batch += [torch.zeros(size, length), torch.ones(size, length),
                  torch.ones(size, length), [str(i) for i in range(size)]]
        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        result = trainer.train_epoch(model, [batch, batch], optimizer, config,
                                     trainer._schedules(config, "train"), 0, (1, 1, 1),
                                     torch.device("cpu"))
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["source_conversation_count"], 64)
        self.assertEqual(result["masked_view_count"], 64)
        self.assertEqual(result["jepa_loss"], 0.0)
        counts = result["conversation_mixed_regime_counts"]
        self.assertEqual(sum(counts.values()), 64)
        self.assertEqual(sum(result["rate_conversation_counts"].values()), counts["random"])
        self.assertEqual(sum(result["conversation_mixed_pattern_counts"].values()), counts["persistent"])
        self.assertEqual(result["rate_statistics_scope"], "random-regime-only")
        self.assertEqual(len(result["conversation_mixed_assignment_hash"]), 64)
        for regime, count in counts.items():
            self.assertEqual(result["conversation_mixed_valid_utterance_counts"][regime], count * length)
            missing = result["conversation_mixed_missing_modality_counts"][regime]
            self.assertAlmostEqual(result["conversation_mixed_realized_missing_fraction"][regime],
                                   missing / (3 * length * count))

    def test_mixed_mode_rejects_unrelated_objectives(self):
        with self.assertRaisesRegex(ValueError, "conversation-mixed"):
            trainer.TrainConfig(train_rate_mode="conversation-mixed")


if __name__ == "__main__":
    unittest.main()
