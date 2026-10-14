import unittest

import torch

from experiments.osram_meld_no_skip_evidence_ablation_20261014.evaluate import (
    replay_no_skip_hidden,
)


class NoSkipEvidenceAblationTests(unittest.TestCase):
    def test_masks_only_requested_evidence_and_never_adds_skip(self):
        local_dim, context_dim, output_dim = 2, 3, 5
        adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(local_dim + 4 * context_dim),
            torch.nn.Linear(local_dim + 4 * context_dim, output_dim),
        )
        norm = torch.nn.LayerNorm(output_dim)
        value = torch.arange(2 * (local_dim + 4 * context_dim), dtype=torch.float32).reshape(1, 2, -1)
        umask = torch.ones(2, 1)
        captured = []
        handle = adapter.register_forward_pre_hook(lambda _m, args: captured.append(args[0].clone()))
        outputs = {}
        for name in ("LBG", "LG", "LB", "L"):
            outputs[name] = replay_no_skip_hidden(
                adapter, norm, value, umask, local_dim=local_dim,
                context_dim=context_dim, condition=name,
            )
        handle.remove()
        torch.testing.assert_close(captured[0], value)
        self.assertEqual(captured[1][..., local_dim:local_dim + context_dim].count_nonzero(), 0)
        self.assertEqual(captured[2][..., local_dim + context_dim:].count_nonzero(), 0)
        self.assertEqual(captured[3][..., local_dim:].count_nonzero(), 0)
        for seen in captured:
            torch.testing.assert_close(seen[..., :local_dim], value[..., :local_dim])
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs.values()))


if __name__ == "__main__":
    unittest.main()
