import importlib.util
import unittest

import torch


class CommonTests(unittest.TestCase):
    def test_helpers_exist_and_preserve_true_head_membership(self):
        self.assertIsNotNone(importlib.util.find_spec('gcnet_missing_m3.meaningful_blocks_common'))
        from gcnet_missing_m3.meaningful_blocks_common import HeadTokenizer, active_groups, typed_means
        torch.manual_seed(1)
        model = HeadTokenizer(12, 8, 64, normalize=True)
        local = torch.randn(3, 12)
        evidence = torch.randn(3, 4, 512)
        active = torch.tensor([[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]]).bool()
        expected, mask = model(local, evidence, active)
        actual, _ = model(local, evidence.masked_fill(~active[..., None], float('nan')), active)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(mask.sum(1).tolist(), [9, 17, 25])
        self.assertEqual(tuple(expected.shape), (3, 33, 128))
        self.assertEqual(model.role_ids.tolist(), [0] + [1]*8 + [2]*8 + [3]*8 + [4]*8)
        self.assertEqual(model.head_ids.tolist(), [-1] + list(range(8))*4)
        means = typed_means(expected, mask, 8)
        self.assertEqual(means[:, 1:][~active].count_nonzero(), 0)
        for rows, cols, packed in active_groups(expected, mask):
            self.assertTrue(torch.equal(packed, expected[rows][:, cols]))
            self.assertTrue(mask[rows][:, cols].all())


if __name__ == '__main__':
    unittest.main()
