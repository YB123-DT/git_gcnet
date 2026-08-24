import unittest

import torch

from missing_patterns import encode_missing_patterns, flatten_valid_node_masks


class PatternEncodingTests(unittest.TestCase):
    def test_encodes_all_seven_patterns(self):
        masks = torch.tensor(
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
        expected = torch.cat((torch.eye(6), torch.zeros(1, 6)), dim=0)

        encoded, complete = encode_missing_patterns(masks)

        torch.testing.assert_close(encoded, expected)
        torch.testing.assert_close(
            complete, torch.tensor([False] * 6 + [True])
        )

    def test_rejects_invalid_masks(self):
        invalid = (
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[1, 2, 0]]),
            torch.tensor([[1, 0]]),
        )
        for mask in invalid:
            with self.subTest(mask=mask.tolist()):
                with self.assertRaises(ValueError):
                    encode_missing_patterns(mask)

    def test_flattens_conversations_in_batch_graphify_order(self):
        mask = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0]],
                [[1, 1, 0], [0, 0, 1]],
                [[1, 1, 1], [1, 0, 1]],
            ]
        )
        expected = torch.tensor(
            [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0]]
        )

        actual = flatten_valid_node_masks(mask, [3, 1])

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
