import unittest

import torch

from graph import batch_graphify, edge_perms


class GraphDeterminismTests(unittest.TestCase):
    def test_edge_permutations_have_canonical_order(self):
        edges = edge_perms(4, 1, 1)

        self.assertEqual(edges, sorted(edges))

    def test_relation_ids_are_semantically_stable(self):
        features = torch.zeros(2, 1, 1, 3)
        speakers = torch.tensor([[0.0, 1.0]])

        *_, temporal = batch_graphify(
            features, speakers, [2], 2, 1, 1, "temporal", True
        )
        *_, speaker = batch_graphify(
            features, speakers, [2], 2, 1, 1, "speaker", True
        )

        self.assertEqual(temporal, {"past": 0, "now": 1, "future": 2})
        self.assertEqual(speaker, {"00": 0, "01": 1, "10": 2, "11": 3})


if __name__ == "__main__":
    unittest.main()
