import unittest

import torch

from gcnet_missing_m3.fwc_sampling import make_trajectories, pair_hash, sample_pairs, validate_pairs


class FWCSamplingTest(unittest.TestCase):
    def setUp(self):
        self.a = torch.tensor([[[1, 1, 1], [1, 0, 1]], [[1, 0, 1], [0, 1, 1]],
                               [[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 0]]]).float()
        self.u = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        self.ids = ['first', 'second']

    def test_pairs_valid_reproducible_and_rng_independent(self):
        state = torch.random.get_rng_state().clone()
        p = sample_pairs(self.a, self.u, self.ids, seed=7, horizon=2, pairs_per_conversation=100)
        self.assertTrue(p)
        validate_pairs(p, self.a, self.u, self.ids, horizon=2)
        self.assertEqual(p, sample_pairs(self.a, self.u, self.ids, seed=7, horizon=2, pairs_per_conversation=100))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(pair_hash(p), pair_hash(p))
        self.assertTrue(any(x.source_modality == x.missing_modality for x in p))
        self.assertTrue(any(x.source_modality != x.missing_modality for x in p))
        self.assertEqual(len(p), len(set(p)))

    def test_no_pairs(self):
        a = torch.zeros_like(self.a); a[:, :, 0] = 1
        self.assertEqual(sample_pairs(a, self.u, self.ids, seed=2), [])
        self.assertEqual(sample_pairs(self.a[:1], self.u[:, :1], self.ids, seed=2), [])

    def test_random_identity_and_safe_padding(self):
        a = self.a.clone(); a[-1, 1] = float('nan')
        state = torch.random.get_rng_state().clone()
        out, regimes, d = make_trajectories(a, self.u, self.ids, seed=4, ratios=(1, 0, 0))
        self.assertTrue(torch.equal(out, self.a))
        self.assertEqual(regimes, ['random', 'random'])
        self.assertEqual(d['valid_utterances'], 7)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_persistent_and_transition(self):
        p, regimes, _ = make_trajectories(self.a, self.u, self.ids, seed=9, ratios=(0, 0, 1))
        for b in range(2):
            valid = p[:int(self.u[b].sum()), b]
            self.assertTrue((valid == valid[0]).all())
            self.assertIn(int(valid[0].sum()), [1, 2])
        pairs = sample_pairs(p, self.u, self.ids, seed=1, pairs_per_conversation=100, regimes=regimes)
        self.assertTrue(all(x.source_modality != x.missing_modality for x in pairs))
        t, regimes, d = make_trajectories(self.a, self.u, self.ids, seed=9, ratios=(0, 1, 0))
        self.assertTrue((t[0] == 1).all())
        self.assertTrue(all(1 <= c < int(self.u[b].sum()) for b, c in enumerate(d['transition_points'])))
        validate_pairs(sample_pairs(t, self.u, self.ids, seed=1, regimes=regimes), t, self.u, self.ids)

    def test_transition_short_fallback_and_invalid_inputs(self):
        _, r, d = make_trajectories(self.a[:1], self.u[:, :1], self.ids, seed=3, ratios=(0, 1, 0))
        self.assertEqual(r, ['random', 'random'])
        self.assertEqual(d['transition_short_fallbacks'], 2)
        with self.assertRaises(ValueError):
            make_trajectories(self.a, self.u, self.ids, seed=3, ratios=(0, 0, 0))
        with self.assertRaises(ValueError):
            sample_pairs(self.a, self.u, self.ids, seed=3, horizon=0)
        with self.assertRaises(ValueError):
            sample_pairs(self.a, torch.tensor([[1, 0, 1, 1], [1, 1, 1, 0]]), self.ids, seed=3)

    def test_source_balanced(self):
        pairs = sample_pairs(self.a, self.u, self.ids, seed=1, strategy='source-balanced', pairs_per_conversation=6)
        validate_pairs(pairs, self.a, self.u, self.ids)
        first = [x for x in pairs if x.batch == 0]
        counts = [sum(x.source_modality == m for x in first) for m in range(3)]
        self.assertLessEqual(max(counts) - min(counts), 1)


if __name__ == '__main__':
    unittest.main()
