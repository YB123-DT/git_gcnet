import unittest
from types import SimpleNamespace

import torch

from gcnet_missing_m3.r03_mesa import MesaStorage


def memory():
    osram = SimpleNamespace(num_heads=2, key_dim=3, value_dim=4, latent_dim=16,
                            alpha_logits=torch.tensor([0.7, 1.2]),
                            beta_logits=torch.zeros(2, 3), write_step=0.6)
    return MesaStorage(SimpleNamespace(), osram).double()


class MesaTests(unittest.TestCase):
    def test_weighted_history_closed_form_and_joint_slots(self):
        torch.manual_seed(11)
        model = memory()
        reference = torch.zeros(1, dtype=torch.float64)
        state = model.initial_state(reference, 1)
        active = torch.tensor([True])
        expected_g, expected_h = state['G'].clone(), state['H'].clone()
        gamma = model.alpha_logits.sigmoid()[None, :, None, None]
        beta = model.beta_logits.sigmoid() * model.write_step
        for _ in range(3):
            query = torch.randn(1, 4, 2, 3, dtype=torch.float64)
            reads, state = model.read(state, query, active)
            expected_g, expected_h = expected_g * gamma, expected_h * gamma
            solved = torch.linalg.solve(expected_h + torch.diag_embed(model.ridge)[None],
                                        query.permute(0, 2, 3, 1))
            expected = (expected_g @ solved).permute(0, 3, 1, 2)
            torch.testing.assert_close(reads, expected)
            k = torch.randn(1, 2, 3, 3, dtype=torch.float64)
            v = torch.randn(1, 2, 4, 3, dtype=torch.float64)
            observed = torch.tensor([[True, False, True]])
            weight = beta[None] * observed[:, None]
            expected_g += (v * weight[:, :, None]) @ k.transpose(-1, -2)
            expected_h += (k * weight[:, :, None]) @ k.transpose(-1, -2)
            new = model.write(state, k, v, observed, active, query)
            torch.testing.assert_close(new['G'], expected_g)
            torch.testing.assert_close(new['H'], expected_h)
            # Move beta with each slot: an arbitrary A/T/V permutation must
            # not affect the joint weighted association sum.
            permutation = torch.tensor([2, 1, 0])
            permuted = model.write(state, k[..., permutation], v[..., permutation],
                                   observed[:, permutation], active, query)
            torch.testing.assert_close(new['G'], permuted['G'])
            torch.testing.assert_close(new['H'], permuted['H'])
            state = new

    def test_scan_mask_padding_causality_reset_and_parameter_updates(self):
        torch.manual_seed(21)
        model = memory()
        names = ('audio', 'text', 'visual')
        keys = {m: torch.randn(5, 2, 2, 3, dtype=torch.float64, requires_grad=True) for m in names}
        values = {m: torch.randn(5, 2, 2, 4, dtype=torch.float64, requires_grad=True) for m in names}
        query = torch.randn(5, 2, 4, 2, 3, dtype=torch.float64, requires_grad=True)
        available = torch.ones(5, 2, 3, dtype=torch.bool)
        available[::2, :, 0] = False
        valid = torch.ones(5, 2, dtype=torch.bool)
        valid[2, 0] = False
        base, gap, _ = model.scan(keys, values, query, available, valid)
        self.assertEqual(base.shape, (5, 2, 8))
        self.assertEqual(gap.shape, (5, 2, 3, 8))
        self.assertEqual(base[0].count_nonzero().item(), 0)
        self.assertEqual(base[2, 0].count_nonzero().item(), 0)
        dirty = [{m: x.detach().clone() for m, x in group.items()} for group in (keys, values)]
        for group in dirty:
            for i, m in enumerate(names):
                group[m][~(available[:, :, i] & valid)] = float('nan')
        altered = model.scan(*dirty, query, available, valid)
        torch.testing.assert_close(base, altered[0])
        torch.testing.assert_close(gap, altered[1])
        for group in dirty:
            for x in group.values():
                x[3:] = 1000
        altered = model.scan(*dirty, query, available, valid)
        torch.testing.assert_close(base[:4], altered[0][:4])
        select = torch.tensor([0, 1, 3, 4])
        compact = [{m: x[select, :1] for m, x in group.items()} for group in (keys, values)]
        compact_out = model.scan(*compact, query[select, :1], available[select, :1], valid[select, :1])
        torch.testing.assert_close(base[select, :1], compact_out[0])
        torch.testing.assert_close(base, model.scan(keys, values, query, available, valid)[0])
        (base.square().mean() + gap.square().mean()).backward()
        for group in (keys, values):
            for i, m in enumerate(names):
                self.assertTrue(torch.isfinite(group[m].grad).all())
                self.assertEqual(group[m].grad[~(available[:, :, i] & valid)].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(query.grad).all())
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        before = [p.detach().clone() for p in model.parameters()]
        torch.optim.SGD(model.parameters(), lr=.01).step()
        self.assertTrue(all(not torch.equal(old, p) for old, p in zip(before, model.parameters())))

    def test_padding_freezes_both_statistics_and_zero_observation_only_forgets(self):
        model = memory()
        state = {'G': torch.randn(2, 2, 4, 3, dtype=torch.float64),
                 'H': torch.eye(3, dtype=torch.float64).expand(2, 2, 3, 3).clone()}
        original = {name: x.clone() for name, x in state.items()}
        active = torch.tensor([True, False])
        query = torch.randn(2, 4, 2, 3, dtype=torch.float64)
        _, decayed = model.read(state, query, active)
        result = model.write(decayed, torch.full((2, 2, 3, 3), float('nan'), dtype=torch.float64),
                             torch.full((2, 2, 4, 3), float('nan'), dtype=torch.float64),
                             torch.zeros(2, 3, dtype=torch.bool), active, query)
        for name in original:
            torch.testing.assert_close(result[name][1], original[name][1])
            torch.testing.assert_close(result[name][0], original[name][0] * model.alpha_logits.sigmoid()[:, None, None])


if __name__ == '__main__':
    unittest.main()
