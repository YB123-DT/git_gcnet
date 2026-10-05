"""Bounded CPU source algebra, mask, gradients and parameter-update checks."""
import copy
import unittest

import torch
from torch import nn

from gcnet_missing_m3.core20_representation import FactorCL, FishrPenalty, MODALITIES, VQVAEValues


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261005)
        torch.set_num_threads(1)
        self.valid = torch.tensor([[1, 1], [1, 0], [1, 0]], dtype=torch.bool)
        self.availability = torch.tensor([[[1, 1, 1], [0, 1, 1]],
                                          [[1, 0, 1], [1, 1, 1]],
                                          [[0, 0, 0], [1, 1, 1]]])

    def test_vq_nearest_code_ste_reconstruction_and_update(self):
        raw_dims = {'audio': 5, 'text': 7, 'visual': 6}
        model = VQVAEValues(value_dim=4, num_embeddings=8, num_heads=2,
                            reconstruction_dims=raw_dims)
        values = {m: torch.randn(3, 2, 2, 4, requires_grad=True) for m in MODALITIES}
        poisoned = {m: torch.where((self.valid & self.availability[..., i].bool())[..., None, None],
                                   values[m], float('nan')) for i, m in enumerate(MODALITIES)}
        raw = {m: torch.randn(3, 2, raw_dims[m], requires_grad=True) for m in MODALITIES}
        raw_poisoned = {m: torch.where((self.valid & self.availability[..., i].bool())[..., None],
                                       raw[m], float('nan')) for i, m in enumerate(MODALITIES)}
        output, loss = model.transform_values(poisoned, self.availability, self.valid, raw_poisoned)
        reconstruction_terms = []
        for i, m in enumerate(MODALITIES):
            mask = self.valid & self.availability[..., i].bool()
            x = values[m][mask].reshape(-1, 4)
            codebook = model.codebooks[m].weight
            codes = codebook[torch.cdist(x, codebook).argmin(-1)]
            reconstruction_terms.append(
                (model.decoders[m](codes.reshape(-1, 8)) - raw[m][mask]).square().mean()
                + (codes - x.detach()).square().mean()
                + .25 * (x - codes.detach()).square().mean())
        torch.testing.assert_close(loss, torch.stack(reconstruction_terms).mean())
        with self.assertRaisesRegex(ValueError, 'reconstruction_targets'):
            model.transform_values(poisoned, self.availability, self.valid)
        for i, m in enumerate(MODALITIES):
            mask = self.valid & self.availability[..., i].bool()
            x, codebook = values[m][mask].flatten(0, 1), model.codebooks[m].weight
            closest = torch.cdist(x, codebook).argmin(-1)
            torch.testing.assert_close(output[m][mask].flatten(0, 1), codebook[closest])
            self.assertTrue((output[m][~mask] == 0).all())
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        (loss + sum(o.square().mean() for o in output.values())).backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        for i, m in enumerate(MODALITIES):
            mask = self.valid & self.availability[..., i].bool()
            self.assertTrue((values[m].grad[~mask] == 0).all())
            self.assertIsNone(raw[m].grad)
        optimizer.step()
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
                            if n.startswith('codebooks.')))
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
                            if n.startswith('decoders.')))
        empty, zero = model.transform_values(poisoned, torch.zeros_like(self.availability), self.valid, raw_poisoned)
        self.assertEqual(zero.item(), 0)
        self.assertTrue(all((x == 0).all() for x in empty.values()))

    def test_vq_constant_encoder_cannot_zero_raw_reconstruction(self):
        model = VQVAEValues(value_dim=4, num_embeddings=8, num_heads=2,
                            reconstruction_dims={m: 3 for m in MODALITIES})
        for p in model.parameters():
            nn.init.zeros_(p)
        values = {m: torch.zeros(3, 2, 2, 4) for m in MODALITIES}
        raw = {m: torch.ones(3, 2, 3) for m in MODALITIES}
        _, loss = model.transform_values(values, self.availability, self.valid, raw)
        torch.testing.assert_close(loss, torch.tensor(1.))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.decoders.parameters()))

    def test_factorcl_mask_label_free_forward_full_objective_and_update(self):
        model = FactorCL(latent_dim=8, representation_dim=4, critic_hidden=8)
        latents = {m: torch.randn(3, 2, 8, requires_grad=True) for m in MODALITIES}
        poisoned = {m: torch.where((self.valid & self.availability[..., i].bool())[..., None],
                                   latents[m], float('nan')) for i, m in enumerate(MODALITIES)}
        output, metadata = model(poisoned, self.availability, self.valid.T)
        self.assertEqual(set(metadata['representations']['audio']), set(model.branches))
        for i, m in enumerate(MODALITIES):
            self.assertTrue(torch.isfinite(output[m]).all())
            self.assertTrue((output[m][~(self.valid & self.availability[..., i].bool())] == 0).all())
        labels = torch.tensor([[.2, -.5, .7], [-.9, float('nan'), float('nan')]])
        loss = model.training_loss(metadata, latents, labels, self.valid.T)
        self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        (loss + sum(x.square().mean() for x in output.values())).backward()
        for n, p in model.named_parameters():
            self.assertIsNotNone(p.grad, n)
            self.assertTrue(torch.isfinite(p.grad).all(), n)
        optimizer.step()
        for prefix in ('heads.audio.', 'critics.audio_text.unique_upper.', 'task_critics.audio.', 'output.audio.'):
            self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
                                if n.startswith(prefix)), prefix)
        # Critic fitting must never update representations or source inputs.
        model.zero_grad(set_to_none=True)
        detached = {m: x.detach().clone().requires_grad_() for m, x in latents.items()}
        model.critic_learning_loss(detached, labels, self.valid.T).backward()
        self.assertTrue(all(x.grad is None for x in detached.values()))
        self.assertTrue(all(p.grad is None for p in model.heads.parameters()))

    def test_nce_club_matches_author_algebra_and_gradient_partition(self):
        model = FactorCL(latent_dim=8, representation_dim=4, critic_hidden=8)
        critic = model.critics['audio_text']['unique_upper']
        x, y = torch.randn(4, 4, requires_grad=True), torch.randn(4, 4, requires_grad=True)
        scores = critic.scores(x, y)
        expected = scores.diag().mean() - scores.mean()
        torch.testing.assert_close(critic.upper_loss(x, y), expected)
        critic.upper_loss(x, y).backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(all(p.grad is None for p in critic.parameters()))

    def test_factorcl_sum_has_exact_six_author_signed_terms_per_pair(self):
        model = FactorCL(latent_dim=8, representation_dim=4, critic_hidden=8)
        complete = {m: torch.randn(3, 2, 8) for m in MODALITIES}
        labels = torch.randn(2, 3)
        reps = model._represent(complete, {m: self.valid for m in MODALITIES})
        y = labels.T[self.valid].unsqueeze(-1)
        expected = y.sum() * 0
        for a, b in model.pairs:
            critics = model.critics[a + '_' + b]
            for m in (a, b):
                expected = expected + model.task_critics[m].lower_loss(reps[m]['task_lower'][self.valid], y)
            for branch in ('shared_lower', 'unique_upper', 'unique_cond_lower', 'shared_cond_upper'):
                x, z = reps[a][branch][self.valid], reps[b][branch][self.valid]
                if 'cond' in branch:
                    x, z = torch.cat((x, y), -1), torch.cat((z, y), -1)
                loss_fn = critics[branch].upper_loss if 'upper' in branch else critics[branch].lower_loss
                expected = expected + loss_fn(x, z)
        actual = model.training_loss({}, complete, labels, self.valid.T, include_critic_fit=False)
        torch.testing.assert_close(actual, expected)

    def test_fishr_actual_sample_variance_ema_algebra_and_state(self):
        head = nn.Linear(1, 1)
        with torch.no_grad():
            head.weight.fill_(1)
            head.bias.zero_()
        x = torch.tensor([[1.], [2.], [3.], [5.]], requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1])
        losses = head(x).squeeze(-1).square()
        fishr = FishrPenalty(ema=.5)
        penalty = fishr.penalty(losses, head.parameters(), groups)
        # MSE gradients: weight=2*x^2; bias=2*x. Population group
        # variances [[9,1],[256,4]], distance to mean across both params.
        variances = torch.tensor([[9., 1.], [256., 4.]])
        expected = (variances - variances.mean(0)).square().mean()
        torch.testing.assert_close(penalty, expected)
        torch.testing.assert_close(fishr.ema_variances[:2], variances * .5)
        # The source correction is always /(1-ema), so repeat gives 1.5*v.
        second = fishr.penalty(losses, head.parameters(), groups)
        torch.testing.assert_close(second, expected * 1.5 ** 2)
        second.backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all())
        before = head.weight.detach().clone()
        torch.optim.SGD(head.parameters(), lr=1e-7).step()
        self.assertFalse(torch.equal(head.weight, before))
        restored = FishrPenalty(ema=.5)
        restored.load_state_dict(copy.deepcopy(fishr.state_dict()))
        torch.testing.assert_close(restored.ema_variances, fishr.ema_variances)
        torch.testing.assert_close(restored.updates, fishr.updates)
        restored.eval()
        saved = copy.deepcopy(restored.state_dict())
        restored.penalty(head(x).squeeze(-1).square(), head.parameters(), groups)
        for key, value in saved.items():
            torch.testing.assert_close(restored.state_dict()[key], value)
        self.assertTrue((fishr.updates[2:] == 0).all())
        fresh_eval = FishrPenalty().eval()
        initial = copy.deepcopy(fresh_eval.state_dict())
        fresh_eval.penalty(head(x).squeeze(-1).square(), head.parameters(), groups)
        for key, value in initial.items():
            torch.testing.assert_close(fresh_eval.state_dict()[key], value)

    def test_fishr_empty_singleton_and_absent_groups_are_finite(self):
        head, fishr = nn.Linear(2, 1), FishrPenalty()
        x = torch.randn(2, 2)
        loss = head(x).squeeze(-1).square()
        singleton = fishr.penalty(loss, head.parameters(), torch.tensor([2, 5]))
        self.assertEqual(singleton.item(), 0)
        self.assertEqual(fishr.penalty(loss, head.parameters(), torch.tensor([-1, -1])).item(), 0)
        fishr.penalty(loss[:1], head.parameters(), torch.tensor([0]))
        self.assertTrue(torch.isfinite(fishr.ema_variances).all())


if __name__ == '__main__':
    unittest.main()
