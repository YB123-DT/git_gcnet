import unittest
import torch


class CovarianceTests(unittest.TestCase):
    def test_generator_does_not_leak_no_grad_to_caller(self):
        from experiments.osram_value_covariance_20261023.core import replay_value_noise
        from gcnet_missing_m3.osram import OSRAMBackbone, MODALITIES
        osram = OSRAMBackbone(latent_dim=8,output_dim=10,num_heads=1,key_dim=2,value_dim=2,dropout=0.,bidirectional=False)
        keys = {m:torch.ones(1,1,1,2) for m in MODALITIES}
        queries = torch.ones(1,1,4,1,2)
        generator = replay_value_noise(osram,keys,keys,queries,torch.ones(1,1,3),torch.ones(1,1,dtype=torch.bool))
        self.assertTrue(torch.is_grad_enabled())
        next(generator)
        try:
            self.assertTrue(torch.is_grad_enabled())
        finally:
            generator.close()

    def test_original_flat_replay_and_task_covariance_projection(self):
        from experiments.osram_value_covariance_20261023.core import flat_logits, task_direction_variance, evaluate_sequence_draws
        from types import SimpleNamespace
        from torch import nn
        torch.manual_seed(17)
        model = nn.Module()
        model.osram = nn.Module()
        model.osram.local_skip = nn.Linear(8, 10)
        model.osram.emotion_adapter = nn.Sequential(nn.Linear(8+4*2*3*2,10),nn.GELU(),nn.Linear(10,10))
        model.osram.emotion_norm = nn.LayerNorm(10)
        model.smax_fc = nn.Linear(10,1)
        model.requires_grad_(False)
        local, reads = torch.randn(2,8), torch.randn(2,2,3,4)
        valid = torch.tensor([True,False])
        roles = reads.movedim(-1,-3).flatten(-2)
        contexts = torch.cat((roles,torch.zeros_like(roles)),-1)
        x = torch.cat((local,contexts[:,0],contexts[:,1:].flatten(1)),-1)
        expected = model.smax_fc(model.osram.emotion_norm(
            model.osram.local_skip(local)+model.osram.emotion_adapter(x)))
        expected[1] = 0
        self.assertTrue(torch.equal(flat_logits(model,local,reads,valid),expected))
        g = torch.randn(2,2,4,4)
        cov = g @ g.transpose(-1,-2)
        variance = task_direction_variance(model,local,reads,cov,valid)
        self.assertTrue(torch.isfinite(variance).all())
        self.assertTrue(variance[0].gt(0).all())
        self.assertEqual(variance[1].count_nonzero().item(),0)
        variable = reads.detach().requires_grad_(True)
        grad = torch.autograd.grad(flat_logits(model,local,variable,valid).sum(),variable)[0]
        expected_variance = torch.stack([sum(grad[0,h,v] @ cov[0,h] @ grad[0,h,v]
                                                   for v in range(3)) for h in range(2)])
        torch.testing.assert_close(variance[0],expected_variance)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        sequence_local = local.unsqueeze(0).expand(3,-1,-1)
        sequence_reads = reads.unsqueeze(0).expand(3,-1,-1,-1,-1)
        sequence_valid = valid.unsqueeze(0).expand(3,-1)
        zeros = torch.zeros(2,2,*sequence_reads.shape)
        results = evaluate_sequence_draws(model,sequence_local,sequence_reads,zeros,sequence_valid)
        for sample in results['noisy_predictions'].flatten(0,1):
            self.assertTrue(torch.equal(sample,results['clean_prediction']))
        for sample in results['head_predictions']:
            self.assertTrue(torch.equal(sample,results['clean_prediction']))

    def test_omega_matches_explicit_history_value_jacobian(self):
        from experiments.osram_value_covariance_20261023.core import update_omega, read_covariance
        torch.manual_seed(42)
        dtype = torch.float64
        keys = torch.randn(4, 1, 1, 3, 3, dtype=dtype)
        scale = torch.tensor([[[.5, .7, 0.]]], dtype=dtype)
        alpha, eta = torch.tensor([.9], dtype=dtype), .6
        query = torch.randn(1, 1, 3, 4, dtype=dtype)
        omega = torch.zeros(1, 1, 3, 3, dtype=dtype)
        terms = []
        for k in keys:
            kb = k * scale[..., None, :]
            p = torch.linalg.solve(kb.transpose(-1, -2) @ kb + .01 * torch.eye(3), kb.transpose(-1, -2))
            terms.append((kb, p))
            omega = update_omega(omega, alpha, kb, p, scale, eta, torch.tensor([True]))
        expected = read_covariance(omega * alpha.square()[None, :, None, None], query)

        def response(noise):
            memory = noise.new_zeros(1, 1, 1, 3)
            for t, (kb, p) in enumerate(terms):
                before = alpha[None, :, None, None] * memory
                memory = before + eta * (noise[t].reshape(1, 1, 1, 3) * scale[..., None, :] - before @ kb) @ p
            return ((memory * alpha[None, :, None, None]) @ query).reshape(4)
        jac = torch.autograd.functional.jacobian(response, torch.zeros(4, 3, dtype=dtype)).reshape(4, -1)
        torch.testing.assert_close(expected[0, 0], jac @ jac.T, rtol=1e-12, atol=1e-12)
        self.assertGreater(expected[0, 0, 0, 1].abs().item(), 0)

    def test_padding_does_not_decay_or_write_omega(self):
        from experiments.osram_value_covariance_20261023.core import update_omega
        omega = torch.eye(3)[None, None].repeat(2, 1, 1, 1)
        out = update_omega(omega, torch.tensor([.5]), torch.ones(2,1,3,3),
            torch.ones(2,1,3,3), torch.ones(2,1,3), .6, torch.tensor([False,True]))
        self.assertTrue(torch.equal(out[0], omega[0]))

    def test_actual_scan_noise_matches_side_linear_replay_and_mc_covariance(self):
        from experiments.osram_value_covariance_20261023.core import replay_value_noise
        from gcnet_missing_m3.osram import OSRAMBackbone, MODALITIES
        torch.manual_seed(7)
        osram = OSRAMBackbone(latent_dim=8, output_dim=10, num_heads=2,
            key_dim=3, value_dim=16, dropout=0., bidirectional=False, write_step=.6)
        L, B, H, K, V = 5, 2, 2, 3, 16
        keys = {m:torch.nn.functional.normalize(torch.randn(L,B,H,K),dim=-1) for m in MODALITIES}
        values = {m:torch.randn(L,B,H,V) for m in MODALITIES}
        queries = torch.nn.functional.normalize(torch.randn(L,B,4,H,K),dim=-1)
        availability = torch.tensor([[[1,1,1],[1,0,1]],[[1,0,0],[0,1,1]],
            [[1,0,1],[0,0,0]],[[0,1,0],[1,0,1]],[[1,1,1],[1,1,0]]]).float()
        valid = availability.bool().any(-1)
        state = torch.get_rng_state().clone()
        rows = list(replay_value_noise(osram, keys,values,queries,availability,valid,
                                       draws=64,sigmas=(.001,.01),seed=9))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(rows[0][1]['G'].count_nonzero().item(), 0)
        self.assertEqual(rows[0][1]['deltas'].count_nonzero().item(), 0)
        for t, row in rows:
            self.assertLess(row['max_linear_abs_error'], 2e-5)
            self.assertTrue(torch.isfinite(row['G']).all())
            for b in range(B):
                for m in range(3):
                    if availability[t,b,m] or not valid[t,b]:
                        self.assertEqual(row['G'][b,:,m+1].count_nonzero().item(), 0)
                        self.assertEqual(row['deltas'][:,:,b,:,:,m+1].count_nonzero().item(), 0)
            if t and row['G'].norm() > 0:
                relative = (row['cov_empirical'] - row['G']).norm()/row['G'].norm()
                self.assertLess(relative.item(), .15)

        # Independent full production _scan replay, not just the side formula.
        generator = torch.Generator().manual_seed(9)
        noisy_values = {m:v.clone() for m,v in values.items()}
        for t in range(L):
            noise = torch.randn(64,B,H,V,3,generator=generator)
            for i,m in enumerate(MODALITIES):
                noisy_values[m][t] += .01 * noise[0,...,i]
        base,gap,_ = osram._scan(keys,noisy_values,queries,availability,valid,reverse=False)
        for t,row in rows:
            actual = torch.cat((base[t].reshape(B,H,V,1),
                gap[t].reshape(B,3,H,V).permute(0,2,3,1)),dim=-1)
            torch.testing.assert_close(actual, row['clean']+row['deltas'][1,0],rtol=2e-5,atol=2e-5)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
