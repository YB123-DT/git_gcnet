"""Shared CPU sequence contract and targeted core-mechanism checks."""
import unittest

import torch

from gcnet_missing_m3.core20_dynamics import (
    METHODS, MODALITIES, HierarchicalMultiscaleRNN, NeuralCDE,
    RecurrentIndependentMechanisms, TTTMLP, build,
)


class Core20DynamicsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(48)
        torch.set_num_threads(1)

    def inputs(self):
        av = torch.tensor([[[1, 0, 1], [1, 1, 0]], [[0, 1, 0], [0, 0, 0]],
                           [[1, 1, 0], [1, 0, 1]], [[0, 0, 0], [0, 0, 0]]], dtype=torch.bool)
        valid = torch.tensor([[1, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.bool)
        keys = {n: torch.randn(4, 2, 2, 3) for n in MODALITIES}
        values = {n: torch.randn(4, 2, 2, 4) for n in MODALITIES}
        q = torch.randn(4, 2, 4, 2, 3)
        return keys, values, q, av, valid

    def test_sequence_contract(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = build(method, 2, 3, 4, 16).eval()
                keys, values, q, av, valid = self.inputs()
                active = av & valid[..., None]
                dirty_k = {n: x.masked_fill(~active[..., i, None, None], float("nan")).requires_grad_() for i, (n, x) in enumerate(keys.items())}
                dirty_v = {n: x.masked_fill(~active[..., i, None, None], float("nan")).requires_grad_() for i, (n, x) in enumerate(values.items())}
                dirty_q = q.masked_fill(~valid[..., None, None, None], float("nan")).requires_grad_()
                args = (dirty_k, dirty_v, dirty_q, av, valid)
                base, gap, diag = model.scan(*args)
                self.assertEqual(base.shape, (4, 2, 8)); self.assertEqual(gap.shape, (4, 2, 3, 8))
                self.assertTrue(torch.isfinite(base).all() and torch.isfinite(gap).all())
                self.assertTrue(torch.equal(base[~valid], torch.zeros_like(base[~valid])))
                self.assertTrue(torch.equal(gap[~valid], torch.zeros_like(gap[~valid])))
                self.assertTrue(torch.equal(gap[active], torch.zeros_like(gap[active])))
                self.assertTrue(torch.equal(base[0], torch.zeros_like(base[0])))
                for n in MODALITIES:
                    self.assertEqual(set(diag[n]), {"rho", "eta", "cosine"})
                    self.assertEqual(len(diag[n]["rho"]), int(valid.sum()) * 2)
                # Current writes do not change current reads; future writes cannot alter a prefix.
                changed = {n: x.detach().clone() for n, x in dirty_v.items()}
                for n in changed:
                    changed[n][1:] = changed[n][1:] * 9
                other = model.scan(dirty_k, changed, dirty_q, av, valid)
                torch.testing.assert_close(base[:2], other[0][:2], rtol=0, atol=0)
                torch.testing.assert_close(gap[:2], other[1][:2], rtol=0, atol=0)
                # A changed real past observation must affect a later read.
                changed = {n: x.detach().clone() for n, x in dirty_v.items()}
                changed["audio"][0] += 2
                self.assertFalse(torch.equal(base[1], model.scan(dirty_k, changed, dirty_q, av, valid)[0][1]))
                # Temporary state resets each scan, including evaluation adaptation.
                with torch.no_grad():
                    again = model.scan(*args)
                torch.testing.assert_close(base, again[0], rtol=0, atol=0)
                torch.testing.assert_close(gap, again[1], rtol=0, atol=0)
                (base.square().sum() + gap.square().sum()).backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                for i, n in enumerate(MODALITIES):
                    for tensors in (dirty_k, dirty_v):
                        self.assertTrue(torch.isfinite(tensors[n].grad).all())
                        self.assertTrue(torch.equal(tensors[n].grad[~active[..., i]], torch.zeros_like(tensors[n].grad[~active[..., i]])))
                self.assertTrue(any(g.abs().sum() > 0 for g in grads))

    def test_padding_is_state_identity(self):
        for method in METHODS:
            model = build(method, 2, 3, 4, 16)
            k, v, q, av, valid = self.inputs()
            model.scan(k, v, q, av, valid)
            final = model.last_state
            model.scan({n: x[:3] for n, x in k.items()}, {n: x[:3] for n, x in v.items()}, q[:3], av[:3], valid[:3])
            for a, b in zip(final, model.last_state):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            # Removing an interior pad must not change that conversation's state.
            one_valid = torch.ones(4, 1, dtype=torch.bool); one_valid[1] = False
            k = {n: x[:, :1] for n, x in k.items()}; v = {n: x[:, :1] for n, x in v.items()}
            q, av = q[:, :1], av[:, :1]
            model.scan(k, v, q, av, one_valid); full = model.last_state
            inds = torch.tensor([0, 2, 3])
            model.scan({n: x[inds] for n, x in k.items()}, {n: x[inds] for n, x in v.items()}, q[inds], av[inds], one_valid[inds])
            for a, b in zip(full, model.last_state):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_target_specific_residual_queries(self):
        model = build("C08", 2, 3, 4, 16)
        k, v, q, av, valid = self.inputs()
        calls = []
        def residual(slot_keys, target_query):
            self.assertEqual(slot_keys.shape, (2, 2, 3, 3))
            calls.append(target_query.detach().clone())
            return target_query * .25
        raw = model.scan(k, v, q, av, valid)
        modified = model.scan(k, v, q, av, valid, address_residual=residual)
        torch.testing.assert_close(raw[0], modified[0], rtol=0, atol=0)
        self.assertEqual(len(calls), 12)
        for i in range(3):
            torch.testing.assert_close(calls[i], q[0, :, i + 1])
        self.assertFalse(torch.equal(raw[1], modified[1]))

    def test_hm_copy_update_flush_and_boundary_gradient(self):
        h, c = torch.randn(2, 4), torch.randn(2, 4)
        gates = torch.randn(2, 17, requires_grad=True)
        zero, one = torch.zeros(2, 1), torch.ones(2, 1)
        copied = HierarchicalMultiscaleRNN.transition(h, c, zero, zero, gates)
        torch.testing.assert_close(copied[0], h); torch.testing.assert_close(copied[1], c)
        updated = HierarchicalMultiscaleRNN.transition(h, c, zero, one, gates)
        flushed = HierarchicalMultiscaleRNN.transition(h, c, one, zero, gates)
        other = HierarchicalMultiscaleRNN.transition(h, c * 7, one, zero, gates)
        torch.testing.assert_close(flushed[1], other[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(updated[1], flushed[1]))
        boundary_gate = torch.zeros(2, 17, requires_grad=True)
        z = HierarchicalMultiscaleRNN.transition(h, c, zero, one, boundary_gate)[2]
        self.assertTrue(((z == 0) | (z == 1)).all())
        z.sum().backward(); self.assertTrue((boundary_gate.grad[:, -1] != 0).all())

    def test_cde_integrates_control_increment(self):
        core = NeuralCDE(2, 2, 3)
        with torch.no_grad():
            for p in core.field.parameters():
                p.zero_()
            core.field[2].bias.fill_(.2)
        z, delta = torch.randn(2, 3), torch.randn(2, core.control_dim)
        expected = z + math_tanh(.2) * delta.sum(-1, keepdim=True)
        torch.testing.assert_close(core.integrate(z, delta), expected)
        torch.testing.assert_close(core.integrate(z, torch.zeros_like(delta)), z, rtol=0, atol=0)

    def test_cde_checkpoint_output_and_outer_gradient_parity(self):
        import copy
        direct = NeuralCDE(2, 2, 3).double().eval()
        checked = copy.deepcopy(direct).train()
        state = (torch.randn(2, 3, dtype=torch.double),
                 torch.randn(2, direct.control_dim, dtype=torch.double),
                 torch.ones(2, dtype=torch.bool))
        x = torch.randn(2, 15, dtype=torch.double)
        available = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
        time = torch.tensor([4., 5.], dtype=torch.double)
        direct_inputs = tuple(t.clone().requires_grad_() for t in state[:2]) + (state[2],)
        checked_inputs = tuple(t.clone().requires_grad_() for t in state[:2]) + (state[2],)
        direct_x = x.clone().requires_grad_(); checked_x = x.clone().requires_grad_()
        result = direct.step(direct_inputs, direct_x, available, time)
        actual = checked.step(checked_inputs, checked_x, available, time)
        for a, b in zip(actual, result):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        result[0].square().sum().backward(); actual[0].square().sum().backward()
        for a, b in zip(checked.parameters(), direct.parameters()):
            if a.grad is not None or b.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        for a, b in zip(checked_inputs[:2] + (checked_x,), direct_inputs[:2] + (direct_x,)):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        checked.eval()
        with torch.no_grad():
            inference = checked.step(state, x, available, time)
        torch.testing.assert_close(inference[0], result[0], rtol=0, atol=0)
        # The field remains trainable with non-grad observation/state inputs.
        checked.train(); checked.zero_grad(set_to_none=True)
        checked.step(state, x, available, time)[0].square().sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in checked.field.parameters()))

    def test_rim_inactive_states_persist_and_communication_changes_active(self):
        core = RecurrentIndependentMechanisms(7, 8)
        slots = torch.randn(2, 3, 7)
        state = tuple(torch.randn(2, 4, 8) for _ in range(2))
        new = core.step(state, slots, torch.ones(2, 3, dtype=torch.bool))
        selected = core.last_selected
        self.assertTrue((selected.sum(-1) == 2).all())
        for a, b in zip(state, new):
            torch.testing.assert_close(a[~selected], b[~selected], rtol=0, atol=0)
        for f in core.comm_v:
            nn_zero(f)
        no_comm = core.step(state, slots, torch.ones(2, 3, dtype=torch.bool))
        self.assertFalse(torch.equal(new[0][selected], no_comm[0][selected]))
        absent = core.step(state, slots, torch.zeros(2, 3, dtype=torch.bool))
        for a, b in zip(state, absent):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_ttt_inner_gradients_match_autograd_and_eval_adapts(self):
        core = TTTMLP(2, 3, 4).double()
        key, value = torch.randn(2, 2, 3, dtype=torch.double), torch.randn(2, 2, 4, dtype=torch.double)
        state = core.initial(key)
        pred = core.predict(state, key)[0]
        gradients = torch.autograd.grad(.5 * (pred - value).square().sum(), state, create_graph=True)
        eta = core.lr(key).diagonal(dim1=-2, dim2=-1).sigmoid() / 3
        actual = core.step(state, key, value, torch.ones(2, dtype=torch.bool))
        for old, g, new in zip(state, gradients, actual):
            expected = old - eta.reshape(*eta.shape, *((1,) * (old.ndim - 2))) * g
            torch.testing.assert_close(new, expected, rtol=1e-8, atol=1e-8)
        core.eval()
        with torch.inference_mode():
            adapted = core.step(state, key, value, torch.ones(2, dtype=torch.bool))
        self.assertFalse(torch.equal(state[-1], adapted[-1]))
        # Numerical outer derivative through the analytic inner learning rule.
        fn = lambda k: core.predict(core.step(core.initial(k), k, value, torch.ones(2, dtype=torch.bool)), k)[0]
        self.assertTrue(torch.autograd.gradcheck(fn, (key.requires_grad_(),), eps=1e-6, atol=2e-4))


def math_tanh(x):
    import math
    return math.tanh(x)


def nn_zero(layer):
    with torch.no_grad():
        layer.weight.zero_()


if __name__ == "__main__":
    unittest.main()
