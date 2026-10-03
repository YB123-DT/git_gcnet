"""Equation and boundary tests for four current-read grouping adaptations."""

import copy
import importlib
import importlib.util
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F


MODULE = "gcnet_missing_m3.meaningful_blocks_grouping"
grouping = importlib.import_module(MODULE) if importlib.util.find_spec(MODULE) else None
METHODS = (
    "capsule_dynamic_routing",
    "slot_attention",
    "otke",
    "capsule_variational_bayes",
)


def sample_inputs(rows=3, latent_dim=12, heads=2, value_dim=4):
    local = torch.randn(rows, latent_dim)
    evidence = torch.randn(rows, 4, heads * value_dim)
    active = torch.ones(rows, 4, dtype=torch.bool)
    availability = torch.zeros(rows, 3, dtype=torch.bool)
    return local, evidence, active, availability


def reference_dynamic(votes, mask, iterations=3):
    """Literal per-row/per-parent Procedure 1, independently looped."""
    rows, count, parents, width = votes.shape
    result = votes.new_zeros(rows, parents, width)
    for n in range(rows):
        logits = votes.new_zeros(count, parents)
        for step in range(iterations):
            probabilities = logits.softmax(dim=1)
            output = []
            for j in range(parents):
                total = sum(
                    (probabilities[i, j] * votes[n, i, j] for i in range(count) if mask[n, i]),
                    votes.new_zeros(width),
                )
                length2 = total.square().sum()
                output.append(length2 / (1 + length2) * total / length2.clamp_min(1e-8).sqrt())
            result[n] = torch.stack(output)
            if step < iterations - 1:
                for i in range(count):
                    if mask[n, i]:
                        for j in range(parents):
                            logits[i, j] += (votes[n, i, j] * result[n, j]).sum()
    return result


def reference_slot(model, tokens, mask, slots):
    inputs = model.norm_inputs(tokens)
    inputs = torch.where(mask[..., None], inputs, 0.0)
    keys, values = model.key(inputs), model.value(inputs)
    for _ in range(3):
        old = slots
        logits = torch.einsum("nid,njd->nij", keys, model.query(model.norm_slots(slots))) / 8.0
        probabilities = logits.softmax(dim=-1)
        weights = torch.where(mask[..., None], probabilities + 1e-8, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        updates = torch.einsum("nij,nid->njd", weights, values)
        # Explicit GRU equations instead of invoking the implementation's cell.
        input_parts = F.linear(updates, model.gru.weight_ih, model.gru.bias_ih).chunk(3, -1)
        state_parts = F.linear(old, model.gru.weight_hh, model.gru.bias_hh).chunk(3, -1)
        reset = torch.sigmoid(input_parts[0] + state_parts[0])
        update = torch.sigmoid(input_parts[1] + state_parts[1])
        proposal = torch.tanh(input_parts[2] + reset * state_parts[2])
        slots = (1 - update) * proposal + update * old
        slots = slots + model.mlp(model.norm_mlp(slots))
    return slots


class GroupingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.assertIsNotNone(grouping, "grouping implementations have not been added yet")
        torch.manual_seed(71)

    def build(self, method, latent_dim=12, heads=2, value_dim=4):
        return grouping.build_grouping(method, latent_dim, heads, value_dim)

    def test_factory_shapes_and_no_batch_norm_or_dropout(self):
        inputs = sample_inputs()
        for method in METHODS:
            with self.subTest(method=method):
                model = self.build(method).eval()
                self.assertEqual(model.output_dim, 128)
                self.assertEqual(model(*inputs).shape, (3, 128))
                self.assertFalse(any(isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout)) for m in model.modules()))
                self.assertTrue(torch.isfinite(model(*inputs)).all())
        with self.assertRaises(ValueError):
            self.build("row_softmax_is_not_a_grouping_architecture")
        with self.assertRaises(ValueError):
            self.build(METHODS[0], heads=0)

    def test_seven_availability_patterns_ignore_inactive_nans(self):
        availability = torch.tensor([[bool(bits & (1 << i)) for i in range(3)] for bits in range(1, 8)])
        local, evidence, _, _ = sample_inputs(rows=7)
        active = torch.cat((torch.ones(7, 1, dtype=torch.bool), ~availability), dim=1)
        clean = torch.where(active[..., None], evidence, 0.0)
        poisoned = torch.where(active[..., None], evidence, float("nan"))
        for method in METHODS:
            with self.subTest(method=method):
                model = self.build(method).eval()
                expected = model(local, clean, active, availability)
                actual = model(local, poisoned, active, availability)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual).all())

    def test_masked_nan_gradients_and_parameter_updates(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = self.build(method).train()
                local, evidence, active, availability = sample_inputs()
                active[0, 1:] = False
                active[1, 2] = False
                evidence = torch.where(active[..., None], evidence, float("nan")).requires_grad_()
                local.requires_grad_()
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                model(local, evidence, active, availability).square().mean().backward()
                self.assertTrue(torch.isfinite(local.grad).all())
                self.assertTrue(torch.isfinite(evidence.grad).all())
                self.assertEqual(torch.count_nonzero(evidence.grad[~active]).item(), 0)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                optimizer.step()
                self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters()))
                self.assertTrue(torch.isfinite(model(local, evidence, active, availability)).all())

    def test_all_inactive_and_empty_are_exact_zero_after_bias_updates(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = self.build(method)
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if "bias" in name:
                            p.fill_(0.3)
                local, evidence, active, availability = sample_inputs()
                active.zero_()
                local.fill_(float("nan"))
                evidence.fill_(float("nan"))
                self.assertEqual(torch.count_nonzero(model(local, evidence, active, availability)).item(), 0)
                self.assertEqual(model(local[:0], evidence[:0], active[:0], availability[:0]).shape, (0, 128))

    def test_active_zeros_are_finite_and_mixed_empty_rows_are_zero(self):
        inputs = list(sample_inputs())
        inputs[0].zero_()
        inputs[1].zero_()
        inputs[2][1].zero_()
        for method in METHODS:
            with self.subTest(method=method):
                output = self.build(method).eval()(*inputs)
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(torch.count_nonzero(output[1]).item(), 0)

    def test_eval_batch_independence_and_checkpoint_roundtrip(self):
        inputs = sample_inputs()
        for method in METHODS:
            with self.subTest(method=method):
                model = self.build(method).eval()
                expected = model(*inputs)
                separate = torch.cat([model(*(x[i:i + 1] for x in inputs)) for i in range(3)])
                torch.testing.assert_close(expected, separate, rtol=1e-4, atol=2e-5)
                torch.testing.assert_close(expected, model(*inputs), rtol=0, atol=0)
                clone = self.build(method).eval()
                clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                torch.testing.assert_close(expected, clone(*inputs), rtol=0, atol=0)

    def test_local_and_base_condition_outputs_without_gap(self):
        local, evidence, active, availability = sample_inputs()
        active[:, 1:] = False
        availability.fill_(True)
        for method in METHODS:
            with self.subTest(method=method):
                model = self.build(method).eval()
                before = model(local, evidence, active, availability)
                changed_local = local.clone()
                changed_local[:, 0] += 3
                changed_base = evidence.clone()
                changed_base[:, 0, 0] += 3
                self.assertGreater((before - model(changed_local, evidence, active, availability)).abs().max().item(), 1e-6)
                self.assertGreater((before - model(local, changed_base, active, availability)).abs().max().item(), 1e-6)

    def test_tokenizer_preserves_real_heads_and_zeroes_inactive_tokens(self):
        model = self.build(METHODS[0], latent_dim=256, heads=8, value_dim=64)
        local, evidence, active, availability = sample_inputs(2, 256, 8, 64)
        active[0, 1:] = False
        active[1, 2] = False
        evidence = torch.where(active[..., None], evidence, float("nan"))
        tokens, mask = model.tokenizer(local, evidence, active)
        self.assertEqual(tokens.shape, (2, 33, 64))
        self.assertEqual(mask.sum(1).tolist(), [9, 25])
        self.assertEqual(torch.count_nonzero(tokens[~mask]).item(), 0)
        self.assertTrue(torch.isfinite(tokens).all())
        safe = local
        projected = F.gelu(model.tokenizer.local_projection(model.tokenizer.local_norm(safe)))
        torch.testing.assert_close(tokens[:, 0], projected + model.tokenizer.type_embedding[0])

    def test_dynamic_routing_matches_three_round_equations_and_not_one_pass(self):
        votes = torch.randn(2, 5, 4, 32) * 0.4
        mask = torch.tensor([[True, True, False, True, False], [True] * 5])
        actual = grouping.dynamic_routing(votes, mask)
        torch.testing.assert_close(actual, reference_dynamic(votes, mask), atol=3e-7, rtol=1e-5)
        self.assertGreater((actual - reference_dynamic(votes, mask, 1)).abs().max().item(), 1e-3)
        zero = grouping.dynamic_routing(torch.zeros_like(votes), mask)
        self.assertEqual(torch.count_nonzero(zero).item(), 0)

    def test_dynamic_transform_is_parent_specific(self):
        model = self.build(METHODS[0])
        tokens = torch.randn(2, 9, 64)
        mask = torch.ones(2, 9, dtype=torch.bool)
        votes = torch.einsum("nid,ijod->nijo", tokens, model.vote_weight)
        torch.testing.assert_close(model.group_tokens(tokens, mask), reference_dynamic(votes, mask).flatten(1), atol=3e-7, rtol=1e-5)

    def test_slot_refinement_matches_gru_and_competitive_weighting_equations(self):
        model = self.build("slot_attention").eval()
        tokens = torch.randn(2, 9, 64)
        mask = torch.tensor([[True] * 6 + [False] * 3, [True] * 9])
        tokens = torch.where(mask[..., None], tokens, 0.0)
        initial = torch.randn(2, 4, 64)
        actual = model.refine_slots(tokens, mask, initial)
        expected = reference_slot(model, tokens, mask, initial)
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=2e-5)
        logits = torch.tensor([[[2., 0.], [1., 2.], [-1., 3.]]])
        valid = torch.tensor([[True, True, False]])
        weights = grouping.slot_assignment_weights(logits, valid)
        torch.testing.assert_close(weights.sum(1), torch.ones(1, 2))
        self.assertEqual(torch.count_nonzero(weights[:, 2]).item(), 0)
        naive = logits[:, :2].softmax(1)
        self.assertGreater((weights[:, :2] - naive).abs().max().item(), 0.02)

    def test_slot_training_private_rng_does_not_consume_global_rng(self):
        model = self.build("slot_attention").train()
        inputs = sample_inputs()
        global_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        private_before = model.training_noise_rng_state.clone()
        first = model(*inputs)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else [])))
        self.assertFalse(torch.equal(private_before, model.training_noise_rng_state))
        second = model(*inputs)
        self.assertGreater((first - second).abs().max().item(), 1e-6)
        clone = self.build("slot_attention").train()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        expected_next = model(*inputs)
        torch.testing.assert_close(expected_next, clone(*inputs), rtol=0, atol=0)
        self.assertTrue(torch.equal(model.training_noise_rng_state, clone.training_noise_rng_state))

    def test_slot_eval_and_empty_calls_preserve_all_rng_buffers(self):
        model = self.build("slot_attention").eval()
        inputs = sample_inputs()
        buffers_before = {name: b.clone() for name, b in model.named_buffers()}
        global_before = torch.random.get_rng_state().clone()
        first, second = model(*inputs), model(*inputs)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
        for name, buffer in model.named_buffers():
            self.assertTrue(torch.equal(buffers_before[name], buffer), name)
        model.train()
        local, evidence, active, availability = inputs
        active.zero_()
        model(local, evidence, active, availability)
        for name, buffer in model.named_buffers():
            self.assertTrue(torch.equal(buffers_before[name], buffer), name)

    def test_slot_initial_seed_and_configure_seed_are_private(self):
        model = self.build("slot_attention")
        expected = torch.Generator(device="cpu").manual_seed((torch.initial_seed() + 104729) % (2 ** 63))
        self.assertTrue(torch.equal(model.training_noise_rng_state.cpu(), expected.get_state()))
        global_before = torch.random.get_rng_state().clone()
        model.configure_seed(987)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
        expected.manual_seed(987 + 104729)
        self.assertTrue(torch.equal(model.training_noise_rng_state.cpu(), expected.get_state()))
        eval_generator = torch.Generator(device="cpu").manual_seed(1729)
        torch.testing.assert_close(model.epsilon_eval, torch.randn(4, 64, generator=eval_generator), rtol=0, atol=0)

    def test_ot_matches_independent_scaling_and_two_marginals(self):
        cost = torch.tensor([[0.1, 0.5, -0.4, 0.2], [0.3, -0.1, 0.6, 0.4], [-0.2, 0.8, 0.7, -0.5]])
        actual = grouping.sinkhorn_plan(cost)
        kernel = torch.exp(-cost / 0.5)
        v = torch.ones(4)
        for _ in range(30):
            u = (1 / 3) / (kernel @ v)
            v = (1 / 4) / (kernel.T @ u)
        expected = u[:, None] * kernel * v[None, :]
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(actual.sum(0), torch.full((4,), 1 / 4), atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(actual.sum(1), torch.full((3,), 1 / 3), atol=2e-7, rtol=1e-6)
        self.assertGreater((actual - (-cost / 0.5).softmax(-1) / 3).abs().max().item(), 0.01)

    def test_ot_embedding_preserves_four_reference_bins_and_paper_scaling(self):
        model = self.build("otke")
        tokens = torch.randn(1, 9, 64)
        mask = torch.tensor([[True, True, False, True, False, False, False, False, False]])
        actual = model.group_tokens(tokens, mask)
        h = F.normalize(F.relu(model.feature_map(tokens[0, mask[0]])), dim=-1, eps=1e-8)
        supports = F.normalize(model.supports, dim=-1, eps=1e-8)
        plan = grouping.sinkhorn_plan(-h @ supports.T)
        expected = model.output_projection((2 * plan.T @ h).flatten().unsqueeze(0))
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)

    def test_batched_ot_matches_packed_rows_including_empty_nan_rows(self):
        cost = torch.randn(3, 6, 4) * 0.3
        mask = torch.tensor([[True, False, True, True, False, False], [True] * 6, [False] * 6])
        poisoned = torch.where(mask[..., None], cost, float("nan")).requires_grad_()
        actual = grouping.sinkhorn_plan(poisoned, mask=mask)
        expected = torch.zeros_like(cost)
        for row in range(2):
            expected[row, mask[row]] = grouping.sinkhorn_plan(cost[row, mask[row]])
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        self.assertEqual(torch.count_nonzero(actual[~mask]).item(), 0)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(poisoned.grad).all())
        self.assertEqual(torch.count_nonzero(poisoned.grad[~mask]).item(), 0)

    def test_vb_posterior_matches_sufficient_statistics_and_full_covariance(self):
        votes = torch.randn(1, 3, 4, 16) * 0.3
        mass = torch.tensor([[[0.2, 0.3, 0.1, 0.4], [0.1, 0.2, 0.4, 0.1], [0.3, 0.1, 0.1, 0.2]]])
        actual = grouping.vb_posterior(votes, mass)
        for j in range(4):
            count = mass[0, :, j].sum()
            average = sum(mass[0, i, j] * votes[0, i, j] for i in range(3)) / count
            scatter = sum(mass[0, i, j] * torch.outer(votes[0, i, j] - average, votes[0, i, j] - average) for i in range(3))
            inverse_scale = torch.eye(16) + scatter + count / (1 + count) * torch.outer(average, average)
            inverse_scale = inverse_scale + 1e-6 * torch.eye(16)
            mean = count / (1 + count) * average
            torch.testing.assert_close(actual["mean"][0, j], mean)
            torch.testing.assert_close(actual["inverse_scale"][0, j], inverse_scale)
            expected_logdet = 16 * math.log(2) - torch.linalg.slogdet(inverse_scale).logabsdet
            expected_logdet += torch.stack([torch.digamma((17 + count - d) / 2) for d in range(16)]).sum()
            torch.testing.assert_close(actual["expected_logdet"][0, j], expected_logdet, atol=5e-6, rtol=1e-5)
        off_diagonal = actual["inverse_scale"] - torch.diag_embed(actual["inverse_scale"].diagonal(dim1=-2, dim2=-1))
        self.assertGreater(off_diagonal.abs().max().item(), 1e-3)

    def test_vb_routing_matches_three_posterior_rounds_and_entropy_activation(self):
        model = self.build("capsule_variational_bayes")
        votes = torch.randn(2, 5, 4, 16) * 0.3
        activations = torch.rand(2, 5)
        activations[:, -1] = 0
        actual_mean, actual_activation = model.route_votes(votes, activations)
        responsibility = torch.full((2, 5, 4), 0.25)
        for step in range(3):
            posterior = grouping.vb_posterior(votes, activations[..., None] * responsibility)
            if step < 2:
                delta = votes - posterior["mean"][:, None]
                # Explicit inverse is an independent test oracle only; production uses Cholesky solves.
                precision = torch.linalg.inv(posterior["inverse_scale"])
                mahalanobis = torch.einsum("nijd,njde,nije->nij", delta, precision, delta)
                score = posterior["expected_logpi"][:, None] + 0.5 * posterior["expected_logdet"][:, None]
                score = score - 0.5 * (16 / posterior["kappa"][:, None] + posterior["nu"][:, None] * mahalanobis)
                responsibility = score.softmax(-1)
        entropy = 8 * math.log(2 * math.pi * math.e) - 0.5 * posterior["expected_logdet"]
        expected_activation = torch.sigmoid(model.beta_a - posterior["expected_logpi"].exp() * entropy - model.beta_u)
        torch.testing.assert_close(actual_mean, posterior["mean"], rtol=2e-5, atol=4e-7)
        torch.testing.assert_close(actual_activation, expected_activation, rtol=2e-5, atol=4e-7)
        first = grouping.vb_posterior(votes, activations[..., None] * 0.25)
        self.assertGreater((actual_mean - first["mean"]).abs().max().item(), 1e-4)

    def test_vb_matrix_votes_and_output_include_activations(self):
        model = self.build("capsule_variational_bayes")
        tokens = torch.randn(1, 9, 64)
        mask = torch.tensor([[True] * 5 + [False] * 4])
        poses, activation = model.project_capsules(tokens, mask)
        self.assertEqual(poses.shape, (1, 9, 4, 4))
        self.assertEqual(torch.count_nonzero(poses[~mask]).item(), 0)
        self.assertEqual(torch.count_nonzero(activation[~mask]).item(), 0)
        votes = torch.einsum("nirc,ijcd->nijrd", poses, model.vote_weight).flatten(-2)
        mean, parent_activation = model.route_votes(votes, activation)
        expected = model.output_projection(torch.cat(((mean * parent_activation[..., None]).flatten(1), parent_activation), dim=1))
        torch.testing.assert_close(model.group_tokens(tokens, mask), expected, atol=3e-7, rtol=2e-5)

    def test_cpu_autocast_keeps_routing_finite(self):
        inputs = sample_inputs()
        for method in METHODS:
            with self.subTest(method=method):
                model = self.build(method).eval()
                reference = model(*inputs)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    actual = model(*inputs)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_float64_cores_preserve_parameter_precision_and_finite_gradients(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = self.build(method).double().eval()
                local, evidence, active, availability = sample_inputs()
                active[0, 1:] = False
                active[2] = False
                local = torch.where(active.any(-1)[:, None], local.double(), float("nan")).requires_grad_()
                evidence = torch.where(active[..., None], evidence.double(), float("nan")).requires_grad_()
                actual = model(local, evidence, active, availability)
                self.assertEqual(actual.dtype, torch.float64)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(torch.count_nonzero(actual[2]).item(), 0)
                tokens, mask = model.tokenizer(local[:2], evidence[:2], active[:2])
                expected = model.group_tokens(tokens, mask)
                torch.testing.assert_close(actual[:2], expected, rtol=0, atol=0)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    under_autocast = model(local, evidence, active, availability)
                torch.testing.assert_close(actual, under_autocast, rtol=0, atol=0)
                actual.square().mean().backward()
                self.assertTrue(torch.isfinite(local.grad).all())
                self.assertTrue(torch.isfinite(evidence.grad).all())
                self.assertEqual(torch.count_nonzero(local.grad[2]).item(), 0)
                self.assertEqual(torch.count_nonzero(evidence.grad[~active]).item(), 0)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertEqual(parameter.grad.dtype, torch.float64, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_float64_real_head_residual_integration_and_strict_reload(self):
        from gcnet_missing_m3.meaningful_blocks import MeaningfulReadoutResidual

        for method in METHODS:
            with self.subTest(method=method):
                wrapper = MeaningfulReadoutResidual(12, 1024, 1600, method, 8, 64).double().eval()
                with torch.no_grad():
                    wrapper.output.weight.normal_(std=0.003)
                    wrapper.output.bias.fill_(0.1)
                umask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float64)
                valid = umask.T.bool()
                history = valid & (valid.long().cumsum(0) > 1)
                availability = torch.tensor([1., 0., 1.], dtype=torch.float64).expand(4, 2, 3).clone()
                availability[~valid] = 0
                local = torch.where(history[..., None], torch.randn(4, 2, 12, dtype=torch.float64), float("nan")).requires_grad_()
                base = torch.randn(4, 2, 1024, dtype=torch.float64)
                base[..., 512:] = float("nan")
                base = torch.where(history[..., None], base, float("nan")).requires_grad_()
                gap = torch.randn(4, 2, 3, 1024, dtype=torch.float64)
                gap[..., 512:] = float("nan")
                gap_active = history[..., None] & ~availability.bool()
                gap = torch.where(gap_active[..., None], gap, float("nan")).requires_grad_()
                anchor = torch.randn(4, 2, 1600, dtype=torch.float64)
                args = (local, base, gap, availability, umask, anchor)
                actual = wrapper(*args)
                self.assertEqual(actual.dtype, torch.float64)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(torch.count_nonzero(actual[~history]).item(), 0)
                actual.square().mean().backward()
                for value in (local, base, gap):
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all())
                    self.assertEqual(torch.count_nonzero(value.grad[~history]).item(), 0)
                for name, parameter in wrapper.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                clone = MeaningfulReadoutResidual(12, 1024, 1600, method, 8, 64).double().eval()
                clone.load_state_dict(copy.deepcopy(wrapper.state_dict()), strict=True)
                torch.testing.assert_close(actual, clone(*args), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
