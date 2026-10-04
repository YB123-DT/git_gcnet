"""RAT-SPN input adaptation with exact variable marginalization.

Independent equations from Peharz et al., Random Sum-Product Networks (UAI).
Source reference: cambridge-mlg/RAT-SPN @ 366e33a6488d8a8d16ba9a2ba512e7f2779afd7b,
models/RatSpn.py and models/RegionGraph.py. No upstream source is copied; its
license was not established. See replacement_candidates.json for the source card.

Fixed configuration: four scalar coordinates per real input slot, four random
binary trees, three partition depths, eight leaf and internal distributions,
32 normalized roots, Gaussian variance in [0.1, 10]. At eight real memory heads
the circuit covers 132 distinct variables. Architecture uses private seed 1729;
partition permutations and all region scopes are checkpoint buffers.

The density is over projected evidence, not raw modalities. Missing variables
are integrated out, not observed at zero. Only the original task objective is
used. Local is conditioning-only; zero-initialized role-specific affine bridges
correct existing memory reads. No sampling, density loss, EM or batch statistics.
Run this file for the common smoke plus the necessary tiny-circuit reference.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _GaussianCircuit(nn.Module):
    """Smooth/decomposable normalized circuit on a fixed scalar-variable scope."""

    def __init__(self, variables, repeats=4, depth=3, components=8, roots=32):
        super().__init__()
        if variables < 2 ** depth or depth < 1:
            raise ValueError('Every leaf scope must contain at least one variable')
        self.variables, self.repeats = variables, repeats
        self.depth, self.components, self.roots = depth, components, roots
        generator = torch.Generator(device='cpu').manual_seed(1729)
        permutations = torch.stack([torch.randperm(variables, generator=generator)
                                    for _ in range(repeats)])
        scope = torch.zeros(repeats, 2 ** depth, variables, dtype=torch.bool)
        for repetition, permutation in enumerate(permutations):
            regions = [permutation]
            for _ in range(depth):
                regions = [part for region in regions
                           for part in (region[:len(region)//2], region[len(region)//2:])]
            for leaf, region in enumerate(regions):
                scope[repetition, leaf, region] = True
        self.register_buffer('architecture_seed', torch.tensor(1729, dtype=torch.long))
        self.register_buffer('partition_permutations', permutations)
        self.register_buffer('scope_level_0', scope)
        for level in range(1, depth + 1):
            scope = scope[:, 0::2] | scope[:, 1::2]
            self.register_buffer('scope_level_' + str(level), scope)
        self.means = nn.Parameter(torch.randn(repeats, variables, components) * .1)
        self.variance_logits = nn.Parameter(torch.randn(repeats, variables, components) * .1)
        self.sum_logits = nn.ParameterList([
            nn.Parameter(torch.randn(repeats, 2 ** (depth-level), components,
                                     components * components) * .5)
            for level in range(1, depth)])
        # Different repeats are different partitions of the *same* full scope;
        # all their root products enter one normalized mixture for each root.
        self.root_logits = nn.Parameter(torch.randn(roots, repeats * components * components) * .5)

    def log_variances(self):
        return math.log(.1) + (math.log(10.) - math.log(.1)) * self.variance_logits.sigmoid()

    @staticmethod
    def _products(children):
        # Each adjacent pair has disjoint scope. Enumerate ALL K x K products.
        return (children[:, :, 0::2, :, None]
                + children[:, :, 1::2, None, :]).flatten(-2)

    def forward(self, values, observed):
        observed = observed.bool()
        values = torch.where(observed, values, 0.)
        log_variance = self.log_variances()
        log_density = -.5 * ((values[:, None, :, None] - self.means).square()
                              * (-log_variance).exp() + math.log(2 * math.pi) + log_variance)
        # Integrating a normalized univariate Gaussian contributes log(1)=0.
        # This where is essential: a missing variable is NOT a zero observation.
        log_density = torch.where(observed[:, None, :, None], log_density, 0.)
        regions = torch.einsum('nrvc,rlv->nrlc', log_density, self.scope_level_0.to(values))
        for logits in self.sum_logits:
            products = self._products(regions)
            regions = torch.logsumexp(products[..., None, :] + F.log_softmax(logits, -1), -1)
        products = self._products(regions).flatten(1)
        return torch.logsumexp(products[:, None, :] + F.log_softmax(self.root_logits, -1), -1)


class _RATSPNInput(nn.Module):
    def __init__(self, latent_dim, heads, values):
        super().__init__()
        self.heads, self.values = heads, values
        self.local_projection = nn.Linear(latent_dim, 4)
        # Every projected coordinate depends on exactly one real role/head slot.
        self.memory_projection_weight = nn.Parameter(torch.empty(4, heads, 4, values))
        self.memory_projection_bias = nn.Parameter(torch.empty(4, heads, 4))
        nn.init.uniform_(self.memory_projection_weight, -1/math.sqrt(values), 1/math.sqrt(values))
        nn.init.uniform_(self.memory_projection_bias, -1/math.sqrt(values), 1/math.sqrt(values))
        self.circuit = _GaussianCircuit((1 + 4 * heads) * 4)
        self.output_weight = nn.Parameter(torch.zeros(4, heads * values, 32))
        self.output_bias = nn.Parameter(torch.zeros(4, heads * values))

    def forward(self, local, evidence, active, availability):
        del availability  # Effective role validity, not inferred missingness, is authoritative.
        active = active.bool()
        evidence = torch.where(active[..., None], evidence, 0.)
        if evidence.shape[0] == 0:
            return local, evidence
        valid = active.any(-1)
        safe_local = torch.where(valid[:, None], local, 0.)
        memory = evidence.reshape(-1, 4, self.heads, self.values)
        projected = (torch.einsum('nrhv,rhcv->nrhc', memory, self.memory_projection_weight)
                     + self.memory_projection_bias)
        values = torch.cat((self.local_projection(safe_local), projected.flatten(1)), -1)
        observed = torch.cat((valid[:, None].expand(-1, 4),
                              active[:, :, None, None].expand(-1, -1, self.heads, 4).flatten(1)), -1)
        roots = self.circuit(values, observed)
        correction = torch.einsum('nc,rvc->nrv', roots, self.output_weight) + self.output_bias
        changed = torch.where(active[..., None], evidence + correction, 0.)
        return local, changed


def build_circuit(method, latent_dim=256, num_heads=8, value_dim=64):
    """Build the fixed accepted circuit without perturbing caller CPU RNG."""
    if method != 'rat_spn_evidence_circuit':
        raise ValueError('Unknown evidence circuit: ' + str(method))
    if min(latent_dim, num_heads, value_dim) <= 0:
        raise ValueError('Circuit dimensions must be positive')
    with torch.random.fork_rng(devices=[]):
        return _RATSPNInput(latent_dim, num_heads, value_dim)


def _smoke_test():
    import itertools
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(41)
    assert 'build_circuit' in globals(), 'Missing RAT-SPN input circuit'
    state = torch.random.get_rng_state().clone()
    model = build_circuit('rat_spn_evidence_circuit', 12, 2, 4)
    assert torch.equal(state, torch.random.get_rng_state())
    availability = torch.tensor(list(itertools.product([0, 1], repeat=3)))
    valid = availability.bool().any(-1)
    active = torch.cat((valid[:, None], ~availability.bool() & valid[:, None]), -1)
    local = torch.randn(8, 12)
    raw = torch.randn(8, 4, 8)
    expected = torch.where(active[..., None], raw, 0.)
    evidence = torch.where(active[..., None], raw, float('nan'))
    rng = torch.random.get_rng_state().clone()
    lo, out = model(local, evidence, active, availability)
    assert torch.equal(lo, local) and torch.equal(out, expected)
    assert torch.equal(rng, torch.random.get_rng_state())
    (out * torch.randn_like(out)).sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-1e-4)
    model.zero_grad(set_to_none=True)
    _, changed = model(local, evidence, active, availability)
    assert torch.isfinite(changed).all() and not torch.equal(changed, expected)
    for i in range(8):
        _, one = model(local[i:i+1], evidence[i:i+1], active[i:i+1], availability[i:i+1])
        torch.testing.assert_close(one[0], changed[i], rtol=3e-6, atol=3e-6)
    changed.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print('RAT-SPN: identity, private RNG, inactive NaN, update, finite backward, batch independence PASS')


def _circuit_reference_test():
    # An independent probability-domain enumeration, not the production log
    # product/sum operators. Tiny scopes avoid underflow in this reference.
    tiny = _GaussianCircuit(4, repeats=2, depth=2, components=2, roots=2).double()
    values = torch.tensor([[.2, -.5, .3, .8]], dtype=torch.double)
    observed = torch.tensor([[True, False, True, False]])
    repeat_products = []
    for repetition in range(tiny.repeats):
        leaves = []
        for scope in tiny.scope_level_0[repetition]:
            components = []
            for component in range(tiny.components):
                probability = values.new_ones(())
                for variable in scope.nonzero(as_tuple=True)[0].tolist():
                    if observed[0, variable]:
                        distribution = torch.distributions.Normal(
                            tiny.means[repetition, variable, component],
                            (.5 * tiny.log_variances()[repetition, variable, component]).exp())
                        probability = probability * distribution.log_prob(values[0, variable]).exp()
                components.append(probability)
            leaves.append(components)
        regions = leaves
        for logits in tiny.sum_logits:
            new_regions = []
            for region in range(len(regions)//2):
                products = [left * right for left in regions[2*region]
                            for right in regions[2*region+1]]
                weights = logits[repetition, region].softmax(-1)
                new_regions.append([sum(weight * product for weight, product in zip(row, products))
                                    for row in weights])
            regions = new_regions
        repeat_products.extend(left * right for left in regions[0] for right in regions[1])
    expected = torch.stack([sum(weight * product for weight, product in zip(row, repeat_products))
                            for row in tiny.root_logits.softmax(-1)]).log()
    torch.testing.assert_close(tiny(values, observed)[0], expected, atol=1e-12, rtol=1e-12)
    all_missing = tiny(torch.full_like(values, float('nan')), torch.zeros_like(observed))
    torch.testing.assert_close(all_missing, torch.zeros_like(all_missing), atol=1e-12, rtol=0)
    # A real observed zero is not marginalization and has a nontrivial density.
    assert tiny(torch.zeros_like(values), torch.ones_like(observed)).abs().max() > .1
    full = build_circuit('rat_spn_evidence_circuit').circuit
    assert full.variables == 132 and full.repeats == 4 and full.roots == 32
    assert full.scope_level_0.shape == (4, 8, 132)
    assert (full.scope_level_0.sum(1) == 1).all()
    for level in range(1, 4):
        children = getattr(full, 'scope_level_' + str(level-1))
        parent = getattr(full, 'scope_level_' + str(level))
        assert not (children[:, 0::2] & children[:, 1::2]).any()
        assert torch.equal(parent, children[:, 0::2] | children[:, 1::2])
    before = {key: value.clone() for key, value in full.named_buffers()}
    full.eval()
    marginalized = full(torch.full((2, 132), float('inf')), torch.zeros(2, 132, dtype=torch.bool))
    torch.testing.assert_close(marginalized, torch.zeros_like(marginalized), atol=2e-6, rtol=0)
    assert all(torch.equal(value, before[key]) for key, value in full.named_buffers())
    print('RAT-SPN: exhaustive tiny reference, all-marginalized root0, 132-variable disjoint scopes, frozen buffers PASS')


if __name__ == '__main__':
    _smoke_test()
    _circuit_reference_test()
