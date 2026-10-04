"""Finite five-role logic adapters; no labels, auxiliary loss or external KB.

NTP: https://arxiv.org/abs/1705.11040 ; author ntp/prover.py and nunify.py
at 23687f5ccbb089ed78b4bcef5d055091408ee8d2 (max/min, clipped L2 kernel).
DeepProbLog: NeurIPS 2018; author arithmetic_circuit.py / graph_semiring.py
at f64181f66a3a789a96a47d7c2d3fae4d56178990. Here a complete 128-world
decision table replaces the SDD compiler, preserving exact shared-fact WMC.
These are task adapters, not reproductions of either paper's benchmark model.
"""
from itertools import permutations

import torch
from torch import nn


def safe_mask(values, mask):
    return torch.where(mask[..., None], values, torch.zeros_like(values))


def mlp(input_dim, output_dim, hidden=64):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim))


def soft_unify(left, right):
    """Author default exp(-sqrt(clamp(squared L2, 1e-6, 1000)))."""
    squared = (left - right).square().sum(-1).clamp(1e-6, 1000.0)
    return (-squared.sqrt()).exp()


def join_body(left, right):
    """Ground p(X,Y):-q(X,Z),r(Z,Y), using the SAME binding for Z.

    Axes [...,X,Z,Y] enumerate substitutions, AND=min, existential OR=max.
    Independent maxima of each body atom would violate variable consistency.
    """
    return torch.minimum(left[..., :, :, None], right[..., None, :, :]).max(dim=-2).values


def bounded_proofs(direct, head_match, rules, mask, depth):
    """Tensorized bounded backward proof search over a fixed grounded program.

    direct[n,p,x,y] is the best unified fact proof. At each remaining depth,
    each rule branches into TWO subgoals; all bindings and alternative rule
    heads are considered. This dynamic program shares only local subproof
    computations, not persistent facts. A depth of two permits four fact leaves.
    """
    pair_mask = mask[:, :, None] & mask[:, None, :]
    facts = torch.where(pair_mask[:, None], direct, torch.zeros_like(direct))
    proofs = facts
    for _ in range(depth):
        bodies = torch.stack([join_body(proofs[:, left], proofs[:, right])
                              for left, right in rules], dim=1)
        rule_proofs = torch.minimum(head_match[None, :, :, None, None], bodies[:, None])
        proofs = torch.maximum(facts, rule_proofs.max(dim=2).values)
        proofs = torch.where(pair_mask[:, None], proofs, torch.zeros_like(proofs))
    return proofs


def enumerate_worlds(facts):
    return ((torch.arange(1 << facts)[:, None] >> torch.arange(facts)[None]) & 1).bool()


def exact_wmc(probabilities, worlds, queries):
    """Sum disjoint possible-world weights; reused facts occur only once/world."""
    weights = torch.where(worlds[None], probabilities[:, None], 1 - probabilities[:, None]).prod(-1)
    values = weights @ queries.reshape(queries.shape[0], -1).to(probabilities.dtype)
    return values.reshape(probabilities.shape[0], *queries.shape[1:])


class _LogicAdapter(nn.Module):
    width = 64

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        if min(latent_dim, num_heads, value_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.local_in = nn.Linear(latent_dim, self.width)
        self.evidence_in = nn.Linear(num_heads * value_dim, self.width)
        self.roles = nn.Parameter(torch.randn(5, self.width) * 0.02)
        self.availability_in = nn.Linear(3, self.width, bias=False)
        self.local_out = nn.Linear(self.width, latent_dim)
        self.evidence_out = nn.ModuleList(nn.Linear(self.width, num_heads * value_dim) for _ in range(4))
        for head in [self.local_out, *self.evidence_out]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, local, evidence, active, availability):
        active = active.bool()
        clean = safe_mask(evidence, active)
        mask = torch.cat((torch.ones_like(active[:, :1]), active), dim=1)
        x = torch.cat((self.local_in(local)[:, None], self.evidence_in(clean)), dim=1)
        x = safe_mask(x + self.roles + self.availability_in(availability.to(x.dtype))[:, None], mask)
        z = self.core(x, mask)
        residual = torch.stack([head(z[:, i + 1]) for i, head in enumerate(self.evidence_out)], dim=1)
        return local + self.local_out(z[:, 0]), safe_mask(clean + residual, active)


class NeuralTheoremProver(_LogicAdapter):
    """Two query predicates, two learned binary Horn rules, depth two, 16-D symbols."""
    def __init__(self, *args):
        super().__init__(*args)
        self.fact_encoder = mlp(2 * self.width, 16)
        # Symbols 0:2 are queries; 2:6 are the two pairs of rule-body predicates.
        self.symbols = nn.Parameter(torch.randn(6, 16) * 0.2)
        self.rule_heads = nn.Parameter(torch.randn(2, 16) * 0.2)
        self.constants = nn.Parameter(torch.randn(5, 16) * 0.2)
        self.rules = ((2, 3), (4, 5))
        self.decode = mlp(self.width * 3 + 2 * 5, self.width)

    def prove(self, x, mask):
        source = x[:, :, None].expand(-1, -1, 5, -1)
        target = x[:, None, :].expand(-1, 5, -1, -1)
        fact_predicates = self.fact_encoder(torch.cat((source, target), dim=-1))
        predicate_match = soft_unify(self.symbols[None, :, None, None], fact_predicates[:, None])
        constants = soft_unify(self.constants[:, None], self.constants[None, :])
        # Ground goal P(i,j) against every fact F(a,b). Predicate and both
        # constant matches use source soft unification; substitutions for rule
        # variables remain exact shared role indices in bounded_proofs.
        matches = torch.minimum(predicate_match[:, :, None, None], constants[None, None, :, None, :, None])
        matches = torch.minimum(matches, constants[None, None, None, :, None, :])
        valid_facts = mask[:, :, None] & mask[:, None, :]
        matches = torch.where(valid_facts[:, None, None, None], matches, torch.zeros_like(matches))
        direct = matches.flatten(-2).max(-1).values
        head_match = soft_unify(self.symbols[:, None], self.rule_heads[None])
        return bounded_proofs(direct, head_match, self.rules, mask, depth=2)[:, :2]

    def core(self, x, mask):
        proofs = self.prove(x, mask)
        messages = torch.einsum("npij,njd->nipd", proofs, x).flatten(2)
        scores = proofs.permute(0, 2, 1, 3).flatten(2)
        return safe_mask(self.decode(torch.cat((x, messages, scores), dim=-1)), mask)


class DeepProbLog(_LogicAdapter):
    """Seven Bernoulli edge facts, 128 disjoint worlds, exact three query families.

    Undirected support skeleton: Local–Base, and Local–Gap/Base–Gap for each
    Gap. Query families are direct, mediated simple paths of length 2–3, and
    their logical union. An edge is one shared random variable across proofs.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.edges = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4))
        self.edge_encoder = mlp(2 * self.width, 1)
        self.decode = mlp(self.width * 4 + 3 * 5, self.width)
        worlds = enumerate_worlds(len(self.edges))
        direct = torch.zeros(worlds.shape[0], 5, 5, dtype=torch.bool)
        for edge, (source, target) in enumerate(self.edges):
            direct[:, source, target] = worlds[:, edge]
            direct[:, target, source] = worlds[:, edge]
        mediated = torch.zeros_like(direct)
        # Static propositional grounding/decision table, not a cache of examples.
        for source in range(5):
            for target in range(5):
                if source == target:
                    continue
                remaining = [role for role in range(5) if role not in (source, target)]
                for count in (1, 2):
                    for inner in permutations(remaining, count):
                        path = (source, *inner, target)
                        truth = torch.ones(worlds.shape[0], dtype=torch.bool)
                        for left, right in zip(path[:-1], path[1:]):
                            truth = truth & direct[:, left, right]
                        mediated[:, source, target] |= truth
        self.register_buffer("worlds", worlds)
        self.register_buffer("queries", torch.stack((direct, mediated, direct | mediated), dim=1))
        self.register_buffer("edge_sources", torch.tensor([edge[0] for edge in self.edges]))
        self.register_buffer("edge_targets", torch.tensor([edge[1] for edge in self.edges]))

    def marginals(self, x, mask):
        pair = torch.cat((x[:, self.edge_sources], x[:, self.edge_targets]), dim=-1)
        probabilities = self.edge_encoder(pair).squeeze(-1).sigmoid()
        available = mask[:, self.edge_sources] & mask[:, self.edge_targets]
        # Unavailable facts are deterministic false (not missing-role random
        # variables). Their true worlds have exactly zero probability/gradient.
        probabilities = torch.where(available, probabilities, torch.zeros_like(probabilities))
        return exact_wmc(probabilities, self.worlds, self.queries)

    def core(self, x, mask):
        marginals = self.marginals(x, mask)
        messages = torch.einsum("npij,njd->nipd", marginals, x).flatten(2)
        scores = marginals.permute(0, 2, 1, 3).flatten(2)
        return safe_mask(self.decode(torch.cat((x, messages, scores), dim=-1)), mask)


METHODS = {
    "survey80_struct_neural_theorem_prover": NeuralTheoremProver,
    "survey80_struct_deepproblog": DeepProbLog,
}


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    if method not in METHODS:
        raise ValueError(f"Unknown logic method: {method}")
    return METHODS[method](latent_dim, num_heads, value_dim)
