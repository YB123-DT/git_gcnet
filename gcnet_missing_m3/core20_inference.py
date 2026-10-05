"""Mechanism-preserving DEQ, Universal Transformer, NRI and ProtoPNet cores.

These are domain adaptations, not reproductions of the papers' datasets/models.
NRI uses actual utterance nodes; ProtoPNet uses utterance feature exemplars.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _inputs(local, base, gap, availability, umask):
    length, batch, _ = local.shape
    if (base.shape[:2] != (length, batch) or gap.shape != (*base.shape[:2], 3, base.shape[-1])
            or availability.shape != (length, batch, 3) or umask.shape != (batch, length)):
        raise ValueError("readout input shapes do not match")
    valid = umask.T.bool()
    if bool(((availability[valid] != 0) & (availability[valid] != 1)).any()):
        raise ValueError("availability must be binary")
    # Base/Gap are history evidence. No evidence exists before the first write.
    history = valid & (valid.long().cumsum(0) > 1)
    active_gap = history[..., None] & ~availability.bool()
    local = torch.where(valid[..., None], local, torch.zeros_like(local))
    base = torch.where(history[..., None], base, torch.zeros_like(base))
    gap = torch.where(active_gap[..., None], gap, torch.zeros_like(gap))
    return torch.cat((local, base, gap.flatten(2),
                      torch.where(valid[..., None], availability, 0).to(local.dtype)), -1), valid


def fixed_point(function, initial, tolerance=1e-6, max_steps=80):
    """Convergence-controlled Picard solver, with independent row stopping.

    Unlike an unrolled network, callers run this without autograd and attach
    implicit differentiation afterwards. Nonconvergence raises explicitly.
    """
    state = initial
    steps = torch.zeros(state.shape[:-1], device=state.device, dtype=torch.long)
    converged = torch.zeros_like(steps, dtype=torch.bool)
    for _ in range(max_steps):
        proposal = function(state)
        residual = (proposal - state).norm(dim=-1)
        done = residual <= tolerance * (1 + proposal.norm(dim=-1))
        steps = steps + (~converged).long()
        state = torch.where(converged[..., None], state, proposal)
        converged = converged | done
        if bool(converged.all()):
            break
    residual = (function(state) - state).norm(dim=-1)
    converged = residual <= tolerance * (1 + state.norm(dim=-1))
    if not bool(converged.all()):
        raise RuntimeError("fixed-point solver failed to converge")
    return state, {"residual": residual.detach(), "converged": converged.detach(),
                   "iterations": steps.detach()}


class DEQReadout(nn.Module):
    """Input-injected contractive equilibrium; implicit adjoint backward.

    Frobenius bounding guarantees spectral norm <= .8, making both forward
    and the adjoint equation contractive. This deliberately replaces the
    original DEQ transformer's unconstrained attention cell with a dense cell.
    """
    def __init__(self, latent_dim=256, context_dim=1024, output_dim=1600,
                 tolerance=1e-6, max_steps=80):
        super().__init__()
        self.inject = nn.Linear(latent_dim + 4 * context_dim + 3, latent_dim)
        self.recurrent = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.orthogonal_(self.recurrent)
        self.output = nn.Linear(latent_dim, output_dim)
        self.tolerance, self.max_steps = tolerance, max_steps
        self.last_diagnostics = {}

    def cell(self, state, injected):
        weight = self.recurrent * (.8 / self.recurrent.norm().clamp_min(.8))
        return torch.tanh(F.linear(state, weight) + injected)

    def forward(self, local, base, gap, availability, umask):
        inputs, valid = _inputs(local, base, gap, availability, umask)
        output = local.new_zeros(*local.shape[:2], self.output.out_features)
        if not bool(valid.any()):
            return output + self.output.weight.sum() * 0
        injected = self.inject(inputs[valid])
        with torch.no_grad():
            equilibrium, diagnostics = fixed_point(lambda z: self.cell(z, injected),
                torch.zeros_like(injected), self.tolerance, self.max_steps)
        if torch.is_grad_enabled():
            state = equilibrium.detach().requires_grad_()
            equilibrium = self.cell(state, injected)
            # A hook on this one-cell graph implements v = J_f^T v + grad.
            # Remove only during the VJP to prevent recursively invoking itself.
            handle = None
            def implicit(gradient):
                nonlocal handle
                handle.remove()
                with torch.enable_grad():
                    adjoint, backward = fixed_point(
                        lambda v: torch.autograd.grad(equilibrium, state, v,
                            retain_graph=True)[0] + gradient,
                        torch.zeros_like(gradient), self.tolerance, self.max_steps)
                self.last_diagnostics["backward"] = backward
                handle = equilibrium.register_hook(implicit)
                return adjoint
            handle = equilibrium.register_hook(implicit)
        output[valid] = self.output(equilibrium)
        self.last_diagnostics = {"forward": diagnostics}
        return output


def _timing(position, width, reference):
    frequencies = torch.exp(torch.arange(0, width, 2, device=reference.device,
                                        dtype=reference.dtype) * (-math.log(10000) / width))
    angles = torch.as_tensor(position, device=reference.device,
                             dtype=reference.dtype)[..., None] * frequencies
    result = reference.new_zeros(*angles.shape[:-1], width)
    result[..., 0::2] = angles.sin()
    result[..., 1::2] = angles.cos()[..., :width // 2]
    return result


class UniversalACTReadout(nn.Module):
    """Shared causal self-attention/transition with accumulated ACT outputs."""
    def __init__(self, latent_dim=256, context_dim=1024, output_dim=1600,
                 max_steps=12, epsilon=.01, ponder_weight=.01):
        super().__init__()
        heads = next(h for h in (8, 4, 2, 1) if latent_dim % h == 0)
        self.inject = nn.Linear(latent_dim + 4 * context_dim + 3, latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, heads, dropout=0, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(latent_dim), nn.LayerNorm(latent_dim)
        self.transition = nn.Sequential(nn.Linear(latent_dim, 4 * latent_dim), nn.ReLU(),
                                        nn.Linear(4 * latent_dim, latent_dim))
        self.halt = nn.Linear(latent_dim, 1)
        nn.init.constant_(self.halt.bias, 1.)
        self.output = nn.Linear(latent_dim, output_dim)
        self.max_steps, self.epsilon, self.ponder_weight = max_steps, epsilon, ponder_weight
        self.last_diagnostics = {}
        self.ponder_cost = torch.tensor(0.)

    def forward(self, local, base, gap, availability, umask):
        inputs, valid = _inputs(local, base, gap, availability, umask)
        result = local.new_zeros(*local.shape[:2], self.output.out_features)
        costs, records = [], []
        for b in range(local.shape[1]):
            indices = valid[:, b].nonzero().flatten()
            if not indices.numel():
                continue
            state = self.inject(inputs[indices, b]).unsqueeze(0)
            position = _timing(indices, state.shape[-1], state).unsqueeze(0)
            causal = torch.ones(len(indices), len(indices), device=state.device,
                                dtype=torch.bool).triu(1)
            probability = state.new_zeros(1, len(indices))
            remainder, updates = torch.zeros_like(probability), torch.zeros_like(probability)
            accumulated = torch.zeros_like(state)
            for step in range(self.max_steps):
                running = probability < 1 - self.epsilon
                if not bool(running.any()):
                    break
                # Halted tokens remain exactly frozen, including attention keys:
                # later position/depth timing must not alter their representation.
                timed = torch.where(running[..., None],
                    state + position + _timing(step, state.shape[-1], state), state)
                p = self.halt(timed).squeeze(-1).sigmoid()
                halted = running & ((probability + p > 1 - self.epsilon)
                                    | (step == self.max_steps - 1))
                weight = torch.where(halted, 1 - probability,
                                     torch.where(running, p, 0))
                remainder = torch.where(halted, weight, remainder)
                probability = probability + weight
                updates = updates + running.to(updates.dtype)
                attended = self.attention(timed, timed, timed, attn_mask=causal,
                                          need_weights=False)[0]
                transformed = self.norm1(timed + attended)
                transformed = self.norm2(transformed + self.transition(transformed))
                state = torch.where(running[..., None], transformed, state)
                accumulated = accumulated + state * weight[..., None]
            result[indices, b] = self.output(accumulated.squeeze(0))
            costs.append((updates + remainder).reshape(-1))
            records.append({"batch": b, "indices": indices.detach(),
                            "updates": updates.detach(), "remainder": remainder.detach(),
                            "halting_probability": probability.detach()})
        self.ponder_cost = (torch.cat(costs).mean() * self.ponder_weight if costs
                            else self.output.weight.sum() * 0)
        self.last_diagnostics = {"act": records}
        return result


class NRIReadout(nn.Module):
    """Causal latent categorical interactions between actual utterance nodes.

    Each prefix infers edges among its utterances, predicts a successor feature
    with typed messages, and emits the current decoded node. Type zero carries
    no message. Conversation nodes differ from the original persistent physical
    objects; this is explicitly a domain adaptation of NRI's generative core.
    """
    def __init__(self, latent_dim=256, context_dim=1024, output_dim=1600,
                 edge_types=3, temperature=.5, kl_weight=.01, prediction_weight=1.):
        super().__init__()
        if edge_types < 2:
            raise ValueError("NRI requires no-edge and interaction edge categories")
        self.inject = nn.Linear(latent_dim + 4 * context_dim + 3, latent_dim)
        self.node_encoder = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ELU())
        self.edge_encoder = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim), nn.ELU())
        self.node_factor = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ELU())
        self.posterior = nn.Sequential(nn.Linear(3 * latent_dim, latent_dim), nn.ELU(),
                                       nn.Linear(latent_dim, edge_types))
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(), nn.Linear(latent_dim, latent_dim), nn.ReLU()) for _ in range(edge_types - 1)])
        self.decoder = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim), nn.ReLU(),
                                     nn.Linear(latent_dim, latent_dim))
        self.predict = nn.Linear(latent_dim, latent_dim)
        self.output = nn.Linear(latent_dim, output_dim)
        self.temperature, self.kl_weight, self.prediction_weight = temperature, kl_weight, prediction_weight
        self.auxiliary_loss = torch.tensor(0.)
        self.last_diagnostics = {}

    def decode(self, nodes):
        count = nodes.shape[0]
        pairs = (~torch.eye(count, device=nodes.device, dtype=torch.bool)).nonzero()
        if not pairs.numel():
            return nodes + self.decoder(torch.cat((nodes, torch.zeros_like(nodes)), -1)), None
        sender, receiver = pairs.unbind(-1)
        encoded = self.node_encoder(nodes)
        edge = self.edge_encoder(torch.cat((encoded[sender], encoded[receiver]), -1))
        aggregate = torch.zeros_like(encoded).index_add(0, receiver, edge) / (count - 1)
        factored = self.node_factor(aggregate)
        logits = self.posterior(torch.cat((factored[sender], factored[receiver], edge), -1))
        posterior = logits.softmax(-1)
        categories = (F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
                      if self.training else posterior)
        pair_state = torch.cat((nodes[sender], nodes[receiver]), -1)
        messages = sum(message(pair_state) * categories[:, k + 1:k + 2]
                       for k, message in enumerate(self.messages))
        incoming = torch.zeros_like(nodes).index_add(0, receiver, messages) / (count - 1)
        decoded = nodes + self.decoder(torch.cat((nodes, incoming), -1))
        return decoded, (pairs, posterior)

    def forward(self, local, base, gap, availability, umask):
        inputs, valid = _inputs(local, base, gap, availability, umask)
        nodes = self.inject(inputs)
        result = local.new_zeros(*local.shape[:2], self.output.out_features)
        predictions = local.new_zeros(local.shape)
        kl_terms, records = [], []
        for b in range(local.shape[1]):
            indices = valid[:, b].nonzero().flatten()
            for n, t in enumerate(indices):
                prefix = indices[:n + 1]
                decoded, edges = self.decode(nodes[prefix, b])
                result[t, b] = self.output(decoded[-1])
                predictions[t, b] = self.predict(decoded[-1])
                if edges is not None:
                    pairs, posterior = edges
                    kl_terms.append((posterior * (posterior.clamp_min(1e-8).log()
                                    + math.log(posterior.shape[-1]))).sum(-1).mean())
                    records.append({"batch": b, "time": int(t), "nodes": prefix.detach(),
                                    "pairs": pairs.detach(), "posterior": posterior.detach()})
        # Future observations enter the TRAIN loss only, never the readout.
        target_mask = valid[:-1] & valid[1:]
        prediction_loss = (F.mse_loss(predictions[:-1][target_mask], local[1:][target_mask].detach())
                           if self.training and bool(target_mask.any()) else result.sum() * 0)
        kl = torch.stack(kl_terms).mean() if kl_terms else result.sum() * 0
        self.auxiliary_loss = self.prediction_weight * prediction_loss + self.kl_weight * kl
        self.last_diagnostics = {"edges": records, "prediction_loss": prediction_loss.detach(),
                                 "kl": kl.detach()}
        return result


class PrototypeClassifier(nn.Module):
    """Class-specific prototype classification with actual TRAIN exemplar push.

    Input [N,D] treats each utterance as one feature patch. For MOSI this must
    be run as binary sentiment classification with a matched binary baseline.
    """
    def __init__(self, input_dim=1600, prototype_dim=256, num_classes=2,
                 prototypes_per_class=10):
        super().__init__()
        if num_classes < 2 or prototypes_per_class < 1:
            raise ValueError("prototype classification requires at least two classes")
        self.features = nn.Sequential(nn.Linear(input_dim, prototype_dim), nn.ReLU(),
                                      nn.Linear(prototype_dim, prototype_dim), nn.Sigmoid())
        count = num_classes * prototypes_per_class
        self.prototype_vectors = nn.Parameter(torch.rand(count, prototype_dim))
        identity = torch.arange(num_classes).repeat_interleave(prototypes_per_class)
        self.register_buffer("prototype_class", identity)
        self.register_buffer("projected", torch.zeros(count, dtype=torch.bool))
        self.last_layer = nn.Linear(count, num_classes, bias=False)
        with torch.no_grad():
            self.last_layer.weight.fill_(-.5)
            self.last_layer.weight[identity, torch.arange(count)] = 1.
        self.projection_records = []

    def forward(self, inputs):
        features = self.features(inputs)
        distances = (features[:, None] - self.prototype_vectors[None]).square().sum(-1)
        similarities = torch.log((distances + 1) / (distances + 1e-4))
        return self.last_layer(similarities), distances

    def loss(self, logits, distances, labels):
        correct = labels[:, None] == self.prototype_class[None]
        cluster = distances.masked_fill(~correct, torch.inf).min(-1).values.mean()
        separation = distances.masked_fill(correct, torch.inf).min(-1).values.mean()
        wrong = torch.arange(logits.shape[-1], device=logits.device)[:, None] != self.prototype_class
        l1 = (self.last_layer.weight * wrong).abs().sum()
        return F.cross_entropy(logits, labels) + .8 * cluster - .08 * separation + 1e-4 * l1

    @torch.no_grad()
    def project_train_exemplars(self, batches, *, split):
        """batches yield (input[N,D], labels[N], stable training exemplar IDs).

        The caller must supply TRAIN-only batches under the current encoder.
        Projection is atomic and fails if any class has no eligible exemplar.
        """
        if split != "train":
            raise ValueError("prototype projection is permitted on TRAIN only")
        best = self.prototype_vectors.new_full((len(self.prototype_vectors),), torch.inf)
        vectors = torch.empty_like(self.prototype_vectors)
        records = [None] * len(best)
        was_training = self.training
        self.eval()
        try:
            for inputs, labels, identifiers in batches:
                if len(identifiers) != len(inputs):
                    raise ValueError("every training exemplar requires an ID")
                features = self.features(inputs)
                distance = (features[:, None] - self.prototype_vectors[None]).square().sum(-1)
                eligible = labels[:, None] == self.prototype_class[None]
                values, rows = distance.masked_fill(~eligible, torch.inf).min(0)
                for p in (values < best).nonzero().flatten().tolist():
                    best[p], vectors[p] = values[p], features[rows[p]]
                    records[p] = {"prototype": p, "class": int(self.prototype_class[p]),
                                  "exemplar_id": str(identifiers[int(rows[p])]),
                                  "distance_before_projection": float(values[p]), "split": "train"}
            if not bool(torch.isfinite(best).all()):
                raise ValueError("missing training exemplars for at least one prototype class")
            self.prototype_vectors.copy_(vectors)
            self.projected.fill_(True)
            self.projection_records = records
        finally:
            self.train(was_training)
        return records


def build_readout(method, latent_dim=256, context_dim=1024, output_dim=1600, **kwargs):
    methods = {"C13": DEQReadout, "deq": DEQReadout,
               "C14": UniversalACTReadout, "universal_act": UniversalACTReadout,
               "C16": NRIReadout, "nri": NRIReadout}
    if method not in methods:
        raise ValueError(f"{method} is not a readout; C17 uses PrototypeClassifier")
    return methods[method](latent_dim, context_dim, output_dim, **kwargs)
