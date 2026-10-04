"""Five-role OSRAM adapters, not reproductions of the source task models.

PrediNet: Shanahan et al. §2, https://proceedings.mlr.press/v119/shanahan20a.html
Soft MoE: Puigcerver et al. §2.1, https://arxiv.org/abs/2308.00951
SCL: Wu et al. §3.1, https://arxiv.org/abs/2007.04212
ESBN: Webb et al. Algorithm 1, https://arxiv.org/abs/2012.14601
No auxiliary objectives,
persistent sample memory, or changes to OSRAM's query/write/Flat operations.
"""
import torch
from torch import nn
from torch.nn import functional as F


def safe_mask(values, mask):
    return torch.where(mask[..., None], values, torch.zeros_like(values))


def masked_softmax(logits, mask, dim):
    weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)


def mlp(input_dim, output_dim, hidden=64):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim))


class _Adapter(nn.Module):
    """Local is always present; active is authoritative for the four evidence roles."""
    width = 64

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        if min(latent_dim, num_heads, value_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.local_in = nn.Linear(latent_dim, self.width)
        self.evidence_in = nn.Linear(num_heads * value_dim, self.width)
        self.role = nn.Parameter(torch.randn(5, self.width) * 0.02)
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
        x = safe_mask(x + self.role + self.availability_in(availability.to(x.dtype))[:, None], mask)
        z = self.core(x, mask)
        delta = torch.stack([head(z[:, i + 1]) for i, head in enumerate(self.evidence_out)], dim=1)
        return local + self.local_out(z[:, 0]), safe_mask(clean + delta, active)


class PrediNet(_Adapter):
    """Four ordered entity pairs; shared 16-predicate map; both argument IDs retained."""
    def __init__(self, *args):
        super().__init__(*args)
        self.heads = 4
        self.keys = nn.Linear(self.width, 16, bias=False)
        self.queries = nn.Linear(5 * self.width, 2 * self.heads * 16)
        self.predicate = nn.Linear(self.width, 16, bias=False)
        self.records = mlp(self.heads * (16 + 2 * 5), 5 * self.width)
        self.register_buffer("role_ids", torch.eye(5), persistent=False)

    def core(self, x, mask):
        q = self.queries(x.flatten(1)).reshape(x.shape[0], 2, self.heads, 16)
        logits = torch.einsum("nphd,nrd->nphr", q, self.keys(x)) / 4.0
        weights = masked_softmax(logits, mask[:, None, None, :], -1)
        entities = torch.einsum("nphr,nrd->nphd", weights, x)
        predicates = self.predicate(entities[:, 0]) - self.predicate(entities[:, 1])
        ids = torch.einsum("nphr,ri->nphi", weights, self.role_ids.to(x.dtype))
        record = torch.cat((predicates, ids[:, 0], ids[:, 1]), dim=-1)
        return self.records(record.flatten(1)).reshape(x.shape[0], 5, self.width)


class SoftMoE(_Adapter):
    """Four independent experts, one slot each, with tied dual-axis routing logits."""
    def __init__(self, *args):
        super().__init__(*args)
        self.slots = nn.Parameter(torch.randn(4, self.width) * 0.02)
        self.logit_scale = nn.Parameter(torch.ones(()))
        self.experts = nn.ModuleList(mlp(self.width, self.width, 128) for _ in range(4))

    def routing(self, x, mask):
        logits = self.logit_scale * torch.einsum(
            "nrd,ed->nre", F.normalize(x, dim=-1), F.normalize(self.slots, dim=-1))
        dispatch = masked_softmax(logits, mask[:, :, None], 1)
        combine = safe_mask(logits.softmax(-1), mask)
        return dispatch, combine

    def core(self, x, mask):
        dispatch, combine = self.routing(x, mask)
        slots = torch.einsum("nre,nrd->ned", dispatch, x)
        outputs = torch.stack([expert(slots[:, i]) for i, expert in enumerate(self.experts)], dim=1)
        return safe_mask(torch.einsum("nre,ned->nrd", combine, outputs), mask)


class SCL(_Adapter):
    """4 facets × 4 shared attribute functions × 4 shared relation functions.

    Facets are learned feature groups, not image objects. Each relation sees all
    five ordered roles for one facet/attribute combination plus their masks.
    Every Cartesian composition is retained; no symbolic supervision is used.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.facets = nn.Linear(self.width, 4 * 16)
        self.attributes = nn.ModuleList(mlp(16, 8, 32) for _ in range(4))
        self.relations = nn.ModuleList(mlp(5 * 8 + 5, 8, 32) for _ in range(4))
        self.records = mlp(4 * 4 * 4 * 8, 5 * self.width, 128)

    def core(self, x, mask):
        facets = self.facets(x).reshape(x.shape[0], 5, 4, 16)
        # Each attribute network is shared over all role/facet groups.
        attributes = torch.stack([attribute(facets) for attribute in self.attributes], dim=3)
        attributes = torch.where(mask[:, :, None, None, None], attributes, torch.zeros_like(attributes))
        groups = attributes.permute(0, 2, 3, 1, 4).reshape(x.shape[0], 4, 4, 5 * 8)
        role_mask = mask[:, None, None, :].expand(-1, 4, 4, -1).to(x.dtype)
        groups = torch.cat((groups, role_mask), dim=-1)
        # Each relation network is shared over every facet/attribute group.
        records = torch.stack([relation(groups) for relation in self.relations], dim=3)
        return self.records(records.flatten(1)).reshape(x.shape[0], 5, self.width)


class ESBN(_Adapter):
    """64-wide controller, 32-wide abstract keys; at most five temporary bindings.

    Controller inputs are only retrieved keys plus confidence, never sensory
    vectors. Source order is Local, Base, Gap1, Gap2, Gap3; masked steps are no-ops.
    Optional source temporal normalization is omitted, avoiding count-one issues.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.controller = nn.LSTMCell(33, self.width)
        self.write_key = nn.Linear(self.width, 32)
        self.read_gate = nn.Linear(self.width, 1)
        self.confidence_log_gain = nn.Parameter(torch.zeros(()))
        self.confidence_bias = nn.Parameter(torch.zeros(()))
        self.records = mlp(self.width, 5 * self.width)

    def core(self, x, mask):
        h = x.new_zeros(x.shape[0], self.width)
        c = torch.zeros_like(h)
        read = x.new_zeros(x.shape[0], 33)
        # These lists are local to a forward, never buffers or sample caches.
        keys, values, valid = [], [], []
        for role in range(5):
            keep = mask[:, role, None]
            new_h, new_c = self.controller(read, (h, c))
            h, c = torch.where(keep, new_h, h), torch.where(keep, new_c, c)
            key = F.relu(self.write_key(h))
            if keys:
                memory_keys = torch.stack(keys, dim=1)
                memory_values = torch.stack(values, dim=1)
                memory_mask = torch.stack(valid, dim=1)
                similarity = torch.einsum("nmd,nd->nm", memory_values, x[:, role])
                weights = masked_softmax(similarity, memory_mask, 1)
                confidence = torch.sigmoid(self.confidence_log_gain.exp() * similarity + self.confidence_bias)
                symbolic_records = torch.cat((memory_keys, confidence[..., None]), dim=-1)
                new_read = self.read_gate(h).sigmoid() * torch.einsum("nm,nmd->nd", weights, symbolic_records)
            else:
                new_read = torch.zeros_like(read)
            read = torch.where(keep, new_read, read)
            keys.append(torch.where(keep, key, torch.zeros_like(key)))
            values.append(x[:, role])
            valid.append(mask[:, role])
        # Consume the last value-driven lookup, as in the source's extra step.
        h, _ = self.controller(read, (h, c))
        return self.records(h).reshape(x.shape[0], 5, self.width)


METHODS = {
    "routing_predinet_bound_predicates": PrediNet,
    "routing_soft_moe_dispatch_expert_combine": SoftMoE,
    "routing_scl_shared_compositional_maps": SCL,
    "routing_esbn_ephemeral_symbol_binding": ESBN,
}


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    if method not in METHODS:
        raise ValueError(f"Unknown routing method: {method}")
    return METHODS[method](latent_dim, num_heads, value_dim)
