"""Source-grounded storage cores adapted to OSRAM's causal K/V interface.

These replace the scan storage, not the Local/Flat or task heads. Source pins,
equations and adaptations are recorded in docs/osram_core20_20261005/storage.json.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ('audio', 'text', 'visual')


class SequenceStorage(nn.Module):
    def __init__(self, num_heads, key_dim, value_dim, latent_dim):
        super().__init__()
        self.num_heads, self.key_dim, self.value_dim = num_heads, key_dim, value_dim
        self.latent_dim = latent_dim
        self.auxiliary_loss = torch.tensor(0.)

    def scan(self, keys, values, queries, availability, valid, *, address_residual=None):
        length, batch = valid.shape
        state = self.initial_state(queries, batch)
        base, gaps = [], []
        diagnostics = {m: {metric: [] for metric in ('rho', 'eta', 'cosine')} for m in MODALITIES}
        self.auxiliary_loss = queries.new_zeros(())
        self.compression_events = 0
        for t in range(length):
            active = valid[t].bool()
            if not bool(active.any()):
                base.append(queries.new_zeros(batch, self.num_heads * self.value_dim))
                gaps.append(queries.new_zeros(batch, 3, self.num_heads * self.value_dim))
                continue
            observed = availability[t].bool() & active[:, None]
            # Mask before any projection, product, controller or norm operation.
            k = torch.stack([torch.where(observed[:, i, None, None], keys[m][t], 0.)
                             for i, m in enumerate(MODALITIES)], -1)
            v = torch.stack([torch.where(observed[:, i, None, None], values[m][t], 0.)
                             for i, m in enumerate(MODALITIES)], -1)
            q = torch.where(active[:, None, None, None], queries[t], 0.)
            read_queries = [q[:, 0]]
            for i, m in enumerate(MODALITIES):
                raw = q[:, i + 1]
                residual = raw if address_residual is None else address_residual(k, raw)
                norm = raw.norm(dim=-1).clamp_min(1e-8)
                rho = residual.norm(dim=-1) / norm
                cosine = F.cosine_similarity(raw, residual, dim=-1, eps=1e-8)
                for metric, x in (('rho', rho), ('eta', 1 - cosine * rho), ('cosine', cosine)):
                    diagnostics[m][metric].extend(x[active].detach().cpu().reshape(-1).tolist())
                read_queries.append(residual)
            reads, state = self.read(state, torch.stack(read_queries, 1), active)
            reads = torch.where(active[:, None, None, None], reads, 0.)
            base.append(reads[:, 0].reshape(batch, -1))
            gap = reads[:, 1:].reshape(batch, 3, -1)
            gaps.append(torch.where((~availability[t].bool() & active[:, None])[:, :, None], gap, 0.))
            state = self.write(state, k, v, observed, active, q)
        if not length:
            return (queries.new_zeros(0, batch, self.num_heads * self.value_dim),
                    queries.new_zeros(0, batch, 3, self.num_heads * self.value_dim), diagnostics)
        return torch.stack(base), torch.stack(gaps), diagnostics


class DNCStorage(SequenceStorage):
    """DNC controller + content/allocation writes + free gates + temporal reads.

    One independent DNC per OSRAM head, four target-specific read heads, one
    observed-only pooled write. Memory words contain both supplied keys/values.
    """
    def __init__(self, num_heads, key_dim, value_dim, latent_dim):
        super().__init__(num_heads, key_dim, value_dim, latent_dim)
        self.slots = 16
        self.width = key_dim + value_dim
        self.controller = nn.GRUCell(self.width * 2, latent_dim)
        self.interface = nn.Linear(latent_dim, 2 * self.width + 7)
        self.read_interface = nn.Linear(key_dim, self.width + 4)

    @staticmethod
    def allocation(usage):
        u = 1e-6 + (1 - 1e-6) * usage
        sorted_u, order = u.sort(dim=-1, stable=True)
        prefix = torch.cat([torch.ones_like(sorted_u[..., :1]), sorted_u[..., :-1].cumprod(-1)], -1)
        return torch.zeros_like(u).scatter(-1, order, (1 - sorted_u) * prefix)

    @staticmethod
    def usage_update(usage, previous_write, free, previous_read):
        written = previous_write.detach()
        return (usage + (1 - usage) * written) * (1 - free[..., None] * previous_read).prod(-2)

    @staticmethod
    def link_update(link, precedence, write):
        result = ((1 - write[..., :, None] - write[..., None, :]) * link
                  + write[..., :, None] * precedence[..., None, :])
        identity = torch.eye(link.shape[-1], dtype=torch.bool, device=link.device)
        result = torch.where(identity, 0., result)
        return result, (1 - write.sum(-1, keepdim=True)) * precedence + write

    @staticmethod
    def content(memory, key, strength):
        similarity = F.normalize(key, dim=-1) @ F.normalize(memory, dim=-1).transpose(-1, -2)
        return (similarity * (1 + F.softplus(strength))[..., None]).softmax(-1)

    def initial_state(self, reference, batch):
        n = batch * self.num_heads
        z = reference.new_zeros
        return dict(memory=z(n, self.slots, self.width), usage=z(n, self.slots),
                    write=z(n, self.slots), read=z(n, 4, self.slots),
                    link=z(n, self.slots, self.slots), precedence=z(n, self.slots),
                    hidden=z(n, self.latent_dim), words=z(n, 4, self.width))

    def read(self, state, queries, active):
        batch = queries.shape[0]
        q = queries.permute(0, 2, 1, 3).reshape(-1, 4, self.key_dim)
        interface = self.read_interface(q)
        content = self.content(state['memory'], interface[..., :self.width], interface[..., self.width])
        mode = interface[..., self.width + 1:].softmax(-1)
        forward = state['read'] @ state['link'].transpose(-1, -2)
        backward = state['read'] @ state['link']
        weights = mode[..., :1] * backward + mode[..., 1:2] * forward + mode[..., 2:] * content
        words = weights @ state['memory']
        # Freeing uses previous read weights, so preserve until the write phase.
        state = dict(state, next_read=weights, next_words=words)
        output = words[..., self.key_dim:].reshape(batch, self.num_heads, 4, self.value_dim).permute(0, 2, 1, 3)
        return torch.where(active[:, None, None, None], output, 0.), state

    def write(self, state, keys, values, observed, active, queries):
        batch = keys.shape[0]
        count = observed.sum(-1).clamp_min(1).to(keys.dtype)
        word = torch.cat([keys.sum(-1), values.sum(-1)], -1) / count[:, None, None]
        word = word.reshape(-1, self.width)
        hidden = self.controller(torch.cat([word, state['words'].mean(1)], -1), state['hidden'])
        control = self.interface(hidden)
        erase = control[:, :self.width].sigmoid()
        vector = control[:, self.width:2 * self.width]
        free = control[:, 2 * self.width:2 * self.width + 4].sigmoid()
        allocation_gate, write_gate = control[:, -3].sigmoid(), control[:, -2].sigmoid()
        usage = self.usage_update(state['usage'], state['write'], free, state['read'])
        content = self.content(state['memory'], word[:, None], control[:, -1:]).squeeze(1)
        write = write_gate[:, None] * (allocation_gate[:, None] * self.allocation(usage)
                                      + (1 - allocation_gate[:, None]) * content)
        enabled = observed.any(-1)[:, None].expand(-1, self.num_heads).reshape(-1)
        write = torch.where(enabled[:, None], write, 0.)
        memory = state['memory'] * (1 - write[:, :, None] * erase[:, None]) + write[:, :, None] * vector[:, None]
        link, precedence = self.link_update(state['link'], state['precedence'], write)
        candidate = dict(memory=memory, usage=usage, write=write, read=state['next_read'],
                         link=link, precedence=precedence, hidden=hidden, words=state['next_words'])
        # No observation means no write/controller/freeing update; valid queries
        # may still advance DNC read-head addresses through existing links.
        result = {}
        valid_head = active[:, None].expand(-1, self.num_heads).reshape(-1)
        for name, value in candidate.items():
            update = valid_head if name in ('read', 'words') else enabled
            result[name] = torch.where(update.reshape(-1, *([1] * (value.ndim - 1))), value, state[name])
        return result


class CompressiveStorage(SequenceStorage):
    """Exact FIFO, convolution-compressed FIFO, attention reconstruction loss."""
    def __init__(self, num_heads, key_dim, value_dim, latent_dim):
        super().__init__(num_heads, key_dim, value_dim, latent_dim)
        self.short_size, self.long_size, self.rate = 8, 16, 2
        self.compressor = nn.Conv1d(key_dim + value_dim, key_dim + value_dim, self.rate, stride=self.rate, bias=False)
        self.position_key = nn.Linear(key_dim, key_dim, bias=False)
        self.content_bias = nn.Parameter(torch.zeros(num_heads, key_dim))
        self.position_bias = nn.Parameter(torch.zeros(num_heads, key_dim))

    def initial_state(self, reference, batch):
        return [dict(short=[], long=[], pending=[], short_positions=[], long_positions=[],
                     pending_positions=[], clock=0) for _ in range(batch)]

    def attend(self, query, words, positional=True, positions=None):
        # query [R,H,K], words [N,H,K+V].
        keys, values = words[..., :self.key_dim], words[..., self.key_dim:]
        content_query = query + self.content_bias if positional else query
        logits = torch.einsum('rhk,nhk->rhn', content_query, keys) / math.sqrt(self.key_dim)
        if positional:
            if positions is None:
                positions = torch.arange(words.shape[0], 0, -1, device=query.device, dtype=query.dtype)
            frequency = torch.exp(torch.arange(self.key_dim, device=query.device, dtype=query.dtype)
                                  * (-math.log(10000.) / self.key_dim))
            angles = positions[:, None] * frequency[None]
            encoding = torch.where(torch.arange(self.key_dim, device=query.device) % 2 == 0,
                                   angles.sin(), angles.cos())
            logits = logits + torch.einsum('rhk,nk->rhn', query + self.position_bias,
                                          self.position_key(encoding)) / math.sqrt(self.key_dim)
        return torch.einsum('rhn,nhv->rhv', logits.softmax(-1), values)

    def read(self, state, queries, active):
        outputs = []
        for b, memory in enumerate(state):
            # An incomplete compression block remains exactly readable.
            words = memory['long'] + memory['pending'] + memory['short']
            ages = memory['long_positions'] + memory['pending_positions'] + memory['short_positions']
            positions = queries.new_tensor([memory['clock'] + 1 - position for position in ages])
            outputs.append(self.attend(queries[b], torch.stack(words), positions=positions) if bool(active[b]) and words
                           else queries[b].new_zeros(4, self.num_heads, self.value_dim))
        return torch.stack(outputs), state

    def write(self, state, keys, values, observed, active, queries):
        for b, memory in enumerate(state):
            for m in range(3):
                if not bool(observed[b, m]):
                    continue
                memory['short'].append(torch.cat([keys[b, ..., m], values[b, ..., m]], -1))
                memory['clock'] += 1
                memory['short_positions'].append(memory['clock'])
                if len(memory['short']) > self.short_size:
                    memory['pending'].append(memory['short'].pop(0))
                    memory['pending_positions'].append(memory['short_positions'].pop(0))
                # Two compressed tokens keep attention reconstruction
                # query-dependent and supervise both compressed K and V.
                if len(memory['pending']) == 2 * self.rate:
                    old = torch.stack(memory['pending']).detach()
                    compressed = self.compressor(old.permute(1, 2, 0)).permute(2, 0, 1)
                    # Frozen targets/queries, train only the compressor in this loss.
                    query = queries[b].detach()
                    target = self.attend(query, old, positional=False).detach()
                    recovered = self.attend(query, compressed, positional=False)
                    self.auxiliary_loss = self.auxiliary_loss + F.mse_loss(recovered, target)
                    self.compression_events += 1
                    memory['long'].extend(compressed.unbind(0))
                    for start in range(0, len(memory['pending_positions']), self.rate):
                        positions = memory['pending_positions'][start:start + self.rate]
                        memory['long_positions'].append(sum(positions) / self.rate)
                    memory['long'] = memory['long'][-self.long_size:]
                    memory['long_positions'] = memory['long_positions'][-self.long_size:]
                    memory['pending'] = []
                    memory['pending_positions'] = []
        return state

    def scan(self, *args, **kwargs):
        result = super().scan(*args, **kwargs)
        if self.compression_events:
            self.auxiliary_loss = self.auxiliary_loss / self.compression_events
        return result


class HiPPOStorage(SequenceStorage):
    """Orthogonal LegS projections with bilinear discretization, observed clocks.

    Key/value coefficient inner products implement a query-conditioned history
    correlation in the orthonormal Legendre basis (Parseval), without a FIFO.
    """
    def __init__(self, num_heads, key_dim, value_dim, latent_dim):
        super().__init__(num_heads, key_dim, value_dim, latent_dim)
        self.order = min(16, max(2, latent_dim // num_heads))
        self.key_map = nn.Linear(key_dim, key_dim, bias=False)
        self.value_map = nn.Linear(value_dim, value_dim, bias=False)
        nn.init.eye_(self.key_map.weight)
        nn.init.eye_(self.value_map.weight)

    def discretize(self, clock, dtype, device):
        n = torch.arange(self.order, dtype=dtype, device=device)
        root = (2 * n + 1).sqrt()
        a = -torch.tril(root[:, None] * root[None, :]) + torch.diag(n)
        at = a / clock[..., None, None].clamp_min(1)
        bt = root / clock[..., None].clamp_min(1)
        identity = torch.eye(self.order, dtype=dtype, device=device)
        system = identity - at / 2
        return torch.linalg.solve(system, identity + at / 2), torch.linalg.solve(system, bt[..., None]).squeeze(-1)

    def initial_state(self, reference, batch):
        return (reference.new_zeros(batch, 3, self.num_heads, self.key_dim + self.value_dim, self.order),
                reference.new_zeros(batch, 3))

    def read(self, state, queries, active):
        coefficients, clock = state
        keys, values = coefficients[..., :self.key_dim, :], coefficients[..., self.key_dim:, :]
        q = self.key_map(queries)
        weights = torch.einsum('brhk,bmhkn->brmhn', q, keys)
        result = torch.einsum('brmhn,bmhvn->brhv', weights, values) / math.sqrt(self.key_dim)
        result = result / (clock > 0).sum(-1).clamp_min(1)[:, None, None, None]
        return result, state

    def write(self, state, keys, values, observed, active, queries):
        coefficients, clock = state
        next_clock = clock + observed.to(clock.dtype)
        a, b = self.discretize(next_clock, keys.dtype, keys.device)
        k = self.key_map(keys.permute(0, 3, 1, 2))
        v = self.value_map(values.permute(0, 3, 1, 2))
        signal = torch.cat([k, v], -1)
        candidate = torch.einsum('bmnk,bmhwk->bmhwn', a, coefficients) + signal[..., None] * b[:, :, None, None]
        updated = torch.where(observed[:, :, None, None, None], candidate, coefficients)
        return updated, next_clock


def build(method, num_heads=8, key_dim=64, value_dim=64, latent_dim=256):
    implementations = {'C01': DNCStorage, 'C02': CompressiveStorage, 'C04': HiPPOStorage}
    if method not in implementations:
        raise ValueError(f'Unsupported storage backend: {method}')
    return implementations[method](num_heads, key_dim, value_dim, latent_dim)
