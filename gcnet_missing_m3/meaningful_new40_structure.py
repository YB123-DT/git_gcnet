"""Current-read graph, cell, partition and tree inference adapters.

All topology is derived from compact identity-bearing tokens. No inference state
survives a forward call and all writes pass through zero-initialized bridges.
"""
import itertools
import math

import torch
from torch import nn

from .meaningful_blocks_common import active_groups, safe_mask
from .meaningful_input_new40 import TokenAdapter, zero_linear


def mlp(inputs, outputs=64):
    return nn.Sequential(nn.Linear(inputs, outputs), nn.Tanh(), nn.Linear(outputs, outputs))


def topology(columns, heads, dtype):
    role = torch.where(columns == 0, -1, (columns - 1) // heads)
    head = (columns - 1) % heads
    adj = ((role[:, None] == role[None, :]) |
           (head[:, None] == head[None, :]) |
           (columns[:, None] == 0) | (columns[None, :] == 0))
    adj.fill_diagonal_(False)
    return adj.to(dtype)


class GraphConv(nn.Module):
    def __init__(self, inputs=64, outputs=64):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.norm = nn.LayerNorm(outputs)

    def forward(self, x, a):
        a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
        return torch.tanh(self.norm(self.linear(a @ x)))


class MatrixTree(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = nn.Linear(64, 32)
        self.child = nn.Linear(64, 32)
        self.root = nn.Linear(64, 1)
        self.root_content = nn.Parameter(torch.zeros(64))
        self.content = nn.Linear(64, 64)
        self.update = mlp(256)

    @staticmethod
    def marginals(scores, root_scores):
        """Single-root directed spanning tree marginals, parent x child."""
        n = scores.shape[-1]
        if n == 1:
            return torch.zeros_like(scores), torch.ones_like(root_scores)
        eye = torch.eye(n, device=scores.device, dtype=torch.bool)
        scores = scores.double().masked_fill(eye, -torch.inf)
        root_scores = root_scores.double()
        shift = torch.maximum(scores.amax(-2), root_scores)
        arcs = (scores - shift.unsqueeze(-2)).exp()
        roots = (root_scores - shift).exp()
        lap = torch.diag_embed(arcs.sum(-2)) - arcs
        matrix = torch.cat((roots.unsqueeze(-2), lap[..., 1:, :]), -2)
        inverse = torch.linalg.solve(matrix, torch.eye(n, device=scores.device,
                                                     dtype=torch.float64).expand_as(matrix))
        root_marg = roots * inverse[..., :, 0]
        # Derivative of log det: dL[j,j]/dw_ij - dL[i,j]/dw_ij;
        # the replaced row has no arc contribution.
        diagonal = inverse.diagonal(dim1=-2, dim2=-1).clone()
        diagonal[..., 0] = 0
        off = inverse.transpose(-1, -2).clone()
        off[..., 0, :] = 0
        arc_marg = arcs * (diagonal.unsqueeze(-2) - off)
        return arc_marg, root_marg

    def forward(self, x, columns, heads):
        scores = self.parent(x) @ self.child(x).transpose(-1, -2) / math.sqrt(32)
        arcs, roots = self.marginals(scores, self.root(x).squeeze(-1))
        arcs, roots = arcs.to(x.dtype), roots.to(x.dtype)
        content = self.content(x)
        return self.update(torch.cat((content, arcs.transpose(-1, -2) @ content,
                                      arcs @ content, roots[..., None] * self.root_content), -1))


class RoleAdapter(nn.Module):
    def __init__(self, core, latent_dim, heads, value_dim):
        super().__init__()
        self.core = core
        self.encoders = nn.ModuleList([nn.Linear(latent_dim, 64)] +
                                     [nn.Linear(heads * value_dim, 64) for _ in range(4)])
        self.role = nn.Embedding(5, 64)
        self.decoders = nn.ModuleList([zero_linear(64, latent_dim)] +
                                     [zero_linear(64, heads * value_dim) for _ in range(4)])

    def forward(self, local, evidence, active, availability):
        mask = torch.cat((active.any(-1, keepdim=True), active.bool()), -1)
        values = [local] + list(evidence.unbind(1))
        tokens = torch.stack([encoder(safe_mask(value, mask[:, i])) + self.role.weight[i]
                              for i, (encoder, value) in enumerate(zip(self.encoders, values))], 1)
        output = torch.zeros_like(tokens)
        for rows, columns, packed in active_groups(tokens, mask):
            output[rows[:, None], columns[None, :]] = self.core(packed)
        delta = [safe_mask(decoder(output[:, i]), mask[:, i])
                 for i, decoder in enumerate(self.decoders)]
        return local + delta[0], safe_mask(evidence + torch.stack(delta[1:], 1), active)


def partitions(n):
    """Restricted growth strings enumerate each set partition exactly once."""
    result = [(0,)]
    for _ in range(1, n):
        result = [p + (j,) for p in result for j in range(max(p) + 2)]
    return result


class SparsePartition(nn.Module):
    def __init__(self):
        super().__init__()
        self.affinity = mlp(128, 1)
        self.message = mlp(128)
        self.update = mlp(128)

    @staticmethod
    def marginal(eta, features):
        """Wolfe active-set projection, followed by differentiable support KKT.

        Only u=M p is consumed; no penalty is imposed on the nonunique p.
        Support discovery is piecewise constant. The final pseudoinverse is
        the reduced-space support derivative, valid away from support changes.
        """
        m = features.double()
        outputs = []
        for target in eta.double():
            with torch.no_grad():
                support = [int((m.T @ target).argmax())]
                p = target.new_ones(1)
                for _ in range(128):
                    u = m[:, support] @ p
                    residual = target - u
                    best = int((m.T @ residual).argmax())
                    if (m[:, best] - u).dot(residual) <= 1e-9:
                        break
                    if best in support:
                        break
                    support.append(best)
                    p = torch.cat((p, p.new_zeros(1)))
                    for _ in range(128):
                        ms = m[:, support]
                        ones = ms.new_ones(len(support), 1)
                        kkt = torch.cat((torch.cat((ms.T @ ms, ones), 1),
                                         torch.cat((ones.T, ms.new_zeros(1, 1)), 1)), 0)
                        rhs = torch.cat((ms.T @ target, target.new_ones(1)))
                        candidate = (torch.linalg.pinv(kkt) @ rhs)[:-1]
                        if candidate.min() >= -1e-10:
                            p = candidate
                            break
                        neg = candidate < 0
                        alpha = (p[neg] / (p[neg] - candidate[neg])).min()
                        p = p + alpha * (candidate - p)
                        keep = p > 1e-10
                        support = [s for s, yes in zip(support, keep.tolist()) if yes]
                        p = p[keep]
                    else:
                        raise RuntimeError('SparseMAP minor cycle did not converge')
                else:
                    raise RuntimeError('SparseMAP major cycle did not converge')
            ms = m[:, support]
            ones = ms.new_ones(len(support), 1)
            kkt = torch.cat((torch.cat((ms.T @ ms, ones), 1),
                             torch.cat((ones.T, ms.new_zeros(1, 1)), 1)), 0)
            rhs = torch.cat((ms.T @ target, target.new_ones(1)))
            outputs.append(ms @ (torch.linalg.pinv(kkt) @ rhs)[:-1])
        return torch.stack(outputs).to(eta.dtype)

    def forward(self, x):
        b, n, d = x.shape
        pairs = list(itertools.combinations(range(n), 2))
        if not pairs:
            return self.update(torch.cat((x, self.message(torch.cat((x, x), -1))), -1))
        left, right = zip(*pairs)
        pair = torch.cat((x[:, left] + x[:, right], (x[:, left] - x[:, right]).abs()), -1)
        eta = self.affinity(pair).squeeze(-1)
        m = x.new_tensor([[float(p[i] == p[j]) for p in partitions(n)] for i, j in pairs])
        marginal = self.marginal(eta, m)
        u = torch.eye(n, device=x.device, dtype=x.dtype).expand(b, n, n).clone()
        u[:, left, right] = marginal
        u[:, right, left] = marginal
        receivers = x[:, :, None].expand(b, n, n, d)
        senders = x[:, None].expand(b, n, n, d)
        messages = self.message(torch.cat((receivers, senders), -1))
        pooled = (u[..., None] * messages).sum(2) / u.sum(-1, keepdim=True).clamp_min(1e-8)
        return self.update(torch.cat((x, pooled), -1))


class Janossy(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.LSTM(64, 64, batch_first=True)
        self.update = mlp(192)

    def forward(self, x):
        b, n, d = x.shape
        order = torch.tensor(list(itertools.permutations(range(n))), device=x.device)
        sequences = x[:, order].reshape(-1, n, d)
        _, (h, _) = self.sequence(sequences)
        pooled = h[0].reshape(b, len(order), d).mean(1)
        return self.update(torch.cat((x, pooled[:, None].expand_as(x),
                                      x[:, :1].expand_as(x)), -1))


class DiffPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeds = nn.ModuleList([GraphConv() for _ in range(3)])
        self.assign = nn.ModuleList([GraphConv(outputs=4), GraphConv(outputs=2)])
        self.update = mlp(192)

    def forward(self, x, columns, heads):
        a = topology(columns, heads, x.dtype)
        z0 = self.embeds[0](x, a)
        s0 = self.assign[0](x, a).softmax(-1)
        a1 = s0.transpose(-1, -2) @ a @ s0
        x1 = s0.transpose(-1, -2) @ z0
        z1 = self.embeds[1](x1, a1)
        s1 = self.assign[1](x1, a1).softmax(-1)
        a2 = s1.transpose(-1, -2) @ a1 @ s1
        z2 = self.embeds[2](s1.transpose(-1, -2) @ z1, a2)
        return self.update(torch.cat((z0, s0 @ z1, s0 @ s1 @ z2), -1))


def cell_complex(a, squares=False):
    """Complete edge boundaries for all triangles and chordless four-cycles."""
    n = a.shape[-1]
    connected = a.bool().tolist()
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if connected[i][j]]
    lookup = {e: k for k, e in enumerate(edges)}
    cycles = [list(t) for t in itertools.combinations(range(n), 3)
              if all(connected[i][j] for i, j in itertools.combinations(t, 2))]
    if squares:
        for vertices in itertools.combinations(range(n), 4):
            i = vertices[0]
            for tail in itertools.permutations(vertices[1:]):
                cycle = (i,) + tail
                if cycle[1] > cycle[-1]:
                    continue
                if (all(connected[cycle[k]][cycle[(k + 1) % 4]] for k in range(4))
                        and not connected[cycle[0]][cycle[2]]
                        and not connected[cycle[1]][cycle[3]]):
                    cycles.append(cycle)
    b1 = a.new_zeros(n, len(edges))
    b2 = a.new_zeros(len(edges), len(cycles))
    for e, (i, j) in enumerate(edges):
        b1[i, e], b1[j, e] = -1, 1
    for c, cycle in enumerate(cycles):
        for i, j in zip(cycle, list(cycle[1:]) + [cycle[0]]):
            b2[lookup[tuple(sorted((i, j)))], c] = 1 if i < j else -1
    return edges, b1, b2


class Cellular(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = mlp(128)
        self.ring = mlp(64)
        self.updates = nn.ModuleList([nn.ModuleList([mlp(192) for _ in range(3)]) for _ in range(3)])
        self.upper_messages = nn.ModuleList([nn.ModuleList([mlp(128) for _ in range(2)]) for _ in range(3)])
        self.norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(64) for _ in range(3)]) for _ in range(3)])
        self.readout = mlp(192)

    @staticmethod
    def upper(states, cofaces, incidence, message):
        # Shared coface-conditioned message from each distinct upper neighbor.
        node, face = incidence.nonzero(as_tuple=True)
        source = message(torch.cat((states[:, node], cofaces[:, face]), -1))
        face_index = face[None, :, None].expand_as(source)
        node_index = node[None, :, None].expand_as(source)
        totals = torch.zeros_like(cofaces).scatter_add(1, face_index, source)
        return torch.zeros_like(states).scatter_add(1, node_index, totals[:, face] - source)

    def forward(self, x, columns, heads):
        edges, b1, b2 = cell_complex(topology(columns, heads, x.dtype), squares=True)
        i, j = zip(*edges) if edges else ([], [])
        h0 = x
        h1 = self.edge(torch.cat((x[:, i] + x[:, j], (x[:, i] - x[:, j]).abs()), -1))
        c1, c2 = b1.abs(), b2.abs()
        h2 = self.ring(c2.T @ h1)
        for updates, messages, norms in zip(self.updates, self.upper_messages, self.norms):
            upper0 = self.upper(h0, h1, c1, messages[0])
            upper1 = self.upper(h1, h2, c2, messages[1])
            h0, h1, h2 = [norm(update(torch.cat(parts, -1))) for norm, update, parts in zip(
                norms, updates, ((h0, torch.zeros_like(h0), upper0),
                                 (h1, c1.T @ h0, upper1),
                                 (h2, c2.T @ h1, torch.zeros_like(h2))))]
        return self.readout(torch.cat((h0, c1 @ h1, (c1 @ c2) @ h2), -1))


class Hodge(nn.Module):
    def __init__(self):
        super().__init__()
        self.endpoint = mlp(128)
        self.filters = nn.ModuleList([nn.ModuleList([nn.Linear(64, 64, bias=False)
                                                    for _ in range(3)]) for _ in range(3)])
        self.readout = mlp(192)

    def forward(self, x, columns, heads):
        edges, b1, b2 = cell_complex(topology(columns, heads, x.dtype))
        if not edges:
            return self.readout(torch.cat((x, torch.zeros_like(x), torch.zeros_like(x)), -1))
        i, j = zip(*edges)
        e = self.endpoint(torch.cat((x[:, i], x[:, j]), -1)) - self.endpoint(torch.cat((x[:, j], x[:, i]), -1))
        lap = b1.T @ b1 + b2 @ b2.T
        scale = torch.linalg.eigvalsh(lap.double()).amax().to(x.dtype).clamp_min(1e-8)
        operator = 2 * lap / scale - torch.eye(len(edges), device=x.device, dtype=x.dtype)
        for weights in self.filters:
            t1 = operator @ e
            t2 = 2 * operator @ t1 - e
            e = torch.tanh(weights[0](e) + weights[1](t1) + weights[2](t2))
        return self.readout(torch.cat((x, b1 @ e, b1.abs() @ e.abs()), -1))


class NestedGNN(nn.Module):
    def __init__(self, root_aware=False):
        super().__init__()
        self.root_embedding = nn.Embedding(2, 64)
        self.distance = nn.Embedding(2, 64)
        self.layers = nn.ModuleList([mlp(64) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(64) for _ in range(3)])
        self.epsilon = nn.Parameter(torch.zeros(3))
        self.pool = mlp(192)
        self.readout = mlp(192)
        self.root_aware = root_aware
        if root_aware:
            # Preserve both common parameters and subsequent adapter RNG draws.
            with torch.random.fork_rng(devices=[]):
                self.root_pool = nn.Linear(128, 64)

    def forward(self, x, columns, heads):
        a = topology(columns, heads, x.dtype)
        roots = []
        for root in range(x.shape[1]):
            neighbors = (a[root].bool() | (torch.arange(len(columns), device=x.device) == root)).nonzero(as_tuple=True)[0]
            induced = a[neighbors][:, neighbors]
            marker = (neighbors == root).long()
            h = x[:, neighbors] + self.root_embedding(marker) + self.distance(1 - marker)
            states = []
            for layer, norm, eps in zip(self.layers, self.norms, self.epsilon):
                h = norm(layer((1 + eps) * h + induced @ h))
                if self.root_aware:
                    root_hidden = h[:, marker.bool()].squeeze(1)
                    nonroot = h[:, ~marker.bool()]
                    neighbor_hidden = (nonroot.mean(1) if nonroot.shape[1]
                                       else torch.zeros_like(root_hidden))
                    states.append(self.root_pool(torch.cat((root_hidden, neighbor_hidden), -1)))
                else:
                    states.append(h.mean(1))
            roots.append(self.pool(torch.cat(states, -1)))
        rooted = torch.stack(roots, 1)
        return self.readout(torch.cat((x, rooted, rooted.mean(1, keepdim=True).expand_as(x)), -1))


class GraphUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([GraphConv() for _ in range(2)])
        self.score = nn.ParameterList([nn.Parameter(torch.randn(64) / 8) for _ in range(2)])
        self.bottom = GraphConv()
        self.decoder = nn.ModuleList([GraphConv() for _ in range(2)])

    def forward(self, x, columns, heads):
        b = x.shape[0]
        a = topology(columns, heads, x.dtype).expand(b, -1, -1)
        saved = []
        for conv, projection in zip(self.encoder, self.score):
            x = conv(x, a)
            scores = (x @ projection) / projection.norm().clamp_min(1e-8)
            k = max(1, x.shape[1] // 2)
            indices = scores.topk(k, dim=1).indices
            saved.append((x, a, indices))
            x = x.gather(1, indices[..., None].expand(-1, -1, 64)) * scores.gather(1, indices).sigmoid()[..., None]
            augmented = a + torch.eye(a.shape[-1], device=x.device, dtype=x.dtype)
            augmented = (augmented @ augmented > 0).to(x.dtype)
            a = augmented.gather(1, indices[..., None].expand(-1, -1, a.shape[-1]))
            a = a.gather(2, indices[:, None].expand(-1, k, -1))
        x = self.bottom(x, a)
        for conv, (skip, a, indices) in zip(self.decoder, reversed(saved)):
            restored = torch.zeros_like(skip).scatter(1, indices[..., None].expand(-1, -1, 64), x)
            x = conv(restored, a) + skip
        return x


class CRFMeanField(nn.Module):
    def __init__(self):
        super().__init__()
        self.unary = nn.Linear(64, 8)
        self.descriptor = nn.Linear(64, 16)
        self.kernel_weights = nn.Parameter(torch.ones(2))
        self.compatibility = nn.Parameter(torch.eye(8))
        self.readout = mlp(88)

    def forward(self, x, columns, heads):
        unary = self.unary(x)
        descriptor = self.descriptor(x)
        content = torch.exp(-((descriptor[:, :, None] - descriptor[:, None]) ** 2).sum(-1) / 16)
        role = torch.where(columns == 0, 0, 1 + (columns - 1) // heads)
        head = torch.where(columns == 0, heads, (columns - 1) % heads)
        identity = torch.exp(-((role[:, None] != role[None, :]).to(x.dtype) +
                               (head[:, None] != head[None, :]).to(x.dtype)))
        offdiag = 1 - torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        kernel = (self.kernel_weights[0] * content + self.kernel_weights[1] * identity) * offdiag
        compatibility = (self.compatibility + self.compatibility.T) / 2
        initial = unary.softmax(-1)
        q = initial
        for _ in range(5):
            q = (unary - (kernel @ q) @ compatibility.T).softmax(-1)
        return self.readout(torch.cat((x, initial, q, q - initial), -1))


class GraphMatching(nn.Module):
    """Full Base-head / Gap-head graph pairs; Local is conditioning only."""
    def __init__(self, latent_dim, heads, value_dim):
        super().__init__()
        self.heads, self.value_dim = heads, value_dim
        self.local = nn.Linear(latent_dim, 64)
        self.base = nn.Linear(value_dim, 64)
        self.gap = nn.Linear(value_dim, 64)
        self.head = nn.Embedding(heads, 64)
        self.role = nn.Embedding(3, 64)
        self.message = mlp(130)
        self.gru = nn.GRUCell(192, 64)
        self.decoder = zero_linear(64, value_dim)

    def within(self, x):
        b, n, d = x.shape
        sender = x[:, None].expand(b, n, n, d)
        receiver = x[:, :, None].expand(b, n, n, d)
        eye = torch.eye(n, device=x.device, dtype=x.dtype)
        identity = torch.stack((eye, 1 - eye), -1).expand(b, -1, -1, -1)
        msg = self.message(torch.cat((receiver, sender, identity), -1))
        return (msg * (1 - eye)[None, :, :, None]).sum(2)

    def forward(self, local, evidence, active, availability):
        clean = safe_mask(evidence, active)
        correction = torch.zeros_like(evidence)
        count = torch.zeros_like(active[:, :1], dtype=evidence.dtype)
        for gap in range(1, 4):
            rows = (active[:, 0].bool() & active[:, gap].bool()).nonzero(as_tuple=True)[0]
            if not rows.numel():
                continue
            condition = self.local(local[rows])[:, None].expand(-1, self.heads, -1)
            base = self.base(clean[rows, 0].reshape(-1, self.heads, self.value_dim)) + self.head.weight + condition
            other = self.gap(clean[rows, gap].reshape(-1, self.heads, self.value_dim)) + self.head.weight + condition + self.role.weight[gap - 1]
            for _ in range(4):
                similarity = base @ other.transpose(-1, -2) / 8
                base_delta = base - similarity.softmax(-1) @ other
                gap_delta = other - similarity.transpose(-1, -2).softmax(-1) @ base
                base_input = torch.cat((self.within(base), base_delta, condition), -1)
                gap_input = torch.cat((self.within(other), gap_delta, condition), -1)
                new_base = self.gru(base_input.flatten(0, 1), base.flatten(0, 1)).reshape_as(base)
                other = self.gru(gap_input.flatten(0, 1), other.flatten(0, 1)).reshape_as(other)
                base = new_base
            correction[rows, 0] = correction[rows, 0] + self.decoder(base).flatten(1)
            correction[rows, gap] = self.decoder(other).flatten(1)
            count[rows] += 1
        correction[:, 0] = correction[:, 0] / count.clamp_min(1)
        return local, safe_mask(clean + correction, active)


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    roles = {'sparsemap_role_partition': SparsePartition,
             'janossy_full_role_symmetrization': Janossy}
    if method in roles:
        return RoleAdapter(roles[method](), latent_dim, num_heads, value_dim)
    if method == 'graph_matching_base_gap_pairs':
        return GraphMatching(latent_dim, num_heads, value_dim)
    if method == 'nested_gnn_rootaware_evidence':
        return TokenAdapter(NestedGNN(root_aware=True), latent_dim, num_heads, value_dim, dim=64)
    cores = {'matrix_tree_nonprojective_evidence': MatrixTree,
             'diffpool_hierarchical_evidence_graph': DiffPool,
             'cwn_cellular_evidence': Cellular,
             'simplicial_hodge_evidence': Hodge,
             'nested_gnn_rooted_evidence': NestedGNN,
             'graph_unet_evidence_encoder_decoder': GraphUNet,
             'crfrnn_latent_evidence_states': CRFMeanField}
    if method not in cores:
        raise ValueError('Unknown structure method: ' + method)
    return TokenAdapter(cores[method](), latent_dim, num_heads, value_dim, dim=64)
