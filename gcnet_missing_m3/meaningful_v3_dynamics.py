"""Current-utterance algorithmic readouts; five roles, never attention heads.

Independent mathematical implementations of Irie et al. (2022), Soulos et al.
(2023), and Freivalds et al. (2019). See the dynamics survey for sources.
All working weights/trees/switch ports are allocated per forward. The original
OSRAM memory, queries, Flat head and task objective are outside these modules.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


METHODS = (
    "dynamics_srwm_self_modifying_program",
    "dynamics_differentiable_tree_machine",
    "dynamics_neural_shuffle_exchange",
)


class SelfReferentialMatrix(nn.Module):
    """Full self-modification: output, key, query AND rate-generating weights."""

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.initial = nn.Parameter(torch.randn(3 * dim + 4, dim) / math.sqrt(dim))

    def forward(self, tokens, mask):
        d = self.dim
        matrix = self.initial.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        outputs = [None] * 5
        # Base, three Gap roles, then Local: a computational dependency order.
        for index in (1, 2, 3, 4, 0):
            x = tokens[:, index].softmax(-1)
            emitted = torch.einsum("bod,bd->bo", matrix, x)
            y, key, query, rate = emitted.split((d, d, d, 4), -1)
            key, query = key.softmax(-1), query.softmax(-1)
            difference = torch.einsum("bod,bd->bo", matrix, query - key)
            rates = rate.sigmoid()
            rates = torch.cat((rates[:, :1].expand(-1, d),
                               rates[:, 1:2].expand(-1, d),
                               rates[:, 2:3].expand(-1, d),
                               rates[:, 3:4].expand(-1, 4)), -1)
            proposal = matrix + (rates * difference).unsqueeze(-1) * key.unsqueeze(1)
            matrix = torch.where(mask[:, index, None, None], proposal, matrix)
            outputs[index] = torch.where(mask[:, index, None], y, torch.zeros_like(y))
        return torch.stack(outputs, 1)


class TreeMachine(nn.Module):
    """TPR car/cdr/cons and argument-tree selection, with exact finite roles.

    An orthonormal one-hot role basis permits storing TPR coefficients directly.
    Five initial heap positions have depth <= 2. Three cons steps can reach at
    most depth 5; all 63 positions are retained, so no reachable value is cut.
    """

    def __init__(self, dim=32, steps=3):
        super().__init__()
        self.steps = steps
        self.nodes = 2 ** (3 + steps) - 1
        self.encode = nn.Linear(self.nodes * dim, dim)
        self.control_tokens = nn.Parameter(torch.randn(2, dim) / math.sqrt(dim))
        self.controllers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0.,
                                       activation="gelu", batch_first=True)
            for _ in range(steps)])
        self.operations = nn.ModuleList([nn.Linear(dim, 3) for _ in range(steps)])
        self.arguments = nn.ModuleList([nn.Linear(dim, 4) for _ in range(steps)])
        self.roots = nn.ModuleList([nn.Linear(dim, dim) for _ in range(steps)])
        left, right = torch.zeros(self.nodes, self.nodes), torch.zeros(self.nodes, self.nodes)
        for source in range((self.nodes - 1) // 2):
            suffix = bin(source + 1)[3:]
            left[int("10" + suffix, 2) - 1, source] = 1
            right[int("11" + suffix, 2) - 1, source] = 1
        self.register_buffer("left", left)
        self.register_buffer("right", right)

    def interpret(self, memory, argument_weights, operation_weights, root):
        # memory [batch, prior trees, node roles, filler dimensions]
        arguments = torch.einsum("bmnf,bma->banf", memory, argument_weights)
        car = self.left.T @ arguments[:, 0]
        cdr = self.right.T @ arguments[:, 1]
        cons = self.left @ arguments[:, 2] + self.right @ arguments[:, 3]
        root_tree = F.pad(root.unsqueeze(1), (0, 0, 0, self.nodes - 1))
        candidates = torch.stack((car, cdr, cons + root_tree), 1)
        return torch.einsum("banf,ba->bnf", candidates, operation_weights)

    def forward(self, tokens, mask):
        tree = F.pad(torch.where(mask[..., None], tokens, torch.zeros_like(tokens)),
                     (0, 0, 0, self.nodes - 5))
        memory = [tree]
        control = self.control_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for step in range(self.steps):
            control = torch.cat((control, self.encode(memory[-1].flatten(1)).unsqueeze(1)), 1)
            control = self.controllers[step](control)
            operations = self.operations[step](control[:, 0]).softmax(-1)
            arguments = self.arguments[step](control[:, 2:]).softmax(1)
            root = self.roots[step](control[:, 1])
            tree = self.interpret(torch.stack(memory, 1), arguments, operations, root)
            memory.append(tree)
        return memory[-1][:, :5]


class PairSwitch(nn.Module):
    """Two reset-conditioned candidates plus gated half-channel swapping."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.resets = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(2)])
        self.candidates = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(2)])
        self.gate = nn.Linear(2 * dim, 2 * dim)
        for layer in (*self.resets, self.gate):
            nn.init.constant_(layer.bias, .5)

    def forward(self, ports):
        n, count, dim = ports.shape
        pairs = ports.reshape(n, count // 2, 2 * dim)
        candidate = torch.cat([
            transform(pairs * reset(pairs).sigmoid())
            for reset, transform in zip(self.resets, self.candidates)], -1).tanh()
        paired = ports.reshape(n, count // 2, 2, dim)
        swapped = torch.cat((paired.flip(2)[..., :dim // 2], paired[..., dim // 2:]), -1)
        gate = self.gate(pairs).sigmoid()
        return (gate * swapped.flatten(2) + (1 - gate) * candidate).reshape(n, count, dim)


class ShuffleExchange(nn.Module):
    """One complete 8-port Beneš block: five switches, four permutations.

    Five real role vectors and three initially zero working ports. Empty ports
    can carry intermediate values; they are never emitted as extra evidence.
    """

    def __init__(self, dim=32):
        super().__init__()
        self.forward_switch = PairSwitch(dim)
        self.reverse_switch = PairSwitch(dim)
        self.last_switch = PairSwitch(dim)
        self.register_buffer("rotate_left", torch.tensor([((i << 1) & 7) | (i >> 2) for i in range(8)]))
        self.register_buffer("rotate_right", torch.tensor([(i >> 1) | ((i & 1) << 2) for i in range(8)]))

    def forward(self, tokens, mask):
        ports = F.pad(torch.where(mask[..., None], tokens, torch.zeros_like(tokens)), (0, 0, 0, 3))
        for _ in range(2):
            ports = self.forward_switch(ports)[:, self.rotate_left]
        for _ in range(2):
            ports = self.reverse_switch(ports)[:, self.rotate_right]
        return self.last_switch(ports)[:, :5]


class DynamicsReadout(nn.Module):
    def __init__(self, core, latent_dim, num_heads, value_dim, dim=32):
        super().__init__()
        self.width = num_heads * value_dim
        self.local_encoder = nn.Linear(latent_dim, dim)
        self.evidence_encoder = nn.Linear(self.width, dim)
        self.role = nn.Parameter(torch.randn(5, dim) / math.sqrt(dim))
        self.core = core
        self.local_decoder = nn.Linear(dim, latent_dim)
        self.evidence_decoder = nn.Linear(dim, self.width)
        for layer in (self.local_decoder, self.evidence_decoder):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, local, evidence, active, availability):
        if active.dtype != torch.bool or evidence.shape[1:] != (4, self.width):
            raise ValueError("Expected four raw evidence slots and a boolean active mask")
        valid = active.any(-1)
        clean = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
        safe_local = torch.where(valid[:, None], local, torch.zeros_like(local))
        mask = torch.cat((valid[:, None], active), 1)
        tokens = torch.cat((self.local_encoder(safe_local).unsqueeze(1),
                            self.evidence_encoder(clean)), 1) + self.role
        tokens = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        result = self.core(tokens, mask)
        local_delta = torch.where(valid[:, None], self.local_decoder(result[:, 0]), torch.zeros_like(local))
        evidence_delta = self.evidence_decoder(result[:, 1:])
        return local + local_delta, torch.where(active[..., None], clean + evidence_delta, torch.zeros_like(clean))


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    cores = dict(zip(METHODS, (SelfReferentialMatrix, TreeMachine, ShuffleExchange)))
    if method not in cores:
        raise ValueError("Unknown dynamics method: " + method)
    return DynamicsReadout(cores[method](), latent_dim, num_heads, value_dim)
