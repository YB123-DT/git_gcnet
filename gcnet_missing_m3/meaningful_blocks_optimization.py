"""Independent mathematical implementations of three optimization readouts.

References and adaptation decisions: optimization.json in the meaningful20
experiment. No external source code is copied. All inferred state is local to
one forward call; none of these blocks reads or writes OSRAM memory.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_blocks_common import HeadTokenizer, active_groups, typed_means, safe_mask


class _TokenDifferenceBlock(nn.Module):
    output_dim = 128

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        self.num_heads = num_heads
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim,
                                       dim=128, shared_projection=True, normalize=False)
        self.readout = nn.Sequential(nn.Linear(640, 128), nn.GELU())

    def transform(self, packed):
        raise NotImplementedError

    def forward(self, local, evidence, active, availability):
        tokens, mask = self.tokenizer(local, evidence, active)
        difference = torch.zeros_like(tokens)
        for rows, cols, packed in active_groups(tokens, mask):
            difference[rows[:, None], cols[None, :]] = self.transform(packed) - packed
        return safe_mask(self.readout(typed_means(difference, mask, self.num_heads).flatten(1)), active.any(-1))


class HamburgerBlock(_TokenDifferenceBlock):
    """Two breads plus rank-four NMF, with one-step gradient differentiation."""

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.lower = nn.Linear(128, 128)
        self.upper = nn.Linear(128, 128, bias=False)
        self.norm = nn.LayerNorm(128)
        generator = torch.Generator(device='cpu').manual_seed(66)
        basis = torch.rand(128, 4, generator=generator)
        self.register_buffer('initial_basis', F.normalize(basis, dim=0))

    def factorize(self, x):
        # [batch,tokens,channels]; factor inference is deliberately truncated.
        with torch.no_grad():
            basis = self.initial_basis.unsqueeze(0).expand(x.shape[0], -1, -1).clone()
            coefficient = (x @ basis).softmax(dim=-1)
            for _ in range(6):
                coefficient = coefficient * (x @ basis) / (
                    coefficient @ (basis.transpose(1, 2) @ basis) + 1e-6)
                basis = basis * (x.transpose(1, 2) @ coefficient) / (
                    basis @ (coefficient.transpose(1, 2) @ coefficient) + 1e-6)
        # Only this coefficient refinement carries gradients from reconstruction.
        coefficient = coefficient * (x @ basis) / (
            coefficient @ (basis.transpose(1, 2) @ basis) + 1e-6)
        return coefficient @ basis.transpose(1, 2)

    def transform(self, packed):
        reconstructed = self.factorize(F.relu(self.lower(packed)))
        return F.relu(packed + self.norm(self.upper(reconstructed)))


class CRATEIteration(nn.Module):
    """Complete tied-subspace interaction followed by dictionary ISTA."""

    def __init__(self, dim=128, heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError('CRATE dimension must divide into heads')
        self.heads, self.head_dim = heads, dim // heads
        self.norm_attention = nn.LayerNorm(dim)
        self.subspace = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.norm_sparse = nn.LayerNorm(dim)
        self.dictionary = nn.Parameter(torch.empty(dim, dim))
        nn.init.kaiming_uniform_(self.dictionary)

    def forward(self, x):
        batch, length, dim = x.shape
        w = self.subspace(self.norm_attention(x)).reshape(
            batch, length, self.heads, self.head_dim).transpose(1, 2)
        weights = ((w @ w.transpose(-1, -2)) / math.sqrt(self.head_dim)).softmax(-1)
        compression = (weights @ w).transpose(1, 2).reshape(batch, length, dim)
        z = self.norm_sparse(x + self.output(compression))
        reconstruction = F.linear(F.linear(z, self.dictionary), self.dictionary.T)
        dictionary_pull = F.linear(z, self.dictionary.T)
        # No second Transformer residual: ISTA replaces, rather than adds to, z.
        return F.relu(z + .1 * (dictionary_pull - reconstruction) - .01)


class CRATEBlock(_TokenDifferenceBlock):
    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__(latent_dim, num_heads, value_dim)
        self.iterations = nn.Sequential(CRATEIteration(), CRATEIteration())

    def transform(self, packed):
        return self.iterations(packed)


class _PotentialStage(nn.Module):
    def __init__(self, input_dim, output_dim, nonlinear):
        super().__init__()
        self.main = nn.Linear(input_dim, output_dim)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if nonlinear else None

    def forward(self, x):
        h = self.main(x)
        if self.norm is not None:
            h = torch.tanh(self.norm(h))
        return self.skip(x) + h


class EquilibriumBlock(nn.Module):
    """Learned squared energy and ten explicit Nesterov inference updates.

    The inner energy defines the forward map, not an auxiliary training loss.
    No optimizer, tensor state, or inferred aggregate survives this call.
    """
    output_dim = 128

    def __init__(self, latent_dim, num_heads, value_dim):
        super().__init__()
        self.tokenizer = HeadTokenizer(latent_dim, num_heads, value_dim,
                                       dim=128, shared_projection=True, normalize=False)
        self.potential = nn.Sequential(_PotentialStage(256, 256, True),
                                      _PotentialStage(256, 128, True),
                                      _PotentialStage(128, 32, False))
        self.raw_learning_rate = nn.Parameter(torch.tensor(math.log(math.expm1(.125))))
        self.raw_momentum = nn.Parameter(torch.tensor(math.log(.9 / .1)))
        self.raw_regularizer = nn.Parameter(torch.tensor(math.log(math.expm1(.1))))

    @property
    def learning_rate(self):
        return F.softplus(self.raw_learning_rate)

    @property
    def momentum(self):
        return torch.sigmoid(self.raw_momentum)

    def energy(self, tokens, mask, y):
        state = y[:, None, :].expand(-1, tokens.shape[1], -1)
        values = self.potential(torch.cat((tokens, state), dim=-1)).square().mean(-1)
        total = torch.where(mask, values, torch.zeros_like(values)).sum(-1)
        total = total + F.softplus(self.raw_regularizer) * y.square().sum(-1)
        count = mask.sum(-1).to(y.dtype)
        return total * torch.log2(count + 1) / (count + 1e-8)

    def solve(self, energy, y, create_graph):
        y = y.requires_grad_(True)
        velocity = torch.zeros_like(y)
        eta, mu = self.learning_rate, self.momentum
        if not create_graph:
            eta, mu = eta.detach(), mu.detach()
        for _ in range(10):
            lookahead = y + mu * velocity
            gradient = torch.autograd.grad(energy(lookahead).sum(), lookahead,
                                           create_graph=create_graph)[0]
            velocity = mu * velocity - eta * gradient
            moving = (gradient.abs().amax(-1) >= .001).to(y.dtype).unsqueeze(-1)
            y = y + moving * velocity
            if not create_graph:
                y, velocity = y.detach().requires_grad_(True), velocity.detach()
        return y

    def forward(self, local, evidence, active, availability):
        outer_grad = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        create_graph = bool(self.training and outer_grad)
        # inference_mode tensors cannot be saved for differentiating inner energy.
        with torch.inference_mode(False), torch.enable_grad():
            if not create_graph:
                local, evidence = local.detach().clone(), evidence.detach().clone()
                active, availability = active.clone(), availability.clone()
            tokens, mask = self.tokenizer(local, evidence, active)
            if not create_graph:
                tokens, mask = tokens.detach(), mask.detach()
            zero = tokens.new_zeros((tokens.shape[0], self.output_dim))
            if not tokens.shape[0]:
                return zero
            result = self.solve(lambda y: self.energy(tokens, mask, y), zero, create_graph)
        return result if create_graph else result.detach()


def build_optimization(method, latent_dim, num_heads, value_dim):
    classes = {'hamburger_nmf_full': HamburgerBlock,
               'crate_mssa_ista_full': CRATEBlock,
               'equilibrium_aggregation': EquilibriumBlock}
    if method not in classes:
        raise ValueError(f'Unknown optimization block: {method}')
    with torch.random.fork_rng(devices=[]):
        return classes[method](latent_dim, num_heads, value_dim)
