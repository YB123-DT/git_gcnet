"""Independent N3 and X-conv equation adaptations, not upstream code copies.

Sources and migration limitations: docs/osram_new40_retrieval_cards.json.
All neighborhoods are within a single already-read evidence set, never a
second OSRAM query or a database of other utterances.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .meaningful_input_new40 import TokenAdapter


class RecursiveNeighborVolumes(nn.Module):
    """Three separate soft neighbor ranks with recursive exclusion (N3)."""
    def __init__(self, dim=64):
        super().__init__()
        self.metric = nn.Linear(dim, 32)
        self.temperature = nn.Linear(dim, 1)
        self.mix = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU())

    def forward(self, x, columns, num_heads):
        embedded = self.metric(x)
        distances = (embedded[:, :, None] - embedded[:, None]).square().mean(-1)
        temperature = 0.1 + 1.9 * self.temperature(x).sigmoid()
        logits = -distances / temperature
        diagonal = torch.eye(x.shape[1], device=x.device, dtype=torch.bool)
        logits = logits.masked_fill(diagonal, -torch.inf)
        volumes = []
        for rank in range(3):
            weights = logits.softmax(-1)
            volumes.append(weights @ x - x)
            if rank < 2:
                logits = logits + torch.log1p(-weights.clamp_max(1.0 - 1e-6))
        return self.mix(torch.cat((x, *volumes), -1))


class XConvolution(nn.Module):
    """Learned chart -> dense neighbor X transform -> separable contraction."""
    def __init__(self, dim=64, neighbors=4):
        super().__init__()
        self.neighbors = neighbors
        self.chart = nn.Linear(dim, 8)
        self.lift = nn.Sequential(nn.Linear(8, 16), nn.ELU(),
                                  nn.Linear(16, 16), nn.ELU())
        self.x_global = nn.Linear(neighbors * 8, neighbors * neighbors)
        self.x_rows1 = nn.Linear(neighbors, neighbors)
        self.x_rows2 = nn.Linear(neighbors, neighbors)
        self.channel_kernel = nn.Parameter(torch.empty(neighbors, dim + 16))
        nn.init.uniform_(self.channel_kernel, -1 / math.sqrt(neighbors),
                         1 / math.sqrt(neighbors))
        self.pointwise = nn.Linear(dim + 16, dim)

    def forward(self, x):
        coordinates = self.chart(x)
        distances = (coordinates[:, :, None] - coordinates[:, None]).square().sum(-1)
        distances = distances.masked_fill(
            torch.eye(x.shape[1], device=x.device, dtype=torch.bool), torch.inf)
        indices = distances.topk(self.neighbors, dim=-1, largest=False).indices
        rows = torch.arange(x.shape[0], device=x.device)[:, None, None]
        relative = coordinates[rows, indices] - coordinates[:, :, None]
        neighbors = torch.cat((self.lift(relative), x[rows, indices]), -1)
        transform = F.elu(self.x_global(relative.flatten(-2))).reshape(
            x.shape[0], x.shape[1], self.neighbors, self.neighbors)
        transform = self.x_rows2(F.elu(self.x_rows1(transform)))
        transformed = transform @ neighbors
        contracted = (transformed * self.channel_kernel).sum(-2)
        return F.elu(self.pointwise(contracted))


class EvidencePointCNN(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.blocks = nn.ModuleList([XConvolution(dim), XConvolution(dim)])

    def forward(self, x, columns, num_heads):
        for block in self.blocks:
            x = block(x)
        return x


def build(method, latent_dim=256, num_heads=8, value_dim=64):
    cores = {
        'n3_recursive_neighbor_volumes': RecursiveNeighborVolumes,
        'pointcnn_x_transformed_evidence': EvidencePointCNN,
    }
    if method not in cores:
        raise ValueError('Unknown retrieval adaptation: ' + method)
    return TokenAdapter(cores[method](), latent_dim, num_heads, value_dim, dim=64)
