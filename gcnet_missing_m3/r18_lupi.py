"""R18 training-only privileged Gaussian noise, after the causal task readout.

The released CNN's unsquared standard-deviation norm is normalized here by
sqrt(number of valid activations), an explicit adaptation for padded batches.
Only the variance head sees complete current-utterance raw training features.
"""

import math

import torch
from torch import nn


class PrivilegedNoise(nn.Module):
    def __init__(self, raw_dim: int, hidden_dim: int):
        super().__init__()
        self.raw_dim = raw_dim
        self.hidden_dim = hidden_dim
        self.variance_head = nn.Sequential(
            nn.Linear(raw_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.ReLU(),
        )

    def forward(self, hidden, complete_features, valid):
        if hidden.shape[-1] != self.hidden_dim or valid.shape != hidden.shape[:-1]:
            raise ValueError("hidden and valid shapes must match the task readout")
        valid = valid.bool()
        safe_hidden = torch.where(valid[..., None], hidden, torch.zeros_like(hidden))
        zero = safe_hidden.sum() * 0
        # Return before even inspecting privileged features during evaluation.
        if not self.training:
            return safe_hidden, zero
        if complete_features is None:
            raise ValueError("R18 training requires complete raw utterance features")
        if complete_features.shape != (*hidden.shape[:-1], self.raw_dim):
            raise ValueError("complete_features must match readout rows and raw_dim")
        count = int(valid.sum().item())
        if count == 0:
            return safe_hidden, zero
        logvar = self.variance_head(complete_features[valid])
        sigma = torch.exp(0.5 * logvar)
        noisy_hidden = safe_hidden.clone()
        noisy_hidden[valid] = safe_hidden[valid] * (1 + torch.randn_like(sigma) * sigma)
        regularizer = torch.linalg.vector_norm(sigma) / math.sqrt(count * self.hidden_dim)
        return noisy_hidden, regularizer
