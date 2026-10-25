"""Conditional DDPM in fixed modality-specific 64-dimensional coordinates."""
import math
import hashlib

import torch
from torch import nn


class FixedLatentProjection(nn.Module):
    def __init__(self, mean, basis):
        super().__init__()
        if mean.shape != (3, 256) or basis.shape != (3, 64, 256):
            raise ValueError("expected mean[3,256] and basis[3,64,256]")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("basis", basis.detach().clone())

    def forward(self, latents, availability=None):
        projected = torch.einsum("tbmd,mkd->tbmk", latents - self.mean, self.basis)
        if availability is not None:
            projected = projected.masked_fill(~availability.bool().unsqueeze(-1), 0)
        return projected


class _Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, 2 * channels, 3, padding=1)
        self.condition = nn.Linear(channels, 2 * channels)
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, x, condition):
        gate, value = (self.conv(x) + self.condition(condition).unsqueeze(-1)).chunk(2, 1)
        return (x + self.out(torch.sigmoid(gate) * torch.tanh(value))) / math.sqrt(2)


class LatentCSDI(nn.Module):
    def __init__(self, steps=20, channels=64, layers=4):
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        self.steps = steps
        beta = torch.linspace(1e-4, .02, steps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", 1 - beta)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))
        self.input = nn.Conv1d(1, channels, 1)
        self.condition = nn.Linear(2304 + 192 + 3, channels)
        self.step_embedding = nn.Embedding(steps, channels)
        self.modality_embedding = nn.Embedding(3, channels)
        self.blocks = nn.ModuleList([_Residual(channels) for _ in range(layers)])
        self.output = nn.Conv1d(channels, 1, 1)

    def _predict(self, noisy, observed, availability, context, step):
        n = noisy.shape[0]
        # Only available clean coordinates can enter the conditioning branch.
        clean = observed.masked_fill(~availability.unsqueeze(-1), 0)
        condition = self.condition(torch.cat((context, clean.flatten(1), availability.to(clean.dtype)), -1))
        condition = condition + self.step_embedding(step)
        condition = condition[:, None, :] + self.modality_embedding.weight[None, :, :]
        condition = condition.reshape(n * 3, -1)
        h = self.input(noisy.reshape(n * 3, 1, 64))
        for block in self.blocks:
            h = block(h, condition)
        return self.output(h).reshape(n, 3, 64)

    def noise_loss(self, observed, target, a, context, valid, generator=None):
        a = a.bool().reshape(-1, 3)
        valid = valid.bool().reshape(-1)
        missing = (~a) & valid[:, None]
        if not missing.any():
            return self.output.weight.sum() * 0
        observed = observed.reshape(-1, 3, 64)
        target = target.reshape_as(observed)
        context = context.reshape(-1, 2304)
        step = torch.randint(self.steps, (target.shape[0],), device=target.device, generator=generator)
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        abar = self.alpha_bar[step, None, None]
        noisy = abar.sqrt() * target + (1 - abar).sqrt() * noise
        noisy = torch.where(a[..., None], observed, noisy)
        prediction = self._predict(noisy, observed, a, context, step)
        return (prediction - noise).square().masked_select(missing[..., None].expand_as(noise)).mean()

    @torch.no_grad()
    def sample(self, observed, a, context, valid, n_samples=4, generator=None, query_keys=None):
        t, b = observed.shape[:2]
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        # N order is T, B, K, preserving a separate draw for every hypothesis.
        clean = observed.unsqueeze(2).expand(t, b, n_samples, 3, 64).reshape(-1, 3, 64)
        available = a.bool().unsqueeze(2).expand(t, b, n_samples, 3).reshape(-1, 3)
        ctx = context.unsqueeze(2).expand(t, b, n_samples, 2304).reshape(-1, 2304)
        keys = None
        if query_keys is not None:
            keys = list(query_keys)
            if len(keys) == t and all(isinstance(row, (list, tuple)) for row in keys):
                keys = [key for row in keys for key in row]
            if len(keys) != t * b or not all(isinstance(key, str) for key in keys):
                raise ValueError("query_keys must contain one string per T,B position")

        def draw(draw_index):
            if keys is None:
                return torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
            draws = []
            for key, active in zip(keys, valid.bool().reshape(-1).tolist()):
                if not active:
                    draws.append(clean.new_zeros(n_samples, 3, 64))
                    continue
                # The caller's key identifies parent/rate/conversation/utterance;
                # neither its batch position nor any label participates in RNG.
                digest = hashlib.sha256(f"{len(key)}:{key}:{draw_index}".encode()).digest()
                seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
                rng = torch.Generator(device=clean.device).manual_seed(seed)
                draws.append(torch.randn((n_samples, 3, 64), device=clean.device,
                                         dtype=clean.dtype, generator=rng))
            return torch.stack(draws).reshape_as(clean)

        x = draw(self.steps)
        x = torch.where(available[..., None], clean, x)
        for s in reversed(range(self.steps)):
            step = torch.full((x.shape[0],), s, device=x.device, dtype=torch.long)
            # Keyed noise is independent of batch composition. Batched kernels
            # may introduce ordinary FP rounding changes across batch shapes.
            eps = self._predict(x, clean, available, ctx, step)
            mean = (x - self.beta[s] / (1 - self.alpha_bar[s]).sqrt() * eps) / self.alpha[s].sqrt()
            if s:
                variance = self.beta[s] * (1 - self.alpha_bar[s - 1]) / (1 - self.alpha_bar[s])
                noise = draw(s)
                mean = mean + variance.sqrt() * noise
            x = torch.where(available[..., None], clean, mean)
        result = x.reshape(t, b, n_samples, 3, 64)
        return result.masked_fill(~valid.bool()[..., None, None, None], 0)


class HypothesisBridge(nn.Module):
    def __init__(self, output_dim=1600):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(451), nn.Linear(451, 128), nn.GELU(), nn.Linear(128, output_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, local, samples, a, valid):
        k = samples.shape[2]
        features = torch.cat((local.unsqueeze(2).expand(-1, -1, k, -1), samples.flatten(-2),
                              a.to(local.dtype).unsqueeze(2).expand(-1, -1, k, -1)), -1)
        delta = self.net(features)
        active = valid.bool() & ~a.bool().all(-1)
        return delta.masked_fill(~active[..., None, None], 0)
