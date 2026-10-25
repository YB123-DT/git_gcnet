"""Pathwise DDPM sampling for the one-stage joint models."""

import hashlib

import torch
from torch.utils.checkpoint import checkpoint

from osram_four_block_wiring.csdi_latent import LatentCSDI


class DifferentiableLatentCSDI(LatentCSDI):
    """The original denoiser and schedule with differentiable reverse steps.

    Random draws cover only valid queries with missing modalities, in T,B,K
    order. Consequently their RNG shapes differ from the frozen full sampler.
    No history or target coordinates participate in conditioning.
    """

    def __init__(self, steps=20, channels=64, layers=4, checkpointed=True):
        super().__init__(steps=steps, channels=channels, layers=layers)
        self.checkpointed = checkpointed
        # CPU noise only, never predictions: weights change every training epoch.
        # Eight MOSI evaluation rates require roughly 340 MB at K=4, steps=20.
        # Private state is reconstructible from keys and excluded from state_dict.
        self._query_noise_cache = {}

    def rsample(self, observed, a, context, valid, n_samples=4, generator=None, query_keys=None):
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        t, b = observed.shape[:2]
        available = a.bool().reshape(-1, 3)
        valid = valid.bool().reshape(-1)
        active = valid & ~available.all(-1)
        keys = None
        if query_keys is not None:
            keys = list(query_keys)
            if len(keys) == t and all(isinstance(row, (list, tuple)) for row in keys):
                keys = [key for row in keys for key in row]
            if len(keys) != t * b or not all(isinstance(key, str) for key in keys):
                raise ValueError("query_keys must contain one string per T,B position")
        result = observed.reshape(-1, 3, 64)[:, None].expand(-1, n_samples, -1, -1)
        result = result.masked_fill(~valid[:, None, None, None], 0)
        if not active.any():
            return result.reshape(t, b, n_samples, 3, 64)
        clean = observed.reshape(-1, 3, 64)[active, None].expand(-1, n_samples, -1, -1).reshape(-1, 3, 64)
        mask = available[active, None].expand(-1, n_samples, -1).reshape(-1, 3)
        ctx = context.reshape(-1, 2304)[active, None].expand(-1, n_samples, -1).reshape(-1, 2304)

        keyed_noise = None
        if keys is not None:
            noise = []
            for index in active.nonzero(as_tuple=True)[0].tolist():
                key = keys[index]
                cache_key = (key, self.steps, n_samples)
                if cache_key not in self._query_noise_cache:
                    digest = hashlib.sha256(f"{len(key)}:{key}".encode()).digest()
                    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
                    rng = torch.Generator(device="cpu").manual_seed(seed)
                    self._query_noise_cache[cache_key] = torch.randn(
                        self.steps + 1, n_samples, 3, 64, generator=rng)
                noise.append(self._query_noise_cache[cache_key])
            # One host-to-device transfer per forward; retained noise is CPU only.
            keyed_noise = torch.stack(noise, dim=1).to(device=clean.device, dtype=clean.dtype)

        def draw(index):
            if keyed_noise is not None:
                return keyed_noise[index].reshape_as(clean)
            return torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)

        x = torch.where(mask[..., None], clean, draw(self.steps))
        for s in reversed(range(self.steps)):
            step = torch.full((x.shape[0],), s, device=x.device, dtype=torch.long)
            if self.checkpointed and torch.is_grad_enabled():
                # Non-reentrant checkpointing (available in Torch 1.13) also
                # supports trainable parameters when input tensors are constants.
                eps = checkpoint(self._predict, x, clean, mask, ctx, step,
                                 use_reentrant=False, preserve_rng_state=False)
            else:
                eps = self._predict(x, clean, mask, ctx, step)
            mean = (x - self.beta[s] / (1 - self.alpha_bar[s]).sqrt() * eps) / self.alpha[s].sqrt()
            if s:
                variance = self.beta[s] * (1 - self.alpha_bar[s - 1]) / (1 - self.alpha_bar[s])
                mean = mean + variance.sqrt() * draw(s)
            x = torch.where(mask[..., None], clean, mean)
        result = result.index_copy(0, active.nonzero(as_tuple=True)[0], x.reshape(-1, n_samples, 3, 64))
        return result.reshape(t, b, n_samples, 3, 64)
