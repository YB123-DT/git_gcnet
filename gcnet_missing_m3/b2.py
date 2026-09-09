"""Source-only completion and read fusion; no historical context or writes."""
from typing import Mapping

import torch
from torch import nn

from .model import MODALITIES, DualGateTopKMMoE, MissingM3Predictions


def _valid_latents(latents, availability, umask, latent_dim):
    if set(latents) != set(MODALITIES):
        raise ValueError("latents must contain audio/text/visual")
    reference = latents["audio"]
    if reference.ndim != 3 or reference.shape[-1] != latent_dim:
        raise ValueError("latents must have shape [L,B,latent_dim]")
    if any(z.shape != reference.shape for z in latents.values()):
        raise ValueError("latent shapes differ")
    if availability.shape != (*reference.shape[:2], 3) or umask.shape != reference.shape[:2][::-1]:
        raise ValueError("availability/umask shapes differ from latents")
    if not bool(((availability == 0) | (availability == 1)).all()):
        raise ValueError("availability must be binary")
    valid = umask.T.bool()
    if bool(availability[~valid].any()) or bool((availability[valid].sum(-1) == 0).any()):
        raise ValueError("padding must be empty and valid patterns nonempty")
    return valid


class SourceOnlyM3Predictor(nn.Module):
    """Six directed tasks, mean over real observed sources of each target."""

    def __init__(self, latent_dim, num_experts=4, top_k=2, dropout=.1,
                 mmoe_variant="dual-gate", target_private_rank=0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input_norm = nn.LayerNorm(latent_dim)
        self.mmoe = DualGateTopKMMoE(latent_dim, num_experts, top_k, dropout,
                                   variant=mmoe_variant, target_private_rank=target_private_rank)

    def forward(self, latents: Mapping[str, torch.Tensor], availability, umask):
        valid = _valid_latents(latents, availability, umask, self.latent_dim)
        ref = latents["audio"]
        shape = ref.shape[:2]
        avail = availability.reshape(-1,3).bool()
        flat = {m:z.reshape(-1,self.latent_dim) for m,z in latents.items()}
        regs, cls, masks, counts_out = [], [], [], []
        for q in range(3):
            target = valid.reshape(-1) & ~avail[:,q]
            reg = ref.new_zeros(target.numel(),self.latent_dim)
            cl = torch.zeros_like(reg)
            counts = torch.zeros_like(target,dtype=torch.long)
            for p,m in enumerate(MODALITIES):
                if p == q:
                    continue
                indices = (target & avail[:,p]).nonzero(as_tuple=False).flatten()
                if indices.numel():
                    r,c = self.mmoe(self.input_norm(flat[m][indices]),p,q)
                    reg = reg.index_add(0,indices,r)
                    cl = cl.index_add(0,indices,c)
                    counts = counts.index_add(0,indices,torch.ones_like(indices))
            divisor = counts.clamp_min(1).to(ref.dtype).unsqueeze(-1)
            regs.append(reg/divisor)
            cls.append(cl/divisor)
            masks.append(target & (counts>0))
            counts_out.append(counts)
        return MissingM3Predictions(
            torch.stack(regs,1).reshape(*shape,3,self.latent_dim),
            torch.stack(cls,1).reshape(*shape,3,self.latent_dim),
            torch.stack(masks,1).reshape(*shape,3),
            torch.stack(counts_out,1).reshape(*shape,3))


class CompletedReadFusion(nn.Module):
    """Fixed A/T/V slots plus observed/predicted identity, never confidence."""

    def __init__(self, latent_dim, dropout=.1):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.slot_norm = nn.LayerNorm(latent_dim)
        width = 3*(latent_dim+2)
        self.residual = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,latent_dim),
            nn.GELU(),nn.Dropout(dropout),nn.Linear(latent_dim,latent_dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def fill_slots(self,latents,predictions,availability,umask):
        valid = _valid_latents(latents,availability,umask,self.latent_dim)
        observed = torch.stack([latents[m] for m in MODALITIES],dim=2)
        if predictions.shape != observed.shape:
            raise ValueError("reg predictions must have shape [L,B,3,latent_dim]")
        filled = torch.where(availability.bool().unsqueeze(-1),observed,predictions)
        return torch.where(valid[...,None,None],filled,torch.zeros_like(filled))

    def forward(self,observed_node,latents,reg_predictions,availability,umask):
        filled = self.fill_slots(latents,reg_predictions,availability,umask)
        if observed_node.shape != filled.shape[:2]+(self.latent_dim,):
            raise ValueError("observed_node shape differs from latents")
        active = umask.T.bool() & (availability == 0).any(-1)
        correction = torch.zeros_like(observed_node)
        if bool(active.any()):
            status = torch.stack((availability,1-availability),dim=-1).to(filled.dtype)
            slots = torch.cat((self.slot_norm(filled[active]),status[active]),dim=-1)
            correction[active] = self.residual(slots.flatten(-2))
        return observed_node + correction
