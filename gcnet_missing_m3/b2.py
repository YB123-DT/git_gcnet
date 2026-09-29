"""Source-only completion, read fusion, and opt-in predicted memory writes."""
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


def predicted_latent_write_callback(
    osram, node, latents, predictions, availability, qmask, umask
):
    """Fill missing K/V slots AFTER reads, without changing real availability.

    Real slots remain online-projector observations. Missing slots use the
    frozen predictor's latent directly through the existing online K/V maps;
    no coordinate adapter, extra gate, or separate write strength is added.
    Query construction and Gap residualization still see real observations
    only. The returned write mask must never be reused as input availability.
    """
    valid = _valid_latents(latents, availability, umask, node.shape[-1])
    if predictions.shape != (*node.shape[:2], 3, node.shape[-1]):
        raise ValueError("reg predictions must have shape [L,B,3,latent_dim]")
    missing = valid.unsqueeze(-1) & ~availability.bool()
    if not bool(missing.any()):
        return None
    if not bool(torch.isfinite(predictions[missing]).all()):
        raise ValueError("missing-latent write predictions must be finite")
    completed = {
        name: torch.where(
            missing[..., index, None], predictions[..., index, :], latents[name]
        )
        for index, name in enumerate(MODALITIES)
    }
    keys, values, _ = osram._project_sequence(
        node, completed, availability, qmask, write_node=node
    )
    predicted_keys = torch.stack([keys[name] for name in MODALITIES], dim=-1)
    predicted_values = torch.stack([values[name] for name in MODALITIES], dim=-1)
    write_mask = valid.unsqueeze(-1).expand_as(availability).to(availability.dtype)

    def complete_write(time_index, _base, _gap, observed_keys, observed_values):
        selected = missing[time_index, :, None, None, :]
        return (
            torch.where(selected, predicted_keys[time_index], observed_keys),
            torch.where(selected, predicted_values[time_index], observed_values),
            write_mask[time_index],
        )

    return complete_write


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
