"""Dense expert mixture over twelve distinct historical source/role slots.

All experts process the identical, unpooled evidence vector. Their identities
have no prescribed modality semantics. No auxiliary balancing loss is used.
"""
import torch
from torch import nn


class SourceRoleMixtureOfExperts(nn.Module):
    def __init__(self, local_dim=256, forward_dim=512, output_dim=1600):
        super().__init__()
        self.memory_projection = nn.Linear(forward_dim, 64)
        self.local_projection = nn.Linear(local_dim, 64)
        self.role_embedding = nn.Embedding(4, 64)
        self.source_embedding = nn.Embedding(3, 64)
        # Twelve identity-preserving slots, Local, trajectories, availability,
        # and the explicit twelve-slot activity mask. No source/role pooling.
        input_dim = 12 * 64 + 64 + 12 + 3 + 12
        self.experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 256),
                          nn.GELU(), nn.Linear(256, 128)) for _ in range(4)])
        self.router = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 128),
                                    nn.GELU(), nn.Linear(128, 4))
        self.output_projection = nn.Linear(128, output_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.last_diagnostics = {}
        self.last_weights = self.last_active = None

    def forward(self, local, source_base, source_gap, trajectory, availability, umask):
        valid = umask.T.bool()
        seen = trajectory[..., 3].bool() & valid[..., None]
        missing = ~availability.bool()
        roles = torch.cat((torch.ones_like(missing[..., :1]), missing), -1)
        active = roles[..., :, None] & seen[..., None, :]
        source = torch.cat((source_base.unsqueeze(2), source_gap), 2)
        source = torch.where(active[..., None], source, torch.zeros_like(source))
        tokens = (self.memory_projection(source)
                  + self.role_embedding.weight[None, None, :, None]
                  + self.source_embedding.weight[None, None, None, :])
        tokens = torch.where(active[..., None], tokens, torch.zeros_like(tokens))
        safe_local = torch.where(valid[..., None], local, torch.zeros_like(local))
        meta = torch.stack((trajectory[..., 0].clamp_min(0).log1p(),
                            trajectory[..., 1], trajectory[..., 2].clamp_min(0).log1p(),
                            trajectory[..., 3]), -1)
        meta = torch.where(valid[..., None, None], meta, torch.zeros_like(meta))
        features = torch.cat((tokens.flatten(-3), self.local_projection(safe_local),
                              meta.flatten(-2), availability.to(local.dtype),
                              active.flatten(-2).to(local.dtype)), -1)
        features = torch.where(valid[..., None], features, torch.zeros_like(features))
        weights = self.router(features).softmax(-1)
        values = torch.stack([expert(features) for expert in self.experts], -2)
        mixed = (weights[..., None] * values).sum(-2)
        output = self.output_projection(mixed)
        output = torch.where(valid[..., None], output, torch.zeros_like(output))
        self.last_weights = weights.detach()
        self.last_active = active.detach()
        with torch.no_grad():
            w = weights[valid]
            unit_values = torch.nn.functional.normalize(values[valid], dim=-1)
            cosine = unit_values @ unit_values.transpose(-1, -2)
            off_diagonal = ~torch.eye(4, device=local.device, dtype=torch.bool)
            self.last_diagnostics = {
                'valid_count': int(valid.sum()),
                'expert_weight_mean': w.mean(0).cpu().tolist() if w.numel() else [0.] * 4,
                'expert_weight_std': w.std(0, unbiased=False).cpu().tolist() if w.numel() else [0.] * 4,
                'router_entropy': float(-(w * w.clamp_min(1e-12).log()).sum(-1).mean()) if w.numel() else 0.,
                'output_norm': float(output[valid].norm(dim=-1).mean()) if w.numel() else 0.,
                'active_slot_mean': float(active[valid].float().sum((-1, -2)).mean()) if w.numel() else 0.,
                'expert_output_cosine_offdiag': float(cosine[:, off_diagonal].mean()) if w.numel() else 0.,
            }
        return output
