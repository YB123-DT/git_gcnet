"""Training-only prediction of an EMA complete-view Local state."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class CompleteViewLocalStateJEPA(nn.Module):
    def __init__(self, encoder, local_path, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.teacher_encoder = copy.deepcopy(encoder).requires_grad_(False).eval()
        self.teacher_local = copy.deepcopy(local_path).requires_grad_(False).eval()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.pattern = nn.Embedding(8, 16, padding_idx=0)
        self.predictor = nn.Sequential(nn.Linear(hidden_dim + 16, latent_dim),
                                       nn.GELU(), nn.Linear(latent_dim, latent_dim))

    def train(self, mode=True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_local.eval()
        return self

    def predict(self, hidden, availability, umask):
        valid = umask.T.bool()
        result = hidden.new_zeros(*hidden.shape[:2], self.latent_dim)
        if valid.any():
            ids = (availability[..., 0].long() * 4 + availability[..., 1].long() * 2
                   + availability[..., 2].long())
            result[valid] = self.predictor(torch.cat([
                self.input_norm(hidden[valid]), self.pattern(ids[valid])], -1))
        return result

    @torch.no_grad()
    def target(self, features, umask):
        valid = umask.T.bool()
        full = valid[..., None].expand(-1, -1, 3).to(features.dtype)
        node, _ = self.teacher_encoder(features, full, umask)
        state = node + self.teacher_local(node)
        return state.masked_fill(~valid[..., None], 0)

    def loss(self, hidden, features, availability, umask):
        selected = umask.T.bool() & (availability.sum(-1) < 3)
        count = int(selected.sum().item())
        if not count:
            return hidden.sum() * 0, 0
        pred = self.predict(hidden, availability, umask)[selected]
        target = self.target(features, umask)[selected]
        loss = F.smooth_l1_loss(F.layer_norm(pred, (self.latent_dim,)),
                                F.layer_norm(target, (self.latent_dim,)))
        return loss, count

    @torch.no_grad()
    def update(self, encoder, local_path, tau):
        if not 0 <= tau <= 1:
            raise ValueError('EMA tau must be in [0,1]')
        for teacher, online in ((self.teacher_encoder, encoder),
                                (self.teacher_local, local_path)):
            source = dict(online.named_parameters())
            for name, parameter in teacher.named_parameters():
                parameter.mul_(tau).add_(source[name], alpha=1-tau)
            source_buffers = dict(online.named_buffers())
            for name, buffer in teacher.named_buffers():
                buffer.copy_(source_buffers[name])
