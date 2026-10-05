"""Source-grounded VQ-VAE, supervised FactorCL and Fishr transfers.

Pinned sources and explicit protocol adaptations: representation.json in
docs/osram_core20_20261005. No labels enter representation inference.
"""
import math
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ('audio', 'text', 'visual')


class VQVAEValues(nn.Module):
    """VQ-VAE Eq. (3): decoder MSE + codebook MSE + beta commitment MSE.

    The existing Value projection serves as encoder. Its output is quantized
    per head; a learned decoder reconstructs original observed input features.
    Decoder outputs are auxiliary targets, never missing memory writes.
    """

    def __init__(self, value_dim=64, num_embeddings=64, commitment_cost=.25,
                 num_heads=8, reconstruction_dims=None):
        super().__init__()
        if reconstruction_dims is None or set(reconstruction_dims) != set(MODALITIES):
            raise ValueError('VQ-VAE requires original raw reconstruction_dims for all modalities')
        self.commitment_cost = commitment_cost
        self.num_heads = num_heads
        self.codebooks = nn.ModuleDict({
            m: nn.Embedding(num_embeddings, value_dim) for m in MODALITIES})
        self.decoders = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(num_heads * value_dim, num_heads * value_dim), nn.ReLU(),
                             nn.Linear(num_heads * value_dim, reconstruction_dims[m]))
            for m in MODALITIES})
        for codebook in self.codebooks.values():
            nn.init.uniform_(codebook.weight, -1 / num_embeddings, 1 / num_embeddings)

    def transform_values(self, values, availability, valid, reconstruction_targets=None):
        if reconstruction_targets is None:
            raise ValueError('VQ-VAE requires original observed raw feature reconstruction_targets')
        result, terms = {}, []
        for index, modality in enumerate(MODALITIES):
            value = values[modality]
            target = reconstruction_targets[modality]
            if value.shape[-2] != self.num_heads:
                raise ValueError('VQ-VAE Value head count does not match decoder input')
            if target.shape[:-1] != valid.shape or target.shape[-1] != self.decoders[modality][-1].out_features:
                raise ValueError('VQ-VAE raw target must match [L,B,reconstruction_dim]')
            observed = availability[..., index].bool() & valid.bool()
            # Index BEFORE distances: poisoned missing/padded tensors never
            # enter the codebook, reconstruction objective or gradients.
            x = value[observed].reshape(-1, value.shape[-1])
            output = torch.zeros_like(value)
            if x.shape[0]:
                embeddings = self.codebooks[modality].weight
                distances = (x.square().sum(-1, keepdim=True)
                             - 2 * x @ embeddings.T
                             + embeddings.square().sum(-1)[None])
                codes = F.embedding(distances.argmin(-1), embeddings)
                quantized = x + (codes - x).detach()
                output[observed] = quantized.reshape_as(value[observed])
                decoded = self.decoders[modality](quantized.reshape(-1, self.num_heads * value.shape[-1]))
                raw_target = target[observed].detach().to(decoded.dtype)
                terms.append(F.mse_loss(decoded, raw_target)
                             + F.mse_loss(codes, x.detach())
                             + self.commitment_cost * F.mse_loss(x, codes.detach()))
            result[modality] = output
        zero = self.codebooks['audio'].weight.sum() * 0
        return result, torch.stack(terms).mean() if terms else zero


def _mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(),
                         nn.Linear(input_dim, output_dim))


class _NCECritic(nn.Module):
    """Author concatenated critic and InfoNCE / NCE-CLUB estimators."""

    def __init__(self, x_dim, y_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(x_dim + y_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def scores(self, x, y, freeze=False):
        n = x.shape[0]
        xy = torch.cat((x[None].expand(n, -1, -1),
                        y[:, None].expand(-1, n, -1)), -1)
        if freeze:
            # Critic fixed for encoder minimization; gradients through inputs
            # retained. Fitting is a separate detached-input objective.
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    xy = F.linear(xy, layer.weight.detach(), layer.bias.detach())
                else:
                    xy = layer(xy)
            return xy.squeeze(-1)
        return self.net(xy).squeeze(-1)

    def lower_loss(self, x, y, freeze=False):
        scores = self.scores(x, y, freeze)
        return scores.logsumexp(1).mean() - scores.diag().mean() - math.log(x.shape[0])

    def upper_loss(self, x, y):
        scores = self.scores(x, y, freeze=True)
        return scores.diag().mean() - scores.mean()


class FactorCL(nn.Module):
    """FactorCL-SUP with all five source projection branches per modality.

    Pairwise extension to three modalities; continuous sentiment replaces
    scalar class IDs and their conditional one-hot coding. The concatenated
    five branches are projected back to the existing latent dimension.
    ``include_critic_fit=False`` and ``critic_learning_loss`` permit source
    alternating updates; default combined loss separates the same gradients
    but updates simultaneously (an explicitly recorded protocol adaptation).
    """

    branches = ('shared_lower', 'unique_upper', 'task_lower',
                'unique_cond_lower', 'shared_cond_upper')

    def __init__(self, latent_dim=256, representation_dim=32, critic_hidden=128):
        super().__init__()
        self.heads = nn.ModuleDict({
            m: nn.ModuleDict({b: _mlp(latent_dim, representation_dim)
                              for b in self.branches}) for m in MODALITIES})
        self.output = nn.ModuleDict({
            m: nn.Linear(5 * representation_dim, latent_dim) for m in MODALITIES})
        self.pairs = tuple(combinations(MODALITIES, 2))
        self.critics = nn.ModuleDict()
        for a, b in self.pairs:
            key = a + '_' + b
            self.critics[key] = nn.ModuleDict({
                'shared_lower': _NCECritic(representation_dim, representation_dim, critic_hidden),
                'unique_upper': _NCECritic(representation_dim, representation_dim, critic_hidden),
                'unique_cond_lower': _NCECritic(representation_dim + 1, representation_dim + 1, critic_hidden),
                'shared_cond_upper': _NCECritic(representation_dim + 1, representation_dim + 1, critic_hidden)})
        self.task_critics = nn.ModuleDict({
            m: _NCECritic(representation_dim, 1, critic_hidden) for m in MODALITIES})

    def _represent(self, latents, masks):
        return {m: {b: torch.where(masks[m][..., None],
                                  head(torch.where(masks[m][..., None], latents[m], 0)), 0)
                    for b, head in self.heads[m].items()} for m in MODALITIES}

    def forward(self, latents, availability, umask):
        valid = umask.T.bool()
        masks = {m: valid & availability[..., i].bool()
                 for i, m in enumerate(MODALITIES)}
        reps = self._represent(latents, masks)
        output = {m: torch.where(masks[m][..., None], self.output[m](
            torch.cat([reps[m][b] for b in self.branches], -1)), 0) for m in MODALITIES}
        return output, {'representations': reps, 'masks': masks}

    def _losses(self, complete_latents, labels, umask):
        valid = umask.T.bool()
        if any(complete_latents[m].shape[:-1] != valid.shape for m in MODALITIES):
            raise ValueError('FactorCL expects complete train latents [L,B,D]')
        if labels.shape == umask.shape:
            labels = labels.T
        if labels.shape != valid.shape:
            raise ValueError('FactorCL labels must have shape [B,L] or [L,B]')
        # Complete inputs and gold live only in this training objective.
        reps = self._represent(complete_latents, {m: valid for m in MODALITIES})
        zero = self.output['audio'].weight.sum() * 0
        if valid.sum() < 2:
            return zero, zero
        y = labels[valid].to(complete_latents['audio'].dtype).unsqueeze(-1)
        reps = {m: {b: r[valid] for b, r in branches.items()} for m, branches in reps.items()}
        lower, upper, fit = [], [], []
        for a, b in self.pairs:
            # Six terms for each author two-view FactorCLSUP objective;
            # task relevance is repeated when a modality occurs in two pairs.
            for m in (a, b):
                lower.append(self.task_critics[m].lower_loss(reps[m]['task_lower'], y))
            critics = self.critics[a + '_' + b]
            for branch in ('shared_lower', 'unique_upper',
                           'unique_cond_lower', 'shared_cond_upper'):
                x, z = reps[a][branch], reps[b][branch]
                if 'cond' in branch:
                    x, z = torch.cat((x, y), -1), torch.cat((z, y), -1)
                if 'upper' in branch:
                    upper.append(critics[branch].upper_loss(x, z))
                    fit.append(critics[branch].lower_loss(x.detach(), z.detach()))
                else:
                    lower.append(critics[branch].lower_loss(x, z))
        return torch.stack(lower + upper).sum(), torch.stack(fit).sum()

    def training_loss(self, metadata, complete_latents, labels, umask, include_critic_fit=True):
        del metadata  # Inference metadata intentionally carries no gold.
        representation, fit = self._losses(complete_latents, labels, umask)
        return representation + fit if include_critic_fit else representation

    def critic_learning_loss(self, complete_latents, labels, umask):
        return self._losses(complete_latents, labels, umask)[1]

    def upper_critic_parameters(self):
        for critics in self.critics.values():
            for branch in ('unique_upper', 'shared_cond_upper'):
                yield from critics[branch].parameters()


class FishrPenalty(nn.Module):
    """Population per-example predictor gradient variances, author EMA rule.

    Pass ONLY all parameters of the original prediction head, as in the source
    classifier-only approximation. Group IDs are availability bit codes 0..7;
    negative IDs can mark padding. Absent groups do not advance their EMA.
    The task remains its original MSE; this returns an unweighted extra term.
    """

    def __init__(self, ema=.95, num_groups=8):
        super().__init__()
        if not 0 <= ema < 1:
            raise ValueError('ema must be in [0,1)')
        self.ema = float(ema)
        # Dynamic width determined by predictor scope; serializable buffers.
        self.register_buffer('ema_variances', torch.empty(num_groups, 0))
        self.register_buffer('updates', torch.zeros(num_groups, dtype=torch.long))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'ema_variances'
        if key in state_dict:
            self.ema_variances = self.ema_variances.new_empty(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def availability_groups(availability):
        bits = torch.tensor([1, 2, 4], device=availability.device)
        return (availability.bool().long() * bits).sum(-1)

    def penalty(self, loss_per_sample, parameters, group_ids):
        losses, groups = loss_per_sample.reshape(-1), group_ids.reshape(-1).long()
        if losses.shape != groups.shape:
            raise ValueError('loss and group IDs must have the same number of samples')
        params = tuple(p for p in parameters if p.requires_grad)
        if not params:
            raise ValueError('Fishr requires trainable prediction head parameters')
        active = groups >= 0
        losses, groups = losses[active], groups[active]
        if groups.numel() and groups.max() >= self.updates.numel():
            raise ValueError('availability group ID exceeds configured groups')
        zero = sum(p.sum() * 0 for p in params)
        if not losses.numel():
            return zero
        individual = []
        for loss in losses:
            grads = torch.autograd.grad(loss, params, create_graph=True,
                                        retain_graph=True, allow_unused=False)
            individual.append(torch.cat([g.reshape(-1) for g in grads]))
        gradients = torch.stack(individual)
        width = gradients.shape[-1]
        if self.ema_variances.shape[-1] == 0:
            history = gradients.new_zeros(self.updates.numel(), width)
            if self.training:
                self.ema_variances = history
        elif self.ema_variances.shape[-1] != width:
            raise ValueError('Fishr predictor parameter scope changed after initialization')
        else:
            history = self.ema_variances
        variances = []
        for group in groups.unique(sorted=True):
            samples = gradients[groups == group]
            variance = (samples - samples.mean(0)).square().mean(0)
            # Source MovingAverage(oneminusema_correction=True) divides every
            # step by (1-ema), NOT (1-ema**t); historic state is detached.
            moving = self.ema * history[group].detach().clone() + (1 - self.ema) * variance
            variances.append(moving / (1 - self.ema))
            if self.training:
                with torch.no_grad():
                    self.ema_variances[group].copy_(moving.detach())
                    self.updates[group].add_(1)
        if len(variances) < 2:
            return zero
        variances = torch.stack(variances)
        return (variances - variances.mean(0)).square().mean()


VQVAE = VQVAEValues
Fishr = FishrPenalty
