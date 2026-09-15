"""Train a fixed Text target subspace; leave the 256d online predictor intact."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .b2_training import file_sha256, state_sha256
from .loss import MissingM3Loss, _symmetric_info_nce

FORMAT = 'text-predictable-subspace-v1'
PATTERNS = ('A', 'V', 'AV')


def variance_covariance(z):
    """Pre-normalization safeguards, not a guarantee of full effective rank."""
    if z.ndim != 2 or z.shape[1] < 1:
        raise ValueError('Subspace samples must be [N,D]')
    if z.shape[0] < 2:
        zero = z.sum() * 0.
        return zero, zero
    centered = z - z.mean(0)
    covariance = centered.T @ centered / (z.shape[0] - 1)
    variance = F.relu(1. - torch.sqrt(covariance.diag() + 1e-4)).mean()
    off_diagonal = covariance - torch.diag_embed(covariance.diag())
    return variance, off_diagonal.square().sum() / z.shape[1]


@torch.no_grad()
def representation_stats(z):
    z = z.detach().double()
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError('Representation statistics require nonempty [N,D]')
    centered = z - z.mean(0)
    covariance = centered.T @ centered / max(z.shape[0] - 1, 1)
    std = covariance.diag().clamp_min(0).sqrt()
    spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = spectrum.sum()
    rank = 0.
    if total > 0:
        p = spectrum / total
        p = p[p > 0]
        rank = float(torch.exp(-(p * p.log()).sum()))
    off = covariance - torch.diag_embed(covariance.diag())
    return dict(per_dim_std=std.cpu().tolist(), mean_std=float(std.mean()),
                effective_rank=rank,
                covariance_off_diagonal=float(off.square().sum() / z.shape[1]),
                # A conservative flag, not an assertion that rank=32 is necessary.
                collapsed=bool(std.mean() < 1e-4 or rank < 1.5),
                count=z.shape[0], dimension=z.shape[1])


class TextSubspacePretrainer(nn.Module):
    """R/C/Q only. Inputs are frozen Teacher latents, never OSRAM context."""

    def __init__(self, latent_dim=256, subspace_dim=32, hidden_dim=128):
        super().__init__()
        if latent_dim < 1 or subspace_dim != 32 or hidden_dim < 1:
            raise ValueError('Positive dimensions and fixed subspace_dim=32 required')
        self.latent_dim = latent_dim
        self.subspace_dim = subspace_dim
        self.hidden_dim = hidden_dim
        self.projector = nn.Linear(latent_dim, subspace_dim)
        self.sentiment = nn.Linear(subspace_dim, 1)
        self.predictor = nn.Sequential(
            nn.Linear(2 * latent_dim + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, subspace_dim))

    def predict(self, audio, visual, pattern):
        if pattern not in PATTERNS:
            raise ValueError('pattern must be A, V, or AV')
        if audio.ndim != 2 or audio.shape != visual.shape or audio.shape[1] != self.latent_dim:
            raise ValueError('Audio/Visual must have shape [N,latent_dim]')
        # Fixed slots distinguish modality identity; bits describe observation, not confidence.
        a, v = int('A' in pattern), int('V' in pattern)
        bits = audio.new_tensor([a, v]).expand(audio.shape[0], -1)
        return self.predictor(torch.cat((audio.detach() * a, visual.detach() * v, bits), -1))

    def loss(self, latents, labels, beta=1.):
        if beta < 0:
            raise ValueError('beta must be nonnegative')
        text = latents['text'].detach()
        if text.ndim != 2 or text.shape != (labels.numel(), self.latent_dim) or text.shape[0] == 0:
            raise ValueError('Expected nonempty valid Teacher latents aligned with labels')
        target = self.projector(text)  # IMPORTANT: do not detach R output.
        sentiment = F.mse_loss(self.sentiment(target).squeeze(-1), labels.reshape(-1).detach())
        prediction = torch.stack([
            F.smooth_l1_loss(self.predict(latents['audio'], latents['visual'], p), target)
            for p in PATTERNS]).mean()
        variance, covariance = variance_covariance(target)
        return dict(total=sentiment + prediction + beta * (variance + covariance),
                    sentiment=sentiment, predictability=prediction,
                    variance=variance, covariance=covariance)


def text_jepa_loss(predictions, teacher_targets, subspace=None, temperature=.03):
    """One target per missing-Text utterance; fixed .5/.5 even with singleton NCE."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if subspace is not None and any(p.requires_grad for p in subspace.parameters()):
        raise ValueError('Stage2 subspace must be frozen')
    selected = predictions.target_mask[..., 1]
    count = int(selected.sum())
    zero = (predictions.reg_predictions.sum() + predictions.cl_predictions.sum()) * 0.
    if count == 0:
        return MissingM3Loss(zero, zero, zero, 0)
    reg = predictions.reg_predictions[..., 1, :][selected]
    cl = predictions.cl_predictions[..., 1, :][selected]
    target = teacher_targets['text'][selected].detach()
    if subspace is not None:
        # Frozen parameters do NOT mean no_grad on prediction projection.
        reg, cl = subspace(reg), subspace(cl)
        with torch.no_grad():
            target = subspace(target)
    regression = F.smooth_l1_loss(reg, target)
    contrastive = _symmetric_info_nce(cl, target, temperature) if count >= 2 else zero
    return MissingM3Loss(.5 * regression + .5 * contrastive, regression, contrastive, count)


def save_subspace(path, model, teacher_sha256, epoch, validation_loss, config):
    """Stage-1 artifact; Stage 2 imports only projector, never Q/C or Student weights."""
    projector = {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()}
    payload = dict(format=FORMAT, projector=projector,
                   projector_sha256=state_sha256(projector),
                   teacher_projector_sha256=teacher_sha256,
                   latent_dim=model.latent_dim, subspace_dim=model.subspace_dim,
                   selection_split='validation', selection_metric='subspace_validation_loss',
                   validation_loss=float(validation_loss), epoch=int(epoch), config=config,
                   stage1_model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_frozen_subspace(path, teacher_sha256, latent_dim):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    if (cp.get('format') != FORMAT or cp.get('selection_split') != 'validation'
            or cp.get('selection_metric') != 'subspace_validation_loss'):
        raise ValueError('Subspace must be a validation-selected Stage1 artifact')
    if cp.get('teacher_projector_sha256') != teacher_sha256:
        raise ValueError('Subspace Teacher hash differs from Stage2 Teacher')
    if cp.get('latent_dim') != latent_dim or cp.get('subspace_dim') != 32:
        raise ValueError('Subspace dimensions differ (expected latent_dim→32)')
    state = cp.get('projector', {})
    if (set(state) != {'weight', 'bias'}
            or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values())
            or state_sha256(state) != cp.get('projector_sha256')):
        raise ValueError('Invalid subspace projector state/hash')
    with torch.random.fork_rng(devices=[]):
        projector = nn.Linear(latent_dim, 32)
    for k, v in projector.state_dict().items():
        if state[k].shape != v.shape or state[k].dtype != v.dtype:
            raise ValueError(f'Subspace shape/dtype mismatch: {k}')
    projector.load_state_dict(state, strict=True)
    projector.requires_grad_(False).eval()
    return projector, dict(checkpoint=str(Path(path).resolve()), checkpoint_sha256=file_sha256(path),
                           projector_sha256=cp['projector_sha256'],
                           teacher_projector_sha256=teacher_sha256,
                           epoch=cp['epoch'], selection_split='validation',
                           validation_loss=cp['validation_loss'], config=cp['config'])
