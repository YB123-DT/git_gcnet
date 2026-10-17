"""Utterance-only MMIN with the existing GCNet pre-extracted-feature adaptation.

Temporal encoders are not added. FcClassifier/ResidualAE are vendored upstream
classes. Teacher access to missing training features is restricted to losses().
"""
import copy
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .networks import FcClassifier, ResidualAE


@dataclass(frozen=True)
class MMINConfig:
    input_dims: Tuple[int, int, int]
    num_classes: int
    embedding_dim: int = 128
    cls_layers: Tuple[int, ...] = (128, 128)
    ae_layers: Tuple[int, ...] = (256, 128, 64)
    n_blocks: int = 5
    dropout: float = 0.5
    teacher_dropout: float = 0.3
    ce_weight: float = 1.0
    reconstruction_weight: float = 4.0
    cycle_weight: float = 2.0
    use_bn: bool = False
    init_type: str = 'normal'
    init_gain: float = 0.02
    teacher_clip: float = 5.0
    missing_clip: float = 5.0


def _initialize(module, config):
    # Match the selected upstream/GCNet configuration; Kaiming ignores gain.
    if config.init_type not in ('normal', 'kaiming'):
        raise ValueError('unsupported MMIN initialization: ' + config.init_type)
    if isinstance(module, nn.Linear):
        if config.init_type == 'normal':
            nn.init.normal_(module.weight, mean=0.0, std=config.init_gain)
        else:
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def _classifier(config, in_dim, out_dim, dropout):
    return FcClassifier(in_dim, list(config.cls_layers), out_dim,
                        dropout=dropout, use_bn=config.use_bn)


def _masked_inputs(a, t, v, visible):
    features = (a, t, v)
    if any(x.ndim != 2 or x.shape[0] != a.shape[0] for x in features):
        raise ValueError('MMIN expects aligned utterance feature matrices [N,D]')
    if visible.shape != (a.shape[0], 3):
        raise ValueError('visible must have shape [N,3] in A/T/V order')
    if not torch.all((visible == 0) | (visible == 1)) or not torch.all(visible.sum(-1) >= 1):
        raise ValueError('visible must be binary with at least one visible modality')
    # where also removes NaN/Inf in features unavailable to the predictor.
    return tuple(torch.where(visible[:, i:i + 1].bool(), x, torch.zeros_like(x))
                 for i, x in enumerate(features))


class Teacher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.netA = _classifier(config, config.input_dims[0], config.embedding_dim, config.teacher_dropout)
        self.netT = _classifier(config, config.input_dims[1], config.embedding_dim, config.teacher_dropout)
        self.netV = _classifier(config, config.input_dims[2], config.embedding_dim, config.teacher_dropout)
        self.netC = _classifier(config, 3 * config.embedding_dim, config.num_classes, config.teacher_dropout)
        self.apply(lambda module: _initialize(module, config))

    def encode(self, a, t, v):
        return torch.cat([net(x)[0] for net, x in zip((self.netA, self.netT, self.netV), (a, t, v))], dim=-1)

    def forward(self, a, t, v):
        return self.netC(self.encode(a, t, v))[0]


class MMIN(nn.Module):
    def __init__(self, config, teacher):
        super().__init__()
        if config.input_dims != teacher.config.input_dims or config.num_classes != teacher.config.num_classes:
            raise ValueError('teacher input dimensions/classes are incompatible')
        if config.embedding_dim != teacher.config.embedding_dim or config.cls_layers != teacher.config.cls_layers:
            raise ValueError('teacher encoder dimensions are incompatible')
        self.config = config
        self.netA = _classifier(config, config.input_dims[0], config.embedding_dim, config.dropout)
        self.netT = _classifier(config, config.input_dims[1], config.embedding_dim, config.dropout)
        self.netV = _classifier(config, config.input_dims[2], config.embedding_dim, config.dropout)
        fusion_dim = config.embedding_dim * 3
        self.netAE = ResidualAE(list(config.ae_layers), config.n_blocks, fusion_dim, dropout=0, use_bn=False)
        self.netAE_cycle = ResidualAE(list(config.ae_layers), config.n_blocks, fusion_dim, dropout=0, use_bn=False)
        self.netC = _classifier(config, config.ae_layers[-1] * config.n_blocks, config.num_classes, config.dropout)
        self.apply(lambda module: _initialize(module, config))
        for target, source in zip((self.netA, self.netT, self.netV), (teacher.netA, teacher.netT, teacher.netV)):
            target.load_state_dict(source.state_dict())
        self.teacher = copy.deepcopy(teacher).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _features(self, a, t, v, visible):
        masked = _masked_inputs(a, t, v, visible)
        fusion = torch.cat([net(x)[0] for net, x in zip((self.netA, self.netT, self.netV), masked)], dim=-1)
        reconstruction, latent = self.netAE(fusion)
        return self.netC(latent)[0], fusion, reconstruction

    def forward(self, a, t, v, visible):
        # No teacher, reconstruction targets, labels, or future context needed.
        return self._features(a, t, v, visible)[0]

    def losses(self, a, t, v, visible, labels):
        if not self.training:
            raise RuntimeError('losses with complete training features are training-only')
        logits, fusion, reconstruction = self._features(a, t, v, visible)
        cycle, _ = self.netAE_cycle(reconstruction)
        # Original MMIN target is the teacher encoding of the complementary
        # modalities, not the complete teacher fusion. This is training-only.
        complement = tuple(torch.where(visible[:, i:i + 1].bool(), torch.zeros_like(x), x)
                           for i, x in enumerate((a, t, v)))
        with torch.no_grad():
            target = self.teacher.encode(*complement)
        ce = self.config.ce_weight * F.cross_entropy(logits, labels.long())
        recon = self.config.reconstruction_weight * F.mse_loss(reconstruction, target)
        cycle_loss = self.config.cycle_weight * F.mse_loss(cycle, fusion.detach())
        return {'total': ce + recon + cycle_loss, 'ce': ce,
                'reconstruction': recon, 'cycle': cycle_loss}


def _step(model, optimizer, losses, clip):
    if not torch.isfinite(losses['total']):
        raise FloatingPointError('non-finite MMIN training loss')
    optimizer.zero_grad(set_to_none=True)
    losses['total'].backward()
    # Upstream clips each named network separately.
    for name in ('netA', 'netT', 'netV', 'netC', 'netAE', 'netAE_cycle'):
        if hasattr(model, name):
            torch.nn.utils.clip_grad_norm_(getattr(model, name).parameters(), clip)
    optimizer.step()
    return {name: float(loss.detach().cpu()) for name, loss in losses.items()}


def train_teacher_step(teacher, optimizer, a, t, v, labels):
    teacher.train()
    loss = F.cross_entropy(teacher(a, t, v), labels.long())
    return _step(teacher, optimizer, {'total': loss}, teacher.config.teacher_clip)['total']


def train_missing_step(model, optimizer, a, t, v, visible, labels):
    model.train()
    return _step(model, optimizer, model.losses(a, t, v, visible, labels), model.config.missing_clip)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _teacher_signature(signature):
    required = {'dataset', 'fold', 'seed', 'feature_sha256', 'train_split_sha256', 'teacher_epochs'}
    if not required.issubset(signature):
        raise ValueError('teacher signature missing: ' + ','.join(sorted(required - set(signature))))
    if any(k in signature for k in ('mr', 'missing_rate', 'mask_rate')):
        raise ValueError('teacher identity must not depend on missing rate')
    return dict(signature)


def save_teacher(path, teacher, signature, epoch):
    """Write an immutable final training-epoch teacher, independently of MR.

    This is a weights artifact. Resuming optimization requires the run-level
    optimizer/RNG/progress checkpoint and is not implied by this function.
    """
    signature = _teacher_signature(signature)
    if epoch != signature['teacher_epochs']:
        raise ValueError('refusing to mark an incomplete teacher reusable')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'format_version': 1, 'config': asdict(teacher.config),
                'signature': signature, 'epoch': epoch,
                'selection': 'final_train_epoch',
                'state_dict': {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}}
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix='teacher-', suffix='.pt', delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(metadata, temp)
        os.link(temp, path)  # Atomic publication; never overwrite a reusable teacher.
    finally:
        temp.unlink(missing_ok=True)


def load_teacher(path, config, signature, device='cpu'):
    signature = _teacher_signature(signature)
    artifact = torch.load(path, map_location='cpu')
    if artifact.get('format_version') != 1:
        raise ValueError('unknown teacher checkpoint format')
    if _canonical(artifact['config']) != _canonical(asdict(config)):
        raise ValueError('teacher configuration mismatch')
    if _canonical(artifact['signature']) != _canonical(signature):
        raise ValueError('teacher identity mismatch')
    if artifact['epoch'] != signature['teacher_epochs']:
        raise ValueError('incomplete teacher checkpoint')
    teacher = Teacher(config)
    teacher.load_state_dict(artifact['state_dict'], strict=True)
    teacher.to(device).eval()
    metadata = {k: v for k, v in artifact.items() if k != 'state_dict'}
    metadata['checkpoint_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return teacher, metadata
