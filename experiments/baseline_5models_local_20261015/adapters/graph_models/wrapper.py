"""Thin adapters around vendored original graph models (no simplified GNNs).

Forward inputs are time-major masked A/T/V concatenation, integer speaker IDs
[B,T], utterance validity [B,T], and dialogue lengths. Complete training features
are accepted only by the loss helper, never the model forward path.
"""
from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F
from .gcnet.model import GraphModel as GCNet
from .sdrgnn.model_SDRGNN import GraphModel as SDRGNN


def _dims(dims):
    if isinstance(dims, Mapping):
        return tuple(int(dims[key]) for key in ('a', 't', 'v'))
    if len(dims) != 3:
        raise ValueError('dims must have A/T/V dimensions')
    return tuple(map(int, dims))


def _float_alias(config, key, official_key, default):
    values = [float(config[name]) for name in (key, official_key) if name in config]
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError(f'Conflicting {key} and {official_key} values')
    return values[0] if values else default


class GraphAdapter(nn.Module):
    def __init__(self, config, dims, nclasses, nspeakers):
        super().__init__()
        self.config = dict(config)
        self.config['dropout'] = _float_alias(config, 'drop_rate', 'dropout', .5)
        self.config['recon_loss_weight'] = _float_alias(
            config, 'recon_weight', 'recon_loss_weight', .5)
        self.dims = _dims(dims)
        model_name = str(config.get('model', config.get('model_name', 'GCNet'))).lower().replace('-', '').replace('_','')
        self.is_sdr = 'sdr' in model_name
        hidden = int(config.get('hidden', 200))
        cls = SDRGNN if self.is_sdr else GCNet
        self.model = cls(
            base_model=config.get('base_model', 'GRU' if self.is_sdr else 'LSTM'),
            adim=self.dims[0], tdim=self.dims[1], vdim=self.dims[2],
            D_e=hidden, graph_hidden_size=int(config.get('graph_hidden', hidden // 2)),
            n_speakers=nspeakers,
            window_past=int(config.get('window_past', config.get('windowp', 3 if self.is_sdr else 2))),
            window_future=int(config.get('window_future', config.get('windowf', 3 if self.is_sdr else 2))),
            n_classes=nclasses, dropout=self.config['dropout'],
            time_attn=bool(config.get('time_attn',False)), no_cuda=True,
        )
        self.version = ('SDR-GNN-Implementation-fixed' if self.is_sdr else 'GCNet-Original')
        if nspeakers > 2:
            self.version += '-MELD-adapted'

    def forward(self, x, speakers, valid, lengths):
        if x.ndim != 3 or x.shape[-1] != sum(self.dims):
            raise ValueError('x must be [T,B,A+T+V]')
        if speakers.shape != valid.shape or tuple(valid.shape) != (x.shape[1],x.shape[0]):
            raise ValueError('speakers and valid must be [B,T]')
        if len(lengths) != x.shape[1] or any(int(n) < 1 for n in lengths):
            raise ValueError('one positive dialogue length is required per batch element')
        # Original internals use .cuda() only when no_cuda=False. CUDA_VISIBLE_DEVICES
        # is assigned by the local dispatcher; its sole GPU appears as cuda:0.
        for module in self.model.modules():
            if hasattr(module, 'no_cuda'):
                module.no_cuda = x.device.type != 'cuda'
        logits, reconstruction, hidden = self.model([x], speakers, valid, lengths)
        return {'logits': logits, 'reconstruction': reconstruction[0], 'hidden': hidden}

    def loss(self, outputs, labels, target, availability, valid):
        """Original objective, correcting the upstream T/B padding-mask alignment.

        labels/valid are B,T; target is T,B,D; availability T,B,3 is 1=visible.
        GCNet: CE + reconstructed missing-feature modality-normalized MSE.
        SDR: (1-w)*CE + w*reconstruction, w=.5 by default.
        """
        selector = valid.bool().transpose(0,1)
        logits = outputs['logits']
        ce = F.cross_entropy(logits[selector],labels.transpose(0,1)[selector])
        squared = (outputs['reconstruction'] - target).square()
        recon = squared.sum() * 0.
        start = 0
        for modality, width in enumerate(self.dims):
            missing = (1-availability[...,modality])*selector
            recon = recon + (squared[...,start:start+width]*missing.unsqueeze(-1)).sum()/width
            start += width
        recon = recon/selector.sum().clamp_min(1)
        if not bool(self.config.get('loss_recon', True)):
            total = ce
        elif self.is_sdr:
            weight = float(self.config.get('recon_loss_weight',.5))
            total = (1-weight)*ce+weight*recon
        else:
            total = ce+recon
        return total, {'classification':ce, 'reconstruction':recon}


def build_model(config, dims, nclasses, nspeakers):
    return GraphAdapter(config,dims,nclasses,nspeakers)
