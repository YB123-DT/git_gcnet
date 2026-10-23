"""TRAIN-only exemplar reading; experimental wrapper, not a CHM change.

TabR adaptation: shared keys, negative squared L2, and label/difference Values.
The caller must construct the support bank from TRAIN indices only. No target
labels are accepted by the forward interface. Bank identity belongs in run and
checkpoint metadata; raw examples are intentionally not model state_dict data.
"""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn


class TrainSupportBank:
    def __init__(self, features, labels, conversation_ids, utterance_indices):
        self.features = features.detach().cpu().contiguous().clone()
        self.labels = labels.detach().cpu().reshape(-1).contiguous().clone()
        self.conversation_ids = tuple(map(str, conversation_ids))
        self.utterance_indices = tuple(map(int, utterance_indices))
        n = len(self.conversation_ids)
        if (not n or self.features.ndim != 2 or self.features.shape[0] != n
                or self.labels.shape != (n,) or len(self.utterance_indices) != n
                or not torch.isfinite(self.features).all() or not torch.isfinite(self.labels).all()):
            raise ValueError('Support bank must contain finite, aligned nonempty TRAIN examples')
        if len(set(zip(self.conversation_ids, self.utterance_indices))) != n:
            raise ValueError('Duplicate TRAIN utterance IDs')

    def manifest(self):
        ids = json.dumps(list(zip(self.conversation_ids, self.utterance_indices)),
                         separators=(',', ':')).encode()
        return dict(N=len(self.labels), feature_dim=self.features.shape[-1],
                    conversations=len(set(self.conversation_ids)),
                    ids_sha256=hashlib.sha256(ids).hexdigest(),
                    features_sha256=hashlib.sha256(self.features.numpy().tobytes()).hexdigest(),
                    labels_sha256=hashlib.sha256(self.labels.numpy().tobytes()).hexdigest(),
                    support_split='TRAIN', same_conversation_excluded=True)


class TrainExemplarReadout(nn.Module):
    def __init__(self, local_dim=256, output_dim=1600, *, key_dim=128,
                 neighbors=16, use_labels=True):
        super().__init__()
        if min(local_dim, output_dim, key_dim, neighbors) <= 0:
            raise ValueError('Positive dimensions and neighbor count required')
        self.neighbors, self.use_labels = int(neighbors), bool(use_labels)
        self.norm = nn.LayerNorm(local_dim)
        self.key = nn.Linear(local_dim, key_dim)
        self.label = nn.Linear(1, key_dim)
        self.difference = nn.Sequential(nn.Linear(key_dim, key_dim), nn.ReLU(),
                                        nn.Linear(key_dim, key_dim, bias=False))
        self.output = nn.Linear(key_dim, output_dim)

    def forward(self, local, support_local, support_labels, query_ids, support_ids):
        if len(query_ids) != len(local) or len(support_ids) != len(support_local):
            raise ValueError('Retrieval ID/feature alignment differs')
        if not len(support_local):
            return self.output.weight.new_zeros(len(local), self.output.out_features), torch.empty(
                len(local), 0, dtype=torch.long, device=local.device)
        q = self.key(self.norm(local))
        k = self.key(self.norm(support_local))
        # Do not normalize keys or add a temperature: preserve key-only L2.
        scores = -(q[:, None, :] - k[None, :, :]).square().sum(-1)
        legal = torch.tensor([[str(qid) != str(sid) for sid in support_ids]
                              for qid in query_ids], device=local.device, dtype=torch.bool)
        scores = scores.masked_fill(~legal, -torch.inf)
        top, indices = scores.topk(min(self.neighbors, len(k)), dim=-1)
        active = torch.isfinite(top)
        has_neighbor = active.any(-1)
        # Entire-conversation exclusion can leave no legal neighbor. Prevent
        # softmax(-inf,...,-inf), then hard-zero both Value and output bias.
        safe_top = torch.where(has_neighbor[:, None], top, torch.zeros_like(top))
        weights = safe_top.softmax(-1) * active.to(top.dtype)
        neighbor_y = support_labels.to(local).reshape(-1)[indices, None]
        if not self.use_labels:
            neighbor_y = torch.zeros_like(neighbor_y)
        values = self.label(neighbor_y) + self.difference(q[:, None, :] - k[indices])
        pooled = (weights[..., None] * values).sum(1)
        delta = self.output(pooled)
        delta = torch.where(has_neighbor[:, None], delta, torch.zeros_like(delta))
        return delta, torch.where(active, indices, torch.full_like(indices, -1))


class ExemplarAugmentedModel(nn.Module):
    """One causal backbone scan; supports only run shared current encoders."""
    def __init__(self, backbone, bank, *, use_labels=True, seed=66):
        super().__init__()
        if backbone.backbone_type != 'osram' or backbone.osram_readout_fusion != 'flat':
            raise ValueError('Exemplar wrapper requires the original OSRAM Flat')
        self.backbone, self.bank, self.enabled = backbone, bank, True
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 314627)
            self.retrieval = TrainExemplarReadout(backbone.osram.latent_dim,
                                                 backbone.osram.output_dim,
                                                 use_labels=use_labels)
        self.retrieval.to(next(backbone.parameters()).device)
        self._support_cache = {}
        self.last_diagnostics = {}

    def train(self, mode=True):
        self._support_cache.clear()
        return super().train(mode)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        self._support_cache.clear()
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def _support_local(self, pattern, device):
        if not self.training and pattern in self._support_cache:
            return self._support_cache[pattern]
        features = self.bank.features.to(device)
        n = len(features)
        availability = features.new_tensor(pattern)[None, None, :].expand(1, n, 3)
        valid = features.new_ones(n, 1)
        # Protect original query Dropout and mask RNG streams. Supports have
        # genuine gradients; this is not frozen or stale-bank encoding.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            node, _ = self.backbone.observed_set(features[None], availability, valid)
            local = node + self.backbone.osram.local_path(node)
        result = local[0]
        if not self.training:
            self._support_cache[pattern] = result.detach()
        return result

    def forward(self, view):
        # Never read view['labels'] or its complete raw current input.
        args = ([view['incomplete']], view['availability'], view['qmask'],
                view['umask'], view['lengths'])
        if not self.enabled:
            return self.backbone(*args, predict_missing=False)[0]
        conversation_ids = tuple(map(str, view['conversation_ids']))
        if len(conversation_ids) != view['umask'].shape[0]:
            raise ValueError('Query conversation IDs are not batch aligned')
        indices_by_pattern = []
        callback_calls = 0

        def residual(local, availability, umask):
            nonlocal callback_calls
            callback_calls += 1
            valid = umask.T.bool()
            out = local.new_zeros(*local.shape[:2], self.retrieval.output.out_features)
            for bits in availability[valid].long().unique(dim=0).tolist():
                pattern = tuple(bits)
                if not any(pattern):
                    raise ValueError('Valid query must retain a modality')
                selected = valid & (availability == availability.new_tensor(bits)).all(-1)
                batch_indices = selected.nonzero(as_tuple=True)[1].tolist()
                query_ids = [conversation_ids[b] for b in batch_indices]
                support = self._support_local(pattern, local.device)
                delta, indices = self.retrieval(local[selected], support, self.bank.labels,
                                               query_ids, self.bank.conversation_ids)
                out[selected] = delta
                indices_by_pattern.append((pattern, indices.detach().cpu()))
            self.last_diagnostics = dict(valid_N=int(valid.sum()),
                residual_norm=float(out[valid].detach().norm(dim=-1).mean()) if valid.any() else 0.,
                neighbor_N=sum(int(index.ge(0).sum()) for _, index in indices_by_pattern))
            return out

        result = self.backbone(*args, predict_missing=False, flat_anchor_residual=residual)[0]
        if callback_calls != 1:
            raise RuntimeError('Exemplar callback must run exactly once on the original Flat path')
        # Classifier bias can make padding logits nonzero; enforce the wrapper
        # contract without changing any valid baseline prediction.
        return torch.where(view['umask'].T.bool()[..., None], result, torch.zeros_like(result))
