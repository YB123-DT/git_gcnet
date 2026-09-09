"""Detached, streaming probes; never a model parameter or forward input."""
import torch
from torch.nn import functional as F

MODALITIES = ('audio', 'text', 'visual')


class MemoryRetentionDiagnostics:
    """One collector per forward/batch. Sink receives scalar JSON-safe records.

    Probe state is O(B H 3 (dk+dv)); records are streamed, not retained here.
    sample_ids must be globally unique within a dataset/rate analysis input.
    """

    def __init__(self, sink, *, dataset, missing_rate, sample_ids, eps=1e-8):
        self.sink = sink
        self.dataset = dataset
        self.missing_rate = missing_rate
        self.sample_ids = list(sample_ids)
        self.eps = eps
        self.last_key = self.last_value = None
        self.last_seen_time = self.has_history = None

    @torch.no_grad()
    def observe(self, t, pre, decay, post, keys, values, availability, valid):
        # Detach before any probe computation, including state allocation.
        pre, decay, post = pre.detach(), decay.detach(), post.detach()
        keys, values = keys.detach(), values.detach()
        availability, valid = availability.detach().bool(), valid.detach().bool()
        batch, heads, _, _ = keys.shape
        if self.last_key is None:
            if len(self.sample_ids) != batch:
                raise ValueError('sample_ids must match batch size')
            self.last_key = torch.zeros_like(keys).transpose(-1, -2).contiguous()
            self.last_value = torch.zeros_like(values).transpose(-1, -2).contiguous()
            self.last_seen_time = torch.full((batch, 3), -1, device=keys.device, dtype=torch.long)
            self.has_history = torch.zeros((batch, 3), device=keys.device, dtype=torch.bool)
        for b in range(batch):
            if not bool(valid[b]):
                continue
            observed = availability[b]
            for m, name in enumerate(MODALITIES):
                if bool(observed[m]):
                    continue
                record = dict(dataset=self.dataset, missing_rate=self.missing_rate,
                              sample_id=self.sample_ids[b], time_index=t,
                              target_modality=name,
                              missing_pattern=''.join(str(int(v)) for v in observed.tolist()))
                if not bool(self.has_history[b, m]):
                    self.sink(dict(record, status='NO_HISTORY'))
                    continue
                k, v = self.last_key[b, :, m], self.last_value[b, :, m]
                metrics = {}
                for suffix, memory in (('pre', pre), ('decay', decay), ('post', post)):
                    prediction = (memory[b] @ k.unsqueeze(-1)).squeeze(-1)
                    denominator = v.norm(dim=-1) + self.eps
                    metrics['err_'+suffix] = (prediction-v).norm(dim=-1)/denominator
                    metrics['cos_'+suffix] = F.cosine_similarity(prediction,v,dim=-1,eps=self.eps)
                    metrics['norm_ratio_'+suffix] = prediction.norm(dim=-1)/denominator
                metrics['decay_damage'] = metrics['err_decay']-metrics['err_pre']
                metrics['write_damage'] = metrics['err_post']-metrics['err_decay']
                if bool(observed.any()):
                    current = keys[b, :, :, observed].transpose(-1,-2)
                    overlap = F.cosine_similarity(k.unsqueeze(1),current,dim=-1,eps=self.eps).abs().clamp(max=1)
                    metrics['max_key_overlap'] = overlap.max(dim=-1).values
                    metrics['mean_key_overlap'] = overlap.mean(dim=-1)
                for h in range(heads):
                    row = dict(record, status='retention', head=h,
                               history_distance=t-int(self.last_seen_time[b,m]),
                               max_key_overlap=None, mean_key_overlap=None)
                    row.update({key: float(value[h]) for key,value in metrics.items()})
                    self.sink(row)
            # Update only after measuring historical targets, from raw real slots.
            for m in range(3):
                if bool(observed[m]):
                    self.last_key[b,:,m].copy_(keys[b,:,:,m])
                    self.last_value[b,:,m].copy_(values[b,:,:,m])
                    self.last_seen_time[b,m] = t
                    self.has_history[b,m] = True
