"""Read-only history-window replay; no model, sampler or training changes."""
import torch

MODALITIES = ('audio','text','visual')


class ScanCapture:
    """Capture the actual plain causal scan inputs, without re-projecting."""
    def __init__(self, osram):
        self.osram, self.calls = osram, 0

    def __enter__(self):
        self.original = self.osram._scan
        self.had_instance = '_scan' in self.osram.__dict__
        self.instance = self.osram.__dict__.get('_scan')

        def capture(keys,values,queries,availability,valid,reverse,**kwargs):
            if reverse or self.calls or any(v is not None for v in kwargs.values()):
                raise ValueError('Require exactly one unmodified causal Flat scan')
            self.calls += 1
            self.keys = {m:v.detach().clone() for m,v in keys.items()}
            self.values = {m:v.detach().clone() for m,v in values.items()}
            self.queries = queries.detach().clone()
            self.availability = availability.detach().clone()
            self.valid = valid.detach().clone()
            result = self.original(keys,values,queries,availability,valid,reverse,**kwargs)
            self.base,self.gap = (v.detach().clone() for v in result[:2])
            return result

        self.osram._scan = capture
        return self

    def __exit__(self,*args):
        if self.had_instance:
            self.osram._scan = self.instance
        else:
            del self.osram._scan


def read(osram,memory,queries,keys,availability,valid):
    """Original read after target decay, before target write."""
    count = len(memory)
    scale = valid.to(queries.dtype)[:,None]
    base = (memory @ queries[:,0].unsqueeze(-1)).squeeze(-1).reshape(count,-1)*scale
    strength = 0. if osram.osram_gap_read == 'raw' else osram.gap_residual_strength
    gap = []
    for m in range(3):
        q = queries[:,m+1]
        effective = q-strength*(q-osram._address_residual(keys,q))
        r = (memory @ effective.unsqueeze(-1)).squeeze(-1).reshape(count,-1)
        gap.append(r*(1.-availability[:,m].to(r.dtype))[:,None]*scale)
    return base,torch.stack(gap,1)


@torch.no_grad()
def replay_reads(osram,trace,window,chunk_size=64):
    if window is not None and (type(window) is not int or window < 0):
        raise ValueError('window must be a nonnegative integer or None (full)')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    q,a,valid = trace.queries,trace.availability,trace.valid
    length,batch = valid.shape
    keys = torch.stack([trace.keys[m] for m in MODALITIES],-1)*a[...,None,None,:]
    values = torch.stack([trace.values[m] for m in MODALITIES],-1)*a[...,None,None,:]
    base = q.new_zeros(length,batch,osram.num_heads*osram.value_dim)
    gap = q.new_zeros(length,batch,3,osram.num_heads*osram.value_dim)
    alpha = osram.alpha_logits.sigmoid().to(q.dtype).view(1,osram.num_heads,1,1)
    beta = osram.beta_logits.sigmoid().to(q.dtype)
    if window is None:
        memory = q.new_zeros(batch,osram.num_heads,osram.value_dim,osram.key_dim)
        for t in range(length):
            active = valid[t]
            if not bool(active.any()):
                continue
            memory = torch.where(active[:,None,None,None],memory*alpha,memory)
            base[t],gap[t] = read(osram,memory,q[t],keys[t],a[t],active)
            updated = osram.block_write(memory,keys[t],values[t],a[t],beta=beta)
            memory = torch.where(active[:,None,None,None],updated,memory)
        return base,gap
    if window == 0:
        return base,gap
    targets,history = [],[]
    for b in range(batch):
        times = valid[:,b].nonzero().flatten().tolist()
        for rank,t in enumerate(times):
            past = times[max(0,rank-window):rank]
            targets.append((t,b))
            history.append([-1]*(window-len(past))+past)
    for start in range(0,len(targets),chunk_size):
        selected = targets[start:start+chunk_size]
        t,b = torch.tensor(selected,device=q.device).unbind(-1)
        prior = torch.tensor(history[start:start+chunk_size],device=q.device)
        memory = q.new_zeros(len(selected),osram.num_heads,osram.value_dim,osram.key_dim)
        for j in range(window):
            active = prior[:,j] >= 0
            index = prior[:,j].clamp_min(0)
            mask = a[index,b]*active[:,None]
            memory = torch.where(active[:,None,None,None],memory*alpha,memory)
            updated = osram.block_write(memory,keys[index,b],values[index,b],mask,beta=beta)
            memory = torch.where(active[:,None,None,None],updated,memory)
        # The target's own decay is part of the original read-before-write.
        base[t,b],gap[t,b] = read(osram,memory*alpha,q[t,b],keys[t,b],a[t,b],valid[t,b])
    return base,gap


@torch.no_grad()
def decode(model,frame,base,gap):
    """Whole-batch original Flat, with original Local/Skip and zero backward."""
    osram = model.osram
    base = torch.cat((base,torch.zeros_like(base)),-1)
    gap = torch.cat((gap,torch.zeros_like(gap)),-1)
    gap = gap*(1.-frame.availability.to(gap.dtype))[...,None]
    inputs = torch.cat((frame.local,base,gap.flatten(-2)),-1)
    anchor = osram.local_skip(frame.local.contiguous())+osram.emotion_adapter(inputs)
    hidden = osram.emotion_norm(anchor)*frame.valid[...,None]
    return model.smax_fc(hidden)
