"""Temporary inference-only write interventions; never modify trained weights."""
import torch


@torch.no_grad()
def protect_update(delta, history, protect_mask, ridge=1e-3, eps=1e-8):
    if ridge<=0:raise ValueError('protection ridge must be positive')
    h=history*protect_mask[:,None,None,:].to(history.dtype)
    gram=h.transpose(-1,-2)@h
    eye=torch.eye(3,device=h.device,dtype=h.dtype)
    # Delta (I-H (H^T H + ridge I)^-1 H^T); never explicitly invert.
    projection=torch.linalg.solve(gram+ridge*eye,h.transpose(-1,-2))
    protected=delta-(delta@h)@projection
    dn=delta.norm(dim=(-2,-1));pn=protected.norm(dim=(-2,-1))
    ratio=pn/(dn+eps)
    global_update=delta*ratio[...,None,None]
    active=protect_mask.any(-1)[:,None,None,None]
    # No eligible history must be EXACT identity, not epsilon weakening.
    protected=torch.where(active,protected,delta)
    global_update=torch.where(active,global_update,delta)
    ratio=torch.where(protect_mask.any(-1)[:,None],ratio,torch.ones_like(ratio))
    return protected,global_update,ratio


class WriteIntervention:
    """Context-managed method override, reset historical keys for each forward.

    Global matches its own same-state hypothetical protected update per head.
    Different intervention trajectories need not have equal subsequent norms.
    Fixed-step modes scale every update, even complete/no-history steps.
    Keys are from real observed slots only. No parameter or state_dict additions.
    """
    def __init__(self,backbone,mode,ridge=1e-3):
        if mode not in ('reference','protected','global','fixed0.95','fixed0.9','fixed0.8'):raise ValueError(mode)
        if backbone.training or backbone.bidirectional:raise ValueError('eval forward-only required')
        self.model=backbone;self.mode=mode;self.ridge=ridge;self.records=[]
        self.observed_records=[]

    def _reset(self,module,args,kwargs):
        if module.training:raise ValueError('No training with inference intervention')
        self.valid=args[4].T.bool()
        self.times=self.valid.any(-1).nonzero().flatten().tolist()
        self.index=0;self.history=None;self.seen=None

    @torch.no_grad()
    def _write(self,memory,keys,values,availability,beta=None):
        baseline=self.original(memory,keys,values,availability,beta=beta)
        t=self.times[self.index];self.index+=1
        active=self.valid[t]
        if self.history is None:
            self.history=torch.zeros_like(keys)
            self.seen=torch.zeros_like(availability,dtype=torch.bool)
        observed=availability.bool() & active[:,None]
        mask=self.seen & ~availability.bool() & active[:,None]
        delta=baseline-memory
        protected,glob,ratio=protect_update(delta,self.history,mask,self.ridge)
        if self.mode=='reference':
            post=baseline
        elif self.mode in ('fixed0.95','fixed0.9','fixed0.8'):
            # Unconditional global control, including complete input/no history.
            strength={'fixed0.95':.95,'fixed0.9':.9,'fixed0.8':.8}[self.mode]
            post=memory+strength*delta
        else:
            update=protected if self.mode=='protected' else glob
            post=torch.where(mask.any(-1)[:,None,None,None],memory+update,baseline)
        # Current observed association fit. All tensors are detached by no_grad;
        # scores do not feed back into memory or choose an intervention strength.
        denominator=values.norm(dim=-2)+1e-8
        err_before=(memory@keys-values).norm(dim=-2)/denominator
        err_after=(post@keys-values).norm(dim=-2)/denominator
        err_original=(baseline@keys-values).norm(dim=-2)/denominator
        for b in active.nonzero().flatten().tolist():
            for h in range(keys.shape[1]):
                self.records.append(dict(time_index=t,batch_index=b,head=h,
                    has_protection=bool(mask[b].any()),original_norm=float(delta[b,h].norm()),
                    protected_norm=float(protected[b,h].norm()),global_norm=float(glob[b,h].norm()),
                    global_scale=float(ratio[b,h]),applied_norm=float((post-memory)[b,h].norm())))
                for m in observed[b].nonzero().flatten().tolist():
                    before=float(err_before[b,h,m]);after=float(err_after[b,h,m])
                    self.observed_records.append(dict(time_index=t,batch_index=b,head=h,
                        target_modality=('audio','text','visual')[m],
                        missing_pattern=availability[b].int().tolist(),
                        has_protection=bool(mask[b].any()),err_before=before,err_after=after,
                        err_original_after=float(err_original[b,h,m]),fit_gain=before-after))
        self.history=torch.where(observed[:,None,None,:],keys.detach(),self.history)
        self.seen|=observed
        return post

    def __enter__(self):
        if 'block_write' in self.model.__dict__:raise ValueError('Existing block_write override')
        self.original=self.model.block_write
        self.hook=self.model.register_forward_pre_hook(self._reset,with_kwargs=True)
        self.model.block_write=self._write
        return self

    def __exit__(self,*exc):
        del self.model.block_write
        self.hook.remove()
