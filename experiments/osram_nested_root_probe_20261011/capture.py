"""Read-only capture of actual rooted states and pre-decoder graph outputs."""
import torch
from gcnet_missing_m3.meaningful_blocks_common import active_groups

class RootCollector:
    def __init__(self,block):
        self.block=block
        self.handles=[]
        self.chunks=[]
        self.columns=[]
        self.raw=None
        self.token=None

    def __enter__(self):
        def raw(m,args):
            assert self.raw is None, 'Only one Memory/readout scan per capture'
            self.raw=tuple(x.detach().clone() for x in args)
        def tokenizer(m,args,out):
            self.token=tuple(x.detach().clone() for x in out)
        def core_pre(m,args):
            self.columns.append(args[1].detach().clone())
        def rooted(m,args):
            assert args[0].shape[-1]==192
            self.chunks.append([args[0][...,64:128].detach().clone()])
        def core_post(m,args,out):
            self.chunks[-1].append(out.detach().clone())
        adapter=self.block.core
        self.handles=[self.block.register_forward_pre_hook(raw),
            adapter.tokenizer.register_forward_hook(tokenizer),
            adapter.core.register_forward_pre_hook(core_pre),
            adapter.core.readout.register_forward_pre_hook(rooted),
            adapter.core.register_forward_hook(core_post)]
        return self

    def __exit__(self,*args):
        for h in self.handles:h.remove()

    def finish(self):
        local,base,gap,av,umask=self.raw
        valid=umask.T.bool()
        active=torch.cat((valid[...,None],valid[...,None]&~av.bool()),-1)
        evidence=torch.cat((base[...,:512].unsqueeze(2),
            torch.where(active[...,1:,None],gap[...,:512],0.)),2)
        history=valid&(valid.long().cumsum(0)>1)&evidence.ne(0).any(-1).any(-1)
        output=[local.new_zeros(*local.shape[:2],33*64) for _ in range(2)]
        role_active=torch.zeros(*local.shape[:2],33,dtype=torch.bool,device=local.device)
        if history.any():
            tokens,mask=self.token
            groups=list(active_groups(tokens,mask))
            assert len(groups)==len(self.chunks)==len(self.columns)
            full=[torch.zeros_like(tokens) for _ in range(2)]
            for (rows,columns,_),seen,chunk in zip(groups,self.columns,self.chunks):
                assert torch.equal(columns,seen)
                for target,value in zip(full,chunk):
                    target[rows[:,None],columns[None,:]]=value
            for target,value in zip(output,full):
                target[history]=value.flatten(1)
            role_active[history]=mask
        assert all(torch.isfinite(v).all() for v in output)
        return *output,role_active
