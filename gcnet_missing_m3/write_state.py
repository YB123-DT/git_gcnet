"""Predict write addresses/content; only subsequent reads consume the update."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

MODALITIES=('audio','text','visual')


class WriteStatePredictor(nn.Module):
    def __init__(self, latent_dim, heads, key_dim, value_dim, parameter_budget):
        super().__init__()
        self.heads,self.key_dim,self.value_dim=heads,key_dim,value_dim
        self.input_dim=2*latent_dim+2*heads*value_dim+16
        self.output_dim=heads*(key_dim+value_dim)
        self.parameter_budget=parameter_budget
        # Parameter count is affine in width, so only floor/ceil can be nearest.
        ideal=(parameter_budget-(2*self.input_dim+48+self.output_dim))/(self.input_dim+self.output_dim+1)
        widths={max(1,math.floor(ideal)),max(1,math.ceil(ideal))}
        self.width=min(widths,key=lambda w:(abs(self.parameter_count(w)-parameter_budget),w))
        self.target_type=nn.Embedding(3,16)
        self.norm=nn.LayerNorm(self.input_dim)
        self.mlp=nn.Sequential(nn.Linear(self.input_dim,self.width),nn.GELU(),
                               nn.Linear(self.width,self.output_dim))

    def parameter_count(self,width):
        return 48+2*self.input_dim+self.output_dim+width*(self.input_dim+self.output_dim+1)

    def forward(self,node,observed_mean,base,gap,availability,valid):
        batch=node.shape[0]
        selected=valid[:,None] & ~availability.bool()
        keys=node.new_zeros(batch,self.heads,self.key_dim,3)
        values=node.new_zeros(batch,self.heads,self.value_dim,3)
        rows,targets=selected.nonzero(as_tuple=True)
        if len(rows):
            inputs=torch.cat([node[rows],observed_mean[rows],base[rows],gap[rows,targets],
                              self.target_type(targets)],-1)
            out=self.mlp(self.norm(inputs))
            key,value=torch.split(out,[self.heads*self.key_dim,self.heads*self.value_dim],-1)
            # Assign through [B,3,H,d] views to keep target indexing explicit.
            keys=keys.permute(0,3,1,2).contiguous()
            values=values.permute(0,3,1,2).contiguous()
            keys[rows,targets]=F.normalize(key.reshape(-1,self.heads,self.key_dim),dim=-1)
            values[rows,targets]=value.reshape(-1,self.heads,self.value_dim)
            keys=keys.permute(0,2,3,1);values=values.permute(0,2,3,1)
        return keys,values


class WriteStateTeacher(nn.Module):
    def __init__(self,encoder,osram):
        super().__init__()
        self.encoder=copy.deepcopy(encoder)
        for name in ('latent_norm','node_norm','key_projectors','value_projectors',
                     'speaker_embedding','availability_embedding'):
            setattr(self,name,copy.deepcopy(getattr(osram,name)))
        self.heads,self.key_dim,self.value_dim=osram.num_heads,osram.key_dim,osram.value_dim
        self.requires_grad_(False);self.eval()

    @torch.no_grad()
    def forward(self,features,availability,qmask,umask):
        valid=umask.T.bool();length,batch=valid.shape
        # Encoder selects only observed blocks, even though targets are complete.
        node,_=self.encoder(features,availability,umask)
        node=self.node_norm(node)
        ae=self.availability_embedding(availability.to(features.dtype))
        speaker=self.speaker_embedding(qmask.long().masked_fill(~umask.bool(),0).T)
        keys=[];values=[]
        for name,part in zip(MODALITIES,torch.split(features,self.encoder.dimensions,-1)):
            latent=features.new_zeros(length,batch,self.encoder.latent_dim)
            latent[valid]=self.encoder.projectors[name](part[valid])
            latent=self.latent_norm(latent)
            k=self.key_projectors[name](torch.cat([latent,node,ae,speaker],-1))
            k=F.normalize(k.reshape(length,batch,self.heads,self.key_dim),dim=-1)
            v=self.value_projectors[name](latent).reshape(length,batch,self.heads,self.value_dim)
            keys.append(k.masked_fill(~valid[...,None,None],0))
            values.append(v.masked_fill(~valid[...,None,None],0))
        return torch.stack(keys,-1),torch.stack(values,-1)


class WriteStateCompletion(nn.Module):
    def __init__(self,encoder,osram,hidden_dim,latent_dim):
        super().__init__()
        budget=2*hidden_dim+8*16+(hidden_dim+16+1)*latent_dim+(latent_dim+1)*latent_dim
        self.predictor=WriteStatePredictor(latent_dim,osram.num_heads,osram.key_dim,osram.value_dim,budget)
        self.teacher=WriteStateTeacher(encoder,osram)
        self.register_buffer('ema_updates',torch.zeros((),dtype=torch.long))
        self.records=[]

    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self

    def begin(self,node,latents,availability,valid):
        self.records=[]
        observed_mean=sum(latents.values())/availability.sum(-1,keepdim=True).clamp_min(1)
        def complete(t,base,gap,keys,values):
            pk,pv=self.predictor(node[t],observed_mean[t],base,gap,availability[t],valid[t])
            if self.training:
                self.records.append((t,pk,pv))
            observed=availability[t].bool()[:,None,None,:]
            k=torch.where(observed,keys,pk)
            v=torch.where(observed,values,pv)
            return k,v,valid[t,:,None].expand(-1,3).to(availability.dtype)
        return complete

    def loss(self,features,availability,qmask,umask):
        selected=umask.T.bool()[...,None] & ~availability.bool()
        count=int(selected.sum())
        if not count:
            return next(self.predictor.parameters()).sum()*0,0
        if not self.records:
            raise ValueError('write-state loss requires preceding training forward')
        tk,tv=self.teacher(features,availability,qmask,umask)
        key_loss=value_loss=None
        for t,pk,pv in self.records:
            chosen=selected[t]
            if not chosen.any():continue
            k=F.smooth_l1_loss(pk.permute(0,3,1,2)[chosen],tk[t].permute(0,3,1,2)[chosen],reduction='sum')
            v=F.smooth_l1_loss(pv.permute(0,3,1,2)[chosen],tv[t].permute(0,3,1,2)[chosen],reduction='sum')
            key_loss=k if key_loss is None else key_loss+k
            value_loss=v if value_loss is None else value_loss+v
        p=self.predictor
        return .5*(key_loss/(count*p.heads*p.key_dim)+value_loss/(count*p.heads*p.value_dim)),count

    @torch.no_grad()
    def update(self,encoder,osram,tau):
        if not 0<=tau<=1:raise ValueError('EMA tau must be in [0,1]')
        pairs=[(self.teacher.encoder,encoder)]
        pairs += [(getattr(self.teacher,n),getattr(osram,n)) for n in
                  ('latent_norm','node_norm','key_projectors','value_projectors','speaker_embedding','availability_embedding')]
        for teacher,online in pairs:
            ps=dict(online.named_parameters())
            for n,p in teacher.named_parameters():p.mul_(tau).add_(ps[n],alpha=1-tau)
            bs=dict(online.named_buffers())
            for n,b in teacher.named_buffers():b.copy_(bs[n])
        self.ema_updates.add_(1)
