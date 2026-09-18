"""Train a utility retriever from cached frozen-Reader candidate utilities."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/data2/yb/remote_experiments/osram_utility_text_retrieval_20260918')
RATES = [round(i/10,1) for i in range(8)]
METHODS = ('U1','U2','U3')

def rate_tag(rate): return f'{rate:.1f}'.replace('.','p')

class QueryEncoder(nn.Module):
    def __init__(self, input_dim, hidden=256, out_dim=64, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,out_dim))
    def forward(self,x): return F.normalize(self.net(x),dim=-1)

class KeyEncoder(nn.Module):
    def __init__(self, dim=256, out_dim=64, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,out_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(out_dim,out_dim))
    def forward(self,x): return F.normalize(self.net(x),dim=-1)

class SetScorer(nn.Module):
    def __init__(self, key_dim=64, hidden=128, heads=4, dropout=0.1):
        super().__init__()
        self.proj=nn.Sequential(nn.LayerNorm(4*key_dim),nn.Linear(4*key_dim,hidden),nn.GELU(),nn.Dropout(dropout))
        self.attn=nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True)
        self.out=nn.Linear(hidden,1)
    def forward(self,q,k,mask):
        b,c,_=k.shape
        qe=q.unsqueeze(1).expand(-1,c,-1)
        e=torch.cat((qe,k,qe*k,(qe-k).abs()),dim=-1)
        h=self.proj(e)
        key_padding_mask=~mask
        h2,_=self.attn(h,h,h,key_padding_mask=key_padding_mask)
        return self.out(h2).squeeze(-1)

class Retriever(nn.Module):
    def __init__(self, context_dim, key_dim=64, method='U1', dropout=0.1):
        super().__init__()
        self.method=method
        self.query=QueryEncoder(context_dim,out_dim=key_dim,dropout=dropout)
        self.key=KeyEncoder(out_dim=key_dim,dropout=dropout)
        self.k0=nn.Parameter(torch.zeros(key_dim))
        if method=='U3':
            self.set_scorer=SetScorer(key_dim=key_dim,dropout=dropout)
    def forward(self, context, values, mask):
        q=self.query(context)
        keys=self.key(values)
        keys=torch.cat((self.k0.view(1,1,-1).expand(keys.shape[0],1,-1),keys[:,1:,:]),dim=1)
        if self.method=='U3':
            r=self.set_scorer(q,keys,mask)
        else:
            r=torch.einsum('bd,bcd->bc',q,keys)
        return r

def masked_softmax(logits, mask, temperature=0.1):
    logits=logits/temperature
    logits=logits.masked_fill(~mask, -1e9)
    return F.softmax(logits,dim=1)*mask.float()

def weighted_f1(labels,predictions):
    labels=np.asarray(labels); predictions=np.asarray(predictions)
    nz=labels!=0
    labels=labels[nz]>0; predictions=predictions[nz]>0
    if labels.size==0: return 0.0
    return float(f1_score(labels,predictions,average='weighted'))

def load_cache(seed,rate,split):
    p=ROOT/'utility_cache_v2'/f'seed_{seed}'/f'rate_{rate_tag(rate)}'/f'{split}.pt'
    if not p.exists():
        return None
    return torch.load(p,map_location='cpu',weights_only=False)['tensors']

def batch_to_device(batch,device):
    return {k:v.to(device) for k,v in batch.items()}

def retrieval_loss(method,retriever,batch,utility_temp=0.1,retriever_temp=0.1,margin=0.1):
    context,values,mask,losses=batch['context'],batch['values'],batch['mask'],batch['losses']
    r=retriever(context,values,mask)
    if method=='U1':
        q=masked_softmax(-losses, mask, utility_temp)
        p=masked_softmax(r, mask, retriever_temp).clamp_min(1e-9)
        return (q*(q.clamp_min(1e-9).log()-p.log())).sum(1).mean()
    if method=='U2':
        valid=mask
        l=losses
        ri=r.unsqueeze(2); rj=r.unsqueeze(1)
        li=l.unsqueeze(2); lj=l.unsqueeze(1)
        pair=valid.unsqueeze(2)&valid.unsqueeze(1)&(li<lj)
        weights=(li-lj).abs()*pair.float()
        rank=F.relu(margin - ri + rj)
        return (weights*rank).sum()/(weights.sum().clamp_min(1.0))
    if method=='U3':
        q=masked_softmax(-losses, mask, utility_temp)
        p=masked_softmax(r, mask, retriever_temp).clamp_min(1e-9)
        return (q*(q.clamp_min(1e-9).log()-p.log())).sum(1).mean()
    raise ValueError(method)

def infer(r,batch,mode):
    scores,losses,mask=batch['scores'],batch['losses'],batch['mask']
    p=masked_softmax(r,mask,temperature=0.1)
    if mode=='Original': return scores[:,0]
    if mode=='Uniform':
        return (scores*mask.float()).sum(1)/mask.float().sum(1).clamp_min(1)
    if mode=='Soft': return (p*scores).sum(1)
    if mode=='Top1':
        idx=p.argmax(1); return scores[torch.arange(scores.shape[0]),idx]
    if mode=='Top2':
        top=torch.topk(p,2,dim=1)
        w=top.values/top.values.sum(1,keepdim=True).clamp_min(1e-9)
        return (scores.gather(1,top.indices)*w).sum(1)
    if mode=='Oracle-Best':
        idx=losses.masked_fill(~mask,1e9).argmin(1); return scores[torch.arange(scores.shape[0]),idx]
    if mode=='Oracle-Soft':
        q=masked_softmax(-losses,mask,0.1); return (q*scores).sum(1)
    raise ValueError(mode)

def evaluate(retriever,batch,method,mode):
    retriever.eval()
    with torch.no_grad():
        r=retriever(batch['context'],batch['values'],batch['mask'])
        pred=infer(r,batch,mode)
        return weighted_f1(batch['labels'].cpu().numpy(),pred.cpu().numpy())


def evaluate_full(retriever,batch,mode='Soft'):
    """Full split W-F1 with T-present frozen Reader predictions preserved."""
    retriever.eval()
    with torch.no_grad():
        r=retriever(batch['context'],batch['values'],batch['mask'])
        pred_missing=infer(r,batch,mode)
        full=batch['full_scores'].clone()
        full[batch['full_index']]=pred_missing
        return weighted_f1(batch['full_labels'].cpu().numpy(),full.cpu().numpy())

def train_one(seed,rate,method,device,epochs=100,batch_size=128):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tr=load_cache(seed,rate,'train'); va=load_cache(seed,rate,'validation')
    if tr is None or va is None:
        return float('nan'), [], None
    context_dim=tr['context'].shape[1]
    retriever=Retriever(context_dim,method=method).to(device)
    optimizer=torch.optim.Adam(retriever.parameters(),lr=1e-3,weight_decay=1e-5)
    n=tr['labels'].shape[0]; best=-1; best_state=None; history=[]
    for epoch in range(epochs):
        retriever.train(); order=torch.randperm(n)
        for start in range(0,n,batch_size):
            idx=order[start:start+batch_size]
            batch={k:v[idx].to(device) for k,v in tr.items()}
            optimizer.zero_grad(set_to_none=True)
            loss=retrieval_loss(method,retriever,batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retriever.parameters(),1.0)
            optimizer.step()
        val_batch=batch_to_device(va,device)
        score=evaluate_full(retriever,val_batch,mode='Soft')
        history.append(float(score))
        if score>best:
            best=score; best_state={k:v.detach().cpu().clone() for k,v in retriever.state_dict().items()}
    retriever.load_state_dict(best_state)
    out_dir=ROOT/'methods_valid_v2'/method/f'seed_{seed}'
    out_dir.mkdir(parents=True,exist_ok=True)
    ckpt=out_dir/f'rate_{rate_tag(rate)}.pt'
    torch.save({'state_dict':best_state,'method':method,'seed':seed,'rate':rate,'validation_soft_wf1':best,'history':history,'context_dim':context_dim},ckpt)
    # Evaluate all inference modes on train/val/test and save test artifacts.
    rows=[]
    artifacts={}
    for split,data in (('train',tr),('validation',va),('test',load_cache(seed,rate,'test'))):
        batch=batch_to_device(data,device)
        with torch.no_grad():
            r=retriever(batch['context'],batch['values'],batch['mask'])
            for mode in ('Original','Uniform','Soft','Top1','Top2','Oracle-Best','Oracle-Soft'):
                pred=infer(r,batch,mode)
                rows.append({'method':method,'seed':seed,'rate':rate,'split':split,'mode':mode,'weighted_f1':weighted_f1(batch['labels'].cpu().numpy(),pred.cpu().numpy())*100.0})
                if split=='test':
                    artifacts[f'pred_{mode}']=pred.detach().cpu().numpy()
        if split=='test':
            artifacts['labels']=batch['labels'].detach().cpu().numpy()
            artifacts['patterns']=batch['patterns'].detach().cpu().numpy()
            artifacts['scores']=batch['scores'].detach().cpu().numpy()
            artifacts['losses']=batch['losses'].detach().cpu().numpy()
            artifacts['mask']=batch['mask'].detach().cpu().numpy()
            artifacts['r']=r.detach().cpu().numpy()
    odd=ROOT/'retrieval_artifacts_v2'/method/f'seed_{seed}'
    odd.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(odd/f'rate_{rate_tag(rate)}.npz',**artifacts)
    return best,rows,ckpt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--rate',type=float,required=True); ap.add_argument('--method',choices=METHODS,required=True); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    device=torch.device(args.device)
    best,rows,ckpt=train_one(args.seed,args.rate,args.method,device)
    print(json.dumps({'method':args.method,'seed':args.seed,'rate':args.rate,'validation_soft_wf1':best,'checkpoint':str(ckpt) if ckpt else None,'n_rows':len(rows)}))
if __name__=='__main__': main()
