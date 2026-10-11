"""Training-only counterfactual future credit; no parameters or inference changes.

The scalar softplus defines utility weighting. E4 intentionally uses a restricted
first-order VJP, NOT the full parameter gradient of that scalar objective.
Independent conversations may be batched together; pairs of the SAME
conversation are separated before source masking to prevent credit leakage.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import math

import torch
import torch.nn.functional as F


def pair_terms(keep_loss, omit_loss, tau=1.):
    if not math.isfinite(float(tau)) or tau <= 0:
        raise ValueError('tau must be finite and positive in task-loss units')
    reference = omit_loss.detach()
    loss = tau * F.softplus((keep_loss-reference)/tau)
    utility = reference-keep_loss.detach()
    return loss, torch.sigmoid(-utility/tau), utility


def gradient_norm(grads):
    return math.sqrt(sum(float(g.detach().double().square().sum())
                         for g in grads if g is not None))


def writer_parameters(model):
    return tuple(p for name,p in model.named_parameters()
                 if p.requires_grad and name.startswith(
                     ('osram.key_projectors.','osram.value_projectors.')))


def source_writer_vjp(loss, nodes, sources, parameters):
    """sources=(time,batch,modality); at most one pair per conversation here.

    `nodes` must be the actual write-only nodes consumed by the Keep scan.
    Never create posthoc stacks and ask autograd to differentiate into them.
    Detaching adjoints explicitly avoids Hessian terms/gradients through w.
    """
    times=sorted({int(t) for t,b,m in sources})
    tensors=[x for t in times for x in nodes[t]]
    adjoints=torch.autograd.grad(loss,tensors,retain_graph=True,
                                create_graph=False,allow_unused=True)
    outputs, cotangents=[],[]
    connected=nonzero=0
    for idx,(x,g) in enumerate(zip(tensors,adjoints)):
        if g is None: continue
        connected+=1
        masked=torch.zeros_like(g)
        t=times[idx//2]
        for source_t,b,m in sources:
            if source_t==t: masked[b,:,:,m]=g.detach()[b,:,:,m]
        nonzero+=int(bool(masked.count_nonzero()))
        outputs.append(x);cotangents.append(masked)
    if not outputs:
        result=tuple(None for _ in parameters)
    else:
        result=torch.autograd.grad(outputs,parameters,grad_outputs=cotangents,
                                  retain_graph=True,create_graph=False,allow_unused=True)
    return result,dict(connected_nodes=connected,nonzero_nodes=nonzero,
                      requested_nodes=len(tensors),
                      connected_parameters=sum(g is not None for g in result))


def _rng_state(device):
    return (torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def _set_rng(state,device):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1],device)


def forward_view(model,view,**kwargs):
    return model([view['incomplete']],view['availability'],view['qmask'],
                 view['umask'],view['lengths'],predict_missing=False,**kwargs)[0]


def _task_loss(logits,view,dataset):
    from .train_gcnet import _task_loss as original_task_loss
    return original_task_loss(dataset,logits,view['labels'],view['umask'],
                              'regression','mse',1.)


def _target_loss(logits,view,pair,dataset):
    pred=logits[pair.future_j,pair.batch]
    y=view['labels'][pair.batch,pair.future_j]
    if dataset in ('CMUMOSI','CMUMOSEI'):
        return (pred.reshape(())-y.to(pred.dtype)).square()
    if dataset not in ('IEMOCAPFour','IEMOCAPSix'):
        raise ValueError('only original MOSI/MOSEI MSE and IEMOCAP CE supported')
    return F.cross_entropy(pred.unsqueeze(0),y.long().reshape(1))


def _validate_pairs(view,pairs):
    seen=set()
    a,u=view['availability'],view['umask']
    for p in pairs:
        identity=(p.batch,p.source_t,p.source_modality,p.future_j,p.missing_modality)
        if identity in seen: raise ValueError('duplicate credit pair')
        seen.add(identity)
        if not (0<=p.batch<a.shape[1] and 0<=p.source_t<p.future_j<a.shape[0]
                and 0<=p.source_modality<3 and 0<=p.missing_modality<3):
            raise ValueError('invalid credit pair indices')
        if str(view['conversation_ids'][p.batch])!=str(p.conversation_id):
            raise ValueError('credit pair conversation mismatch')
        if not (u[p.batch,p.source_t] and u[p.batch,p.future_j]):
            raise ValueError('credit pair touches padding')
        if not a[p.source_t,p.batch,p.source_modality] or a[p.source_t,p.batch].sum()<2:
            raise ValueError('source omission must retain an observed modality')
        if a[p.future_j,p.batch,p.missing_modality]:
            raise ValueError('target condition must be missing')


def credit_backward(model,view,pairs,*,mode='E4',tau=1.,lambda_fwc=.1,
                    dataset='CMUMOSI',shuffle_seed=0):
    """One Keep plus paired no-grad Omits, then normal+extra backward.

    Caller must zero gradients BEFORE and clip/optimizer.step AFTER this call.
    E2 uses fixed weight .5 (neutral-utility weight), with identical Omit budget.
    E0/E1 do not perform counterfactual computation. No-pair batches are exactly
    ordinary training. The global RNG advances only as the Keep forward would.
    """
    if mode not in ('E0','E1','E2','E3','E4','shuffled'):
        raise ValueError('unknown MC-FWC mode')
    if not math.isfinite(float(lambda_fwc)) or lambda_fwc<0:
        raise ValueError('lambda_fwc must be finite and nonnegative')
    pair_terms(torch.tensor(0.),torch.tensor(0.),tau)  # validate also empty batches
    if not model.training: raise ValueError('credit_backward is training-only')
    pairs=list(pairs) if mode not in ('E0','E1') else []
    _validate_pairs(view,pairs)
    device=view['incomplete'].device
    before=_rng_state(device)
    nodes={}
    logits=forward_view(model,view,**({'write_node_collector':nodes} if pairs else {}))
    after=_rng_state(device)
    task=_task_loss(logits,view,dataset)
    if not bool(torch.isfinite(task)): raise FloatingPointError('nonfinite normal task loss')
    stats=dict(mode=mode,pair_count=len(pairs),omit_forwards=0,omit_requires_grad=False,
               connected_nodes=0,nonzero_nodes=0,requested_nodes=0,
               connected_parameters=0,extra_gradient_norm=0.,pairs=[],
               lambda_fwc=float(lambda_fwc),tau=float(tau),fwc_loss=0.)
    if not pairs:
        task.backward()
        stats['normal_gradient_norm']=gradient_norm(p.grad for p in model.parameters())
        return logits.detach(),task.detach(),stats
    # Each round contains at most one pair from any conversation. cfg84 uses
    # sample-independent operations (no BatchNorm/cross-conversation attention).
    per_conversation=defaultdict(list)
    for i,p in enumerate(pairs): per_conversation[p.batch].append(i)
    rounds=[]
    for r in range(max(map(len,per_conversation.values()))):
        rounds.append([items[r] for items in per_conversation.values() if r<len(items)])
    normalizers=[1./len(per_conversation)/len(per_conversation[p.batch]) for p in pairs]
    keep=[_target_loss(logits,view,p,dataset) for p in pairs]
    omit=[None]*len(pairs)
    # Preserve user-visible Keep context references; Omit is not a replacement
    # normal prediction and must not leave last_* pointing at a no-grad replay.
    context=getattr(model,'last_osram_context',None)
    diag=getattr(model.osram,'last_diagnostics',None)
    try:
        for group in rounds:
            override=view['availability'].clone()
            for idx in group:
                p=pairs[idx];override[p.source_t,p.batch,p.source_modality]=0
            _set_rng(before,device)
            with torch.no_grad():
                prediction=forward_view(model,view,write_mask_override=override)
                for idx in group: omit[idx]=_target_loss(prediction,view,pairs[idx],dataset)
            if prediction.requires_grad: raise RuntimeError('Omit must be detached')
            stats['omit_forwards']+=1
    finally:
        _set_rng(after,device)
        model.last_osram_context=context
        model.osram.last_diagnostics=diag
    terms,weights,utility=pair_terms(torch.stack(keep),torch.stack(omit),tau)
    if not bool(torch.isfinite(terms).all()): raise FloatingPointError('nonfinite pair loss')
    used_weights=weights.clone()
    shuffled_count=0
    if mode=='E2': used_weights=torch.full_like(weights,.5)
    if mode=='shuffled':
        generator=torch.Generator().manual_seed(int(shuffle_seed))
        groups=defaultdict(list)
        for i,p in enumerate(pairs):
            groups[(p.source_modality,p.missing_modality,p.future_j-p.source_t,p.regime)].append(i)
        for ids in groups.values():
            if len(ids)<2: continue
            permutation=torch.randperm(len(ids),generator=generator).tolist()
            for k,j in enumerate(ids): used_weights[j]=weights[ids[permutation[k]]]
            shuffled_count+=len(ids)
    if mode in ('E2','shuffled'):
        losses=[used_weights[i].detach()*keep[i] for i in range(len(pairs))]
    else:
        losses=list(terms.unbind())
    weighted=[loss*normalizers[i] for i,loss in enumerate(losses)]
    extra=sum(weighted)
    parameters=writer_parameters(model)
    if not parameters: raise ValueError('no original Writer parameters found')
    extra_grads=[None]*len(parameters)
    if mode!='E3':
        for group in rounds:
            group_loss=sum(weighted[i] for i in group)
            selected=[(pairs[i].source_t,pairs[i].batch,pairs[i].source_modality) for i in group]
            gradients,coverage=source_writer_vjp(group_loss,nodes,selected,parameters)
            for name,value in coverage.items(): stats[name]+=value
            for i,g in enumerate(gradients):
                if g is not None:
                    if not bool(torch.isfinite(g).all()): raise FloatingPointError('nonfinite source VJP')
                    extra_grads[i]=g.detach() if extra_grads[i] is None else extra_grads[i]+g.detach()
        stats['extra_gradient_norm']=gradient_norm(extra_grads)
    else:
        # Explicitly compute extra gradients to measure them independently of
        # the ordinary task, then add only once after normal backward.
        parameters=tuple(p for p in model.parameters() if p.requires_grad)
        extra_grads=torch.autograd.grad(extra,parameters,retain_graph=True,allow_unused=True)
        stats['extra_gradient_norm']=gradient_norm(extra_grads)
    task.backward()
    stats['normal_gradient_norm']=gradient_norm(p.grad for p in model.parameters())
    for p,g in zip(parameters,extra_grads):
        if g is None: continue
        if not bool(torch.isfinite(g).all()): raise FloatingPointError('nonfinite extra parameter gradient')
        scaled=g.detach()*lambda_fwc
        if p.grad is None: p.grad=scaled.clone()
        else: p.grad.add_(scaled)
    stats['fwc_loss']=float(extra.detach())
    stats['utility_positive_fraction']=float((utility>0).float().mean())
    stats['utility_negative_fraction']=float((utility<0).float().mean())
    stats['weight_mean']=float(used_weights.mean())
    stats['weight_saturation_fraction']=float(((used_weights<.01)|(used_weights>.99)).float().mean())
    stats['shuffled_eligible_pairs']=shuffled_count
    stats['scaled_extra_gradient_norm']=lambda_fwc*stats['extra_gradient_norm']
    for i,p in enumerate(pairs):
        stats['pairs'].append(dict(asdict(p),normalizer=normalizers[i],
            loss_keep=float(keep[i].detach()),loss_omit=float(omit[i]),
            utility=float(utility[i]),weight=float(used_weights[i]),
            native_weight=float(weights[i])))
    return logits.detach(),task.detach(),stats
