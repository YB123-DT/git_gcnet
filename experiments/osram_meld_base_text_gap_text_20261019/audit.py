"""Same-cohort Base<-T / Gap-T<-T interventions, no training machinery."""
from __future__ import annotations

import numpy as np
import torch

from experiments.osram_12_source_evidence_audit_20261019.analysis import _metrics, _paired

CONDITIONS = ('full', 'without_base_text', 'without_gap_text', 'without_both')


def eligible_mask(availability, trajectory):
    availability, trajectory = np.asarray(availability), np.asarray(trajectory)
    if (availability.ndim != 2 or availability.shape[-1] != 3
            or trajectory.shape != (len(availability), 3, 4)
            or not np.isfinite(trajectory).all()
            or not np.isin(availability, (0,1)).all()
            or np.any(availability.sum(-1)<1)
            or not np.isin(trajectory[...,3], (0,1)).all()):
        raise ValueError('Need valid utterance availability[N,3] and strictly-past trajectory[N,3,4]')
    return ~availability[:,1].astype(bool) & trajectory[:,1,3].astype(bool)


@torch.no_grad()
def replay_four(osram, head, local, emotion_input, source_reads, umask, selected, full_logits):
    if any(module.training for module in (osram.local_skip, osram.emotion_adapter,
                                          osram.emotion_norm, head)):
        raise ValueError('Frozen eval modules required')
    length,batch,ld = local.shape
    fd = osram.num_heads*osram.value_dim
    if (source_reads.shape != (length,batch,4,3,fd)
            or emotion_input.shape != (length,batch,ld+8*fd)
            or umask.shape != (batch,length) or selected.shape != (length,batch)
            or selected.dtype != torch.bool or bool((selected & ~umask.T.bool()).any())):
        raise ValueError('Invalid four-condition replay shapes or selected padding')
    contexts=emotion_input[...,ld:].reshape(length,batch,4,2*fd)
    if bool(contexts[...,fd:].ne(0).any()):
        raise ValueError('Only original causal forward context is supported')
    torch.testing.assert_close(source_reads.sum(-2),contexts[...,:fd],rtol=5e-5,atol=1e-5)
    torch.testing.assert_close(local,emotion_input[...,:ld],rtol=0,atol=0)
    skip=osram.local_skip(local)

    def classify(value):
        hidden=osram.emotion_norm(skip+osram.emotion_adapter(value))
        return head(hidden*umask.T.bool()[...,None].to(hidden.dtype))

    torch.testing.assert_close(classify(emotion_input),full_logits,rtol=0,atol=0)
    result={'full':full_logits}
    for name,rows in (('without_base_text',(0,)),('without_gap_text',(2,)),('without_both',(0,2))):
        candidate=emotion_input.clone()
        for row in rows:
            start=ld+row*2*fd
            candidate[...,start:start+fd]-=torch.where(
                selected[...,None],source_reads[...,row,1,:],0.)
        result[name]=classify(candidate)
        torch.testing.assert_close(result[name][~selected],full_logits[~selected],rtol=0,atol=0)
    return result


def summarize_arrays(arrays, classes=7):
    labels=np.asarray(arrays['labels'])
    selected=eligible_mask(arrays['availability'],arrays['trajectory'])
    if not isinstance(classes,(int,np.integer)) or classes<2:
        raise ValueError('Need an explicit integer class count >=2')
    classes=np.arange(classes)
    if labels.shape!=(len(selected),) or not np.isin(labels,classes).all():
        raise ValueError('Invalid aligned labels for the specified class count')
    logits={name:np.asarray(arrays[name],dtype=np.float64) for name in CONDITIONS}
    if any(v.shape!=(len(labels),len(classes)) or not np.isfinite(v).all() for v in logits.values()):
        raise ValueError('All four logits must be aligned and finite')

    def cohort(mask):
        mask=mask & selected
        target=labels[mask]
        metrics={name:_metrics(target,values[mask],classes) for name,values in logits.items()}
        paired={name:_paired(target,logits['full'][mask],logits[name][mask],classes)
                for name in CONDITIONS[1:]}
        conditional={}
        for name,present,absent in (
                ('base_with_gap','full','without_base_text'),
                ('base_without_gap','without_gap_text','without_both'),
                ('gap_with_base','full','without_gap_text'),
                ('gap_without_base','without_base_text','without_both')):
            conditional[name]={
                'CE_improvement':metrics[absent]['CE']-metrics[present]['CE'] if len(target) else None,
                **{m+'_improvement':metrics[present][m]-metrics[absent][m] if len(target) else None
                   for m in ('ACC','UA','WF1')}}
        interaction=(metrics['without_both']['CE']-metrics['without_base_text']['CE']
                     -metrics['without_gap_text']['CE']+metrics['full']['CE']) if len(target) else None
        return dict(N=len(target),metrics=metrics,paired_vs_full=paired,
                    conditional_effects=conditional,interaction_CE=interaction)

    out=cohort(np.ones(len(labels),dtype=bool))
    bits=np.asarray(arrays['availability']).astype(int)@np.array([1,2,4])
    out['by_current_availability']={name:cohort(bits==value) for name,value in (('A',1),('V',4),('AV',5))}
    out['definition']=f'Same cohort: current Text missing AND strictly-past Text ever seen; {len(classes)}-class metrics; corrections/harms are deletion versus original.'
    return out
