"""Bounded real-MOSI GPU verification, never a performance experiment."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from gcnet_missing_m3 import train_mc_fwc as run
from gcnet_missing_m3.mc_fwc import credit_backward, forward_view, _rng_state, _set_rng
from gcnet_missing_m3.fwc_sampling import sample_pairs


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise ValueError('use a new smoke report path')
    torch.set_num_threads(1)
    c=run.baseline_config(66,'cuda',150)
    run.old.set_random_seed(c.seed)
    train,_,_,dims,provenance=run.make_loaders(SimpleNamespace(dataset_root=args.dataset_root),c)
    run.old.set_random_seed(run.old.SeedBundle(c.seed).derive('missing_m3_model_init:fold:5'))
    model=run.build_model(c,dims).train()
    optimizer=torch.optim.Adam(run.old._optimizer_parameter_groups(model,c)[0],weight_decay=c.weight_decay)
    raw=run.old._move_batch(next(iter(train)),torch.device('cuda'))
    view=run.old._prepare_view(raw,run.old._schedules(c,'train')[.5],0,dims)
    pairs=sample_pairs(view['availability'],view['umask'],view['conversation_ids'],seed=812,horizon=4)
    if not pairs: raise ValueError('real smoke must exercise actual pairs')
    step_reports=[];torch.cuda.reset_peak_memory_stats()
    # Observe all real BlockWrite states without changing their computation.
    original=model.osram.block_write
    states=[]
    def checked(*a,**kw):
        result=original(*a,**kw)
        if not bool(torch.isfinite(result).all()): raise FloatingPointError('nonfinite memory')
        states.append(float(result.detach().norm()))
        return result
    model.osram.block_write=checked
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        before=_rng_state(torch.device('cuda'))
        # Same real batch, same model and dropout: establish ordinary gradients.
        base,base_loss,_=credit_backward(model,view,[],mode='E0')
        after=_rng_state(torch.device('cuda'))
        baseline={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
        optimizer.zero_grad(set_to_none=True);_set_rng(before,torch.device('cuda'))
        started=time.monotonic();states.clear()
        pred,loss,stats=credit_backward(model,view,pairs,mode='E4')
        actual_rng=_rng_state(torch.device('cuda'))
        assert all(torch.equal(a,b) for a,b in zip(after,actual_rng))
        torch.testing.assert_close(pred,base,rtol=0,atol=0)
        torch.testing.assert_close(loss,base_loss,rtol=0,atol=0)
        changed=[]
        for name,param in model.named_parameters():
            expected=baseline[name]
            if expected is None:
                assert param.grad is None,name
            else:
                assert torch.isfinite(param.grad).all(),name
                if not torch.equal(param.grad,expected): changed.append(name)
        assert all(n.startswith(('osram.key_projectors.','osram.value_projectors.')) for n in changed),changed
        if step==0: assert stats['extra_gradient_norm']==0 and not changed
        else: assert stats['extra_gradient_norm']>0 and changed
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        step_reports.append(dict(step=step+1,task_loss=float(loss),extra_norm=stats['extra_gradient_norm'],
            pair_count=len(pairs),changed_parameter_names=changed,source_nodes_connected=stats['connected_nodes'],
            source_nodes_nonzero=stats['nonzero_nodes'],memory_states_checked=len(states),
            maximum_memory_norm=max(states),combined_preclip_norm=float(norm),seconds=time.monotonic()-started,
            identical_keep_prediction=True,cpu_cuda_rng_exact=True,nonwriter_gradients_exact=True))
        del baseline
    model.osram.block_write=original
    # Exact no-intervention replay in train mode, including GPU dropout.
    before=_rng_state(torch.device('cuda'))
    with torch.no_grad(): plain=forward_view(model,view)
    _set_rng(before,torch.device('cuda'))
    with torch.no_grad(): replay=forward_view(model,view,write_mask_override=view['availability'])
    assert torch.equal(plain,replay)
    # Complete epoch-boundary recovery: compare the next optimizer update.
    recovery=output.with_suffix('.recovery.pt')
    run.save_training(recovery,model,optimizer,{'smoke':True},1,[],{'epoch':1,'score':0.})
    def next_update():
        optimizer.zero_grad(set_to_none=True)
        credit_backward(model,view,pairs,mode='E4')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    uninterrupted=next_update()
    run.restore_training(recovery,model,optimizer,{'smoke':True})
    resumed=next_update()
    assert all(torch.equal(uninterrupted[k],resumed[k]) for k in uninterrupted)
    model.eval()
    with torch.no_grad(): assert torch.isfinite(forward_view(model,view)).all()
    run.write_json(output,dict(label='ENGINEERING VERIFICATION ONLY',steps=step_reports,
        valid_utterances=int(view['umask'].sum()),conversations=len(view['lengths']),
        no_intervention_replay_exact=True,resume_next_update_exact=True,
        ordinary_inference_without_credit=True,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        config=asdict(c),data=provenance))
    print(json.dumps(dict(output=str(output),steps=step_reports,passed=True)),flush=True)


if __name__=='__main__': main()
