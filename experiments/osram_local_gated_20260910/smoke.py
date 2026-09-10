"""Exactly one real training batch, two updates; never starts a training run."""
import argparse
from dataclasses import asdict, replace
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel
from gcnet_missing_m3.loss import missing_m3_loss


def make(config, dims):
    names=inspect.signature(MissingM3GraphModel.__init__).parameters
    kw={k:v for k,v in asdict(config).items() if k in names}
    kw.update(adim=dims[0],tdim=dims[1],vdim=dims[2],D_e=config.hidden,
              graph_hidden_size=config.hidden//2,n_speakers=1,n_classes=1,
              time_attn=config.time_attention,no_cuda=config.device=='cpu')
    return MissingM3GraphModel(**kw).to(config.device)


def count(model, trainable=False):
    return sum(p.numel() for p in model.parameters() if not trainable or p.requires_grad)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--feature-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cp=torch.load(args.base_checkpoint,map_location='cpu',weights_only=False)
    cfg=tr.TrainConfig(**dict(cp['config'],device=args.device))
    assert cfg.dataset=='CMUMOSI' and not cfg.osram_bidirectional and cfg.osram_write_step==.6
    tr.set_random_seed(cfg.seed)
    paths=[str(args.feature_root / n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    loader,_,_,*dims=tr.get_loaders(audio_root=paths[0],text_root=paths[1],video_root=paths[2],
        num_folder=1,dataset=cfg.dataset,batch_size=cfg.batch_size,num_workers=0,seed=cfg.seed,
        evaluation_protocol=cfg.evaluation_protocol,validation_fraction=cfg.validation_fraction)
    raw=next(iter(loader[cfg.fold-1]))
    view=tr._prepare_view(tr._move_batch(raw,torch.device(args.device)),tr._build_schedule(cfg,'train',.5),0,tuple(dims))
    flat=make(cfg,dims).eval()
    flat.load_state_dict(cp['model'],strict=True)
    gated_cfg=replace(cfg,osram_readout_fusion='local-gated',checkpoint_selection='test-oracle-per-rate')
    gated=make(gated_cfg,dims).eval()
    result=gated.load_state_dict(cp['model'],strict=False)
    expected={k for k in gated.state_dict() if k.startswith('osram.local_centered_fusion.')}
    assert set(result.missing_keys)==expected and not result.unexpected_keys
    gated.osram.local_centered_fusion.local_skip.load_state_dict(gated.osram.local_skip.state_dict())
    gated.osram.local_centered_fusion.emotion_norm.load_state_dict(gated.osram.emotion_norm.state_dict())
    call=([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'])
    traces=[];outputs=[]
    with torch.no_grad():
        for model in (flat,gated):
            states=[];old=model.osram.block_write
            def capture(*a,**k):
                post=old(*a,**k);states.append(post.cpu().clone());return post
            with patch.object(model.osram,'block_write',side_effect=capture), patch.object(
                    model,'encode_teacher_targets',side_effect=AssertionError('Teacher called in forward')):
                outputs.append(model(*call,predict_missing=True))
            traces.append(states)
    assert len(traces[0])==len(traces[1]) and all(torch.equal(a,b) for a,b in zip(*traces))
    for k in ('local','base','gap'):
        assert torch.equal(flat.last_osram_context[k],gated.last_osram_context[k])
    for k in ('reg_predictions','cl_predictions'):
        assert torch.equal(getattr(outputs[0][3],k),getattr(outputs[1][3],k))
    report=dict(mode='one-batch smoke only; no full training',device=args.device,
        source_checkpoint=str(args.base_checkpoint),conversations=len(view['lengths']),
        valid_utterances=int(view['umask'].sum()),rate=.5,
        flat_strict_checkpoint_load=True,flat_config_default=cfg.osram_readout_fusion,
        memory_equal_every_step=True,scan_steps=len(traces[0]),
        predictor_and_contexts_equal=True,forward_teacher_calls=0,
        flat_total_parameters=count(flat),gated_total_parameters=count(gated),
        flat_trainable_parameters=count(flat,True),gated_trainable_parameters=count(gated,True),
        added_parameters=count(gated)-count(flat),
        readout_parameters=count(gated.osram.local_centered_fusion),steps=[])
    del flat,outputs,traces
    optimizer=torch.optim.Adam((p for p in gated.parameters() if p.requires_grad),
                               lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    fusion=gated.osram.local_centered_fusion
    for step in range(2):
        gated.train();optimizer.zero_grad(set_to_none=True)
        logits,_,_,pred=gated(*call,predict_missing=True)
        cls=tr._task_loss(cfg.dataset,logits,view['labels'],view['umask'],cfg.mosi_task_mode,
                          cfg.task_regression_loss,cfg.task_smooth_l1_beta)
        groups={name:list(getattr(fusion,name).parameters()) for name in ('query','key','value','gate','context_out')}
        parameters=[p for ps in groups.values() for p in ps]
        grads=torch.autograd.grad(cls,parameters,retain_graph=True)
        norms={};offset=0
        for name,ps in groups.items():
            norms[name]=sum(float(g.abs().sum()) for g in grads[offset:offset+len(ps)]);offset+=len(ps)
        if step==1:assert all(x>0 for x in norms.values()),norms
        teacher=gated.encode_teacher_targets([view['complete']])
        jepa=missing_m3_loss(pred,teacher,temperature=cfg.temperature,
            regression_aggregation=cfg.jepa_regression_aggregation,
            contrastive_prediction_source=cfg.jepa_contrastive_source)
        loss=cls+cfg.jepa_weight*jepa.total;assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in gated.parameters() if p.grad is not None)
        assert all(p.grad is None for p in gated.teacher.parameters())
        memory_grad=sum(float(p.grad.abs().sum()) for p in gated.osram.key_projectors.parameters() if p.grad is not None)
        assert memory_grad>0
        report['steps'].append(dict(step=step+1,loss=float(loss),emotion_loss=float(cls),
            jepa_loss=float(jepa.total),emotion_gradient_l1=norms,key_total_gradient_l1=memory_grad,
            diagnostics=dict(fusion.last_diagnostics)))
        torch.nn.utils.clip_grad_norm_(gated.parameters(),cfg.gradient_clip_norm)
        optimizer.step();gated.update_teacher(cfg.ema_tau)
    report['selection_protocol']='per-rate-test-oracle; no epoch selected in smoke'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
