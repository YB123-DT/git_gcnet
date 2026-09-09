"""Explicit B2 stage runners. Nothing launches unless a stage is selected."""
import argparse
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from . import train_gcnet as tr
from .b2_training import SourceOnlyPretrainer, pretrain_step, load_b2_initialization, validate_base
from .loss import missing_m3_loss
from .mixed_rate import BalancedBatchRateSchedule
from .model import MODALITIES, MissingM3GraphModel


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("stage1","stage2","smoke"),required=True)
    parser.add_argument("--base-checkpoint",type=Path,required=True)
    parser.add_argument("--pretrain-checkpoint",type=Path)
    parser.add_argument("--feature-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=None,help="Default: inherit base training budget")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--num-threads",type=int,default=1)
    return parser


def feature_paths(root):
    return [str(root/name) for name in ("wav2vec-large-c-UTT","deberta-large-4-UTT","manet_UTT")]


def loaders(config,root):
    shape=tr._resolve_task_contract(config.dataset,config.mosi_task_mode)
    paths=feature_paths(root)
    return tr.get_loaders(audio_root=paths[0],text_root=paths[1],video_root=paths[2],
        num_folder=int(shape["num_folds"]),dataset=config.dataset,batch_size=config.batch_size,
        num_workers=0,seed=config.seed,validation_fraction=config.validation_fraction,
        evaluation_protocol=config.evaluation_protocol)


def make_b2(config,dimensions):
    shape=tr._resolve_task_contract(config.dataset,config.mosi_task_mode)
    names=inspect.signature(MissingM3GraphModel.__init__).parameters
    kwargs={k:v for k,v in asdict(config).items() if k in names}
    kwargs.update(adim=dimensions[0],tdim=dimensions[1],vdim=dimensions[2],D_e=config.hidden,
        graph_hidden_size=config.hidden//2,n_speakers=int(shape["num_speakers"]),
        n_classes=int(shape["num_classes"]),time_attn=config.time_attention,no_cuda=config.device=="cpu")
    return MissingM3GraphModel(**kwargs).to(config.device)


@torch.no_grad()
def latent_audit(stage,loader,config,device,dimensions):
    # Deliberately retain the old metric definitions, including singular-value entropy.
    from experiments.missing_m3_mosi_latent_diagnostic_20260831.analyze_checkpoint import _metrics
    stage.eval()
    schedule=tr._build_schedule(config,"test",.5)
    collected={}
    for raw in loader:
        view=tr._prepare_view(tr._move_batch(raw,device),schedule,0,dimensions)
        a=view["availability"]
        predictions=stage(view["incomplete"],a,view["umask"])
        target=stage.targets(view["complete"])
        for pattern,bits in (("A",(1,0,0)),("V",(0,0,1)),("AV",(1,0,1))):
            selected_pattern=(a==a.new_tensor(bits)).all(-1) & view["umask"].T.bool()
            for q,m in enumerate(MODALITIES):
                if bits[q]:
                    continue
                selected=selected_pattern & predictions.target_mask[...,q]
                for branch,values in (("regression",predictions.reg_predictions),("contrastive",predictions.cl_predictions)):
                    pair=collected.setdefault((pattern,m,branch),([],[]))
                    pair[0].append(values[...,q,:][selected].cpu())
                    pair[1].append(target[m][selected].cpu())
    rows=[]
    for (pattern,target,branch),(ps,ts) in collected.items():
        p,t=torch.cat(ps),torch.cat(ts)
        result=dict(pattern=pattern,target=target,branch=branch,count=len(p))
        if len(p)>=2:
            if not torch.isfinite(p).all() or not torch.isfinite(t).all():
                raise ValueError("nonfinite Stage1 latent audit")
            result.update(_metrics(p,t,config.temperature,config.seed+100*MODALITIES.index(target)))
            result["std_ratio"]=result["channel_std"]/max(result["target_channel_std"],1e-12)
        else:
            result["status"]="insufficient samples"
        rows.append(result)
    return dict(split="test",rate=.5,seed=config.seed,selection="none; fixed-final audit only",
                rows=rows,diagnostic_only=True)


def run_stage1(args,config):
    stage=SourceOnlyPretrainer.from_checkpoint(args.base_checkpoint).to(args.device)
    train,_,test,*dims=loaders(config,args.feature_root)
    if tuple(dims)!=stage.dimensions:
        raise ValueError("feature dimensions differ from source checkpoint")
    optimizer=torch.optim.Adam(stage.predictor.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    schedules=tr._schedules(config,"train")
    rate_schedule=BalancedBatchRateSchedule()
    history=[]
    epochs=args.epochs if args.epochs is not None else config.epochs
    for epoch in range(epochs):
        batches=[]
        for index,raw in enumerate(train[config.fold-1]):
            rate=rate_schedule.rate_for(epoch,index)
            view=tr._prepare_view(tr._move_batch(raw,torch.device(args.device)),schedules[rate],epoch,tuple(dims))
            batches.append(dict(rate=rate,**pretrain_step(stage,view,optimizer,config)))
        history.append(dict(epoch=epoch+1,batches=batches))
        tr._write_json(args.output_dir/"history.json",history)
        print(f"Stage1 epoch {epoch+1}: {len(batches)} batches",flush=True)
    stage.save(args.output_dir/"source_only_completion_pretrain.pt",epochs,optimizer)
    tr._write_json(args.output_dir/"latent_audit.json",latent_audit(stage,test[config.fold-1],config,
                   torch.device(args.device),tuple(dims)))
    tr._write_json(args.output_dir/"stage1_summary.json",dict(stage="source-only fixed-space pretrain",
        epochs=epochs,config=asdict(config),source_checkpoint_sha256=stage.source_hash,
        frozen_space_sha256=stage.frozen_hash,parameter_count=sum(p.numel() for p in stage.predictor.parameters()),
        osram_forwards=0,emotion_forwards=0,teacher_updates=0))


def stage2_config(config,args,pretrain_path):
    return replace(config,completion_path="pre_osram_b2",b2_base_checkpoint=str(args.base_checkpoint),
        b2_pretrain_checkpoint=str(pretrain_path),initial_backbone_checkpoint=None,pretrained_learning_rate=None,
        training_objective="joint",device=args.device,epochs=args.epochs or config.epochs)


def run_smoke(args,config):
    """One real batch reused for two Stage2 steps to cross zero initialization."""
    stage=SourceOnlyPretrainer.from_checkpoint(args.base_checkpoint).to(args.device)
    train,_,_,*dims=loaders(config,args.feature_root)
    raw=next(iter(train[config.fold-1]))
    device=torch.device(args.device)
    # Explicit debug .5 view, not a replacement for either stage's cyclic schedule.
    view=tr._prepare_view(tr._move_batch(raw,device),tr._build_schedule(config,"train",.5),0,tuple(dims))
    optimizer=torch.optim.Adam(stage.predictor.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    stage1=pretrain_step(stage,view,optimizer,config)
    pretrain=args.output_dir/"source_only_completion_pretrain.pt"
    stage.save(pretrain,0,optimizer)
    config2=stage2_config(config,args,pretrain)
    model=make_b2(config2,tuple(dims))
    provenance=load_b2_initialization(model,args.base_checkpoint,pretrain)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,weight_decay=config.weight_decay)
    calls=[]
    handle=model.osram.register_forward_hook(lambda *unused:calls.append(1))
    steps=[]
    model.train()
    for index in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits,_,_,pred=model([view["incomplete"]],view["availability"],view["qmask"],view["umask"],
                             view["lengths"],predict_missing=True)
        emotion=tr._task_loss(config.dataset,logits,view["labels"],view["umask"],config.mosi_task_mode,
                             config.task_regression_loss,config.task_smooth_l1_beta)
        completion=missing_m3_loss(pred,model.encode_teacher_targets([view["complete"]]),
            temperature=config.temperature,regression_aggregation=config.jepa_regression_aggregation,
            contrastive_prediction_source=config.jepa_contrastive_source)
        grads=torch.autograd.grad(emotion,tuple(model.source_only_predictor.parameters()),retain_graph=True,allow_unused=True)
        emotion_grad=sum(float(g.norm()) for g in grads if g is not None)
        total=emotion+config.jepa_weight*completion.total
        total.backward()
        if not torch.isfinite(total) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite smoke gradient/loss")
        assert all(p.grad is None for p in model.teacher.parameters())
        if config.gradient_clip_norm>0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config.gradient_clip_norm)
        optimizer.step()
        model.update_teacher(config.ema_tau)
        steps.append(dict(step=index+1,emotion=float(emotion.detach()),completion=float(completion.total.detach()),
                          emotion_to_predictor_gradient=emotion_grad))
    assert steps[0]["emotion_to_predictor_gradient"]==0
    assert steps[1]["emotion_to_predictor_gradient"]>0
    assert len(calls)==2
    handle.remove()
    model.eval()
    teacher_calls=[]
    hooks=[p.register_forward_pre_hook(lambda *unused:teacher_calls.append(1)) for p in model.teacher.values()]
    with torch.no_grad():
        inference=model([view["incomplete"]],view["availability"],view["qmask"],view["umask"],view["lengths"])
    assert inference[3] is None and not teacher_calls
    for hook in hooks:
        hook.remove()
    smoke_state=args.output_dir/"b2_smoke.pt"
    torch.save(dict(model=tr._state_to_cpu(model),config=asdict(config2),epoch=0,ema_step=model.ema_step,
                    purpose="SMOKE ONLY; not a trained experiment"),smoke_state)
    restored=make_b2(config2,tuple(dims)).eval()
    restored.load_state_dict(torch.load(smoke_state,map_location=device,weights_only=False)["model"],strict=True)
    with torch.no_grad():
        restored_output=restored([view["incomplete"]],view["availability"],view["qmask"],view["umask"],view["lengths"])
    assert torch.equal(inference[0],restored_output[0])
    np.savez_compressed(args.output_dir/"smoke_predictions.npz",
        predictions=inference[0].detach().cpu().numpy(),labels=view["labels"].cpu().numpy(),
        availability=view["availability"].cpu().numpy(),umask=view["umask"].cpu().numpy(),
        conversation_ids=np.asarray(view["conversation_ids"],dtype=str))
    report=dict(purpose="ONE-BATCH SMOKE ONLY; no F1 conclusion",stage1=stage1,stage2=steps,
        source_provenance=provenance,batch_conversations=len(view["lengths"]),
        valid_utterances=int(view["umask"].sum()),predictor_parameters=sum(p.numel() for p in model.source_only_predictor.parameters()),
        fusion_parameters=sum(p.numel() for p in model.completed_read_fusion.parameters()),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        osram_calls_per_forward=1,inference_teacher_calls=0,strict_checkpoint_restore_exact=True)
    tr._write_json(args.output_dir/"SMOKE.json",report)
    print(json.dumps(report,indent=2),flush=True)


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.epochs is not None and args.epochs<1:
        raise ValueError("epochs must be positive")
    if args.stage=="stage2" and args.pretrain_checkpoint is None:
        raise ValueError("stage2 requires --pretrain-checkpoint")
    torch.set_num_threads(args.num_threads)
    checkpoint=torch.load(args.base_checkpoint,map_location="cpu",weights_only=False)
    config=tr.TrainConfig(**checkpoint["config"])
    validate_base(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("use an empty output directory; do not overwrite runs")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=="stage1":
        run_stage1(args,config)
    elif args.stage=="smoke":
        run_smoke(args,config)
    else:
        config2=stage2_config(config,args,args.pretrain_checkpoint)
        tr.run_experiment(config2,*feature_paths(args.feature_root),output_dir=args.output_dir)


if __name__=="__main__":
    main()
