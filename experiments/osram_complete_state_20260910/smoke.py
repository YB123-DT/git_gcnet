"""One real batch, one update, no extra epoch or sweep."""
import inspect
import json
from dataclasses import asdict
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel
from experiments.osram_complete_state_20260910.run import configuration,ROOT,base


def main():
    torch.set_num_threads(4);cfg,_,_=configuration(66);tr.set_random_seed(cfg.seed)
    roots=[str(base.runner.FEATURES/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    loaders,_,_,*dims=tr.get_loaders(audio_root=roots[0],text_root=roots[1],video_root=roots[2],
        num_folder=1,dataset=cfg.dataset,batch_size=cfg.batch_size,num_workers=0,seed=cfg.seed,
        evaluation_protocol=cfg.evaluation_protocol,validation_fraction=cfg.validation_fraction)
    view=tr._prepare_view(tr._move_batch(next(iter(loaders[0])),torch.device(cfg.device)),
                          tr._build_schedule(cfg,'train',.5),0,tuple(dims))
    names=inspect.signature(MissingM3GraphModel.__init__).parameters
    kw={k:v for k,v in asdict(cfg).items() if k in names}
    kw.update(adim=dims[0],tdim=dims[1],vdim=dims[2],D_e=cfg.hidden,graph_hidden_size=cfg.hidden//2,
              n_speakers=1,n_classes=1,time_attn=cfg.time_attention,complete_state_jepa=True)
    model=MissingM3GraphModel(**kw).to(cfg.device).train()
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg.learning_rate)
    with patch.object(model.missing_predictor,'forward',side_effect=AssertionError('MMoE called')):
        logits,h,_,_=model([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'])
        state,n=model.complete_state_loss(h,view['complete'],view['availability'],view['umask'])
        cls=tr._task_loss(cfg.dataset,logits,view['labels'],view['umask'],cfg.mosi_task_mode,
                          cfg.task_regression_loss,cfg.task_smooth_l1_beta)
        loss=cls+cfg.jepa_weight*state;loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    grads={name:sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
           for name,module in [('state_predictor',model.state_jepa.predictor),('encoder',model.observed_set),('osram',model.osram)]}
    assert all(v>0 for v in grads.values())
    assert all(p.grad is None for p in model.state_jepa.teacher_encoder.parameters())
    assert all(p.grad is None for p in model.state_jepa.teacher_local.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip_norm);opt.step();model.update_teacher(cfg.ema_tau)
    model.eval()
    with torch.no_grad(),patch.object(model.state_jepa,'target',side_effect=AssertionError('test Teacher')):
        model([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'])
    report=dict(loss=float(loss),state_loss=float(state),emotion_loss=float(cls),state_targets=n,
        gradient_l1=grads,ema_step=model.ema_step,total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        state_module_parameters=sum(p.numel() for p in model.state_jepa.parameters()),
        state_module_trainable=sum(p.numel() for p in model.state_jepa.parameters() if p.requires_grad),
        teacher_forward_has_no_memory=True,test_has_no_teacher=True)
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'SMOKE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
