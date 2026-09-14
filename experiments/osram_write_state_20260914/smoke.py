"""Two updates on one real batch; no epoch or experiment launched."""
import inspect
import json
from dataclasses import asdict,replace
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel
from experiments.osram_causal_nojepa_20260910.run import configuration,runner

OUTPUT=Path('/data2/yb/remote_experiments/osram_write_state_20260914')


def main():
    torch.set_num_threads(4);cfg,_,_=configuration(66)
    cfg=replace(cfg,training_objective='write-state');tr.set_random_seed(cfg.seed)
    roots=[str(runner.FEATURES/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    train,_,_,*dims=tr.get_loaders(audio_root=roots[0],text_root=roots[1],video_root=roots[2],
        num_folder=1,dataset=cfg.dataset,batch_size=cfg.batch_size,num_workers=0,seed=cfg.seed,
        evaluation_protocol=cfg.evaluation_protocol,validation_fraction=cfg.validation_fraction)
    v=tr._prepare_view(tr._move_batch(next(iter(train[0])),torch.device(cfg.device)),
                       tr._build_schedule(cfg,'train',.5),0,tuple(dims))
    names=inspect.signature(MissingM3GraphModel.__init__).parameters
    kw={k:value for k,value in asdict(cfg).items() if k in names}
    kw.update(adim=dims[0],tdim=dims[1],vdim=dims[2],D_e=cfg.hidden,graph_hidden_size=cfg.hidden//2,
              n_speakers=1,n_classes=1,time_attn=cfg.time_attention,write_state_completion=True)
    m=MissingM3GraphModel(**kw).to(cfg.device)
    opt=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    report=dict(steps=[],total_parameters=sum(p.numel() for p in m.parameters()),
        trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
        predictor_parameters=sum(p.numel() for p in m.write_state.predictor.parameters()),
        predictor_budget=m.write_state.predictor.parameter_budget,predictor_width=m.write_state.predictor.width,
        teacher_parameters=sum(p.numel() for p in m.write_state.teacher.parameters()),
        conversations=len(v['lengths']),utterances=int(v['umask'].sum()))
    call=([v['incomplete']],v['availability'],v['qmask'],v['umask'],v['lengths'])
    for step in range(2):
        m.train();opt.zero_grad(set_to_none=True)
        with patch.object(m.missing_predictor,'forward',side_effect=AssertionError('MMoE')):
            logits,_,_,_=m(*call)
            aux,n=m.write_state_loss(v['complete'],v['availability'],v['qmask'],v['umask'])
            cls=tr._task_loss(cfg.dataset,logits,v['labels'],v['umask'],cfg.mosi_task_mode,
                              cfg.task_regression_loss,cfg.task_smooth_l1_beta)
            pg=list(m.write_state.predictor.parameters())
            task_grad=torch.autograd.grad(cls,pg,retain_graph=True,allow_unused=True)
            task_norm=sum(float(g.abs().sum()) for g in task_grad if g is not None)
            loss=cls+cfg.jepa_weight*aux;loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        assert all(p.grad is None for p in m.write_state.teacher.parameters())
        norm=sum(float(p.grad.abs().sum()) for p in pg if p.grad is not None)
        assert norm>0
        if step:assert task_norm>0
        report['steps'].append(dict(step=step+1,loss=float(loss),write_loss=float(aux),targets=n,
                                   predictor_gradient_l1=norm,emotion_predictor_gradient_l1=task_norm))
        torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.gradient_clip_norm);opt.step();m.update_teacher(cfg.ema_tau)
    m.eval()
    with torch.no_grad(),patch.object(m.write_state.teacher,'forward',side_effect=AssertionError('test Teacher')):
        m(*call)
    report.update(ema_updates=int(m.write_state.ema_updates),test_teacher_calls=0)
    OUTPUT.mkdir(exist_ok=True,parents=True)
    (OUTPUT/'SMOKE.json').write_text(json.dumps(report,indent=2))
    (OUTPUT/'config.seed66.json').write_text(json.dumps(asdict(cfg),indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
