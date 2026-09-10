"""One real MOSI batch, two gradient updates; not an epoch or experiment sweep."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.osram_local_cross_attn_20260910.run import configuration, ROOT, FEATURES
from experiments.osram_local_gated_20260910.smoke import make, count
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.loss import missing_m3_loss


def main():
    torch.set_num_threads(4);cfg,_,_=configuration(66);tr.set_random_seed(66)
    roots=[str(FEATURES/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    train,_,_,*dims=tr.get_loaders(audio_root=roots[0],text_root=roots[1],video_root=roots[2],
        num_folder=1,dataset=cfg.dataset,batch_size=cfg.batch_size,num_workers=0,seed=cfg.seed,
        evaluation_protocol=cfg.evaluation_protocol,validation_fraction=cfg.validation_fraction)
    view=tr._prepare_view(tr._move_batch(next(iter(train[cfg.fold-1])),torch.device(cfg.device)),
                          tr._build_schedule(cfg,'train',.5),0,tuple(dims))
    model=make(cfg,dims)
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    report=dict(total_parameters=count(model),trainable_parameters=count(model,True),
        cross_fusion_parameters=count(model.osram.local_centered_fusion),steps=[],
        dataset=cfg.dataset,conversations=len(view['lengths']),utterances=int(view['umask'].sum()))
    for step in range(2):
        model.train();opt.zero_grad()
        logits,_,_,pred=model([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'],predict_missing=True)
        cls=tr._task_loss(cfg.dataset,logits,view['labels'],view['umask'],cfg.mosi_task_mode,cfg.task_regression_loss,cfg.task_smooth_l1_beta)
        teacher=model.encode_teacher_targets([view['complete']])
        jepa=missing_m3_loss(pred,teacher,temperature=cfg.temperature,
            regression_aggregation=cfg.jepa_regression_aggregation,contrastive_prediction_source=cfg.jepa_contrastive_source)
        loss=cls+cfg.jepa_weight*jepa.total;assert torch.isfinite(loss);loss.backward()
        norms={name:sum(float(p.grad.abs().sum()) for p in getattr(model.osram.local_centered_fusion,name).parameters() if p.grad is not None)
               for name in ('query','key','value','evidence_type','context_out')}
        if step:assert all(v>0 for v in norms.values()),norms
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.teacher.parameters())
        report['steps'].append(dict(step=step+1,loss=float(loss),gradient_l1=norms,
            diagnostics=model.osram.local_centered_fusion.last_diagnostics))
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip_norm);opt.step();model.update_teacher(cfg.ema_tau)
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'SMOKE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
