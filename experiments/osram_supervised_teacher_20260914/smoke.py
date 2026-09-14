"""One supervised Teacher batch + validation, then one frozen-target Student batch."""
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
from gcnet_missing_m3.pretrained_teacher import export_teacher_projectors,state_sha256
from experiments.osram_causal_nojepa_20260910.run import configuration,runner
from experiments.osram_supervised_teacher_20260914.run import teacher_config,student_config,ROOT


def norm(grads):
    return sum(float(g.detach().double().square().sum()) for g in grads if g is not None)**.5


def build(cfg,dims):
    names=inspect.signature(MissingM3GraphModel.__init__).parameters
    kwargs={k:v for k,v in asdict(cfg).items() if k in names}
    kwargs.update(adim=dims[0],tdim=dims[1],vdim=dims[2],D_e=cfg.hidden,
                  graph_hidden_size=cfg.hidden//2,n_speakers=1,n_classes=1,time_attn=cfg.time_attention)
    return MissingM3GraphModel(**kwargs).to(cfg.device)


def main():
    torch.set_num_threads(4)
    base,_,_=configuration(66);first=replace(teacher_config(base),epochs=1)
    tr.set_random_seed(base.seed)
    roots=[str(runner.FEATURES/n) for n in ('wav2vec-large-c-UTT','deberta-large-4-UTT','manet_UTT')]
    train,validation,_,*dims=tr.get_loaders(audio_root=roots[0],text_root=roots[1],video_root=roots[2],
        num_folder=1,dataset=base.dataset,batch_size=base.batch_size,num_workers=0,seed=base.seed,
        evaluation_protocol=base.evaluation_protocol,validation_fraction=base.validation_fraction)
    batch=tr._move_batch(next(iter(train[0])),torch.device(base.device))
    full=tr._prepare_view(batch,tr._build_schedule(first,'train',0.),0,tuple(dims))
    initial_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    teacher_model=build(first,dims)
    initial={k:v.detach().cpu().clone() for k,v in teacher_model.state_dict().items()}
    initial_projectors=state_sha256(teacher_model.observed_set.projectors.state_dict())
    stale=state_sha256(teacher_model.teacher.state_dict())
    opt=torch.optim.Adam((p for p in teacher_model.parameters() if p.requires_grad),
                          lr=first.learning_rate,weight_decay=first.weight_decay)
    teacher_model.train()
    with patch.object(teacher_model.missing_predictor,'forward',side_effect=AssertionError('Stage1 MMoE')):
        logits=teacher_model([full['incomplete']],full['availability'],full['qmask'],full['umask'],full['lengths'])[0]
    loss=tr._task_loss(first.dataset,logits,full['labels'],full['umask'],first.mosi_task_mode,
                       first.task_regression_loss,first.task_smooth_l1_beta)
    loss.backward()
    teacher_grad={m:norm(p.grad for p in module.parameters()) for m,module in teacher_model.observed_set.projectors.items()}
    assert all(v>0 for v in teacher_grad.values())
    torch.nn.utils.clip_grad_norm_(teacher_model.parameters(),first.gradient_clip_norm);opt.step()
    trained=state_sha256(teacher_model.observed_set.projectors.state_dict())
    assert trained!=initial_projectors and stale==state_sha256(teacher_model.teacher.state_dict())
    # Real validation, complete inputs only. No test-set selection or evaluation.
    validation_metrics,_=tr.evaluate_rate(teacher_model,validation[0],tr._build_schedule(first,'validation',0.),
        first.dataset,tuple(dims),torch.device(first.device),False,mosi_task_mode=first.mosi_task_mode,
        task_regression_loss=first.task_regression_loss,task_smooth_l1_beta=first.task_smooth_l1_beta)
    out=ROOT/'smoke';out.mkdir(parents=True,exist_ok=False)
    source=out/'one_batch_teacher.pt'
    torch.save(dict(model=tr._state_to_cpu(teacher_model),config=asdict(first),epoch=1,
        selection_split='validation',validation_mean_weighted_f1=validation_metrics['weighted_f1'],
        diagnostic_only=True),source)
    exported=out/'teacher_projectors.pt';export_teacher_projectors(source,exported)
    del teacher_model,opt
    second=student_config(base,exported)
    torch.set_rng_state(initial_rng);torch.cuda.set_rng_state_all(cuda_rng)
    student=build(second,dims)
    for k,v in initial.items():
        if not k.startswith('teacher.'):assert torch.equal(v,student.state_dict()[k].cpu()),k
    frozen_before=student.teacher_integrity();assert frozen_before==trained
    view=tr._prepare_view(batch,tr._build_schedule(second,'train',.5),0,tuple(dims))
    opt=torch.optim.Adam((p for p in student.parameters() if p.requires_grad),lr=second.learning_rate,
                          weight_decay=second.weight_decay)
    student.train()
    logits,_,_,pred=student([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'],predict_missing=True)
    targets=student.encode_teacher_targets([view['complete']])
    jepa=tr.missing_m3_loss(pred,targets,temperature=second.temperature,
        regression_aggregation=second.jepa_regression_aggregation,contrastive_prediction_source=second.jepa_contrastive_source)
    cls=tr._task_loss(second.dataset,logits,view['labels'],view['umask'],second.mosi_task_mode,
                      second.task_regression_loss,second.task_smooth_l1_beta)
    jepa_predictor_grad=norm(torch.autograd.grad(jepa.total,list(student.missing_predictor.parameters()),
                                               retain_graph=True,allow_unused=True))
    total=cls+second.jepa_weight*jepa.total;total.backward()
    assert torch.isfinite(total) and jepa_predictor_grad>0
    assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
    assert all(p.grad is None for p in student.teacher.parameters())
    torch.nn.utils.clip_grad_norm_(student.parameters(),second.gradient_clip_norm);opt.step()
    with patch.object(student.teacher,'update_from',side_effect=AssertionError('Frozen EMA')):
        student.update_teacher(second.ema_tau)
    assert frozen_before==student.teacher_integrity() and student.ema_step==0
    student.eval()
    with torch.no_grad(),patch.object(student,'encode_teacher_targets',side_effect=AssertionError('inference Teacher')):
        assert torch.isfinite(student([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'])[0]).all()
    report=dict(diagnostic_only=True,full_training_started=False,teacher_optimizer_updates=1,student_optimizer_updates=1,
        teacher_emotion_loss=float(loss),teacher_projector_gradient_l2=teacher_grad,
        trained_online_projectors_changed=True,stale_stage1_teacher_unchanged=True,
        teacher_source_prefix='observed_set.projectors.',student_shared_initialization_exact=True,
        teacher_hash_before=frozen_before,teacher_hash_after=student.teacher_integrity(),student_ema_updates=0,
        student_emotion_loss=float(cls),jepa_loss=float(jepa.total),regression_loss=float(jepa.regression),
        contrastive_loss=float(jepa.contrastive),total_loss=float(total),targets=jepa.target_count,
        jepa_predictor_gradient_l2=jepa_predictor_grad,teacher_gradients_none=True,new_model_parameters=0,
        stage1_validation_only=True,teacher_validation_weighted_f1=validation_metrics['weighted_f1'])
    (out/'SMOKE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
