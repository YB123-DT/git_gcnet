"""One real MOSI batch and one optimizer update; never launch a full run."""
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from gcnet_missing_m3 import train_gcnet as tr
from gcnet_missing_m3.model import MissingM3GraphModel
from experiments.osram_causal_nojepa_20260910.run import configuration, runner

OUTPUT = Path('/data2/yb/remote_experiments/osram_future_state_20260914')


def gradient_norm(grads):
    return sum(float(g.detach().double().square().sum()) for g in grads if g is not None) ** .5


def main():
    torch.set_num_threads(4)
    cfg, reference, _ = configuration(66)
    cfg = replace(cfg, training_objective='future-state')
    tr.set_random_seed(cfg.seed)
    roots = [str(runner.FEATURES / name) for name in (
        'wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    train, _, _, *dims = tr.get_loaders(
        audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset=cfg.dataset, batch_size=cfg.batch_size, num_workers=0,
        seed=cfg.seed, evaluation_protocol=cfg.evaluation_protocol,
        validation_fraction=cfg.validation_fraction)
    batch = tr._move_batch(next(iter(train[0])), torch.device(cfg.device))
    view = tr._prepare_view(batch, tr._build_schedule(cfg, 'train', .5), 0, tuple(dims))
    names = inspect.signature(MissingM3GraphModel.__init__).parameters
    kwargs = {k: v for k,v in asdict(cfg).items() if k in names}
    kwargs.update(adim=dims[0], tdim=dims[1], vdim=dims[2], D_e=cfg.hidden,
                  graph_hidden_size=cfg.hidden//2, n_speakers=1, n_classes=1,
                  time_attn=cfg.time_attention)
    rng = torch.get_rng_state()
    baseline = MissingM3GraphModel(**kwargs)
    after_baseline = torch.get_rng_state()
    torch.set_rng_state(rng)
    model = MissingM3GraphModel(**kwargs, future_state_jepa=True)
    assert torch.equal(after_baseline, torch.get_rng_state())
    for k,v in baseline.state_dict().items(): assert torch.equal(v,model.state_dict()[k]),k
    shared_count = len(baseline.state_dict())
    del baseline
    model.to(cfg.device).train()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    args = ([view['incomplete']],view['availability'],view['qmask'],view['umask'],view['lengths'])
    with patch.object(model.missing_predictor,'forward',side_effect=AssertionError('old MMoE')):
        logits,_,_,_=model(*args)
        future,count=model.future_state_loss(view['complete'],view['umask'])
    expected=sum(max(0,n-1) for n in view['lengths'])
    assert count==expected
    groups={'state_query':[model.future_state.state_query],
            'predictor':list(model.future_state.predictor.parameters()),
            'key_projectors':list(model.osram.key_projectors.parameters()),
            'value_projectors':list(model.osram.value_projectors.parameters()),
            'encoder':list(model.observed_set.parameters())}
    gradients={}
    for name,params in groups.items():
        grads=torch.autograd.grad(future,params,retain_graph=True,allow_unused=True)
        gradients[name]=gradient_norm(grads)
        assert gradients[name]>0
        assert all(torch.isfinite(g).all() for g in grads if g is not None)
    cls=tr._task_loss(cfg.dataset,logits,view['labels'],view['umask'],cfg.mosi_task_mode,
                      cfg.task_regression_loss,cfg.task_smooth_l1_beta)
    total=cls+cfg.jepa_weight*future
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.future_state.teacher_encoder.parameters())
    assert all(p.grad is None for p in model.future_state.teacher_local.parameters())
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip_norm)
    optimizer.step();model.update_teacher(cfg.ema_tau)
    counts={}
    with torch.no_grad():
        for rate in (0.,.7):
            v=tr._prepare_view(batch,tr._build_schedule(cfg,'train',rate),0,tuple(dims))
            model([v['incomplete']],v['availability'],v['qmask'],v['umask'],v['lengths'])
            _,counts[str(rate)]=model.future_state_loss(v['complete'],v['umask'])
        assert counts['0.0']==counts['0.7']==expected
        model.eval()
        with (patch.object(model.future_state,'observe',side_effect=AssertionError('state query')),
             patch.object(model.future_state.predictor,'forward',side_effect=AssertionError('predictor')),
             patch.object(model.future_state,'target',side_effect=AssertionError('Teacher'))):
            assert torch.isfinite(model(*args)[0]).all()
    head=model.future_state
    report=dict(config=asdict(cfg),reference_config=str(reference/'config.json'),
                smoke_only=True,full_training_started=False,shared_state_tensors_exact=shared_count,
                rng_equal=True,conversations=len(view['lengths']),utterances=int(view['umask'].sum()),
                transitions=count,counts_by_rate=counts,emotion_loss=float(cls),future_loss=float(future),
                total_loss=float(total),future_gradient_l2=gradients,total_gradient_l2=float(grad_norm),
                teacher_gradients_none=True,ema_updates=int(head.ema_updates),
                online_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
                parameter_budget=head.parameter_budget,hidden_width=head.width,
                teacher_parameters=sum(p.numel() for p in head.teacher_encoder.parameters())
                                   +sum(p.numel() for p in head.teacher_local.parameters()),
                total_model_parameters=sum(p.numel() for p in model.parameters()),
                eval_auxiliary_calls=0)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'SMOKE.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
