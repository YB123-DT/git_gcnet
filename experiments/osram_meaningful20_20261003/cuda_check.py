"""Actual batch32 CUDA feasibility evidence; never launches formal training."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from .manifest import now, read, sha, write
from .preflight import query_gpus, validate_gpu


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--reference', type=Path, required=True, help='Original seed66 reference directory')
    p.add_argument('--dataset', type=Path, required=True, help='Hashed data manifest with feature_roots')
    p.add_argument('--output', type=Path, required=True, help='New evidence directory; no overwrite')
    p.add_argument('--gpu-index', required=True)
    p.add_argument('--gpu-uuid', required=True)
    return p


def check_gpu(index, uuid, mapping):
    validate_gpu(index, uuid, mapping)


def artifact_budget(model_bytes):
    if model_bytes <= 0: raise ValueError('Positive model size required')
    # 100 immutable model versions + 8 BEST + last model/Adam moments (3),
    # plus serialization/log/prediction headroom; never delete versions.
    return 1.25 * (112 * model_bytes / 1024**3) + 2


def check_finite_state(model, candidate):
    """Audit parameters AND floating buffers after data initialization/update."""
    import torch
    count=0
    for name,value in model.state_dict().items():
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite floating model state: '+name)
            count+=1
    initialized=[]
    if candidate=='node':
        initialized=[(name,module) for name,module in model.named_modules()
                     if all(hasattr(module,k) for k in ('initialized','threshold','log_temperature'))]
        if len(initialized)!=3 or any(not bool(m.initialized) for _,m in initialized):
            raise RuntimeError('NODE must have exactly three initialized tree layers')
    return {'finite_floating_state_tensors':count,'node_initialized_layers':len(initialized)}


def profile(args, log):
    resources = query_gpus()
    check_gpu(args.gpu_index, args.gpu_uuid, {i:r['uuid'] for i,r in resources.items()})
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    if previous not in (None, args.gpu_uuid):
        raise ValueError('Visibility must be unset or the exact audited UUID')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    # The helper is stdlib-only and validates label/split hashes before config
    # captures GCNET_DATASET_ROOT at import time. Never infer an inherited root.
    from .run import candidate_config, bind_data_environment
    roots = bind_data_environment(args.dataset)
    import torch
    from gcnet_missing_m3.train_gcnet import (TrainConfig, get_loaders, _move_batch,
        _prepare_view, _schedules, _emotion_loss, _optimizer_parameter_groups)
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one real CUDA device required')
    raw = candidate_config(read(args.reference / 'config.json'), args.candidate, seed=66)
    config = TrainConfig(**raw)
    loaders = get_loaders(audio_root=roots[0], text_root=roots[1], video_root=roots[2],
        num_folder=1, dataset=config.dataset, batch_size=32, num_workers=0, seed=66,
        validation_fraction=config.validation_fraction, evaluation_protocol='official')
    train_loader = loaders[0][config.fold-1]
    dimensions = tuple(loaders[3:])
    if dimensions != (512,1024,1024): raise ValueError('Unexpected raw feature dimensions')
    data = _move_batch(next(iter(train_loader)), torch.device('cuda'))
    if data[7].shape[0] != 32 or len(data[-1]) != 32:
        raise ValueError('First training batch must contain 32 real conversations')
    view = _prepare_view(data, _schedules(config,'train')[.7], 1, dimensions)
    baseline = _build_model(TrainConfig(**dict(raw,osram_meaningful_block='none')),dimensions).cuda()
    model = _build_model(config,dimensions).cuda()
    if not any('meaningful_block.' in n for n,_ in model.named_parameters()):
        raise RuntimeError('Builder did not instantiate the requested meaningful block')
    model.load_state_dict({**model.state_dict(), **baseline.state_dict()},strict=True)
    def forward(m,v):
        out = m([v['incomplete']],v['availability'],v['qmask'],v['umask'],v['lengths'],predict_missing=False)
        if out[3] is not None: raise RuntimeError('Unexpected completion')
        return out[0]
    def loss(pred,v):
        return _emotion_loss(config.dataset,pred,v['labels'],v['umask'],v['availability'],
            config.emotion_loss_mode,config.mosi_task_mode,config.task_regression_loss,
            config.task_smooth_l1_beta)[0]
    baseline.eval(); model.eval()
    def scanned(m):
        scans=[s for s in m.modules() if callable(getattr(s,'_scan',None))]
        if len(scans)!=1: raise RuntimeError('Expected one OSRAM scan owner')
        return scans[0]
    with torch.no_grad():
        with patch.object(scanned(baseline),'_scan',wraps=scanned(baseline)._scan) as bscan:
            expected=forward(baseline,view)
        with patch.object(scanned(model),'_scan',wraps=scanned(model)._scan) as scan:
            actual=forward(model,view)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        if bscan.call_count!=1 or scan.call_count!=1: raise RuntimeError('Extra memory scan')
    log('zero_bridge_predictions_exact; one causal scan each')
    # Raw shared gradients at the identity initialization, without optimizer or clipping.
    loss(forward(baseline,view),view).backward()
    expected_grad={n:p.grad.detach().cpu().clone() for n,p in baseline.named_parameters() if p.grad is not None}
    loss(forward(model,view),view).backward()
    checked=0
    for n,p in model.named_parameters():
        if n in expected_grad:
            if p.grad is None: raise RuntimeError('Shared gradient disappeared: '+n)
            torch.testing.assert_close(p.grad.cpu(),expected_grad[n],rtol=1e-5,atol=1e-7)
            checked+=1
    model.zero_grad(set_to_none=True)
    # Mock wraps retain their bound OSRAM owners beyond the context manager.
    del bscan, scan
    del baseline,expected_grad,expected,actual
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    groups,_ = _optimizer_parameter_groups(model,config)
    optimizer=torch.optim.Adam(groups,weight_decay=config.weight_decay)
    losses=[]; start=time.monotonic(); model.train()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        value=loss(forward(model,view),view)
        state_checks=check_finite_state(model,args.candidate)
        if not torch.isfinite(value): raise RuntimeError('Nonfinite task loss')
        value.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite gradient')
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm)
        optimizer.step(); losses.append(float(value.detach()))
        state_checks=check_finite_state(model,args.candidate)
        log(f'Adam step {step+1}/3, loss={losses[-1]}')
    model.eval(); schedules=_schedules(config,'test'); eval_checks=[]
    for rate,schedule in schedules.items():
        testview=_prepare_view(data,schedule,0,dimensions)
        with torch.no_grad(): prediction=forward(model,testview)
        with torch.inference_mode(): inference=forward(model,testview)
        torch.testing.assert_close(prediction,inference,rtol=1e-5,atol=1e-6)
        if not torch.isfinite(prediction).all(): raise RuntimeError('Nonfinite evaluation')
        eval_checks.append(rate)
    torch.cuda.synchronize()
    peak=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())/1024**2
    checkpoint=args.output/'strict_reload.pt'
    torch.save(model.state_dict(),checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True),strict=True)
    state_checks=check_finite_state(model,args.candidate)
    with torch.no_grad(): reloaded=forward(model,testview)
    torch.testing.assert_close(reloaded,prediction,rtol=0,atol=0)
    return dict(status='passed',candidate=args.candidate,gpu_index=args.gpu_index,gpu_uuid=args.gpu_uuid,
        profile=dict(batch_size=32,train_and_eval=True,peak_mib=peak,
                     artifact_gib=artifact_budget(checkpoint.stat().st_size)),
        task_losses=losses,training_missing_rate=.7,evaluation_rates=eval_checks,
        seconds=time.monotonic()-start,shared_gradient_tensors=checked,
        state_checks=state_checks,
        conversation_ids=[str(x) for x in data[-1]],valid_utterances=int(data[7].sum()),
        strict_reload_sha256=sha(checkpoint),reference_config_sha256=sha(args.reference/'config.json'),
        data_manifest_sha256=sha(args.dataset),torch_version=torch.__version__,
        not_tested=['full epoch','full resume','task scores','all training batch lengths'],
        note='Evaluation masks on real training conversations; feasibility only, no test-score evaluation.')


def main():
    args=parser().parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    logpath=args.output/'cuda.log'
    with logpath.open('x') as stream:
        def log(message):
            print(message,flush=True); stream.write(message+'\n'); stream.flush()
        log('command: '+' '.join(sys.argv))
        try:
            result=profile(args,log); log('CUDA feasibility passed')
        except Exception as error:
            log(f'{type(error).__name__}: {error}')
            write(args.output/'profile.json',dict(status='failed',error=str(error),candidate=args.candidate))
            raise
    result.update(completed_at=now(),log=str(logpath.resolve()),log_sha256=sha(logpath),command=sys.argv)
    write(args.output/'profile.json',result)


if __name__ == '__main__': main()
