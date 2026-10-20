"""Seed66 MOSI150: change only backbone LR to constant1e-5; internal Test-oracle."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
from experiments.osram_mosi_gcnet_router_3seed_20261019.report import CORE

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2/runs/MOSI/seed_66')
RATES = selected.RATES


def validate_backbone_lr(backbone_lr):
    if not math.isfinite(backbone_lr) or backbone_lr <= 0:
        raise ValueError('Positive finite backbone LR required')


def set_epoch_lrs(optimizer, epoch, *, backbone_lr=1e-5):
    validate_backbone_lr(backbone_lr)
    if not 0 <= epoch < 150:
        raise ValueError('Exactly150epochs')
    if any(g['name'] not in ('backbone', 'router') for g in optimizer.param_groups):
        raise ValueError('Unknown optimizer parameter group')
    for group in optimizer.param_groups:
        group['lr'] = backbone_lr if group['name'] == 'backbone' else (3e-4 if epoch < 100 else 3e-5)


def validate_reference(identity, reference, initial, *, backbone_lr=1e-5):
    validate_backbone_lr(backbone_lr)
    if reference.get('status') != 'complete' or reference.get('epoch') != 150:
        raise ValueError('Require completed150epoch reference')
    if reference['initial_backbone_sha256'] != initial:
        raise ValueError('Fresh initial backbone differs')
    old = reference['identity']
    expected = dict(old['config'], learning_rate=backbone_lr)
    if expected != json.loads(json.dumps(identity['config'])):
        raise ValueError('Training config changed beyond backbone learning rate')
    for key in ('split', 'mask_hashes', 'seed_protocol'):
        if old[key] != identity[key]:
            raise ValueError('Reference '+key+' differs')
    for name in CORE:
        if old['source_hashes'][name] != identity['source_hashes'][name]:
            raise ValueError('Model/sampler numerical core changed: '+name)
    return dict(only_backbone_lr_changed=True, initialization_exact=True,
                split_exact=True, evaluation_masks_exact=True, core_sources_exact=True,
                old_weights_loaded=False)


def check_mask_parity(actual, old):
    for key in ('epoch', 'mask_trace_sha256', 'optimizer_steps'):
        if actual[key] != old[key]:
            raise ValueError('Reference training '+key+' differs')
    return dict(epoch=actual['epoch'], training_masks_exact=True, optimizer_steps_exact=True)


def main(argv=None, *, backbone_lr=1e-5, extra_source_files=()):
    validate_backbone_lr(backbone_lr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--mode', choices=('flat', 'joint'), required=True)
    parser.add_argument('--seed', type=int, choices=(66,), required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as original
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if args.output.exists() and not args.resume:
        raise FileExistsError('Independent fresh output required')
    if args.resume and args.phase != 'train':
        raise ValueError('Resume only exact training state')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick = args.output/'SUMMARY.json', time.monotonic()
    summary = dict(status='preflight', dataset='CMUMOSI', mode=args.mode, phase=args.phase,
        seed=args.seed, source_commit=args.source_commit, pid=os.getpid(),
        experiment_label='INTERNAL TEST-ORACLE SCREEN; backbone LR-only experiment')
    try:
        torch.set_num_threads(1)
        bundle = original.setup(args.gpu_uuid, args.output, args.seed)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        bundle['cfg'] = replace(selected.testbest_config(bundle['cfg']), learning_rate=backbone_lr)
        model = scalar.wrap_model(bundle['model'], args.mode)
        optimizer = scalar.make_optimizer(model, bundle['cfg'], args.mode)
        masks = {split: original.build_masks(bundle, split) for split in ('validation', 'test')}
        hashes = {split: {rate: hashlib.sha256(b''.join(row['availability'].numpy().tobytes()
                    for row in bank)).hexdigest() for rate, bank in rates.items()} for split, rates in masks.items()}
        ref = REFERENCE/args.mode
        prior = json.loads((ref/'SUMMARY.json').read_text())
        sources = tuple(prior['identity']['source_hashes']) + tuple(
            str(Path(__file__).with_name(name).relative_to(ROOT)) for name in ('worker.py', 'run.py', 'report.py')) + tuple(extra_source_files)
        identity = dict(source_commit=args.source_commit, source_hashes={p:h.sha(ROOT/p) for p in sources},
            mode=args.mode, seed=args.seed, config=asdict(bundle['cfg']), split=bundle['split'],
            mask_hashes=hashes, seed_protocol=bundle['seed_protocol'], historical_bridge=bundle['bridge'],
            dataset_manifest_sha256=bundle['data_manifest_sha256'], data_manifest=bundle['data_manifest'],
            data_files=bundle['data_files'], initialization='fresh-original-hardcoded-missing_m3_model_init:fold:5',
            selection='test-oracle-per-rate; weighted_f1; strict improvement; earliest tie',
            backbone_learning_rates=dict(first_100=backbone_lr, final_50=backbone_lr),
            router_learning_rates=[3e-4, 3e-5] if args.mode == 'joint' else None)
        proof = validate_reference(identity, prior, bundle['initial_backbone_sha256'], backbone_lr=backbone_lr)
        proof.update(path=str(ref), summary_sha256=h.sha(ref/'SUMMARY.json'), history_sha256=h.sha(ref/'history.json'))
        reference_history = json.loads((ref/'history.json').read_text())
        if len(reference_history) != 150:
            raise ValueError('Require complete reference history')
        identity['lr_only_reference'] = proof
        torch.save(masks, args.output/'fixed_masks.pt')
        summary.update(status='running', identity=identity, config=identity['config'], environment=bundle['environment'],
            gpu=bundle['gpu'], initial_backbone_sha256=bundle['initial_backbone_sha256'],
            mask_hashes=hashes, anchor_parity_gate=bundle['bridge'], lr_only_reference=proof,
            total_parameters=sum(p.numel() for p in model.parameters()),
            router_parameters=sum(p.numel() for p in model.router.parameters()) if args.mode=='joint' else 0)
        h.write_json(args.output/'config.json', identity['config'])
        h.write_json(path, summary)
        reset_training_stochasticity(args.seed, bundle['cfg'].fold)
        torch.cuda.reset_peak_memory_stats()
        best, history, parity, start = {}, [], [], 0
        if args.resume:
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            selected.validate_selected(args.output, best)
            parity = [check_mask_parity(row, reference_history[i]) for i,row in enumerate(history)]
        before = h.backbone_digest(model) if args.phase=='smoke' else None
        before_router = h.backbone_digest(model.router) if args.phase=='smoke' and args.mode=='joint' else None
        for epoch in (range(1) if args.phase=='smoke' else range(start,150)):
            set_epoch_lrs(optimizer, epoch, backbone_lr=backbone_lr)
            row = scalar.train_epoch(bundle,model,optimizer,epoch,args.mode,
                max_batches=selected.SMOKE_BATCHES if args.phase=='smoke' else None)
            if args.mode=='joint' and not row['gradients']['router']['nonzero']:
                raise RuntimeError('No nonzero Router gradient')
            if args.phase=='train':
                parity.append(check_mask_parity(row,reference_history[epoch]))
            evaluated = {rate:scalar.evaluate(bundle,model,args.mode,'test',masks['test'][rate],
                max_batches=1 if args.phase=='smoke' else None) for rate in RATES}
            row['test']={rate:value[0] for rate,value in evaluated.items()}
            history.append(row)
            improved=selected.persist_epoch(args.output,model,optimizer,epoch=epoch+1,identity=identity,
                history=history,best=best,evaluated=evaluated)
            summary.update(epoch=epoch+1,selected_epoch_by_rate={r:v['epoch'] for r,v in best.items()},
                best_by_rate=best,training_mask_parity=parity,seconds=time.monotonic()-tick,
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(path,summary)
            print(json.dumps(dict(event='epoch',epoch=epoch+1,task_MSE=row['task_MSE'],
                learning_rates=row['learning_rates'],test_wf1={r:v[0]['router']['WF1'] for r,v in evaluated.items()},
                improved_rates=improved)),flush=True)
        if args.phase=='smoke':
            if h.backbone_digest(model)==before or (before_router and h.backbone_digest(model.router)==before_router):
                raise RuntimeError('Smoke must update both backbone and Router')
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            restore(model,optimizer,state,identity)
            if not all(torch.equal(v.cpu(),state['model'][k]) for k,v in model.state_dict().items()):
                raise RuntimeError('Model recovery differs')
            selected.validate_selected(args.output,best)
            _,arrays=scalar.evaluate(bundle,model,args.mode,'test',masks['test']['0.3'],max_batches=1)
            with np.load(args.output/best['0.3']['predictions'],allow_pickle=False) as saved:
                for k in arrays: np.testing.assert_array_equal(arrays[k],saved[k])
            summary.update(smoke=history[0],smoke_only=True,full_checkpoint_reloaded=True,
                selected_artifacts_exact_reload=True)
        else:
            summary['evaluation']=selected.publish_selected(args.output,best)
            summary['aggregate']=scalar.aggregate(summary['evaluation'])
        for name,digest in identity['source_hashes'].items():
            if h.sha(ROOT/name)!=digest: raise RuntimeError('Source changed: '+name)
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(path,summary)


if __name__=='__main__':
    main()
