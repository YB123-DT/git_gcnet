"""Original Nested GNN vs current MOSEI Flat; finite matched-protocol rerun."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
METHOD = 'nested_gnn_rooted_evidence'
RATES = tuple(f'{i/10:.1f}' for i in range(8))
FLAT66 = Path('/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat')
DIRECT = Path('/data2/yb/remote_experiments/osram_source_role_direct_cross_20261020/execution_v1/runs/CMUMOSEI/fold_1')


def mask_reference(seed):
    if seed not in (66, 67, 68): raise ValueError('Only requested seeds66/67/68')
    return FLAT66 if seed == 66 else DIRECT/f'seed_{seed}'/'direct'


def candidate_config(cfg, seed, mode):
    if seed not in (66, 67, 68) or mode not in ('flat', 'nested'):
        raise ValueError('Only original Nested/Flat seeds66/67/68')
    required = dict(dataset='CMUMOSEI', seed=66, fold=1, epochs=60, batch_size=32,
        learning_rate=.001, weight_decay=1e-5, optimizer='adam', lr_schedule='constant',
        missing_mask_protocol='gcnet-random-mask-v1', training_objective='emotion-only',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
        evaluate_test=True, osram_output_dim=1600, osram_num_heads=8, osram_key_dim=64,
        osram_value_dim=64, osram_write_step=.6, osram_bidirectional=False,
        osram_readout_fusion='flat', osram_meaningful_block='none',
        osram_source_role=False, osram_source_relation=False, osram_tasr=False,
        osram_source_tracking=False, paired_history_views=False, completion_path='none',
        task_regression_loss='mse', mosi_task_mode='regression', train_rate_mode='cyclic')
    for key, value in required.items():
        if getattr(cfg, key) != value: raise ValueError('Original MOSEI Flat config changed: '+key)
    if tuple(cfg.train_missing_rates) != tuple(i/10 for i in range(8)):
        raise ValueError('Keep original cyclic0.0..0.7 rates')
    return replace(cfg, seed=seed, osram_meaningful_block=METHOD if mode=='nested' else 'none')


def validate_flat_config(cfg, old):
    """Normalize only four reviewed default-disabled schema additions."""
    expected = dict(old)
    expected.setdefault('osram_source_role', False)
    expected.setdefault('osram_source_role_mode', 'residual')
    expected.setdefault('osram_source_relation', False)
    expected.setdefault('osram_source_relation_layout', 'source-mean')
    actual = asdict(cfg)
    actual['osram_meaningful_block'] = 'none'
    if json.loads(json.dumps(actual)) != json.loads(json.dumps(expected)):
        changed = [k for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k)]
        raise ValueError('Pair differs beyond original Nested switch: '+str(changed))


def original_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('osram.meaningful_block.'): continue
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_nested_gradients(observed):
    required = ('tokenizer.', 'core.layers.', 'core.pool.', 'core.readout.',
                'local_decoder.', 'memory_decoders.')
    absent = [p for p in required if not any(n.startswith(p) and v for n,v in observed.items())]
    if absent: raise RuntimeError('Missing finite nonzero Nested task gradients: '+str(absent))


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('smoke','train'), required=True)
    p.add_argument('--mode', choices=('flat','nested'), required=True)
    p.add_argument('--seed', type=int, choices=(66,67,68), required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    return p


def main():
    args = argument_parser().parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import numpy as np
    import torch
    from experiments.osram_mosei_gcnet_router_20261020 import worker as old
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader, restore
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import persist_epoch, validate_selected, publish_selected
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    if args.phase=='train' and args.mode=='flat' and args.seed==66:
        raise ValueError('Completed matched Flat66 must be reused, not trained again')
    if args.resume and args.phase!='train': raise ValueError('Train-only full-state resume')
    if args.output.exists() and not args.resume: raise FileExistsError('Fresh independent output required')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    path = args.output/'SUMMARY.json'
    if path.exists() and json.loads(path.read_text()).get('status')=='complete':
        raise ValueError('Completed run must not repeat')
    tick = time.monotonic(); hooks = []
    summary = dict(status='preflight',dataset='CMUMOSEI',seed=args.seed,mode=args.mode,
        phase=args.phase,source_commit=args.source_commit,pid=os.getpid(),
        label='INTERNAL TEST-ORACLE SCREEN')
    try:
        torch.set_num_threads(1); h.write_json(path,summary)
        bundle = old.setup(args.gpu_uuid,args.output,66)
        del bundle['model']
        cfg = candidate_config(bundle['cfg'],args.seed,args.mode); bundle['cfg']=cfg
        bundle['seed_protocol'] = seed_training_loader(bundle,args.seed)
        baseline_cfg = replace(cfg,osram_meaningful_block='none')
        baseline = old.fresh_backbone(baseline_cfg,bundle['dimensions'],torch.device('cuda:0'))
        baseline.eval(); initial = h.backbone_digest(baseline)
        model = (old.fresh_backbone(cfg,bundle['dimensions'],torch.device('cuda:0'))
                 if args.mode=='nested' else baseline)
        if original_digest(model)!=initial: raise ValueError('Nested changed original parameter initialization')
        if args.mode=='nested':
            adapter = model.osram.meaningful_block.core
            core = adapter.core
            if (not adapter.residual or core.root_aware or len(core.layers)!=3
                    or core.root_embedding.embedding_dim!=64 or adapter.num_heads!=8):
                raise ValueError('Require exact original64d/3layer/8head residual Nested, not another variant')
        bundle['model']=model
        masks = scalar.build_masks(bundle,'test')
        hashes = {r:hashlib.sha256(b''.join(x['availability'].numpy().tobytes() for x in bank)).hexdigest()
                  for r,bank in masks.items()}
        ref = mask_reference(args.seed); prior = json.loads((ref/'SUMMARY.json').read_text())
        if prior['status']!='complete' or prior['epoch']!=60: raise ValueError('Complete current-mask reference required')
        ref_cfg = prior['config']
        if args.seed!=66:
            # This Direct model is ONLY a data/seed/mask reference, never a Flat score.
            ref_cfg = dict(ref_cfg,osram_source_role=False,osram_source_role_mode='residual')
        validate_flat_config(cfg,ref_cfg)
        if initial!=prior['initial_backbone_sha256']: raise ValueError('Same-seed original initialization differs')
        ref_hashes = prior['identity']['mask_hashes']
        if hashes!=ref_hashes.get('test',ref_hashes): raise ValueError('Same-seed exact GCNet evaluation masks differ')
        for key in ('split','seed_protocol'):
            if bundle[key]!=prior['identity'][key]: raise ValueError('Same-seed '+key+' differs')
        reference_history=json.loads((ref/'history.json').read_text())
        if len(reference_history)!=60: raise ValueError('Reference training history incomplete')
        if args.mode=='nested':
            _,a=scalar.evaluate(bundle,baseline,'flat','test',masks['0.3'],max_batches=1)
            _,b=scalar.evaluate(bundle,model,'flat','test',masks['0.3'],max_batches=1)
            for key in ('prediction','labels','availability'): np.testing.assert_array_equal(a[key],b[key])
            del baseline,a,b
        sources=set(prior['identity']['source_hashes'])
        for package in ('gcnet_missing_m3','gcnet_modality_jepa'):
            sources.update(str(p.relative_to(ROOT)) for p in (ROOT/package).rglob('*.py'))
        sources.update('experiments/osram_mosei_nested_gcnet_3seed_20261021/'+n for n in ('worker.py','run.py','report.py'))
        identity=dict(source_commit=args.source_commit,source_hashes={p:h.sha(ROOT/p) for p in sorted(sources)},
            config=asdict(cfg),seed=args.seed,mode=args.mode,split=bundle['split'],seed_protocol=bundle['seed_protocol'],
            mask_hashes={'test':hashes},dataset_manifest_sha256=bundle['data_manifest_sha256'],
            data_manifest=bundle['data_manifest'],data_files=bundle['data_files'],
            initialization='fresh original seed-specific factory; no checkpoint loaded',
            selection='per-rate TEST weighted_f1; strict improvement/earliest tie',
            reference=dict(path=str(ref),summary_sha256=h.sha(ref/'SUMMARY.json'),
                use='Flat score reference' if args.seed==66 else 'data/seed/mask only; NOT a Flat score',
                initialization_exact=True,masks_exact=True,config_only_nested_switch_changed=True))
        optimizer=scalar.make_optimizer(model,cfg,'flat')
        torch.save({'test':masks},args.output/'fixed_masks.pt')
        summary.update(status='running',identity=identity,config=identity['config'],
            environment=bundle['environment'],gpu=bundle['gpu'],reference_pair_verified=True,
            initial_backbone_sha256=initial,total_parameters=sum(p.numel() for p in model.parameters()),
            nested_parameters=sum(p.numel() for p in model.osram.meaningful_block.parameters()) if args.mode=='nested' else 0,
            zero_init_predictions_exact=True if args.mode=='nested' else None)
        h.write_json(args.output/'config.json',identity['config']); h.write_json(path,summary)
        observed={}; scans=[0]
        if args.mode=='nested':
            def capture(name):
                def hook(gradient):
                    if not torch.isfinite(gradient).all(): raise RuntimeError('Nonfinite Nested gradient: '+name)
                    observed[name]=observed.get(name,False) or bool(gradient.ne(0).any())
                return hook
            for name,p in model.osram.meaningful_block.core.named_parameters():
                hooks.append(p.register_hook(capture(name)))
        hooks.append(model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0,scans[0]+1)))
        reset_training_stochasticity(args.seed,1); torch.cuda.reset_peak_memory_stats()
        best,history,start={},[],0
        if args.resume:
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            restore(model,optimizer,state,identity); start,history,best=state['epoch'],state['history'],state['best_by_rate']
            validate_selected(args.output,best)
        before_model=h.backbone_digest(model) if args.phase=='smoke' else None
        before_nested=h.backbone_digest(model.osram.meaningful_block) if args.phase=='smoke' and args.mode=='nested' else None
        for epoch in (range(1) if args.phase=='smoke' else range(start,60)):
            if [g['lr'] for g in optimizer.param_groups]!=[.001]: raise ValueError('Keep constant1e-3')
            scans[0]=0; observed.clear()
            row=scalar.train_epoch(bundle,model,optimizer,epoch,'flat',max_batches=3 if args.phase=='smoke' else None)
            if scans[0]!=row['optimizer_steps']: raise RuntimeError('One causal OSRAM forward per batch required')
            if args.mode=='nested' and (args.phase=='smoke' or epoch==0): validate_nested_gradients(observed)
            if args.phase=='train':
                for key in ('epoch','optimizer_steps','mask_trace_sha256'):
                    if row[key]!=reference_history[epoch][key]: raise ValueError('Reference training '+key+' changed')
            row.update(actual_memory_scans=scans[0],nested_gradient_paths=dict(observed))
            evaluated={r:scalar.evaluate(bundle,model,'flat','test',masks[r],max_batches=1 if args.phase=='smoke' else None) for r in RATES}
            row['test']={r:v[0] for r,v in evaluated.items()}; history.append(row)
            improved=persist_epoch(args.output,model,optimizer,epoch=epoch+1,identity=identity,
                history=history,best=best,evaluated=evaluated)
            summary.update(epoch=epoch+1,best_by_rate=best,selected_epoch_by_rate={r:v['epoch'] for r,v in best.items()},
                seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(path,summary)
            print(json.dumps(dict(event='epoch',seed=args.seed,mode=args.mode,epoch=epoch+1,
                improved_rates=improved,task_MSE=row['task_MSE'],test_WF1={r:v[0]['router']['WF1'] for r,v in evaluated.items()})),flush=True)
        if args.phase=='smoke':
            if h.backbone_digest(model)==before_model: raise RuntimeError('Smoke made no task update')
            if args.mode=='nested' and h.backbone_digest(model.osram.meaningful_block)==before_nested:
                raise RuntimeError('Nested parameters did not update')
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            restore(model,optimizer,state,identity)
            if any(not torch.equal(v.cpu(),state['model'][n]) for n,v in model.state_dict().items()):
                raise RuntimeError('Full model restoration differs')
            from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
            require_same_rng(state['rng'],h.rng_state()); validate_selected(args.output,best)
            _,arrays=scalar.evaluate(bundle,model,'flat','test',masks['0.3'],max_batches=1)
            with np.load(args.output/best['0.3']['predictions'],allow_pickle=False) as saved:
                for key in arrays: np.testing.assert_array_equal(arrays[key],saved[key])
            summary.update(smoke=history[0],smoke_only=True,full_checkpoint_reloaded=True,selected_artifacts_exact_reload=True)
        else:
            summary['evaluation']=publish_selected(args.output,best)
            summary['aggregate']=scalar.aggregate(summary['evaluation'])
        for name,digest in identity['source_hashes'].items():
            if h.sha(ROOT/name)!=digest: raise RuntimeError('Immutable source changed: '+name)
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick); raise
    finally:
        for hook in hooks: hook.remove()
        h.write_json(path,summary)


if __name__=='__main__': main()
