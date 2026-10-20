"""Finite one-stage direct source weighting; dataset protocols remain separate."""
from dataclasses import asdict, replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(f'{i/10:.1f}' for i in range(8))


def settings(dataset, seed, fold):
    if seed not in (66, 67, 68) or dataset not in ('CMUMOSEI', 'MELD', 'IEMOCAPSix'):
        raise ValueError('Only the three approved datasets and seeds66/67/68')
    if fold not in (range(1, 6) if dataset == 'IEMOCAPSix' else (1,)):
        raise ValueError('IEMOCAP requires LOSO folds1..5; other datasets fold1')
    epochs, lr = {'CMUMOSEI': (60, .001), 'MELD': (50, 1e-5), 'IEMOCAPSix': (100, .001)}[dataset]
    return dict(epochs=epochs, learning_rate=lr, weight_decay=1e-5, batch_size=32,
        selection_metric='weighted_f1' if dataset == 'CMUMOSEI' else 'accuracy',
        test_masks='original-per-epoch-resampled' if dataset == 'MELD' else 'original-fixed-GCNet-bank')


def direct_config(cfg):
    expected = settings(cfg.dataset, cfg.seed, cfg.fold)
    required = dict(epochs=expected['epochs'], learning_rate=expected['learning_rate'],
        weight_decay=1e-5, batch_size=32, lr_schedule='constant', backbone_type='osram',
        osram_readout_fusion='flat', osram_bidirectional=False, osram_output_dim=1600,
        osram_write_step=.6, osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
        osram_tasr=False, osram_source_role=False, osram_meaningful_block='none',
        osram_disable_local_skip=False, paired_history_views=False, completion_path='none',
        training_objective='emotion-only', missing_mask_protocol='gcnet-random-mask-v1',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric=expected['selection_metric'])
    for key, value in required.items():
        if getattr(cfg, key) != value:
            raise ValueError('Accepted configuration mismatch: ' + key)
    return replace(cfg, osram_source_role=True, osram_source_role_mode='direct')


def setup(dataset, seed, fold, uuid, output):
    """Read original data/config; build fresh weights, never a trained initialization."""
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_20261019 import run as joint
    if dataset == 'CMUMOSEI':
        from experiments.osram_mosei_gcnet_router_20261020 import worker as old
        bundle = old.setup(uuid, output, 66)
        cfg = replace(bundle['cfg'], seed=seed)
        bundle['cfg'] = cfg
        bundle['seed_protocol'] = joint.seed_training_loader(bundle, seed)
        del bundle['model']
        baseline = old.fresh_backbone(cfg, bundle['dimensions'], torch.device('cuda:0'))
        builder = old.fresh_backbone
    elif dataset == 'IEMOCAPSix':
        from experiments.osram_iemocap6_gcnet_testbest_20261020 import worker as old
        bundle, prepare = old.setup_bundle(fold, seed, uuid, output)
        baseline = prepare(bundle, 'flat', seed=seed)
        cfg = old.make_config(fold, seed)
        bundle['cfg'] = cfg
        from experiments.osram_mosei_gcnet_router_20261020.worker import fresh_backbone
        builder = fresh_backbone
    else:
        from experiments.osram_tasr_20261018.protocol import build_cases
        from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment, repair_uuid_rng
        from experiments.osram_meld_write_retention_20261014.run import guard_gpu
        from experiments.osram_tasr_joint_20261019.run import fresh_model
        case = build_cases()['MELD']
        feature = Path('/data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl')
        if h.sha(feature) != case['dataset_sha256']:
            raise ValueError('Accepted MELD ECERC feature SHA256 changed')
        os.environ.update(GCNET_MELD_PICKLE=str(feature),
            GCNET_DATASET_ROOT=str(Path(case['numerical_source']['source'])/'dataset'),
            GCNET_CACHE_ROOT=str(output.parent/'feature_cache'/'MELD'))
        from gcnet_missing_m3 import train_gcnet as trainer
        repair_uuid_rng(torch)
        environment = require_replication_environment(torch)
        cfg = replace(trainer.TrainConfig(**case['expected_config']), seed=seed,
            checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='accuracy', evaluate_test=True)
        train, validation, test, *dimensions = trainer.get_loaders(*('MELD-pickle',)*3,
            dataset='MELD', num_folder=1, batch_size=32, num_workers=0, seed=66,
            validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
        groups = [set(loader[0].sampler.indices) for loader in (train, validation, test)]
        if any(groups[a] & groups[b] for a, b in ((0,1),(0,2),(1,2))):
            raise ValueError('Official MELD splits overlap')
        bundle = dict(trainer=trainer, cfg=cfg, dimensions=tuple(dimensions),
            train_loader=train[0], validation_loader=validation[0], test_loader=test[0],
            environment=environment, gpu=guard_gpu(uuid), case=case,
            data_manifest=dict(feature=str(feature), sha256=case['dataset_sha256']),
            split=dict(method='official-train-dev-test', train_indices=sorted(groups[0]),
                validation_indices=sorted(groups[1]), test_indices=sorted(groups[2])))
        bundle['seed_protocol'] = joint.seed_training_loader(bundle, seed)
        # Preserve the original run_experiment factory's hardcoded fold5 seed.
        builder = fresh_model
        baseline = builder(cfg, bundle['dimensions'], torch.device('cuda:0'))
    torch.set_num_threads(1)
    baseline.requires_grad_(True).eval()
    new_cfg = direct_config(cfg)
    model = builder(new_cfg, bundle['dimensions'], torch.device('cuda:0')).requires_grad_(True)
    from experiments.osram_coupled_source_role_20261020.worker import original_digest
    initial = h.backbone_digest(baseline)
    if original_digest(model) != initial:
        raise RuntimeError('New module changed original parameter initialization')
    bundle.update(cfg=new_cfg, model=model, initial_backbone_sha256=initial)
    return bundle, baseline


def evaluate_meld(bundle, *, limited=False):
    """Original resampled evaluation, preserving its RNG consumption and order."""
    import itertools
    from experiments.osram_persistent_patterns_20261018.evaluate import classification_metrics
    trainer, loader = bundle['trainer'], bundle['test_loader']
    schedules = trainer._schedules(bundle['cfg'], 'test')
    results = {}
    for rate in RATES:
        batches = itertools.islice(iter(loader), 1) if limited else loader
        # evaluate_rate is no_grad-decorated in the original trainer.
        _, arrays = trainer.evaluate_rate(bundle['model'], batches, schedules[float(rate)],
            'MELD', bundle['dimensions'], next(bundle['model'].parameters()).device,
            collect=True, collect_identity=True)
        scores = classification_metrics(arrays['labels'], arrays['logits'], 7)
        if not np.array_equal(arrays['predictions'], arrays['logits'].argmax(-1)):
            raise ValueError('Same-call MELD labels/logits/predictions differ')
        shifted = arrays['logits'] - arrays['logits'].max(-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
        arrays.update(log_probs_router=log_probs)
        results[rate] = (dict(router=scores,
            mask_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest()), arrays)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('CMUMOSEI','MELD','IEMOCAPSix'), required=True)
    parser.add_argument('--seed', type=int, choices=(66,67,68), required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--phase', choices=('smoke','train'), required=True)
    parser.add_argument('--mode', choices=('joint',), required=True, help='Queue label; no probability Router')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    spec = settings(args.dataset, args.seed, args.fold)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_20261019 import run as joint
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_coupled_source_role_20261020.worker import serial, original_digest
    from experiments.osram_iemocap6_gcnet_testbest_20261020 import worker as classification
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as regression
    from experiments.osram_iemocap6_router_fivefold_20261019.worker import build_mask_bank
    tick = time.monotonic()
    if args.output.exists() and not args.resume: raise FileExistsError('Fresh output required')
    if args.resume and args.phase != 'train': raise ValueError('Only complete-state train resume')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'SUMMARY.json'
    summary = dict(status='preflight', dataset=args.dataset, seed=args.seed, fold=args.fold,
        mode=args.mode, phase=args.phase, source_commit=args.source_commit, pid=os.getpid(),
        experiment_label='INTERNAL TEST-ORACLE SCREEN', block='coupled-source-role-direct', router_parameters=0)
    hooks = []
    try:
        bundle, baseline = setup(args.dataset, args.seed, args.fold, args.gpu_uuid, args.output)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        model, cfg = bundle['model'], bundle['cfg']
        # A short exact-input prediction check before discarding fresh control.
        raw = next(iter(bundle['test_loader']))
        data = bundle['trainer']._move_batch(raw, torch.device('cuda:0'))
        with h.isolated_rng(9971), torch.no_grad():
            view = bundle['trainer']._prepare_view(data, bundle['trainer']._schedules(cfg,'test')[.3], 0, bundle['dimensions'])
            baseline.eval(); model.eval()
            old = h.forward(baseline, view); new = h.forward(model, view)
            torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-5)
            zero_error = float((old-new).abs().max())
        del baseline, old, new, raw, data, view
        masks = (scalar.build_masks(bundle, 'test') if args.dataset == 'CMUMOSEI'
                 else build_mask_bank(bundle, 'test') if args.dataset == 'IEMOCAPSix' else None)
        if masks is not None: torch.save(masks, args.output/'test_masks.pt')
        sources = ('gcnet_missing_m3/osram.py','gcnet_missing_m3/source_role.py',
            'gcnet_missing_m3/tasr.py','gcnet_missing_m3/model.py','gcnet_missing_m3/train_gcnet.py',
            'gcnet_missing_m3/mixed_rate.py','gcnet_modality_jepa/gcnet_legacy_random_mask.py',
            'gcnet_modality_jepa/protocol.py','gcnet_modality_jepa/train_gcnet.py',
            'experiments/osram_text_path_router_joint_20261019/run.py',
            'experiments/osram_text_path_router_joint_mosi_20261019/run.py',
            'experiments/osram_iemocap6_gcnet_testbest_20261020/worker.py',
            'experiments/osram_mosei_gcnet_router_20261020/worker.py',
            'experiments/osram_tasr_joint_20261019/run.py',
            'experiments/osram_meld_flat_source_combinations_20261013/evaluate.py',
            'experiments/osram_tasr_20261018/runner_helpers.py',
            'experiments/osram_source_role_direct_cross_20261020/worker.py')
        identity = dict(source_commit=args.source_commit, source_hashes={p:h.sha(ROOT/p) for p in sources},
            dataset=args.dataset,seed=args.seed,fold=args.fold,config=asdict(cfg),split=bundle['split'],
            seed_protocol=bundle['seed_protocol'],test_mask_policy=spec['test_masks'],
            mask_hashes={} if masks is None else {r:hashlib.sha256(b''.join(x['availability'].numpy().tobytes() for x in bank)).hexdigest() for r,bank in masks.items()},
            data_manifest=bundle.get('data_manifest',bundle.get('case')),one_stage=True,
            initialization='fresh-original-factory; no trained weights loaded',
            loss='original MSE' if args.dataset=='CMUMOSEI' else 'original CE',
            selection='per-rate TEST '+spec['selection_metric']+'; strict greater; earliest ties')
        summary.update(status='running', identity=identity, config=identity['config'],
            environment=bundle['environment'],gpu=bundle['gpu'],initial_backbone_sha256=bundle['initial_backbone_sha256'],
            zero_init_prediction_max_abs_error=zero_error,zero_init_predictions_within_tolerance=True,
            total_parameters=sum(p.numel() for p in model.parameters()),
            source_role_parameters=sum(p.numel() for p in model.osram.source_role.parameters()))
        h.write_json(args.output/'config.json',identity['config']); h.write_json(path,summary)
        if args.dataset == 'MELD':
            from experiments.osram_tasr_joint_20261019.run import make_optimizer
            optimizer, _ = make_optimizer(model,cfg)
        else: optimizer = joint.make_optimizer(model,cfg,'flat')
        observed = {}
        def capture(name):
            def hook(gradient):
                if not torch.isfinite(gradient).all(): raise RuntimeError('Nonfinite source-role gradient')
                observed[name]=observed.get(name,False) or bool(gradient.ne(0).any())
            return hook
        for name, p in model.osram.source_role.named_parameters(): hooks.append(p.register_hook(capture(name)))
        scans = [0]
        hooks.append(model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0,scans[0]+1)))
        selection = regression if args.dataset == 'CMUMOSEI' else classification
        reset_training_stochasticity(args.seed,args.fold)
        best,history,start = {},[],0
        if args.resume:
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            joint.restore(model,optimizer,state,identity)
            start,history,best=state['epoch'],state['history'],state['best_by_rate']
            selection.validate_selected(args.output,best)
        torch.cuda.reset_peak_memory_stats()
        initial_added=h.backbone_digest(model.osram.source_role)
        for epoch in (range(1) if args.phase=='smoke' else range(start,cfg.epochs)):
            observed.clear(); scans[0]=0
            if args.dataset=='MELD':
                from experiments.osram_tasr_joint_20261019.run import run_epoch
                row=run_epoch(bundle,optimizer,epoch,smoke=args.phase=='smoke')
            else:
                runner=scalar if args.dataset=='CMUMOSEI' else joint
                row=runner.train_epoch(bundle,model,optimizer,epoch,'flat',max_batches=3 if args.phase=='smoke' else None)
            if scans[0]!=row['optimizer_steps']: raise RuntimeError('Expected one Memory scan per training batch')
            if not any(observed.values()): raise RuntimeError('No nonzero source-role task gradients')
            row.update(actual_memory_scans=scans[0],learning_rates=[g['lr'] for g in optimizer.param_groups],
                source_role_gradients=dict(connected=len(observed),nonzero=sum(observed.values())),
                last_training_batch_source_role=serial(model.osram.source_role.last_diagnostics),
                last_training_batch_decomposition=serial(model.osram.last_source_trace['diagnostics']))
            if args.dataset=='MELD': evaluated=evaluate_meld(bundle,limited=args.phase=='smoke')
            elif args.dataset=='CMUMOSEI':
                evaluated={r:scalar.evaluate(bundle,model,'flat','test',masks[r],max_batches=1 if args.phase=='smoke' else None) for r in RATES}
            else:
                evaluated={r:joint.evaluate(bundle,model,'flat',bundle['test_loader'],batches=masks[r],max_batches=1 if args.phase=='smoke' else None) for r in RATES}
            row['test']={r:value[0] for r,value in evaluated.items()};history.append(row)
            improved=selection.persist_epoch(args.output,model,optimizer,epoch=epoch+1,
                identity=identity,history=history,best=best,evaluated=evaluated)
            summary.update(epoch=epoch+1,best_by_rate=best,
                selected_epoch_by_rate={r:v['epoch'] for r,v in best.items()},
                peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,seconds=time.monotonic()-tick)
            h.write_json(path,summary)
            print(json.dumps(dict(event='epoch',epoch=epoch+1,dataset=args.dataset,seed=args.seed,fold=args.fold,
                improved_rates=improved,test={r:v[0]['router'] for r,v in evaluated.items()},
                source_role_gradients=row['source_role_gradients'])),flush=True)
        if args.phase=='smoke':
            if h.backbone_digest(model.osram.source_role)==initial_added: raise RuntimeError('New module did not update')
            if original_digest(model)==bundle['initial_backbone_sha256']: raise RuntimeError('Original model did not update')
            state=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            joint.restore(model,optimizer,state,identity)
            if any(not torch.equal(v.cpu(),state['model'][n]) for n,v in model.state_dict().items()): raise RuntimeError('Reload differs')
            selection.validate_selected(args.output,best)
            summary.update(smoke_only=True,full_checkpoint_reloaded=True,smoke=history[0])
        else:
            summary['evaluation']=selection.publish_selected(args.output,best)
            keys=('ACC','WF1','MSE') if args.dataset=='CMUMOSEI' else ('ACC','UA','WF1','CE')
            summary['aggregate']={'router':{scope:{k:sum(summary['evaluation'][r]['router'][k] for r in rates)/len(rates) for k in keys}
                for scope,rates in (('mean',RATES),('high',RATES[5:]))}}
        for name,digest in identity['source_hashes'].items():
            if h.sha(ROOT/name)!=digest: raise RuntimeError('Source changed during run: '+name)
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        for hook in hooks: hook.remove()
        h.write_json(path,summary)


if __name__=='__main__': main()
