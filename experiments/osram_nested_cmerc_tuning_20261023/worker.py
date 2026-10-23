"""Existing Nested structural screen, matched Flat, original tasks and masks.

INTERNAL TEST-ORACLE SCREEN. IEMOCAP results cover fold5 only. No trained
weights initialize a run; historical Flat artifacts provide configs/masks only.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(f'{i/10:.1f}' for i in range(8))
METHODS = ('none', 'nested_gnn_rooted_evidence', 'nested_dim32', 'nested_dim128',
           'mosei_nested_depth2', 'mosei_nested_mlp128', 'mosei_nested_groups4')
DATASETS = ('CMUMOSEI', 'IEMOCAPFour', 'IEMOCAPSix')
MOSEI = Path('/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1/runs/CMUMOSEI/seed_66/flat')
IEMOCAP = Path('/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016/runs')
DATA_ROOT = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')


def reference_path(dataset):
    if dataset not in DATASETS:
        raise ValueError('Only MOSEI and IEMOCAP4/6 approved')
    return MOSEI if dataset == 'CMUMOSEI' else IEMOCAP / (
        'iemocap4' if dataset == 'IEMOCAPFour' else 'iemocap6') / 'seed_66/fold_5'


def candidate_config(base, method):
    if base.dataset not in DATASETS or method not in METHODS:
        raise ValueError('Outside finite existing-Nested screen')
    scalar = base.dataset == 'CMUMOSEI'
    required = dict(seed=66, fold=1 if scalar else 5, epochs=60 if scalar else 100,
        learning_rate=1e-4 if scalar else 1e-3, weight_decay=1e-5,
        optimizer='adam', lr_schedule='constant', batch_size=32,
        missing_mask_protocol='gcnet-random-mask-v1', train_rate_mode='cyclic',
        training_objective='emotion-only', checkpoint_selection='test-oracle-per-rate',
        checkpoint_selection_metric='weighted_f1' if scalar else 'accuracy', evaluate_test=True,
        osram_meaningful_block='none', osram_output_dim=1600, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_write_step=.6,
        osram_bidirectional=False, osram_readout_fusion='flat', osram_disable_local_skip=False,
        paired_history_views=False, osram_source_role_v3=False, osram_tasr=False,
        osram_source_role=False, osram_source_relation=False, osram_source_tracking=False,
        completion_path='none', initial_backbone_checkpoint=None)
    for key, value in required.items():
        if getattr(base, key) != value:
            raise ValueError('Protected baseline setting differs: ' + key)
    if tuple(base.train_missing_rates) != tuple(float(r) for r in RATES):
        raise ValueError('Original cyclic rates changed')
    return replace(base, osram_meaningful_block=method)


def check_nested_gradients(observed):
    required = ('tokenizer.', 'core.layers.', 'core.pool.', 'core.readout.',
                'local_decoder.', 'memory_decoders.')
    missing = [prefix for prefix in required if not any(
        name.startswith(prefix) and value for name, value in observed.items())]
    if missing:
        raise RuntimeError('No finite nonzero Nested task gradients: ' + str(missing))


def prediction_parity(old, new, *, scalar):
    """Allow float32 roundoff, never call it bitwise equality or allow class flips."""
    import numpy as np
    np.testing.assert_allclose(old, new, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(old > 0, new > 0) if scalar else np.testing.assert_array_equal(
        old.argmax(-1), new.argmax(-1))
    return dict(numerically_equal=True, bitwise_equal=bool(np.array_equal(old, new)),
        max_absolute_error=float(np.max(np.abs(old-new))), atol=1e-6, rtol=1e-6,
        classes_exact=True)


def setup(dataset, method, uuid, output):
    import numpy as np
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_iemocap_b6_transfer_20261015.worker import (
        LABEL_HASHES, require_replication_environment, repair_uuid_rng)
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    environment = require_replication_environment(torch)
    rng_repair = repair_uuid_rng(torch)
    gpu = guard_gpu(uuid)
    ref = reference_path(dataset)
    prior = json.loads((ref / 'SUMMARY.json').read_text())
    if prior['status'] != 'complete':
        raise ValueError('Complete protected Flat reference required')
    raw = json.loads((ref / 'config.json').read_text())
    if dataset == 'CMUMOSEI':
        from experiments.osram_mosei_gcnet_router_20261020.worker import load_manifest
        manifest, proof = load_manifest()
        os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
        roots = manifest['feature_roots']
        data = dict(manifest=proof, files=manifest['files'])
    else:
        label, digest = LABEL_HASHES[dataset]
        if h.sha(DATA_ROOT / 'IEMOCAP' / label) != digest:
            raise ValueError('Accepted IEMOCAP labels changed')
        os.environ['GCNET_DATASET_ROOT'] = str(DATA_ROOT)
        roots = [str(DATA_ROOT / 'IEMOCAP/features' / n) for n in
                 ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        if not all(Path(p).is_dir() for p in roots):
            raise FileNotFoundError('Accepted IEMOCAP features missing')
        data = dict(label_sha256=digest, feature_roots=roots)
    os.environ.pop('GCNET_MELD_PICKLE', None)
    os.environ['GCNET_CACHE_ROOT'] = str(output.parent / 'feature_cache' / dataset)
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    from experiments.osram_iemocap6_gcnet_testbest_20261020.worker import full_training_split
    from experiments.osram_mosei_nested_gcnet_3seed_20261021.worker import original_digest
    base = trainer.TrainConfig(**raw)
    # Retention-only defaults are logged explicitly, never a hidden loss/mask change.
    base = replace(base, device='cuda:0', preserve_selection_artifacts=True)
    cfg = candidate_config(base, method)
    train, validation, test, *dimensions = trainer.get_loaders(*roots, dataset=dataset,
        num_folder=1 if dataset == 'CMUMOSEI' else 5, batch_size=32, num_workers=0, seed=66,
        validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    fold = cfg.fold - 1
    bundle = dict(trainer=trainer, cfg=cfg, dimensions=tuple(dimensions), train_loader=train[fold],
        validation_loader=validation[fold], test_loader=test[fold], environment=environment, gpu=gpu,
        case=dict(dataset=dataset, classes=1 if dataset == 'CMUMOSEI' else (4 if dataset == 'IEMOCAPFour' else 6)))
    bundle['split'] = ({n: loader.protocol_metadata for n, loader in
        (('train', train[fold]), ('validation', validation[fold]), ('test', test[fold]))}
        if dataset == 'CMUMOSEI' else full_training_split(train[fold], test[fold], cfg.fold))
    bundle['seed_protocol'] = seed_training_loader(bundle, 66)
    baseline = _build_model(base, tuple(dimensions), torch.device('cuda:0'))
    initial = h.backbone_digest(baseline)
    model = baseline if method == 'none' else _build_model(cfg, tuple(dimensions), torch.device('cuda:0'))
    if original_digest(model) != initial:
        raise RuntimeError('Nested changed original Flat initialization')
    model.requires_grad_(True)
    bundle.update(model=model, data=data, initial_backbone_sha256=initial, rng_repair=rng_repair,
        reference=dict(path=str(ref), summary_sha256=h.sha(ref / 'SUMMARY.json'),
            config_sha256=h.sha(ref / 'config.json'), use='config/data/masks only; not paired scores',
            current_source_matched_flat_required=True), config_reference_delta={k: v for k, v in asdict(cfg).items()
                if k not in raw or json.loads(json.dumps(v)) != raw[k]})
    if dataset == 'CMUMOSEI':
        masks = torch.load(ref / 'fixed_masks.pt', map_location='cpu', weights_only=False)
        masks = masks.get('test', masks)
        hashes = {r: hashlib.sha256(b''.join(x['availability'].numpy().tobytes() for x in b)).hexdigest()
                  for r, b in masks.items()}
        if hashes != prior['identity']['mask_hashes']['test']:
            raise ValueError('MOSEI exact masks changed')
        selected, mask_refs = masks, dict(kind='accepted-cmerc-MOSEI-fixed-test-bank', hashes=hashes)
    else:
        selected, mask_refs = {}, {}
        for rate in RATES:
            row = prior['testbest']['rates'][rate]
            path = ref / 'selection_versions' / f"epoch_{row['epoch']:06d}_{rate.replace('.', 'p')}.npz"
            if h.sha(path) != row['selection_sha256']:
                raise ValueError('Accepted IEMOCAP selected artifact changed')
            with np.load(path, allow_pickle=False) as stored:
                selected[rate] = {k: stored[k].copy() for k in
                    ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability')}
            if hashlib.sha256(selected[rate]['availability'].tobytes()).hexdigest() != row['mask_sha256']:
                raise ValueError('Accepted IEMOCAP mask changed')
            mask_refs[rate] = dict(path=str(path), sha256=h.sha(path), mask_sha256=row['mask_sha256'])
    bundle['masks'] = selected
    bundle['mask_reference'] = mask_refs
    return bundle, baseline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--seed', type=int, choices=(66,), default=66)
    p.add_argument('--phase', choices=('train', 'smoke'), required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.fold != (1 if args.dataset == 'CMUMOSEI' else 5):
        raise ValueError('Only approved screening fold permitted')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import numpy as np
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_20261019 import run as classification
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_iemocap6_gcnet_testbest_20261020 import worker as class_selected
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as scalar_selected
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh output required; preserve all prior runs')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick, hooks = args.output / 'SUMMARY.json', time.monotonic(), []
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError('Do not repeat completed run')
    summary = dict(status='preflight', dataset=args.dataset, method=args.method, seed=66, fold=args.fold,
        phase=args.phase, source_commit=args.source_commit, pid=os.getpid(), label='INTERNAL TEST-ORACLE SCREEN')
    h.write_json(path, summary)
    try:
        torch.set_num_threads(1)
        bundle, baseline = setup(args.dataset, args.method, args.gpu_uuid, args.output)
        model, cfg, masks = bundle['model'], bundle['cfg'], bundle['masks']
        is_scalar = args.dataset == 'CMUMOSEI'
        task, selection = (scalar, scalar_selected) if is_scalar else (classification, class_selected)
        def evaluate(current, rate, max_batches=None):
            if is_scalar:
                return scalar.evaluate(bundle, current, 'flat', 'test', masks[rate], max_batches=max_batches)
            return classification.evaluate(bundle, current, 'flat', bundle['test_loader'], saved=masks[rate], max_batches=max_batches)
        parity = None
        if args.method != 'none':
            _, old = evaluate(baseline, '0.3', 1)
            _, new = evaluate(model, '0.3', 1)
            for name in ('labels', 'availability'):
                np.testing.assert_array_equal(old[name], new[name], err_msg='Nested zero-init equivalence: ' + name)
            key = 'prediction' if is_scalar else 'log_probs_router'
            parity = prediction_parity(old[key], new[key], scalar=is_scalar)
            if model.osram.meaningful_block.last_diagnostics['input_change_norm'] != 0.:
                raise RuntimeError('Zero decoder must produce exactly zero input correction')
            del baseline, old, new
        optimizer = task.make_optimizer(model, cfg, 'flat')
        if {id(p) for p in model.parameters()} != {id(p) for g in optimizer.param_groups for p in g['params']}:
            raise RuntimeError('One-stage optimizer must include every parameter')
        source_hashes = {str(path.relative_to(ROOT)): h.sha(path)
            for package in ('gcnet_missing_m3', 'gcnet_modality_jepa') for path in sorted((ROOT / package).glob('*.py'))}
        for name in ('worker.py', 'run.py', 'report.py'):
            rel = 'experiments/osram_nested_cmerc_tuning_20261023/' + name
            source_hashes[rel] = h.sha(ROOT / rel)
        identity = dict(source_commit=args.source_commit, source_hashes=source_hashes, config=asdict(cfg),
            dataset=args.dataset, method=args.method, fold=args.fold, seed=66, split=bundle['split'],
            seed_protocol=bundle['seed_protocol'], environment=bundle['environment'], rng_repair=bundle['rng_repair'],
            data=bundle['data'], reference=bundle['reference'], mask_reference=bundle['mask_reference'],
            initialization='fresh canonical original factory; no trained weights',
            selection='per-rate TEST W-F1' if is_scalar else 'per-rate TEST ACC; same-call UA/W-F1',
            protocol_note='fixed accepted mask replay; historical IEMOCAP resampled per-epoch masks; compare new matched Flat')
        torch.save(masks, args.output / 'test_masks.pt')
        h.write_json(args.output / 'config.json', asdict(cfg))
        summary.update(status='running', config=asdict(cfg), identity=identity,
            environment=bundle['environment'], gpu=bundle['gpu'], config_reference_delta=bundle['config_reference_delta'],
            initial_backbone_sha256=bundle['initial_backbone_sha256'], zero_init_parity=parity,
            zero_init_predictions_exact=parity['bitwise_equal'] if parity else None,
            total_parameters=sum(p.numel() for p in model.parameters()),
            nested_parameters=sum(p.numel() for p in model.osram.meaningful_block.parameters()) if args.method != 'none' else 0)
        h.write_json(path, summary)
        observed, cumulative, scans = {}, {}, [0]
        if args.method != 'none':
            def capture(name):
                def hook(grad):
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('Nonfinite Nested gradient: ' + name)
                    observed[name] = observed.get(name, False) or bool(grad.ne(0).any())
                return hook
            for name, parameter in model.osram.meaningful_block.core.named_parameters():
                hooks.append(parameter.register_hook(capture(name)))
        hooks.append(model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0, scans[0]+1)))
        reset_training_stochasticity(66, args.fold)
        if len(h.rng_state()['cuda']) != 1:
            raise RuntimeError('UUID-bound CUDA RNG must actually be captured')
        torch.cuda.reset_peak_memory_stats()
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            if len(state['rng']['cuda']) != 1:
                raise ValueError('Missing CUDA RNG: refuse silent nonexact resume')
            classification.restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            selection.validate_selected(args.output, best)
        for epoch in (range(1) if args.phase == 'smoke' else range(start, cfg.epochs)):
            observed.clear(); scans[0] = 0
            row = task.train_epoch(bundle, model, optimizer, epoch, 'flat', max_batches=3 if args.phase == 'smoke' else None)
            if scans[0] != row['optimizer_steps']:
                raise RuntimeError('Expected one causal memory forward per task update')
            for name, value in observed.items():
                cumulative[name] = cumulative.get(name, False) or value
            if args.method != 'none' and epoch == 0:
                check_nested_gradients(cumulative)
            row.update(learning_rates=[g['lr'] for g in optimizer.param_groups],
                nested_gradient_paths=dict(observed), actual_memory_scans=scans[0])
            evaluated = {r: evaluate(model, r, 1 if args.phase == 'smoke' else None) for r in RATES}
            row['test'] = {r: item[0] for r, item in evaluated.items()}
            history.append(row)
            improved = selection.persist_epoch(args.output, model, optimizer, epoch=epoch+1,
                identity=identity, history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch+1, best_by_rate=best, selected_epoch_by_rate={r: b['epoch'] for r, b in best.items()},
                seconds=time.monotonic()-tick, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            h.write_json(path, summary)
            print(json.dumps(dict(event='epoch', dataset=args.dataset, method=args.method, epoch=epoch+1,
                task_loss=row['task_MSE'] if is_scalar else row['task_NLL'], improved_rates=improved)), flush=True)
        selection.validate_selected(args.output, best)
        if args.phase == 'smoke':
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            if len(state['rng']['cuda']) != 1:
                raise RuntimeError('Saved recovery lacks CUDA state')
            classification.restore(model, optimizer, state, identity)
            _, replay = evaluate(model, '0.3', 1)
            with np.load(args.output / best['0.3']['predictions'], allow_pickle=False) as stored:
                for name, value in replay.items():
                    np.testing.assert_array_equal(value, stored[name], err_msg='Reload replay differs: '+name)
            summary.update(smoke_only=True, smoke=history[0], selected_checkpoint_reload_exact=True,
                cuda_rng_states_saved=1)
        else:
            summary['evaluation'] = selection.publish_selected(args.output, best)
            metrics = {r: v['router'] for r, v in summary['evaluation'].items()}
            summary['aggregate'] = (scalar.aggregate(summary['evaluation']) if is_scalar else
                dict(router=h.aggregate_rates(metrics)))
        for name, digest in source_hashes.items():
            if h.sha(ROOT / name) != digest:
                raise RuntimeError('Immutable source changed: ' + name)
        summary.update(status='complete', seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        for hook in hooks:
            hook.remove()
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
