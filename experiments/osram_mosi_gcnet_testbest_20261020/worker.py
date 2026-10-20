"""MOSI150 selection-only rerun: per-rate TEST-best WF1, original GCNet masks.

INTERNAL TEST-ORACLE SCREEN. One training trajectory produces eight selected
checkpoints; their mean is not single-checkpoint deployment performance.
Prior validation-selected runs remain read-only training/parity references.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(str(i / 10) for i in range(8))
PROTOCOL = 'gcnet-random-mask-v1'
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_gcnet_router_3seed_20261019/execution_v1/runs/MOSI')
MSE_RTOL = 1e-5
MSE_ATOL = 1e-6
SMOKE_BATCHES = 3  # Batch zero has rate0/no eligible Router rows; include .1/.2.


def testbest_config(cfg):
    """Only restore the requested selection rule; no sampler/task/model change."""
    if cfg.missing_mask_protocol != PROTOCOL or cfg.checkpoint_selection != 'validation':
        raise ValueError('Expected the completed GCNet validation-run configuration')
    return replace(cfg, checkpoint_selection='test-oracle-per-rate',
                   checkpoint_selection_metric='weighted_f1', evaluate_test=True)


def check_training_parity(actual, reference):
    """Exact masks/update counts/LRs; explicit tolerance for CUDA task rounding."""
    for key in ('epoch', 'mask_trace_sha256', 'optimizer_steps', 'learning_rates'):
        if actual[key] != reference[key]:
            raise ValueError('Selection-only training parity differs: ' + key)
    error = abs(actual['task_MSE'] - reference['task_MSE'])
    close = math.isclose(actual['task_MSE'], reference['task_MSE'], rel_tol=MSE_RTOL, abs_tol=MSE_ATOL)
    if not close:
        raise ValueError('Selection-only training task MSE differs beyond explicit tolerance')
    return dict(epoch=actual['epoch'], mask_trace_exact=True, optimizer_steps_exact=True,
                learning_rates_exact=True, task_MSE_close=close, task_MSE_abs_error=error,
                task_MSE_rtol=MSE_RTOL, task_MSE_atol=MSE_ATOL)


def selection_reference(path, identity, initial_digest):
    """Read old metadata/history, never old model weights, for selection-only proof."""
    from experiments.osram_tasr_20261018 import runner_helpers as h
    path = Path(path)
    old = json.loads((path / 'SUMMARY.json').read_text())
    if old.get('status') != 'complete' or old.get('epoch') != 150:
        raise ValueError('Completed 150-epoch validation reference required')
    if old['initial_backbone_sha256'] != initial_digest:
        raise ValueError('Selection-only initialization differs')
    old_identity = old['identity']
    expected = dict(old_identity['config'], checkpoint_selection='test-oracle-per-rate')
    if expected != identity['config']:
        raise ValueError('Selection-only config changed beyond checkpoint selection')
    for key in ('split', 'mask_hashes', 'seed_protocol'):
        if old_identity[key] != identity[key]:
            raise ValueError('Selection-only reference differs: ' + key)
    for filename, digest in old_identity['source_hashes'].items():
        if identity['source_hashes'].get(filename) != digest:
            raise ValueError('Previously validated numerical/setup source changed: ' + filename)
    history = json.loads((path / 'history.json').read_text())
    if len(history) != 150:
        raise ValueError('Complete reference training history required')
    return history, dict(path=str(path), summary_sha256=h.sha(path / 'SUMMARY.json'),
        history_sha256=h.sha(path / 'history.json'), initialization_exact=True,
        only_checkpoint_selection_changed=True, core_sources_exact=True,
        training_and_evaluation_seed_protocol_exact=True, fixed_evaluation_masks_exact=True,
        old_weights_loaded=False, task_MSE_rtol=MSE_RTOL, task_MSE_atol=MSE_ATOL)


def _atomic_arrays(path, arrays):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def persist_epoch(output, model, optimizer, *, epoch, identity, history, best, evaluated):
    """Persist same-call arrays/metrics and full state for strictly improved rates.

    Ties retain the earliest epoch. Nothing uses the eight-rate mean or MSE as a
    tie-break. Last is committed after selected artifacts; recovery validates
    them rather than silently accepting mismatched/torn artifacts.
    """
    from experiments.osram_text_path_router_joint_20261019.run import checkpoint_payload
    from experiments.osram_tasr_20261018 import runner_helpers as h
    output = Path(output)
    if set(evaluated) != set(RATES):
        raise ValueError('Require all eight same-epoch TEST evaluations')
    # Validate the whole epoch before any selection artifacts are changed.
    for metrics, arrays in evaluated.values():
        if not all(math.isfinite(metrics['router'][key]) for key in ('ACC', 'WF1', 'MSE')):
            raise ValueError('Selection metrics must be finite')
        if not np.isfinite(arrays['prediction']).all():
            raise ValueError('Selection predictions must be finite')
        actual_mask = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
        if metrics['mask_sha256'] != actual_mask:
            raise ValueError('Same-call prediction mask hash differs')
    improved = [rate for rate, (metrics, _) in evaluated.items()
                if rate not in best or metrics['router']['WF1'] > best[rate]['metrics']['router']['WF1']]
    for rate in improved:
        metrics, arrays = evaluated[rate]
        suffix = rate.replace('.', 'p')
        prediction = 'predictions_miss_' + suffix + '.npz'
        _atomic_arrays(output / prediction, arrays)
        best[rate] = dict(epoch=epoch, checkpoint='best_miss_' + suffix + '.pt', predictions=prediction,
                          prediction_sha256=h.sha(output / prediction), metrics=copy.deepcopy(metrics),
                          criterion='TEST weighted_f1; strict improvement; earliest tie',
                          same_evaluation_call=True)
    state = checkpoint_payload(model, optimizer, epoch=epoch, identity=identity,
                               history=history, best_key={r: row['metrics']['router']['WF1'] for r, row in best.items()})
    state['config'] = copy.deepcopy(identity.get('config'))
    for rate in improved:
        state['selection'] = copy.deepcopy(best[rate])
        state['selection']['rate'] = rate
        h.atomic_torch_save(output / best[rate]['checkpoint'], state)
        best[rate]['checkpoint_sha256'] = h.sha(output / best[rate]['checkpoint'])
    state.pop('selection', None)
    state['best_by_rate'] = copy.deepcopy(best)
    h.atomic_torch_save(output / 'last.pt', state)
    h.write_json(output / 'history.json', history)
    h.write_json(output / 'best_by_rate.json', best)
    return improved


def validate_selected(output, best):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if set(best) != set(RATES):
        raise ValueError('Complete eight-rate selected checkpoint ledger required')
    for rate, row in best.items():
        for field, digest in (('predictions', 'prediction_sha256'), ('checkpoint', 'checkpoint_sha256')):
            path = Path(output) / row[field]
            if h.sha(path) != row[digest]:
                raise ValueError('Selected ' + field + ' integrity failure at rate ' + rate)


def publish_selected(output, best):
    """Publish exact already-selected bytes; never re-infer a random test mask."""
    from experiments.osram_tasr_20261018 import runner_helpers as h
    output = Path(output)
    validate_selected(output, best)
    (output / 'evaluation').mkdir(exist_ok=True)
    evaluation = {}
    for rate, row in best.items():
        target = output / 'evaluation' / ('random_' + rate.replace('.', 'p') + '.npz')
        shutil.copyfile(output / row['predictions'], target)
        metrics = copy.deepcopy(row['metrics'])
        metrics['output_sha256'] = h.sha(target)
        metrics['selected_epoch'] = row['epoch']
        evaluation[rate] = metrics
    return evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('smoke', 'train'))
    parser.add_argument('--mode', required=True, choices=('flat', 'joint'))
    parser.add_argument('--seed', required=True, type=int, choices=(66, 67, 68))
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
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
        raise FileExistsError('Fresh independent output directory required')
    if args.resume and args.phase != 'train':
        raise ValueError('Only complete training runs support exact resume')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary_path, tick = args.output / 'SUMMARY.json', time.monotonic()
    if args.resume and summary_path.exists() and json.loads(summary_path.read_text()).get('status') == 'complete':
        raise ValueError('Completed run cannot be repeated silently')
    summary = dict(status='preflight', dataset='CMUMOSI', mode=args.mode, phase=args.phase,
                   seed=args.seed, source_commit=args.source_commit, pid=os.getpid(),
                   experiment_label='INTERNAL TEST-ORACLE SCREEN')
    try:
        torch.set_num_threads(1)
        bundle = original.setup(args.gpu_uuid, args.output, args.seed)
        # Trainer import follows setup's dataset environment initialization.
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        bundle['cfg'] = testbest_config(bundle['cfg'])
        model = scalar.wrap_model(bundle['model'], args.mode)
        optimizer = scalar.make_optimizer(model, bundle['cfg'], args.mode)
        masks = {split: original.build_masks(bundle, split) for split in ('validation', 'test')}
        hashes = {split: {rate: hashlib.sha256(b''.join(row['availability'].numpy().tobytes()
                    for row in bank)).hexdigest() for rate, bank in rates.items()} for split, rates in masks.items()}
        reference = REFERENCE / f'seed_{args.seed}' / args.mode
        # Include every original source dependency, not just seven numerical cores.
        prior_summary = json.loads((reference / 'SUMMARY.json').read_text())
        source_files = tuple(prior_summary['identity']['source_hashes']) + (str(Path(__file__).relative_to(ROOT)),)
        identity = dict(source_commit=args.source_commit, source_hashes={p: h.sha(ROOT / p) for p in source_files},
            mode=args.mode, seed=args.seed, config=asdict(bundle['cfg']), split=bundle['split'],
            mask_hashes=hashes, seed_protocol=bundle['seed_protocol'], historical_bridge=bundle['bridge'],
            dataset_manifest_sha256=bundle['data_manifest_sha256'], data_manifest=bundle['data_manifest'],
            data_files=bundle['data_files'],
            initialization='fresh-original-hardcoded-missing_m3_model_init:fold:5',
            selection='test-oracle-per-rate; router weighted_f1; strict greater; earliest ties',
            backbone_learning_rates=dict(first_100=1e-3, final_50=1e-4),
            router_learning_rates=[3e-4, 3e-5] if args.mode == 'joint' else None,
            missing_protocol=PROTOCOL, historical_80p747_is_not_paired_score=True)
        reference_history, proof = selection_reference(reference, identity, bundle['initial_backbone_sha256'])
        identity['selection_only_reference'] = proof
        torch.save(masks, args.output / 'fixed_masks.pt')
        summary.update(identity=identity, config=identity['config'], environment=bundle['environment'],
            gpu=bundle['gpu'], initial_backbone_sha256=bundle['initial_backbone_sha256'],
            protected_weights_excluded_from_initialization=True, anchor_parity_gate=bundle['bridge'],
            mask_hashes=hashes, total_parameters=sum(p.numel() for p in model.parameters()),
            router_parameters=sum(p.numel() for p in model.router.parameters()) if args.mode == 'joint' else 0,
            status='running', selection_only_reference=proof, historical_80p747_is_not_paired_score=True)
        h.write_json(args.output / 'config.json', identity['config'])
        h.write_json(summary_path, summary)
        reset_training_stochasticity(args.seed, bundle['cfg'].fold)
        torch.cuda.reset_peak_memory_stats()
        best, history, parity, start = {}, [], [], 0
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            validate_selected(args.output, best)
            parity = [check_training_parity(row, reference_history[i]) for i, row in enumerate(history)]
        smoke_initial_model = h.backbone_digest(model) if args.phase == 'smoke' else None
        smoke_initial_router = (h.backbone_digest(model.router)
                                if args.phase == 'smoke' and args.mode == 'joint' else None)
        epochs = range(1) if args.phase == 'smoke' else range(start, 150)
        for epoch in epochs:
            scalar.set_epoch_lrs(optimizer, epoch)
            row = scalar.train_epoch(bundle, model, optimizer, epoch, args.mode,
                                     max_batches=SMOKE_BATCHES if args.phase == 'smoke' else None)
            if args.mode == 'joint' and not row['gradients']['router']['nonzero']:
                raise RuntimeError('No nonzero Router task gradient reached the added module')
            if args.phase == 'train':
                parity.append(check_training_parity(row, reference_history[epoch]))
            # evaluate() isolates Python/NumPy/Torch/CUDA RNG internally. No
            # validation inference/selection changes subsequent training masks.
            evaluated = {rate: scalar.evaluate(bundle, model, args.mode, 'test', masks['test'][rate],
                         max_batches=1 if args.phase == 'smoke' else None) for rate in RATES}
            row['test'] = {rate: result[0] for rate, result in evaluated.items()}
            history.append(row)
            improved = persist_epoch(args.output, model, optimizer, epoch=epoch + 1, identity=identity,
                                     history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch + 1, selected_epoch_by_rate={r: value['epoch'] for r, value in best.items()},
                best_by_rate=best, training_parity=parity,
                seconds=time.monotonic() - tick, peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
            h.write_json(summary_path, summary)
            print(json.dumps(dict(event='epoch', epoch=epoch + 1, task_MSE=row['task_MSE'],
                test_wf1={rate: result[0]['router']['WF1'] for rate, result in evaluated.items()},
                improved_rates=improved, selected_epochs=summary['selected_epoch_by_rate'],
                learning_rates=row['learning_rates'])), flush=True)
        if args.phase == 'smoke':
            if h.backbone_digest(model) == smoke_initial_model:
                raise RuntimeError('Smoke made no parameter update')
            if args.mode == 'joint' and h.backbone_digest(model.router) == smoke_initial_router:
                raise RuntimeError('Smoke Router parameters did not update')
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            if not all(torch.equal(value.cpu(), state['model'][name]) for name, value in model.state_dict().items()):
                raise RuntimeError('Full model checkpoint reload failed')
            validate_selected(args.output, best)
            metrics, arrays = scalar.evaluate(bundle, model, args.mode, 'test', masks['test']['0.3'], max_batches=1)
            with np.load(args.output / best['0.3']['predictions'], allow_pickle=False) as saved:
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], saved[key], err_msg='Smoke exact selected replay: ' + key)
            summary.update(smoke=history[0], smoke_only=True, full_checkpoint_reloaded=True,
                           selected_artifacts_exact_reload=True, smoke_test=metrics)
        else:
            summary['evaluation'] = publish_selected(args.output, best)
            summary['aggregate'] = scalar.aggregate(summary['evaluation'])
        for filename, digest in identity['source_hashes'].items():
            if h.sha(ROOT / filename) != digest:
                raise RuntimeError('Immutable training source changed: ' + filename)
        summary.update(status='complete', seconds=time.monotonic() - tick,
                       peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        h.write_json(summary_path, summary)


if __name__ == '__main__':
    main()
