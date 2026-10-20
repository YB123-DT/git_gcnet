"""Full-four-session IEMOCAP6 LOSO training; per-rate TEST-best ACC.

INTERNAL TEST-ORACLE SCREEN. Each rate selects its own epoch, with UA and
W-F1 retained from that exact evaluation. No adapter validation holdout or
previously trained weights are used. Eight-checkpoint means are not a single
deployable checkpoint's score.
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
SMOKE_BATCHES = 3
CORE = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py',
        'gcnet_missing_m3/model.py', 'gcnet_missing_m3/train_gcnet.py',
        'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
        'gcnet_modality_jepa/protocol.py', 'gcnet_modality_jepa/train_gcnet.py')
SOURCES = CORE + ('experiments/osram_text_path_router_20261019/router.py',
    'gcnet_missing_m3/mixed_rate.py',
    'experiments/osram_text_path_router_20261019/candidates.py',
    'experiments/osram_12_source_evidence_audit_20261019/analysis.py',
    'experiments/osram_iemocap_b6_transfer_20261015/worker.py',
    'experiments/osram_meld_write_retention_20261014/run.py',
    'experiments/osram_text_path_router_joint_20261019/joint.py',
    'experiments/osram_text_path_router_joint_20261019/run.py',
    'experiments/osram_iemocap6_router_fivefold_20261019/worker.py',
    'experiments/osram_tasr_20261018/protocol.py',
    'experiments/osram_tasr_20261018/runner_helpers.py',
    'experiments/osram_meld_flat_source_combinations_20261013/evaluate.py',
    'experiments/osram_iemocap6_gcnet_testbest_20261020/worker.py')


def make_config(fold, seed):
    if fold not in (1, 2, 3, 4, 5) or seed not in (66, 67, 68):
        raise ValueError('Approved folds1..5 and seeds66/67/68 required')
    from experiments.osram_tasr_20261018.protocol import build_cases
    from gcnet_missing_m3.train_gcnet import TrainConfig
    template = build_cases()['IEMOCAPSix']['expected_config']
    cfg = replace(TrainConfig(**template), fold=fold, seed=seed,
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='accuracy', evaluate_test=True)
    expected = dict(dataset='IEMOCAPSix', epochs=100, batch_size=32, learning_rate=.001,
        weight_decay=1e-5, dropout=.5, lr_schedule='constant', evaluation_protocol='official',
        missing_mask_protocol=PROTOCOL, training_objective='emotion-only',
        osram_readout_fusion='flat', osram_output_dim=1600, osram_write_step=.6,
        osram_bidirectional=False, osram_meaningful_block='none', paired_history_views=False)
    if any(getattr(cfg, name) != value for name, value in expected.items()):
        raise ValueError('Protected original IEMOCAP Flat configuration changed')
    return cfg


def select_fold(train, validation, test, fold):
    if fold not in (1, 2, 3, 4, 5) or any(len(group) != 5 for group in (train, validation, test)):
        raise ValueError('Exactly five original LOSO loaders and a valid fold required')
    return train[fold - 1], validation[fold - 1], test[fold - 1]


def full_training_split(original, test_loader, fold):
    """Record the actual original sampler, without reserving training dialogues."""
    train, test = sorted(original.sampler.indices), sorted(test_loader.sampler.indices)
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise ValueError('Original LOSO train/test overlap or duplicated conversations')
    vids = original.dataset.vids
    return dict(fold=fold, method='original-get_loaders-official-four-session-train',
        no_extra_training_holdout=True, adapter_holdout=False,
        train_indices=train, original_train_indices=train, test_indices=test,
        validation_indices=test, validation_is_test_diagnostic_only=True,
        train_conversations=[str(vids[i]) for i in train],
        test_conversations=[str(vids[i]) for i in test], uses_labels_for_split=False)


def setup_bundle(fold, seed, gpu_uuid, output):
    """Fresh model and full original four-session training; no checkpoint access."""
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.protocol import build_cases
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment, repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    case = build_cases()['IEMOCAPSix']
    data_root = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
    label_path = data_root / 'IEMOCAP/IEMOCAP_features_raw_6way.pkl'
    if h.sha(label_path) != case['label_sha256']:
        raise ValueError('Accepted IEMOCAP label SHA changed')
    roots = tuple(case['feature_roots'])
    if not all(Path(root).is_dir() for root in roots):
        raise FileNotFoundError('Accepted IEMOCAP feature roots unavailable')
    os.environ['GCNET_DATASET_ROOT'] = str(data_root)
    os.environ.pop('GCNET_MELD_PICKLE', None)
    os.environ['GCNET_CACHE_ROOT'] = str(Path(output).parent / 'feature_cache' / 'IEMOCAPSix')
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_text_path_router_joint_20261019.run import prepare_fresh
    environment = require_replication_environment(torch)
    repair_uuid_rng(torch)
    gpu = guard_gpu(gpu_uuid)
    torch.set_num_threads(2)
    cfg = make_config(fold, 66)
    train, validation, test, *dimensions = get_loaders(*roots, dataset='IEMOCAPSix',
        num_folder=5, batch_size=32, num_workers=0, seed=66,
        validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    original, validation_loader, test_loader = select_fold(train, validation, test, fold)
    split = full_training_split(original, test_loader, fold)
    source_case = dict(dataset='IEMOCAPSix', classes=6, fold=fold, seed=seed,
        label_sha256=case['label_sha256'], feature_roots=list(roots), registry_sha256=case['registry_sha256'],
        provenance='accepted-IEMOCAP-data-config-only-no-checkpoint-or-selected-mask-reuse')
    model = _build_model(cfg, tuple(dimensions), torch.device('cuda:0'))
    bundle = dict(model=model, trainer=trainer, cfg=cfg, case=source_case, split=split,
        dimensions=tuple(dimensions), train_loader=original,
        validation_loader=validation_loader, test_loader=test_loader, environment=environment, gpu=gpu)
    return bundle, prepare_fresh


def _atomic_arrays(path, arrays):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def persist_epoch(output, model, optimizer, *, epoch, identity, history, best, evaluated):
    """Select strict TEST ACC improvements and preserve same-call UA/WF1/masks."""
    from experiments.osram_text_path_router_joint_20261019.run import checkpoint_payload
    from experiments.osram_tasr_20261018 import runner_helpers as h
    output = Path(output)
    if set(evaluated) != set(RATES):
        raise ValueError('Require all eight same-epoch TEST evaluations')
    for metrics, arrays in evaluated.values():
        if not all(math.isfinite(metrics['router'][key]) for key in ('ACC', 'UA', 'WF1', 'CE')):
            raise ValueError('Selection metrics must be finite')
        if not np.isfinite(arrays['log_probs_router']).all():
            raise ValueError('Selection predictions must be finite')
        if metrics['mask_sha256'] != hashlib.sha256(arrays['availability'].tobytes()).hexdigest():
            raise ValueError('Same-call prediction mask hash differs')
    improved = [rate for rate, (metrics, _) in evaluated.items()
                if rate not in best or metrics['router']['ACC'] > best[rate]['metrics']['router']['ACC']]
    for rate in improved:
        metrics, arrays = evaluated[rate]
        suffix = rate.replace('.', 'p')
        prediction = 'predictions_miss_' + suffix + '.npz'
        _atomic_arrays(output / prediction, arrays)
        best[rate] = dict(epoch=epoch, checkpoint='best_miss_' + suffix + '.pt', predictions=prediction,
            prediction_sha256=h.sha(output / prediction), metrics=copy.deepcopy(metrics),
            criterion='TEST accuracy; strict improvement; earliest tie', same_evaluation_call=True)
    state = checkpoint_payload(model, optimizer, epoch=epoch, identity=identity,
        history=history, best_key={r: row['metrics']['router']['ACC'] for r, row in best.items()})
    state['config'] = copy.deepcopy(identity.get('config'))
    for rate in improved:
        state['selection'] = dict(copy.deepcopy(best[rate]), rate=rate)
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
            if h.sha(Path(output) / row[field]) != row[digest]:
                raise ValueError('Selected ' + field + ' integrity failure at rate ' + rate)


def publish_selected(output, best):
    """Copy the exact selected predictions, without a new sampler/inference call."""
    from experiments.osram_tasr_20261018 import runner_helpers as h
    output = Path(output)
    validate_selected(output, best)
    (output / 'evaluation').mkdir(exist_ok=True)
    result = {}
    for rate, row in best.items():
        target = output / 'evaluation' / ('random_' + rate.replace('.', 'p') + '.npz')
        shutil.copyfile(output / row['predictions'], target)
        result[rate] = dict(copy.deepcopy(row['metrics']), output_sha256=h.sha(target), selected_epoch=row['epoch'])
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--fold', type=int, choices=(1, 2, 3, 4, 5), required=True)
    result.add_argument('--seed', type=int, choices=(66, 67, 68), required=True)
    result.add_argument('--mode', choices=('flat', 'joint'), required=True)
    result.add_argument('--phase', choices=('smoke', 'train'), required=True)
    result.add_argument('--source-commit', required=True)
    result.add_argument('--gpu-uuid', required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--resume', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.gpu_uuid.startswith('GPU-'):
        raise ValueError('Use an actual healthy GPU UUID')
    if args.resume and args.phase != 'train':
        raise ValueError('Resume requires train phase')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_iemocap6_router_fivefold_20261019.worker import build_mask_bank, selected_from_bank
    from experiments.osram_text_path_router_joint_20261019 import run as joint
    if args.output.exists() and not args.resume:
        raise FileExistsError('Independent output required; never overwrite existing results')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick = args.output / 'SUMMARY.json', time.monotonic()
    if args.resume and path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError('Completed run must not be repeated')
    summary = dict(status='preflight', dataset='IEMOCAPSix', fold=args.fold, seed=args.seed,
        mode=args.mode, phase=args.phase, source_commit=args.source_commit, pid=os.getpid(),
        experiment_label='INTERNAL TEST-ORACLE SCREEN')
    try:
        bundle, prepare_fresh = setup_bundle(args.fold, args.seed, args.gpu_uuid, args.output)
        # This import resolves dataset paths; setup must install its environment
        # first, just as in the original accepted training entry point.
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        # prepare_fresh preserves factory initialization and training-loader seeds;
        # its old validation selection is immediately replaced, without new RNG.
        model = prepare_fresh(bundle, args.mode, seed=args.seed)
        bundle['cfg'] = make_config(args.fold, args.seed)
        backbone = model.backbone if args.mode == 'joint' else model
        initial = h.backbone_digest(backbone)
        optimizer = joint.make_optimizer(model, bundle['cfg'], args.mode)
        test_masks = build_mask_bank(bundle, 'test')
        selected, references = selected_from_bank(test_masks)
        torch.save(test_masks, args.output / 'test_masks.pt')
        torch.save(selected, args.output / 'selected_masks.pt')
        identity = dict(source_commit=args.source_commit, source_hashes={p: h.sha(ROOT / p) for p in SOURCES},
            mode=args.mode, seed=args.seed, fold=args.fold, config=asdict(bundle['cfg']), split=bundle['split'],
            test_masks={r: ref['mask_sha256'] for r, ref in references.items()}, test_mask_origin=references,
            seed_protocol=bundle['seed_protocol'], original_reference_case=bundle['case'],
            initialization='canonical-fresh-factory-no-checkpoint-weights',
            router_learning_rate=3e-4 if args.mode == 'joint' else None,
            selection='test-oracle-per-rate; router accuracy; strict greater; earliest ties',
            no_trajectory_parity_claim_against_old_extra_holdout=True)
        summary.update(identity=identity, config=identity['config'], status='running',
            environment=bundle['environment'], gpu=bundle['gpu'],
            total_parameters=sum(p.numel() for p in model.parameters()),
            router_parameters=sum(p.numel() for p in model.router.parameters()) if args.mode == 'joint' else 0,
            initial_backbone_sha256=initial, protected_weights_excluded_from_initialization=True)
        h.write_json(args.output / 'config.json', identity['config'])
        h.write_json(path, summary)
        reset_training_stochasticity(args.seed, args.fold)
        torch.cuda.reset_peak_memory_stats()
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            joint.restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            validate_selected(args.output, best)
        router_initial = h.backbone_digest(model.router) if args.phase == 'smoke' and args.mode == 'joint' else None
        epochs = range(1) if args.phase == 'smoke' else range(start, bundle['cfg'].epochs)
        for epoch in epochs:
            row = joint.train_epoch(bundle, model, optimizer, epoch, args.mode,
                max_batches=SMOKE_BATCHES if args.phase == 'smoke' else None)
            if args.mode == 'joint' and not row['gradients']['router']['nonzero']:
                raise RuntimeError('No nonzero Router task gradient reached the added module')
            evaluated = {rate: joint.evaluate(bundle, model, args.mode, bundle['test_loader'],
                batches=test_masks[rate], max_batches=1 if args.phase == 'smoke' else None) for rate in RATES}
            row.update(test={r: item[0] for r, item in evaluated.items()},
                learning_rates=[group['lr'] for group in optimizer.param_groups])
            history.append(row)
            improved = persist_epoch(args.output, model, optimizer, epoch=epoch + 1,
                identity=identity, history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch + 1, best_by_rate=best,
                selected_epoch_by_rate={r: item['epoch'] for r, item in best.items()},
                seconds=time.monotonic() - tick, peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
            h.write_json(path, summary)
            print(json.dumps(dict(event='epoch', fold=args.fold, epoch=epoch + 1,
                task_NLL=row['task_NLL'], test_acc={r: item[0]['router']['ACC'] for r, item in evaluated.items()},
                improved_rates=improved, selected_epochs=summary['selected_epoch_by_rate'])), flush=True)
        if args.phase == 'smoke':
            if h.backbone_digest(backbone) == initial:
                raise RuntimeError('Smoke did not update original backbone')
            if args.mode == 'joint' and h.backbone_digest(model.router) == router_initial:
                raise RuntimeError('Smoke did not update Router')
            stored = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            joint.restore(model, optimizer, stored, identity)
            if not all(torch.equal(value.cpu(), stored['model'][name]) for name, value in model.state_dict().items()):
                raise RuntimeError('Full checkpoint restore failed')
            validate_selected(args.output, best)
            metrics, arrays = joint.evaluate(bundle, model, args.mode, bundle['test_loader'], batches=test_masks['0.3'], max_batches=1)
            with np.load(args.output / best['0.3']['predictions'], allow_pickle=False) as saved:
                for key in arrays:
                    np.testing.assert_array_equal(arrays[key], saved[key], err_msg='Smoke exact selected replay: ' + key)
            summary.update(smoke=history[0], smoke_only=True, smoke_test=metrics, backbone_changed=True,
                router_changed=args.mode == 'joint', full_checkpoint_reloaded=True, selected_artifacts_exact_reload=True)
        else:
            summary['evaluation'] = publish_selected(args.output, best)
            summary['aggregate'] = {tag: h.aggregate_rates({r: value[tag] for r, value in summary['evaluation'].items()})
                for tag in joint.TAGS}
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT / name) != digest:
                raise RuntimeError('Immutable source changed during training: ' + name)
        summary.update(status='complete', seconds=time.monotonic() - tick,
            final_backbone_sha256=h.backbone_digest(backbone), peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
