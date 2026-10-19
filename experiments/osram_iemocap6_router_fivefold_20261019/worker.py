"""Remaining IEMOCAP-6 folds for the matched one-stage Router experiment.

Fold5 is deliberately forbidden here: its six completed runs must be reused.
No fold5 checkpoint or selected-mask artifact initializes another LOSO fold.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORE = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py',
        'gcnet_missing_m3/model.py', 'gcnet_missing_m3/train_gcnet.py',
        'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
        'gcnet_modality_jepa/protocol.py', 'gcnet_modality_jepa/train_gcnet.py')
SOURCES = CORE + ('experiments/osram_text_path_router_20261019/router.py',
    'experiments/osram_text_path_router_joint_20261019/joint.py',
    'experiments/osram_text_path_router_joint_20261019/run.py',
    'experiments/osram_iemocap6_router_fivefold_20261019/worker.py')


def require_fold(fold):
    if fold not in (1, 2, 3, 4):
        raise ValueError('Only remaining folds1..4 are allowed; reuse completed fold5')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--fold', type=int, choices=(1, 2, 3, 4), required=True)
    result.add_argument('--seed', type=int, choices=(66, 67, 68), required=True)
    result.add_argument('--mode', choices=('flat', 'joint'), required=True)
    result.add_argument('--phase', choices=('smoke', 'train'), required=True)
    result.add_argument('--source-commit', required=True)
    result.add_argument('--gpu-uuid', required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--resume', action='store_true')
    return result


def make_config(fold, seed):
    require_fold(fold)
    if seed not in (66, 67, 68):
        raise ValueError('Approved seeds are 66/67/68')
    from experiments.osram_tasr_20261018.protocol import build_cases
    from gcnet_missing_m3.train_gcnet import TrainConfig
    template = build_cases()['IEMOCAPSix']['expected_config']
    cfg = replace(TrainConfig(**template), fold=fold, seed=seed,
        checkpoint_selection='validation', checkpoint_selection_metric='accuracy', evaluate_test=True)
    expected = dict(dataset='IEMOCAPSix', epochs=100, batch_size=32, learning_rate=.001,
        weight_decay=1e-5, dropout=.5, lr_schedule='constant',
        missing_mask_protocol='gcnet-random-mask-v1', training_objective='emotion-only',
        osram_readout_fusion='flat', osram_output_dim=1600, osram_write_step=.6,
        osram_bidirectional=False, osram_meaningful_block='none', paired_history_views=False)
    if any(getattr(cfg, name) != value for name, value in expected.items()):
        raise ValueError('Accepted IEMOCAP one-stage template changed')
    return cfg


def select_fold(train, validation, test, fold):
    require_fold(fold)
    if any(len(loaders) != 5 for loaders in (train, validation, test)):
        raise ValueError('Exactly five original LOSO loaders required')
    return train[fold-1], validation[fold-1], test[fold-1]


def partition_fold(training_indices, test_indices, vids, fold):
    require_fold(fold)
    from experiments.osram_tasr_20261018.protocol import adapter_split_indices
    original_train, heldout_test = set(training_indices), set(test_indices)
    if original_train & heldout_test:
        raise ValueError('Original train/test conversations overlap')
    train, validation, split = adapter_split_indices(training_indices, vids, seed=66)
    if set(train) & set(validation) or (set(train) | set(validation)) != original_train:
        raise ValueError('Invalid chosen-fold conversation holdout')
    split.update(fold=fold, backbone_previously_saw_adapter_validation=False,
        train_indices=sorted(train), validation_indices=sorted(validation),
        test_indices=sorted(test_indices), original_train_indices=sorted(original_train))
    return train, validation, split


def build_mask_bank(bundle, split):
    """Original GCNet host/guest sampler, fixed seed66 banks for THIS fold.

    Epoch0 and padded-rectangle generation are preserved. Training RNG is restored,
    and no fold5 selected-mask files are loaded for another test-session cohort.
    """
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.protocol import RATES
    if split not in ('validation', 'test'):
        raise ValueError('Only evaluation mask banks are frozen')
    cfg = replace(bundle['cfg'], seed=66)
    loader = bundle[split+'_loader']
    loader.sampler.set_epoch(0)
    masks = {}
    with h.isolated_rng(66 + (76192 if split == 'validation' else 76194)):
        schedules = bundle['trainer']._schedules(cfg, split)
        for rate in RATES:
            rows = []
            for raw in loader:
                data = bundle['trainer']._move_batch(raw, next(bundle['model'].parameters()).device)
                view = bundle['trainer']._prepare_view(data, schedules[float(rate)], 0, bundle['dimensions'])
                availability = view['availability'].detach().cpu()
                valid = data[7].T.cpu().bool()
                if availability[~valid].ne(0).any() or availability[valid].sum(-1).lt(1).any():
                    raise ValueError('Original GCNet evaluation visibility invariant failed')
                rows.append(dict(h.metadata(data), availability=availability))
            if not rows:
                raise ValueError('No evaluation conversations in chosen fold')
            masks[rate] = rows
    return masks


def selected_from_bank(bank):
    selected, references = {}, {}
    for rate, rows in bank.items():
        keys = ('labels', 'speakers', 'conversation_ids', 'utterance_indices')
        arrays = {key: np.concatenate([row[key] for row in rows]) for key in keys}
        masks = []
        for row in rows:
            availability = row['availability'].transpose(0, 1).numpy()
            # Only padding is zero; valid utterances always contain an observation.
            masks.append(availability[availability.sum(-1) > 0])
        arrays['availability'] = np.concatenate(masks)
        if len(arrays['availability']) != len(arrays['labels']):
            raise ValueError('Fixed mask cohort disagrees with metadata')
        selected[rate] = arrays
        references[rate] = dict(mask_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest(),
            origin='current-fold-original-GCNet-host-guest-padded-rectangle-seed66-fixed-bank',
            selection='not-selected-on-test-labels', N=len(arrays['labels']))
    return selected, references


def setup_bundle(fold, seed, gpu_uuid, output):
    """Create fresh chosen-fold data/model without touching legacy fold5 weights."""
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.protocol import build_cases
    from experiments.osram_iemocap_b6_transfer_20261015.worker import require_replication_environment, repair_uuid_rng
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    case = build_cases()['IEMOCAPSix']
    data_root = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
    if h.sha(data_root/'IEMOCAP/IEMOCAP_features_raw_6way.pkl') != case['label_sha256']:
        raise ValueError('Accepted IEMOCAP label SHA changed')
    roots = tuple(case['feature_roots'])
    if not all(Path(root).is_dir() for root in roots):
        raise FileNotFoundError('Accepted IEMOCAP feature roots unavailable')
    os.environ['GCNET_DATASET_ROOT'] = str(data_root)
    os.environ.pop('GCNET_MELD_PICKLE', None)
    os.environ['GCNET_CACHE_ROOT'] = str(Path(output).parent/'feature_cache'/'IEMOCAPSix')
    # Environment must precede the trainer/config imports that resolve data paths.
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_modality_jepa.train_gcnet import get_loaders, _build_protocol_loader
    from gcnet_modality_jepa.protocol import SeedBundle
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
    original, _, test_loader = select_fold(train, validation, test, fold)
    train_ids, val_ids, split = partition_fold(original.sampler.indices,
        test_loader.sampler.indices, original.dataset.vids, fold)
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()

    def subset(indices, name):
        return _build_protocol_loader(original.dataset, indices, 'IEMOCAPSix', fold, name,
            32, 0, SeedBundle(66), split_hash, 'adapter-training-conversation-holdout')

    case = dict(dataset='IEMOCAPSix', classes=6, fold=fold, seed=seed,
        label_sha256=case['label_sha256'], feature_roots=list(roots),
        registry_sha256=case['registry_sha256'],
        provenance='accepted-IEMOCAP-data-config-only-no-fold5-checkpoint-or-mask-reuse')
    model = _build_model(cfg, tuple(dimensions), torch.device('cuda:0'))
    bundle = dict(model=model, trainer=trainer, cfg=cfg, case=case, split=split,
        dimensions=tuple(dimensions), train_loader=subset(train_ids, 'train'),
        validation_loader=subset(val_ids, 'validation'), test_loader=test_loader,
        environment=environment, gpu=gpu)
    # This helper seeds initialization using cfg.fold and adjusts only training RNG
    # and loader order. It loads no checkpoint and does not call setup_case().
    return bundle, prepare_fresh


def main():
    args = parser().parse_args()
    if not args.gpu_uuid.startswith('GPU-'):
        raise ValueError('Use an actual healthy GPU UUID')
    if args.resume and args.phase != 'train':
        raise ValueError('Resume requires train phase')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_tasr_20261018.protocol import RATES
    from experiments.osram_text_path_router_joint_20261019 import run as joint
    if args.output.exists() and not args.resume:
        raise FileExistsError('Independent output required; never overwrite existing results')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick = args.output/'SUMMARY.json', time.monotonic()
    if args.resume and path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError('Completed run must not be repeated')
    summary = dict(status='preflight', dataset='IEMOCAPSix', fold=args.fold, seed=args.seed,
        mode=args.mode, phase=args.phase, source_commit=args.source_commit, pid=os.getpid())
    try:
        bundle, prepare_fresh = setup_bundle(args.fold, args.seed, args.gpu_uuid, args.output)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        model = prepare_fresh(bundle, args.mode, seed=args.seed)
        backbone = model.backbone if args.mode == 'joint' else model
        initial = h.backbone_digest(backbone)
        optimizer = joint.make_optimizer(model, bundle['cfg'], args.mode)
        validation_masks = build_mask_bank(bundle, 'validation')
        test_masks = build_mask_bank(bundle, 'test')
        selected, references = selected_from_bank(test_masks)
        bundle.update(selected=selected, references=references)
        torch.save(validation_masks, args.output/'validation_masks.pt')
        torch.save(test_masks, args.output/'test_masks.pt')
        mask_hashes = {rate: hashlib.sha256(b''.join(row['availability'].numpy().tobytes()
            for row in rows)).hexdigest() for rate, rows in validation_masks.items()}
        identity = dict(source_commit=args.source_commit, source_hashes={p: h.sha(ROOT/p) for p in SOURCES},
            mode=args.mode, config=asdict(bundle['cfg']), split=bundle['split'],
            validation_mask_hashes=mask_hashes,
            test_masks={rate: ref['mask_sha256'] for rate, ref in references.items()},
            test_mask_origin=references, seed_protocol=bundle['seed_protocol'],
            original_reference_case=bundle['case'],
            initialization='canonical-fresh-factory-no-checkpoint-weights',
            router_learning_rate=3e-4 if args.mode == 'joint' else None,
            selection='validation-eight-rate-ACC-then-lower-CE')
        summary.update(identity=identity, status='running', environment=bundle['environment'],
            gpu=bundle['gpu'], total_parameters=sum(p.numel() for p in model.parameters()),
            router_parameters=sum(p.numel() for p in model.router.parameters()) if args.mode == 'joint' else 0,
            initial_backbone_sha256=initial, validation_mask_hashes=mask_hashes,
            protected_weights_excluded_from_initialization=True)
        h.write_json(args.output/'config.json', identity['config'])
        h.write_json(path, summary)
        reset_training_stochasticity(args.seed, args.fold)
        torch.cuda.reset_peak_memory_stats()
        if args.phase == 'smoke':
            router_before = h.backbone_digest(model.router) if args.mode == 'joint' else None
            row = joint.train_epoch(bundle, model, optimizer, 0, args.mode, max_batches=3)
            if h.backbone_digest(backbone) == initial:
                raise RuntimeError('Smoke did not update original backbone')
            if args.mode == 'joint' and h.backbone_digest(model.router) == router_before:
                raise RuntimeError('Smoke did not update Router')
            validation, _ = joint.evaluate(bundle, model, args.mode, bundle['validation_loader'],
                batches=validation_masks['0.3'], max_batches=1)
            payload = joint.checkpoint_payload(model, optimizer, epoch=1, identity=identity,
                history=[row], best_key=None)
            h.atomic_torch_save(args.output/'smoke.pt', payload)
            stored = torch.load(args.output/'smoke.pt', map_location='cpu', weights_only=False)
            joint.restore(model, optimizer, stored, identity)
            if not all(torch.equal(value.cpu(), stored['model'][name]) for name, value in model.state_dict().items()):
                raise RuntimeError('Full checkpoint restore failed')
            summary.update(smoke=row, smoke_validation=validation, smoke_only=True,
                backbone_changed=True, router_changed=args.mode == 'joint', full_checkpoint_reloaded=True)
        else:
            start, history, best_key = 0, [], None
            if args.resume:
                state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
                joint.restore(model, optimizer, state, identity)
                start, history, best_key = state['epoch'], state['history'], state['best_key']
            for epoch in range(start, bundle['cfg'].epochs):
                row = joint.train_epoch(bundle, model, optimizer, epoch, args.mode)
                validation = {rate: joint.evaluate(bundle, model, args.mode, bundle['validation_loader'],
                    batches=validation_masks[rate])[0] for rate in RATES}
                key = (sum(v['router']['ACC'] for v in validation.values())/8,
                    -sum(v['router']['CE'] for v in validation.values())/8)
                improved = best_key is None or tuple(key) > tuple(best_key)
                if improved:
                    best_key = key
                row.update(validation=validation, validation_key=key)
                history.append(row)
                state = joint.checkpoint_payload(model, optimizer, epoch=epoch+1,
                    identity=identity, history=history, best_key=best_key)
                if improved:
                    h.atomic_torch_save(args.output/'best.pt', state)
                h.atomic_torch_save(args.output/'last.pt', state)
                h.write_json(args.output/'history.json', history)
                summary.update(epoch=epoch+1, best_validation_key=best_key,
                    seconds=time.monotonic()-tick, peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
                h.write_json(path, summary)
                print(json.dumps(dict(event='epoch', fold=args.fold, epoch=epoch+1,
                    loss=row['task_NLL'], validation=key, best=best_key)), flush=True)
            state = torch.load(args.output/'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            summary.update(selected_epoch=state['epoch'], evaluation={})
            output = args.output/'evaluation'
            output.mkdir(exist_ok=True)
            for rate in RATES:
                metrics, arrays = joint.evaluate(bundle, model, args.mode, bundle['test_loader'], saved=selected[rate])
                artifact = output/('random_'+rate.replace('.', 'p')+'.npz')
                np.savez_compressed(artifact, **arrays)
                metrics['output_sha256'] = h.sha(artifact)
                summary['evaluation'][rate] = metrics
                h.write_json(path, summary)
            summary['aggregate'] = {tag: h.aggregate_rates({r: v[tag] for r, v in summary['evaluation'].items()})
                for tag in joint.TAGS}
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT/name) != digest:
                raise RuntimeError('Immutable source changed during training')
        summary.update(status='complete', seconds=time.monotonic()-tick,
            final_backbone_sha256=h.backbone_digest(backbone), peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
        if args.phase == 'train':
            from experiments.osram_text_path_router_joint_20261019.report import write_report
            write_report(args.output, summary)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
