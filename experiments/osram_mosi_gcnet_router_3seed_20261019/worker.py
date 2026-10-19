"""From-scratch MOSI150: matched Flat/one-stage Router under original GCNet masks.

The protected old conversation-mask-v2 checkpoint is used only for a strict
current-code prediction bridge, then discarded. Its 80.747 Test-best score is
not a paired baseline for these new GCNet-missing experiments.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(str(i / 10) for i in range(8))
PROTOCOL = 'gcnet-random-mask-v1'
COMBINED_MANIFEST = ROOT / 'experiments/osram_complete150_20261011/MANIFEST.json'


def load_manifest(path, expected_sha256, *, combined=COMBINED_MANIFEST):
    """Recover missing DATA bytes only when tracked embedded data hashes exactly."""
    path, combined = Path(path), Path(combined)
    if path.exists():
        contents = path.read_bytes()
        reconstructed = False
        origin = path
        extra = {}
    else:
        combined_bytes = combined.read_bytes()
        payload = json.loads(combined_bytes)['data']['CMUMOSI']
        contents = (json.dumps(payload, indent=2, sort_keys=True) + '\n').encode()
        reconstructed = True
        origin = combined
        extra = dict(combined_manifest_sha256=hashlib.sha256(combined_bytes).hexdigest())
    actual = hashlib.sha256(contents).hexdigest()
    if actual != expected_sha256:
        raise ValueError('Original MOSI data manifest hash changed')
    return json.loads(contents), dict(manifest_origin=str(origin), data_sha256=actual,
        byte_identical_reconstruction=reconstructed, **extra)


def configs(raw, seed):
    """Validate accepted historical settings before changing only the new mask."""
    if seed not in (66, 67, 68):
        raise ValueError('Approved MOSI seeds are 66, 67, 68')
    from experiments.osram_mosi_recent_history_20261018 import run as original
    from gcnet_missing_m3.train_gcnet import TrainConfig
    if raw.get('epochs') != 150 or raw.get('learning_rate') != 1e-4:
        raise ValueError('Expected accepted original MOSI150 reference')
    first, _ = original.stage_configs(dict(raw, epochs=100, learning_rate=1e-3))
    legacy = replace(TrainConfig(**first), epochs=150, learning_rate=1e-3,
        missing_mask_protocol='conversation-mask-v2', seed=66, fold=1,
        checkpoint_selection='validation', checkpoint_selection_metric='weighted_f1', device='cuda:0')
    current = replace(legacy, seed=seed, missing_mask_protocol=PROTOCOL)
    return legacy, current


def fresh_backbone(cfg, dimensions, device):
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    # Original run initialized with fold5 despite using official MOSI data fold1.
    return _build_model(replace(cfg, fold=5), dimensions, device).requires_grad_(True)


def validate_bridge(predictions, reference, measured, reference_metrics):
    for key in ('predictions', 'labels', 'availability'):
        np.testing.assert_array_equal(predictions[key], reference[key],
                                      err_msg='Current-source legacy MOSI bridge failed: ' + key)
    if measured['mask_sha256'] != reference_metrics['mask_sha256']['0.3']:
        raise ValueError('Legacy MOSI replay mask differs')
    for key in ('accuracy', 'weighted_f1'):
        if measured[key] != reference_metrics['test']['0.3'][key]:
            raise ValueError('Legacy MOSI replay metric differs: ' + key)


def build_masks(bundle, split):
    """Common evaluation banks seed66 for all six arms; training keeps run seed."""
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    return scalar.build_masks(dict(bundle, cfg=replace(bundle['cfg'], seed=66)), split)


def setup(gpu_uuid, output, seed):
    """No vanished archived source is required: current code must pass the bridge."""
    import torch
    import sklearn
    from experiments.osram_mosi_recent_history_20261018 import run as original
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    if (torch.__version__ != '2.2.2+cu121' or np.__version__ != '1.24.3' or sklearn.__version__ != '1.3.2'):
        raise ValueError('Use archived MOSI s0 Torch2.2.2/NumPy1.24.3/sklearn1.3.2 environment')
    gpu = guard_gpu(gpu_uuid)
    manifest, manifest_proof = load_manifest(original.DATA, original.DATA_SHA)
    for name, digest in manifest['files'].items():
        if h.sha(name) != digest:
            raise ValueError('Original MOSI feature changed: ' + name)
    os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(output.parent / 'feature_cache' / 'CMUMOSI')
    from gcnet_missing_m3 import train_gcnet as trainer
    cfg_path = original.REFERENCE / 'config.json'
    legacy, cfg = configs(json.loads(cfg_path.read_text()), seed)
    train, validation, test, *dimensions = trainer.get_loaders(*manifest['feature_roots'],
        dataset='CMUMOSI', num_folder=1, batch_size=cfg.batch_size, num_workers=0, seed=66,
        validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    dimensions, device = tuple(dimensions), torch.device('cuda:0')
    protected = fresh_backbone(legacy, dimensions, device).eval()
    if sum(p.numel() for p in protected.parameters()) != 13509793:
        raise ValueError('Current LargeFlat parameter count differs from protected MOSI')
    checkpoint = original.REFERENCE / 'best_miss_0p3.pt'
    prediction_file = original.REFERENCE / 'predictions_miss_0p3.npz'
    metrics_file = original.REFERENCE / 'metrics.json'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    protected.load_state_dict(state['model'], strict=True)
    test[0].sampler.set_epoch(0)
    with h.isolated_rng(66 + 12291), torch.no_grad():
        metrics, arrays = trainer.evaluate_rate(protected, test[0], trainer._schedules(legacy, 'test')[.3],
            'CMUMOSI', dimensions, device, collect=True)
    with np.load(prediction_file, allow_pickle=False) as reference:
        validate_bridge(arrays, reference, metrics, json.loads(metrics_file.read_text()))
    proof = dict(passed=True, strict_state_dict=True, exact_prediction_labels_availability=True,
        current_source_used=True, legacy_missing_protocol='conversation-mask-v2',
        archived_source_not_required=True, historical_commit=original.HISTORICAL_COMMIT,
        checkpoint=str(checkpoint), checkpoint_sha256=h.sha(checkpoint),
        config=str(cfg_path), config_sha256=h.sha(cfg_path),
        prediction_reference=str(prediction_file), prediction_reference_sha256=h.sha(prediction_file),
        reference_metrics=str(metrics_file), reference_metrics_sha256=h.sha(metrics_file),
        historical_data_manifest_sha256=original.DATA_SHA, data_manifest=manifest_proof, metrics=metrics)
    protected_digest = h.backbone_digest(protected)
    del protected, state, arrays
    gc.collect()
    torch.cuda.empty_cache()
    backbone = fresh_backbone(cfg, dimensions, device).train()
    initial_digest = h.backbone_digest(backbone)
    if initial_digest == protected_digest:
        raise RuntimeError('Protected checkpoint leaked into fresh initialization')
    bundle = dict(trainer=trainer, cfg=cfg, dimensions=dimensions, model=backbone,
        train_loader=train[0], validation_loader=validation[0], test_loader=test[0],
        bridge=proof, initial_backbone_sha256=initial_digest,
        protected_backbone_sha256=protected_digest, gpu=gpu,
        data_manifest_sha256=original.DATA_SHA, data_manifest=manifest_proof, data_files=manifest['files'],
        environment=dict(python=sys.executable, python_version=sys.version, torch=torch.__version__,
                         numpy=np.__version__, sklearn=sklearn.__version__),
        split={name: loader.protocol_metadata for name, loader in
               (('train', train[0]), ('validation', validation[0]), ('test', test[0]))})
    from experiments.osram_text_path_router_joint_20261019.run import seed_training_loader
    bundle['seed_protocol'] = seed_training_loader(bundle, seed)
    return bundle


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
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import checkpoint_payload, restore
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent output directory required')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary_path, tick = args.output / 'SUMMARY.json', time.monotonic()
    if args.resume and args.phase != 'train':
        raise ValueError('Only complete training runs support exact resume')
    if args.resume and summary_path.exists() and json.loads(summary_path.read_text()).get('status') == 'complete':
        raise ValueError('Completed run cannot be repeated silently')
    summary = dict(status='preflight', dataset='CMUMOSI', mode=args.mode, phase=args.phase,
                   seed=args.seed, source_commit=args.source_commit, pid=os.getpid())
    try:
        torch.set_num_threads(1)
        bundle = setup(args.gpu_uuid, args.output, args.seed)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        model = scalar.wrap_model(bundle['model'], args.mode)
        optimizer = scalar.make_optimizer(model, bundle['cfg'], args.mode)
        masks = {split: build_masks(bundle, split) for split in ('validation', 'test')}
        hashes = {split: {rate: hashlib.sha256(b''.join(row['availability'].numpy().tobytes()
                    for row in bank)).hexdigest() for rate, bank in rates.items()} for split, rates in masks.items()}
        source_files = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py', 'gcnet_missing_m3/model.py',
            'gcnet_missing_m3/train_gcnet.py', 'gcnet_modality_jepa/gcnet_legacy_random_mask.py',
            'gcnet_modality_jepa/mask_schedule.py', 'gcnet_modality_jepa/train_gcnet.py',
            'gcnet_modality_jepa/protocol.py',
            'experiments/osram_text_path_router_joint_20261019/joint.py',
            'experiments/osram_text_path_router_joint_20261019/run.py',
            'experiments/osram_text_path_router_20261019/router.py',
            'experiments/osram_text_path_router_joint_mosi_20261019/router.py',
            'experiments/osram_text_path_router_joint_mosi_20261019/run.py',
            'experiments/osram_mosi_recent_history_20261018/run.py',
            'experiments/osram_complete150_20261011/MANIFEST.json',
            'experiments/osram_meld_flat_source_combinations_20261013/evaluate.py',
            'experiments/osram_tasr_20261018/runner_helpers.py',
            str(Path(__file__).relative_to(ROOT)))
        identity = dict(source_commit=args.source_commit, source_hashes={p: h.sha(ROOT / p) for p in source_files},
            mode=args.mode, seed=args.seed, config=asdict(bundle['cfg']), split=bundle['split'],
            mask_hashes=hashes, seed_protocol=bundle['seed_protocol'], historical_bridge=bundle['bridge'],
            dataset_manifest_sha256=bundle['data_manifest_sha256'], data_manifest=bundle['data_manifest'],
            data_files=bundle['data_files'],
            initialization='fresh-original-hardcoded-missing_m3_model_init:fold:5',
            selection='single-validation-eight-rate-WF1-then-lower-MSE; no test selection',
            backbone_learning_rates=dict(first_100=1e-3, final_50=1e-4),
            router_learning_rates=[3e-4, 3e-5] if args.mode == 'joint' else None,
            missing_protocol=PROTOCOL, historical_80p747_is_not_paired_score=True)
        torch.save(masks, args.output / 'fixed_masks.pt')
        summary.update(identity=identity, config=identity['config'], environment=bundle['environment'],
            gpu=bundle['gpu'], initial_backbone_sha256=bundle['initial_backbone_sha256'],
            protected_weights_excluded_from_initialization=True, anchor_parity_gate=bundle['bridge'],
            mask_hashes=hashes, total_parameters=sum(p.numel() for p in model.parameters()),
            router_parameters=sum(p.numel() for p in model.router.parameters()) if args.mode == 'joint' else 0,
            status='running', historical_80p747_is_not_paired_score=True)
        h.write_json(args.output / 'config.json', identity['config'])
        h.write_json(summary_path, summary)
        reset_training_stochasticity(args.seed, bundle['cfg'].fold)
        torch.cuda.reset_peak_memory_stats()
        if args.phase == 'smoke':
            before = h.backbone_digest(model)
            row = scalar.train_epoch(bundle, model, optimizer, 0, args.mode, max_batches=3)
            if h.backbone_digest(model) == before:
                raise RuntimeError('Smoke made no parameter update')
            if args.mode == 'joint' and not row['gradients']['router']['nonzero']:
                raise RuntimeError('Smoke did not reach Router parameters')
            metrics, _ = scalar.evaluate(bundle, model, args.mode, 'validation', masks['validation']['0.3'], max_batches=1)
            state = checkpoint_payload(model, optimizer, epoch=1, identity=identity, history=[row], best_key=None)
            h.atomic_torch_save(args.output / 'smoke.pt', state)
            stored = torch.load(args.output / 'smoke.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, stored, identity)
            if not all(torch.equal(value.cpu(), stored['model'][name]) for name, value in model.state_dict().items()):
                raise RuntimeError('Full model checkpoint reload failed')
            summary.update(smoke=row, smoke_validation=metrics, smoke_only=True, full_checkpoint_reloaded=True)
        else:
            start, history, best_key = 0, [], None
            if args.resume:
                state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
                restore(model, optimizer, state, identity)
                start, history, best_key = state['epoch'], state['history'], state['best_key']
            for epoch in range(start, 150):
                scalar.set_epoch_lrs(optimizer, epoch)
                row = scalar.train_epoch(bundle, model, optimizer, epoch, args.mode)
                validation = {rate: scalar.evaluate(bundle, model, args.mode, 'validation', masks['validation'][rate])[0]
                              for rate in RATES}
                key = scalar.selection_key(validation)
                improved = best_key is None or tuple(key) > tuple(best_key)
                if improved:
                    best_key = key
                row.update(validation=validation, validation_key=key)
                history.append(row)
                state = checkpoint_payload(model, optimizer, epoch=epoch + 1, identity=identity,
                                           history=history, best_key=best_key)
                if improved:
                    h.atomic_torch_save(args.output / 'best.pt', state)
                h.atomic_torch_save(args.output / 'last.pt', state)
                h.write_json(args.output / 'history.json', history)
                summary.update(epoch=epoch + 1, best_validation_key=best_key,
                    seconds=time.monotonic() - tick, peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
                h.write_json(summary_path, summary)
                print(json.dumps(dict(event='epoch', epoch=epoch + 1, task_MSE=row['task_MSE'],
                    validation=key, best=best_key, learning_rates=row['learning_rates'])), flush=True)
            state = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            evaluation_dir = args.output / 'evaluation'
            evaluation_dir.mkdir(exist_ok=True)
            summary.update(selected_epoch=state['epoch'], evaluation={})
            for rate in RATES:
                metrics, arrays = scalar.evaluate(bundle, model, args.mode, 'test', masks['test'][rate])
                artifact = evaluation_dir / ('random_' + rate.replace('.', 'p') + '.npz')
                np.savez_compressed(artifact, **arrays)
                metrics['output_sha256'] = h.sha(artifact)
                summary['evaluation'][rate] = metrics
                h.write_json(summary_path, summary)
            summary['aggregate'] = scalar.aggregate(summary['evaluation'])
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT / name) != digest:
                raise RuntimeError('Immutable training source changed: ' + name)
        summary.update(status='complete', seconds=time.monotonic() - tick,
                       peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        h.write_json(summary_path, summary)


if __name__ == '__main__':
    main()
