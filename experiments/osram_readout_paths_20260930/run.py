"""Frozen MOSI Flat readout intervention; train/validation only, no fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.osram_cfg84_history_query_random_20260928.run import read, write, sha, configuration_dict

SEEDS = (66, 67, 68)
RATES = tuple(i / 10 for i in range(8))
GPU_UUID = 'GPU-e4cafb17-818e-216a-b94a-7440063a9153'
DEFAULT_DATA = Path('/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset')
DEFAULT_REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full')
LABEL = ('INTERNAL PAIRED READOUT DIAGNOSTIC; original checkpoints are per-rate '
         'Test-oracle selected; current inference uses train/validation only; '
         'no training or deployable coefficient selection')


def update_tensor_hash(digest, name, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        update_tensor_hash(digest, name, value)
    return digest.hexdigest()


def split_ids(loaders):
    """Inspect dataset IDs only: never iterate held-out test loader."""
    result = {}
    for name, loader in loaders.items():
        ids = [str(loader.dataset.vids[i]) for i in loader.protocol_metadata['indices']]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError(f'duplicate or empty {name} conversations')
        result[name] = sorted(ids)
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if set(result[a]) & set(result[b]):
            raise ValueError(f'non-independent conversation splits: {a}/{b}')
    return result


def verify_gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('6', GPU_UUID):
        raise RuntimeError('explicit CUDA_VISIBLE_DEVICES=6 or approved GPU UUID required')
    device = subprocess.check_output([
        'nvidia-smi', '-i', '6', '--query-gpu=uuid,name', '--format=csv,noheader'
    ], text=True).strip()
    if device.split(',')[0].strip() != GPU_UUID:
        raise RuntimeError(f'GPU 6 UUID mismatch: {device}')
    return device


def evaluate_split(model, loader, schedule, dimensions, device, expected_ids):
    import torch
    from gcnet_missing_m3.train_gcnet import _move_batch, _prepare_view
    from experiments.osram_readout_paths_20260930.intervention import (
        SETTINGS, capture_flat_readout, replay_predictions,
    )

    rows = {key: [] for key in ('predictions', 'labels', 'availability',
                               'conversation_ids', 'utterance_indices')}
    upstream = hashlib.sha256()
    batches = 0
    seen = set()
    loader.sampler.set_epoch(0)
    with torch.no_grad():
        for raw in loader:
            view = _prepare_view(_move_batch(raw, device), schedule, 0, dimensions)
            call = lambda: model([view['incomplete']], view['availability'],
                                 view['qmask'], view['umask'], view['lengths'],
                                 predict_missing=False)
            cache = capture_flat_readout(model, call, view['umask'])
            predictions = replay_predictions(model, cache)
            valid = view['umask'].T.bool()
            if predictions.shape[:3] != (len(SETTINGS), *valid.shape):
                raise ValueError(f'unexpected prediction shape {predictions.shape}')
            if not torch.isfinite(predictions[:, valid]).all():
                raise ValueError('non-finite intervention predictions')
            rows['predictions'].append(predictions[:, valid].cpu().numpy())
            rows['labels'].append(view['labels'].T[valid].cpu().numpy())
            rows['availability'].append(view['availability'][valid].cpu().numpy())
            # Boolean indexing is sequence-major, then batch-major, exactly as above.
            positions = valid.nonzero(as_tuple=False).cpu().tolist()
            ids = [str(view['conversation_ids'][b]) for t, b in positions]
            rows['conversation_ids'].extend(ids)
            rows['utterance_indices'].extend(t for t, b in positions)
            seen.update(ids)
            # Hash the cached, single-forward upstream values, not altered readouts.
            for key, value in sorted(vars(cache).items()):
                if isinstance(value, torch.Tensor):
                    update_tensor_hash(upstream, f'batch{batches}:{key}', value)
            batches += 1
    if seen != set(expected_ids):
        raise ValueError('evaluation conversation IDs differ from declared split')
    pairs = list(zip(rows['conversation_ids'], rows['utterance_indices']))
    if len(set(pairs)) != len(pairs):
        raise ValueError('duplicate utterance sample IDs')
    arrays = {
        'predictions': np.concatenate(rows['predictions'], axis=1),
        'labels': np.concatenate(rows['labels']),
        'availability': np.concatenate(rows['availability']),
        'conversation_ids': np.asarray(rows['conversation_ids'], dtype=str),
        'utterance_indices': np.asarray(rows['utterance_indices'], dtype=np.int64),
        'settings': np.asarray(SETTINGS, dtype=np.float64),
    }
    return arrays, dict(batches=batches, utterances=len(pairs), conversations=len(seen),
                        upstream_cache_sha256=upstream.hexdigest())


def run_seed(args, seed):
    import torch
    torch.set_num_threads(2)
    from gcnet_modality_jepa.train_gcnet import get_loaders
    from gcnet_missing_m3.train_gcnet import TrainConfig, _schedules
    from experiments.osram_cfg84_fixed_modality_ablation_20260923.run import _build_model

    output = args.output_root / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', seed=seed, label=LABEL,
                  started_at=datetime.now(timezone.utc).isoformat(), records=[])
    write(output / 'status.json', status)
    try:
        source = args.reference_root / f'seed_{seed}'
        configuration_dict(seed, args.reference_root)  # strict original cfg84 checks
        raw_config = read(source / 'config.json')
        flags = ('osram_history_query_adapter', 'osram_post_grn', 'osram_history_input_gate',
                 'osram_local_evidence_gate', 'osram_hierarchical_evidence_gate',
                 'osram_hierarchical_feature_only', 'completion_write_to_memory')
        if any(raw_config.get(key, False) for key in flags):
            raise ValueError('source is not unmodified Flat')
        config = TrainConfig(**raw_config)
        if config.dataset != 'CMUMOSI':
            raise ValueError('this runner only supports official independent MOSI splits')
        if config.mosi_task_mode != 'regression' or config.task_regression_loss != 'mse':
            raise ValueError('original MOSI regression MSE protocol required')
        write(output / 'source_config.json', raw_config)
        features = [str(args.dataset_root / 'CMUMOSI/features' / name) for name in
                    ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        train, val, test, *dimensions = get_loaders(
            audio_root=features[0], text_root=features[1], video_root=features[2],
            num_folder=1, dataset='CMUMOSI', batch_size=config.batch_size,
            num_workers=0, seed=seed, validation_fraction=config.validation_fraction,
            evaluation_protocol=config.evaluation_protocol)
        loaders = dict(train=train[0], validation=val[0], test=test[0])
        ids = split_ids(loaders)
        write(output / 'split_ids.json', dict(conversation_ids=ids, disjoint=True,
              loader_metadata={k: v.protocol_metadata for k, v in loaders.items()},
              test_access='metadata IDs only; no iteration or metric evaluation'))
        device = torch.device('cuda:0')
        model = _build_model(config, tuple(dimensions)).to(device).requires_grad_(False).eval()
        source_files = [Path(__file__), REPO / 'experiments/osram_readout_paths_20260930/intervention.py',
                        REPO / 'gcnet_missing_m3/osram.py', REPO / 'gcnet_missing_m3/model.py',
                        REPO / 'gcnet_missing_m3/train_gcnet.py']
        write(output / 'provenance.json', dict(
            label=LABEL, command=sys.argv, python=sys.version, torch=torch.__version__,
            cuda=torch.version.cuda, hostname=platform.node(), gpu=args.gpu_record,
            code_sha256={str(p.relative_to(REPO)): sha(p) for p in source_files},
            source_config_sha256=sha(source / 'config.json'),
            dataset_root=str(args.dataset_root), feature_roots=features,
            label_file_sha256=sha(args.dataset_root / 'CMUMOSI/CMUMOSI_features_raw_2way.pkl'),
            feature_directory_mtime_ns={p: Path(p).stat().st_mtime_ns for p in features},
            mask_epoch=0, model_mode='eval', gradients=False, checkpoint_selection='inherited per-rate Test-oracle',
            split_role={'train': 'in-sample path response', 'validation': 'independent conversations'},
            rates=args.rates, splits=args.splits))
        for rate in args.rates:
            tag = f'{rate:.1f}'.replace('.', 'p')
            checkpoint = source / f'best_miss_{tag}.pt'
            checkpoint_sha = sha(checkpoint)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            model.eval()
            before = state_hash(model)
            for split in args.splits:
                arrays, detail = evaluate_split(model, loaders[split], _schedules(config, split)[rate],
                                               tuple(dimensions), device, ids[split])
                after = state_hash(model)
                if before != after:
                    raise RuntimeError('frozen model tensors changed during diagnostic')
                arrays.update(seed=np.asarray(seed), rate=np.asarray(rate), split=np.asarray(split))
                target = output / f'{split}_miss_{tag}.npz'
                with target.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                record = dict(seed=seed, rate=rate, split=split, predictions_file=target.name,
                              predictions_sha256=sha(target), checkpoint=str(checkpoint),
                              checkpoint_sha256=checkpoint_sha, checkpoint_epoch=int(state['epoch']),
                              model_state_before_sha256=before, model_state_after_sha256=after,
                              **detail)
                status['records'].append(record)
                write(output / 'status.json', status)
                print(json.dumps(record, sort_keys=True), flush=True)
        status['status'] = 'complete'
    except BaseException:
        status.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(output / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('/data1/yb/remote_experiments/osram_readout_paths_20260930/results'))
    parser.add_argument('--reference-root', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--rates', type=float, nargs='+', default=list(RATES), choices=RATES)
    parser.add_argument('--splits', nargs='+', choices=('train', 'validation'), default=['train', 'validation'])
    args = parser.parse_args()
    if len(set(args.rates)) != len(args.rates) or len(set(args.splits)) != len(args.splits):
        parser.error('duplicate rates or splits')
    os.environ['GCNET_DATASET_ROOT'] = str(args.dataset_root)
    args.gpu_record = verify_gpu()
    seeds = (args.seed,) if args.seed is not None else SEEDS
    for seed in seeds:
        if (args.output_root / f'seed_{seed}').exists():
            raise FileExistsError(f'refusing overwrite: {args.output_root / f"seed_{seed}"}')
    for seed in seeds:
        run_seed(args, seed)


if __name__ == '__main__':
    main()
