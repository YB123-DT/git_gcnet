"""Frozen original V3 seed66 BEST replays, all rates and IEMOCAP five folds."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path('/data2/yb/remote_experiments')
DATASETS = ('CMUMOSI', 'CMUMOSEI', 'IEMOCAPFour', 'IEMOCAPSix')
RATES = tuple(f'{i/10:.1f}' for i in range(8))


def source_runs(dataset):
    if dataset not in DATASETS:
        raise ValueError('Only original V3 MOSI/MOSEI/IEMOCAP4/6 seed66')
    if dataset.startswith('CMU'):
        return [(1, REMOTE/'osram_source_role_v3_20261021/execution_v1/runs'/dataset)]
    return [(fold, REMOTE/'osram_v3_classification_20261021/execution_v1/runs'/dataset/
             f'fold_{fold}'/'seed_66') for fold in range(1, 6)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def verify_replay(arrays, saved):
    for key in ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability'):
        np.testing.assert_array_equal(arrays[key], saved[key], err_msg='Original replay cohort/mask: '+key)
    actual = arrays['prediction']
    expected = (saved['prediction'].reshape(-1, 1) if actual.shape[-1] == 1
                else saved['candidates'][:, 0, :])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6,
                               err_msg='Frozen checkpoint predictions changed')
    if actual.shape[-1] == 1:
        np.testing.assert_array_equal(actual[:, 0] > 0, expected[:, 0] > 0)
    else:
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    return dict(prediction_max_abs_error=float(np.abs(actual-expected).max(initial=0)),
                predictions_bitwise_exact=bool(np.array_equal(actual, expected)),
                cohort_labels_speakers_masks_exact=True, predicted_classes_exact=True)


def load_dataset(dataset, summary, output):
    from gcnet_missing_m3.train_gcnet import TrainConfig
    cfg = TrainConfig(**summary['identity']['config'])
    if (not cfg.osram_source_role_v3 or cfg.osram_source_role_v3_mode != 'residual'
            or cfg.osram_source_role_v3_dim != 64 or cfg.osram_adapter_hidden_dim != 0
            or cfg.seed != 66 or cfg.missing_mask_protocol != 'gcnet-random-mask-v1'):
        raise ValueError('Need original standard V3 seed66, not capacity/direct')
    identity = summary['identity']
    if dataset.startswith('CMU'):
        files = identity['data_files']
        pkl = next(Path(p) for p in files if p.endswith('.pkl'))
        dataset_root = pkl.parent.parent
        roots = [str(dataset_root/dataset/'features'/name)
                 for name in ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
        proof = dict(source_file_count=len(files), full_data_hash_verification=True)
        for path, digest in files.items():
            if sha(path) != digest:
                raise ValueError('Original feature changed: '+path)
    else:
        manifest = identity['original_reference_case']['data_manifest']
        if sha(manifest['label_file']) != manifest['label_sha256']:
            raise ValueError('Original IEMOCAP labels changed')
        dataset_root = Path(manifest['label_file']).parent.parent
        roots = manifest['feature_roots']
        proof = dict(label_file=manifest['label_file'], label_sha256=manifest['label_sha256'],
                     feature_integrity_check='Same original roots; exact saved prediction replay required')
    os.environ['GCNET_DATASET_ROOT'] = str(dataset_root)
    os.environ['GCNET_CACHE_ROOT'] = str(output/'feature_cache')
    from gcnet_missing_m3 import train_gcnet as trainer
    from gcnet_modality_jepa.train_gcnet import get_loaders
    train, val, test, *dimensions = get_loaders(*roots, dataset=dataset,
        num_folder=1 if dataset.startswith('CMU') else 5, batch_size=32, num_workers=0,
        seed=66, validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    return trainer, test, tuple(dimensions), proof


def evaluate(model, loader, bank, trainer, dimensions, capture):
    import torch
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018.runner_helpers import forward, isolated_rng
    loader.sampler.set_epoch(0)
    pieces, scans = {}, [0]
    hook = model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0, scans[0]+1))
    try:
        with isolated_rng(66+99193), torch.no_grad():
            for index, raw in enumerate(loader):
                data = trainer._move_batch(raw, next(model.parameters()).device)
                meta = metadata(data)
                for key, value in meta.items():
                    np.testing.assert_array_equal(value, bank[index][key])
                availability = bank[index]['availability'].to(data[7].device)
                view = trainer._prepare_view_from_primary_masks(data, availability, availability, dimensions)
                capture.conversation_ids = [str(x) for x in data[-1]]
                prediction = forward(model, view)
                values = dict(meta, **capture.arrays,
                    prediction=prediction.transpose(0, 1)[data[7].bool()].cpu().numpy())
                for key, value in values.items():
                    pieces.setdefault(key, []).append(value)
        if scans[0] != len(bank):
            raise RuntimeError('Require one original CHM scan per test batch')
        return {k: np.concatenate(v) for k, v in pieces.items()}, scans[0]
    finally:
        hook.remove()


def query_sensitivity(arrays):
    from .stats import ROLES
    result = {}
    count = arrays['seen'].sum(-1)
    for i, role in enumerate(ROLES):
        valid = arrays['query_shuffle_eligible'][:, i] & (count >= 2)
        old, new = arrays['attention'][valid, i], arrays['query_shuffle_attention'][valid, i]
        result[role] = dict(N=int(valid.sum()),
            head_average_weight_L1=float(np.abs(old.mean(1)-new.mean(1)).sum(-1).mean()) if valid.any() else None,
            per_head_weight_L1=np.abs(old-new).sum(-1).mean(0).tolist() if valid.any() else None)
    return result


def run_dataset(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    import sklearn
    from gcnet_missing_m3.train_gcnet import TrainConfig
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    from .capture import AttentionCapture
    from .stats import summarize
    if args.output.exists():
        raise FileExistsError('Fresh audit directory required')
    args.output.mkdir(parents=True)
    tick = time.monotonic()
    if not args.dataset.startswith('CMU'):
        from experiments.osram_iemocap_b6_transfer_20261015.worker import repair_uuid_rng
        repair_uuid_rng(torch)
    state = dict(status='preflight', dataset=args.dataset, seed=66, source_commit=args.source_commit,
                 gpu=guard_gpu(args.gpu_uuid), pid=os.getpid(), completed=[], train_steps=0)
    path = args.output/'STATUS.json'
    write(path, state)
    try:
        torch.set_num_threads(1)
        runs = source_runs(args.dataset)
        first = json.loads((runs[0][1]/'SUMMARY.json').read_text())
        expected_environment = first['environment']
        for key, actual in (('torch', torch.__version__), ('numpy', np.__version__), ('sklearn', sklearn.__version__)):
            if expected_environment[key] != actual:
                raise ValueError('Use same original environment: '+key)
        trainer, loaders, dimensions, data_proof = load_dataset(args.dataset, first, args.output)
        state.update(status='running', data_proof=data_proof, environment=dict(
            executable=sys.executable, torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__))
        write(path, state)
        for fold, folder in runs:
            summary = json.loads((folder/'SUMMARY.json').read_text())
            if summary.get('status') != 'complete':
                raise ValueError('Complete selected run required')
            cfg = TrainConfig(**summary['identity']['config'])
            model = _build_model(cfg, dimensions, torch.device('cuda:0')).eval().requires_grad_(False)
            mask_file = folder/('fixed_masks.pt' if args.dataset.startswith('CMU') else 'test_masks.pt')
            banks = torch.load(mask_file, map_location='cpu')
            if args.dataset.startswith('CMU'):
                banks = banks['test']
            mask_sha = sha(mask_file)
            loader = loaders[fold-1]
            for rate in RATES:
                selected = summary['best_by_rate'][rate]
                checkpoint = folder/selected['checkpoint']
                predictions = folder/selected['predictions']
                if sha(checkpoint) != selected['checkpoint_sha256'] or sha(predictions) != selected['prediction_sha256']:
                    raise ValueError('Protected BEST artifacts changed')
                checkpoint_state = torch.load(checkpoint, map_location='cpu')
                if checkpoint_state['epoch'] != selected['epoch']:
                    raise ValueError('BEST epoch mismatch')
                model.load_state_dict(checkpoint_state['model'], strict=True)
                del checkpoint_state
                with np.load(predictions, allow_pickle=False) as f:
                    saved = {key: f[key] for key in f.files}
                # Exact saved predictions gate the full-cohort instrumentation;
                # unit tests also check captured versus uncaptured computation.
                from experiments.osram_tasr_20261018.runner_helpers import backbone_digest
                before = backbone_digest(model)
                with AttentionCapture(model.osram.source_role_v3) as capture:
                    arrays, scans = evaluate(model, loader, banks[rate], trainer, dimensions, capture)
                proof = verify_replay(arrays, saved)
                if backbone_digest(model) != before:
                    raise RuntimeError('Read-only audit changed model parameters')
                result = summarize(arrays, args.dataset, rate, fold)
                result['query_sensitivity'] = query_sensitivity(arrays)
                result['provenance'] = dict(source_run=str(folder), source_sha=summary['source_commit'],
                    checkpoint=str(checkpoint), checkpoint_sha256=selected['checkpoint_sha256'],
                    predictions=str(predictions), prediction_sha256=selected['prediction_sha256'],
                    selected_epoch=selected['epoch'], masks=str(mask_file), masks_sha256=mask_sha,
                    original_config=summary['identity']['config'], scans=scans,
                    frozen_model_digest=before, **proof)
                tag = f'fold_{fold}_rate_{rate.replace(".", "p")}'
                np.savez_compressed(args.output/(tag+'.npz'), **arrays)
                write(args.output/(tag+'.json'), result)
                state['completed'].append(dict(fold=fold, rate=rate, N=len(arrays['labels']),
                    arrays=tag+'.npz', report=tag+'.json', max_error=proof['prediction_max_abs_error']))
                state['seconds'] = time.monotonic()-tick
                write(path, state)
                print(json.dumps(dict(event='replayed', dataset=args.dataset, fold=fold, rate=rate,
                    N=len(arrays['labels']), focus=result['focus'], proof=proof)), flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()
        state.update(status='complete', seconds=time.monotonic()-tick)
    except BaseException as error:
        state.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        write(path, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    run_dataset(p.parse_args())


if __name__ == '__main__':
    main()
