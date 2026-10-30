"""IEMOCAP-six A0--A3, original five-fold LOSO, seed66 only.

Independent extension: the already running immutable MOSEI experiment is not
modified. CE trains all valid labels; strict per-rate TEST ACC selects weights,
and UA/W-F1 come from that same call. INTERNAL TEST-ORACLE SCREEN only.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import protocol as p
from . import runtime as r
from . import train as shared

DATA_ROOT = Path('/data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset')
LABEL_SHA256 = 'e35143e9f087d436779cccd650cb0327e9caa55178d326330f4a8a914e91e023'
PARENTS = Path('/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016/'
               'runs/iemocap6/seed_66')
CONFIG_HASHES = {
    1: '1a75de79a6fa293e848e11a6a2145c80757aa46c05f6c6f72e236d46c99e1acb',
    2: '6459c1e2fdf61bbc5f9144ffe2f1bbc071d3dff1719cb810e4a171367ab55637',
    3: 'ace4947f7ae1d5fc3d519516ebce57c9c4879ddf5e9be3d98e04e717317230e8',
    4: '9d27a9bfb5836e89afc76821f810be911dbf0d9cb6297946fba49bb91c4e5b12',
    5: 'a60b8c6e7bdad07a2bce87321cc2d46bc1072ba50b3e8a9958c5a33270a74a47',
}
INITIAL_SHA256 = '46dc1ec74a63ba007fcce28a5255ebb7f81977ccb3cfef7212caead7ec80934c'
LABEL = 'INTERNAL TEST-ORACLE SCREEN; per-rate TEST ACC; matched UA/W-F1; seed66 five-fold'


def required_config(fold):
    if fold not in CONFIG_HASHES:
        raise ValueError('Only original IEMOCAPSix folds1..5 are approved')
    return dict(p.REQUIRED_CONFIG, dataset='IEMOCAPSix', fold=fold, epochs=100,
                learning_rate=1e-3, checkpoint_selection_metric='accuracy')


def variant_config(raw, variant, fold):
    if variant not in p.VARIANTS:
        raise ValueError('Unknown CAM ablation: ' + variant)
    for key, value in required_config(fold).items():
        # Later disabled opt-in extensions are absent in this protected 20261016
        # config. Only explicit False defaults can be filled, never task values.
        if raw.get(key, False if value is False else object()) != value:
            raise ValueError('Accepted IEMOCAP baseline differs: ' + key)
    if raw.get('osram_emotion_ablation') != 'full' or raw.get('osram_gap_read') != 'residual':
        raise ValueError('Require an unablated original parent config')
    if raw.get('gap_residual_strength') != 1. or tuple(raw.get('train_missing_rates', ())) != tuple(float(v) for v in p.RATES):
        raise ValueError('Original residualization or cyclic rates differ')
    return dict(raw, **p.VARIANTS[variant])


def build_model(cfg, dimensions, device):
    # Native run_experiment explicitly used fold:5 for MODEL initialization even
    # for another LOSO fold. Keep that historical numerical convention; do not
    # change cfg.fold, actual data split, sampler seed or training RNG namespace.
    return r.build_model(replace(cfg, fold=5), dimensions, device)


def artifact_path(parent, artifact):
    parent = Path(parent).resolve()
    path = (parent / artifact['path']).resolve()
    if not path.is_relative_to(parent):
        raise ValueError('Selected artifact path escapes the protected parent')
    if r.sha(path) != artifact['sha256']:
        raise ValueError('Selected artifact hash differs from its checkpoint')
    return path


def read_selected(parent, raw):
    summary = json.loads((parent / 'SUMMARY.json').read_text())
    if summary.get('status') != 'complete' or summary['identity']['label_sha256'] != LABEL_SHA256:
        raise ValueError('Require a complete original corrected-mask IEMOCAP parent')
    arrays, references = {}, {}
    for rate in p.RATES:
        row = summary['testbest']['rates'][rate]
        path = parent / f'best_miss_{rate.replace(".", "p")}.pt'
        if r.sha(path) != row['checkpoint_sha256']:
            raise ValueError('Protected original checkpoint hash changed: ' + rate)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['config'] != raw or checkpoint['epoch'] != row['epoch']:
            raise ValueError('Protected checkpoint config/epoch differs: ' + rate)
        artifact = checkpoint['selection_artifact']
        selected = artifact_path(parent, artifact)
        if artifact['sha256'] != row['selection_sha256']:
            raise ValueError('Selected artifact differs from parent summary: ' + rate)
        with np.load(selected, allow_pickle=False) as stored:
            arrays[rate] = {key: stored[key].copy() for key in stored.files}
        mask_hash = hashlib.sha256(arrays[rate]['availability'].tobytes()).hexdigest()
        if mask_hash != row['mask_sha256'] or mask_hash != artifact['availability_sha256']:
            raise ValueError('Selected original GCNet mask hash differs: ' + rate)
        np.testing.assert_array_equal(arrays[rate]['predictions'], arrays[rate]['logits'].argmax(1))
        references[rate] = dict(checkpoint=str(path), checkpoint_sha256=row['checkpoint_sha256'],
            epoch=row['epoch'], path=str(selected), predictions=str(selected),
            sha256=artifact['sha256'], mask_sha256=mask_hash)
        del checkpoint
    return arrays, references


def build_banks(loader, selected):
    """Recover padded batch masks from immutable selected sample identities."""
    loader.sampler.set_epoch(0)
    banks, cursor = {rate: [] for rate in p.RATES}, 0
    for data in loader:
        meta = r.metadata(data)
        n = len(meta['labels'])
        valid = data[7].T.bool()
        for rate in p.RATES:
            row = selected[rate]
            for key, value in meta.items():
                np.testing.assert_array_equal(value, row[key][cursor:cursor + n],
                                              err_msg=f'Original {rate} test order: {key}')
            availability = torch.zeros((*valid.shape, 3), dtype=torch.float32)
            availability.transpose(0, 1)[data[7].bool()] = torch.as_tensor(row['availability'][cursor:cursor + n])
            if not availability[valid].bool().any(-1).all() or availability[~valid].bool().any():
                raise ValueError('Selected GCNet availability is invalid')
            banks[rate].append(dict(meta, availability=availability))
        cursor += n
    if any(cursor != len(selected[rate]['labels']) for rate in p.RATES):
        raise ValueError('Selected prediction count differs from the original test loader')
    return banks


def full_training_split(original, test_loader, fold):
    train, test = sorted(original.sampler.indices), sorted(test_loader.sampler.indices)
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise ValueError('Original LOSO train/test overlap or duplicated conversations')
    vids = original.dataset.vids
    return dict(fold=fold, method='original-get_loaders-official-four-session-train',
        no_extra_training_holdout=True, adapter_holdout=False, train_indices=train,
        original_train_indices=train, test_indices=test, validation_indices=test,
        validation_is_test_diagnostic_only=True, train_conversations=[str(vids[i]) for i in train],
        test_conversations=[str(vids[i]) for i in test], uses_labels_for_split=False)


def setup(variant, fold, output, gpu_uuid):
    environment, gpu, repair = r.require_environment(gpu_uuid)
    label = DATA_ROOT / 'IEMOCAP/IEMOCAP_features_raw_6way.pkl'
    if r.sha(label) != LABEL_SHA256:
        raise ValueError('Accepted IEMOCAP-six labels changed')
    roots = [str(DATA_ROOT / 'IEMOCAP/features' / name) for name in
             ('wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT')]
    if not all(Path(path).is_dir() for path in roots):
        raise FileNotFoundError('Accepted IEMOCAP feature roots unavailable')
    r.bind_data_environment(dict(dataset_root=str(DATA_ROOT)), Path(output).parent / 'feature_cache' / 'IEMOCAPSix')
    from gcnet_missing_m3 import train_gcnet as trainer

    parent = PARENTS / f'fold_{fold}'
    if r.sha(parent / 'config.json') != CONFIG_HASHES[fold]:
        raise ValueError('Protected original IEMOCAP config changed')
    raw = json.loads((parent / 'config.json').read_text())
    cfg = trainer.TrainConfig(**variant_config(raw, variant, fold))
    train, validation, test, *dimensions = trainer.get_loaders(*roots, dataset='IEMOCAPSix',
        num_folder=5, batch_size=32, num_workers=0, seed=66,
        validation_fraction=cfg.validation_fraction, evaluation_protocol='official')
    train_loader, validation_loader, test_loader = train[fold - 1], validation[fold - 1], test[fold - 1]
    split = full_training_split(train_loader, test_loader, fold)
    if sorted(validation_loader.sampler.indices) != split['test_indices']:
        raise ValueError('Original official IEMOCAP validation/test diagnostic topology changed')
    model = build_model(cfg, tuple(dimensions), torch.device('cuda:0'))
    initial = r.state_digest(model)
    if sum(v.numel() for v in model.parameters()) != 13518054 or initial != INITIAL_SHA256:
        raise ValueError('Original IEMOCAP model dimensions or legacy initialization changed')
    selected, references = read_selected(parent, raw)
    with r.isolated_rng(66001):
        masks = build_banks(test_loader, selected)
    identity = dict(version=2, label=LABEL, dataset='IEMOCAPSix', seed=66, fold=fold, variant=variant,
        config=asdict(cfg), dimensions=dimensions, initial_backbone_sha256=initial,
        initialization='native-run_experiment legacy missing_m3_model_init:fold:5; actual data fold unchanged',
        parent_config=dict(path=str(parent / 'config.json'), sha256=CONFIG_HASHES[fold]),
        original_checkpoint_loaded_for_training=False, split=split,
        loader_protocol={name: loader.protocol_metadata for name, loader in
                         (('train', train_loader), ('validation', validation_loader), ('test', test_loader))},
        data=dict(dataset_root=str(DATA_ROOT), feature_roots=roots, label_sha256=LABEL_SHA256),
        environment=environment, source_hashes=r.source_hashes(),
        mask_reference=dict(kind='exact-original-selected-test-artifacts', rates=references,
                            hashes={rate: ref['mask_sha256'] for rate, ref in references.items()}),
        selection='per-rate TEST ACC; strict improvement; earliest ties; same-call six-class UA/W-F1',
        task_path_ablation='Full architecture retained; blocked task slots zero before original Flat input LN')
    return dict(trainer=trainer, cfg=cfg, dimensions=tuple(dimensions), model=model,
        train_loader=train_loader, validation_loader=validation_loader, test_loader=test_loader,
        test_masks=masks, selected_reference=selected, identity=identity,
        parent_root=parent, selected=references,
        environment=environment, gpu=gpu, rng_repair=repair, initial_backbone_sha256=initial)


def classification_metrics(labels, logits):
    labels = np.asarray(labels).astype(np.int64)
    logits = np.asarray(logits)
    if logits.shape != (len(labels), 6) or not np.isfinite(logits).all() or not len(labels):
        raise ValueError('Finite six-class logits and nonempty labels required')
    if np.any((labels < 0) | (labels > 5)):
        raise ValueError('IEMOCAP-six labels must be in [0,5]')
    pred = logits.argmax(1)
    matrix = confusion_matrix(labels, pred, labels=np.arange(6))
    support = matrix.sum(1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(6, dtype=float), where=support != 0)
    ce = torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels))
    return dict(ACC=100 * float(accuracy_score(labels, pred)), UA=100 * float(recall.mean()),
        WF1=100 * float(f1_score(labels, pred, average='weighted', zero_division=0)), CE=float(ce),
        N=len(labels), recall=(100 * recall).tolist(), support=support.tolist())


def evaluate(bundle, model, rate, max_batches=None):
    loader, trainer = bundle['test_loader'], bundle['trainer']
    loader.sampler.set_epoch(0)
    model.eval()
    pieces = {}
    with r.isolated_rng(66 + 99193), torch.no_grad():
        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            data = trainer._move_batch(raw, next(model.parameters()).device)
            view = r.saved_view(bundle, data, bundle['test_masks'][rate][index])
            logits = r.forward(model, view).transpose(0, 1)[data[7].bool()].cpu().numpy()
            values = dict(r.metadata(data), logits=logits, prediction=logits.argmax(1),
                availability=view['availability'].transpose(0, 1)[data[7].bool()].cpu().numpy())
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
    arrays = {key: np.concatenate(values) for key, values in pieces.items()}
    metrics = classification_metrics(arrays['labels'], arrays['logits'])
    metrics['mask_sha256'] = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
    return metrics, arrays


def aggregate(evaluated):
    if set(evaluated) != set(p.RATES):
        raise ValueError('Eight per-rate IEMOCAP results required')
    return {scope: {key: sum(evaluated[rate][key] for rate in rates) / len(rates)
                    for key in ('ACC', 'UA', 'WF1', 'CE')}
            for scope, rates in (('mean', p.RATES), ('high', p.RATES[5:]))}


def is_improvement(metrics, old):
    if any(not math.isfinite(float(metrics[key])) for key in ('ACC', 'UA', 'WF1', 'CE')):
        raise ValueError('Nonfinite IEMOCAP selection metrics')
    return old is None or metrics['ACC'] > old['ACC']


def persist_selected(output, model, *, epoch, identity, best, evaluated):
    if set(evaluated) != set(p.RATES):
        raise ValueError('Eight same-call test evaluations required')
    for metrics, arrays in evaluated.values():
        is_improvement(metrics, None)
        if hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != metrics['mask_sha256']:
            raise ValueError('Same-call classification mask hash differs')
    improved = [rate for rate in p.RATES if is_improvement(evaluated[rate][0], best.get(rate, {}).get('metrics'))]
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()} if improved else None
    for rate in improved:
        metrics, arrays = evaluated[rate]
        suffix = rate.replace('.', 'p')
        prediction, checkpoint = f'predictions_miss_{suffix}.npz', f'best_miss_{suffix}.pt'
        shared.atomic_arrays(Path(output) / prediction, arrays)
        row = dict(epoch=epoch, checkpoint=checkpoint, predictions=prediction,
            prediction_sha256=r.sha(Path(output) / prediction), metrics=copy.deepcopy(metrics),
            criterion='TEST ACC; strict improvement; earliest ties', same_evaluation_call=True)
        r.atomic_save(Path(output) / checkpoint, dict(version=2, kind='weights-only', epoch=epoch,
            model=state, identity=identity, config=identity['config'], dimensions=identity['dimensions'],
            selection=dict(row, rate=rate)))
        best[rate] = dict(row, checkpoint_sha256=r.sha(Path(output) / checkpoint))
    r.save_json(Path(output) / 'best_by_rate.json', best)
    return improved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=tuple(p.VARIANTS), required=True)
    parser.add_argument('--fold', type=int, choices=tuple(CONFIG_HASHES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.smoke and args.resume:
        raise ValueError('Smoke is not a resumable training run')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent IEMOCAP output required')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick = args.output / 'SUMMARY.json', time.monotonic()
    if path.exists() and json.loads(path.read_text()).get('status') in ('complete', 'smoke-passed'):
        raise ValueError('Completed IEMOCAP output cannot be overwritten')
    summary = dict(status='preflight', dataset='IEMOCAPSix', seed=66, fold=args.fold, variant=args.variant,
        phase='smoke' if args.smoke else 'train', label=LABEL, source_commit=args.source_commit, pid=os.getpid())
    r.save_json(path, summary)
    try:
        torch.set_num_threads(1)
        bundle = setup(args.variant, args.fold, args.output, args.gpu_uuid)
        model, cfg = bundle['model'], bundle['cfg']
        identity = json.loads(json.dumps(dict(bundle['identity'], source_commit=args.source_commit)))
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=cfg.learning_rate, name='backbone')], weight_decay=cfg.weight_decay)
        summary.update(identity=identity, config=identity['config'], environment=bundle['environment'], gpu=bundle['gpu'],
            rng_repair=bundle['rng_repair'], initial_backbone_sha256=bundle['initial_backbone_sha256'],
            total_parameters=sum(v.numel() for v in model.parameters()), status='running')
        r.save_json(args.output / 'config.json', identity['config'])
        r.save_json(args.output / 'IDENTITY.json', identity)
        r.save_json(args.output / 'fixed_masks_reference.json', identity['mask_reference'])
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity

        reset_training_stochasticity(66, args.fold)
        torch.cuda.reset_peak_memory_stats()
        if args.smoke:
            before = r.state_digest(model)
            row = r.train_epoch(bundle, model, optimizer, 0, max_batches=3)
            row['task_CE'] = row.pop('task_MSE')  # native _task_loss dispatched CE by dataset
            if r.state_digest(model) == before:
                raise RuntimeError('Smoke made no original task parameter update')
            metrics, _ = evaluate(bundle, model, '0.3', max_batches=1)
            payload = shared.checkpoint_payload(model, optimizer, epoch=1, identity=identity, history=[row], best={})
            r.atomic_save(args.output / 'smoke.pt', payload)
            shared.restore(model, optimizer, torch.load(args.output / 'smoke.pt', map_location='cpu', weights_only=False), identity)
            summary.update(status='smoke-passed', smoke=row, smoke_test=metrics, optimizer_rng_strict_restore=True, epoch=0)
        else:
            start, history, best = 0, [], {}
            if args.resume:
                payload = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
                shared.restore(model, optimizer, payload, identity)
                start, history, best = payload['epoch'], payload['history'], payload['best_by_rate']
                shared.validate_selected(args.output, best)
            for epoch in range(start, cfg.epochs):
                row = r.train_epoch(bundle, model, optimizer, epoch)
                row['task_CE'] = row.pop('task_MSE')
                evaluated = {rate: evaluate(bundle, model, rate) for rate in p.RATES}
                row['test'] = {rate: result[0] for rate, result in evaluated.items()}
                history.append(row)
                improved = persist_selected(args.output, model, epoch=epoch + 1, identity=identity, best=best, evaluated=evaluated)
                r.atomic_save(args.output / 'last.pt', shared.checkpoint_payload(model, optimizer, epoch=epoch + 1,
                    identity=identity, history=history, best=best))
                r.save_json(args.output / 'history.json', history)
                summary.update(epoch=epoch + 1, best_by_rate=best,
                    selected_epoch_by_rate={rate: value['epoch'] for rate, value in best.items()},
                    aggregate=aggregate({rate: value['metrics'] for rate, value in best.items()}),
                    seconds=time.monotonic() - tick, peak_memory_mib=torch.cuda.max_memory_allocated() / 2 ** 20)
                r.save_json(path, summary)
                print(json.dumps(dict(event='epoch', fold=args.fold, variant=args.variant, epoch=epoch + 1,
                    task_CE=row['task_CE'], aggregate=summary['aggregate'], improved_rates=improved)), flush=True)
            shared.validate_selected(args.output, best)
            if r.source_hashes() != identity['source_hashes']:
                raise RuntimeError('Immutable IEMOCAP source changed')
            summary.update(status='complete', evaluation={rate: row['metrics'] for rate, row in best.items()})
        summary.update(seconds=time.monotonic() - tick, peak_memory_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                       peak_reserved_memory_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        r.save_json(path, summary)


if __name__ == '__main__':
    main()
