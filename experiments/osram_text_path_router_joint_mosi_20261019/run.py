"""Finite seed66 MOSI150 scalar Router plus an identically trained Flat control.

The old 80.747 three-seed Test-best reference is provenance, not a paired score.
Current-source strict checkpoint/prediction parity is required before training;
the protected checkpoint is then discarded and both arms start from scratch.
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
TAGS = ('full', 'router', 'uniform', 'candidate_B', 'candidate_G')


def metadata(data):
    """Do not cast continuous sentiment labels to classification integers."""
    valid = data[7].bool().cpu().numpy()
    ids = np.asarray([str(value) for value in data[-1]])
    return dict(labels=data[8].detach().cpu().numpy()[valid],
        speakers=data[6].detach().cpu().numpy()[valid],
        conversation_ids=np.broadcast_to(ids[:, None], valid.shape)[valid],
        utterance_indices=np.broadcast_to(np.arange(valid.shape[1]), valid.shape)[valid])


def set_epoch_lrs(optimizer, epoch):
    if not 0 <= epoch < 150:
        raise ValueError('MOSI experiment has exactly 150 epochs')
    for group in optimizer.param_groups:
        first = 3e-4 if group['name'] == 'router' else 1e-3
        group['lr'] = first if epoch < 100 else (3e-5 if group['name'] == 'router' else 1e-4)


def wrap_model(backbone, mode, router=None):
    if mode == 'flat':
        if router is not None:
            raise ValueError('Flat cannot contain a Router')
        backbone.osram.osram_source_tracking = False
        backbone.osram.osram_source_audit = False
        backbone.osram.osram_source_grad = False
        return backbone.requires_grad_(True).train()
    if mode != 'joint':
        raise ValueError('Expected flat or joint mode')
    from experiments.osram_text_path_router_joint_20261019.joint import JointTextPathModel
    from .router import RegressionTextPathRouter
    if router is None:
        router = RegressionTextPathRouter().to(next(backbone.parameters()).device)
    return JointTextPathModel(backbone, router).requires_grad_(True).train()


def make_optimizer(model, cfg, mode):
    from experiments.osram_text_path_router_joint_20261019.run import make_optimizer as original
    return original(model, cfg, mode)


def forward_result(model, view, mode):
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if mode == 'joint':
        result = model(view)
        # JointTextPathModel's legacy key is a raw scalar, NOT a log probability.
        return dict(result, prediction=result['log_probs'])
    prediction = h.forward(model, view)
    weights = prediction.new_zeros(*prediction.shape[:2], 3)
    weights[..., 0] = 1
    return dict(prediction=prediction,
        candidates=prediction.unsqueeze(-2).expand(*prediction.shape[:2], 3, 1), weights=weights,
        eligible=torch.zeros(prediction.shape[:2], dtype=torch.bool, device=prediction.device))


def scalar_loss(trainer, cfg, prediction, view):
    if cfg.dataset != 'CMUMOSI' or cfg.mosi_task_mode != 'regression' or cfg.task_regression_loss != 'mse':
        raise ValueError('Preserve the original MOSI scalar MSE task')
    return trainer._task_loss(cfg.dataset, prediction, view['labels'], view['umask'],
        cfg.mosi_task_mode, cfg.task_regression_loss, cfg.task_smooth_l1_beta)


def optimize_view(model, optimizer, view, cfg, mode, trainer):
    import torch
    optimizer.zero_grad(set_to_none=True)
    result = forward_result(model, view, mode)
    n = int(view['umask'].sum())
    if not n:
        raise ValueError('No valid training utterances')
    loss = scalar_loss(trainer, cfg, result['prediction'], view)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite scalar MSE')
    loss.backward()
    gradients = {}
    for group in optimizer.param_groups:
        grads = [p.grad for p in group['params'] if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Nonfinite gradient in ' + group['name'])
        gradients[group['name']] = dict(connected=len(grads), nonzero=sum(bool(g.ne(0).any()) for g in grads))
    if not gradients['backbone']['nonzero']:
        raise RuntimeError('Original model did not receive task gradients')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite global gradient norm')
    optimizer.step()
    return dict(task_MSE=float(loss.detach()), N=n, eligible_N=int(result['eligible'].sum()),
                gradient_norm=float(norm), gradients=gradients, optimizer_steps=1)


def train_epoch(bundle, model, optimizer, epoch, mode, max_batches=None):
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    trainer, cfg, loader = bundle['trainer'], bundle['cfg'], bundle['train_loader']
    loader.sampler.set_epoch(epoch)
    schedules = trainer._schedules(cfg, 'train')
    cycle = BalancedBatchRateSchedule(rates=cfg.train_missing_rates)
    model.train()
    rows, digest = [], hashlib.sha256()
    for index, raw in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        data = trainer._move_batch(raw, next(model.parameters()).device)
        view = trainer._prepare_view(data, schedules[cycle.rate_for(epoch, index)], epoch, bundle['dimensions'])
        rows.append(optimize_view(model, optimizer, view, cfg, mode, trainer))
        digest.update(json.dumps([str(v) for v in data[-1]], separators=(',', ':')).encode())
        digest.update(view['availability'].detach().cpu().numpy().tobytes())
    if not rows:
        raise RuntimeError('No training batches')
    return dict(epoch=epoch + 1, batches=len(rows), memory_scans=len(rows), optimizer_steps=len(rows),
        N=sum(row['N'] for row in rows), eligible_N=sum(row['eligible_N'] for row in rows),
        task_MSE=sum(row['task_MSE'] * row['N'] for row in rows) / sum(row['N'] for row in rows),
        gradient_norm_max=max(row['gradient_norm'] for row in rows),
        gradients={name: {key: max(row['gradients'][name][key] for row in rows)
                          for key in ('connected', 'nonzero')} for name in rows[0]['gradients']},
        mask_trace_sha256=digest.hexdigest(), learning_rates=[g['lr'] for g in optimizer.param_groups])


def summarize_arrays(arrays, trainer):
    labels, candidates = arrays['labels'], arrays['candidates'][..., 0]
    predictions = dict(full=candidates[:, 0], router=arrays['prediction'].reshape(-1),
        uniform=np.where(arrays['eligible'], candidates.mean(1), candidates[:, 0]),
        candidate_B=candidates[:, 1], candidate_G=candidates[:, 2])
    metrics = {}
    for tag, prediction in predictions.items():
        measured = trainer._metrics('CMUMOSI', labels, prediction, 'regression')
        metrics[tag] = dict(ACC=100 * measured['accuracy'], WF1=100 * measured['weighted_f1'],
                           MSE=float(np.mean((prediction - labels)**2)))
    nonzero = labels != 0
    old = candidates[nonzero, 0] > 0
    new = predictions['router'][nonzero] > 0
    truth = labels[nonzero] > 0
    metrics['paired_vs_full'] = dict(corrections=int(((old != truth) & (new == truth)).sum()),
        harms=int(((old == truth) & (new != truth)).sum()), N=int(nonzero.sum()))
    metrics['eligible_N'] = int(arrays['eligible'].sum())
    metrics['weights_mean_eligible'] = (arrays['weights'][arrays['eligible']].mean(0).tolist()
                                       if arrays['eligible'].any() else None)
    return metrics


def selection_key(validation):
    if set(validation) != set(RATES):
        raise ValueError('Selection requires all eight validation rates')
    return (sum(row['router']['WF1'] for row in validation.values()) / 8,
            -sum(row['router']['MSE'] for row in validation.values()) / 8)


def build_masks(bundle, split):
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    trainer, cfg, loader = bundle['trainer'], bundle['cfg'], bundle[split + '_loader']
    schedules = trainer._schedules(cfg, split)
    banks = {}
    with h.isolated_rng(66 + 76192):
        for rate in RATES:
            loader.sampler.set_epoch(0)
            records = []
            for raw in loader:
                data = trainer._move_batch(raw, torch.device('cpu'))
                view = trainer._prepare_view(data, schedules[float(rate)], 0, bundle['dimensions'])
                records.append(dict(metadata(data), availability=view['availability'].cpu()))
            banks[rate] = records
    return banks


def verify_test_mask_hashes(banks, expected, trainer):
    """Original evaluation hashes time-major valid slots, unlike metric order."""
    import torch
    if set(banks) != set(RATES) or set(expected) != set(RATES):
        raise ValueError('Require all eight original test mask hashes')
    actual = {}
    for rate, records in banks.items():
        # Original sampler guarantees nonempty real utterances and zero padding.
        valid_rows = [row['availability'][row['availability'].bool().any(-1)] for row in records]
        actual[rate] = trainer._sha256_tensor(torch.cat(valid_rows))
        if actual[rate] != expected[rate]:
            raise ValueError('Original MOSI test mask differs at rate ' + rate)
    return actual


def evaluate(bundle, model, mode, split, bank, max_batches=None):
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    loader, trainer = bundle[split + '_loader'], bundle['trainer']
    loader.sampler.set_epoch(0)
    model.eval()
    pieces = {}
    with h.isolated_rng(66 + 99193), torch.no_grad():
        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            data = trainer._move_batch(raw, next(model.parameters()).device)
            meta = metadata(data)
            record = bank[index]
            for key, value in meta.items():
                np.testing.assert_array_equal(value, record[key])
            availability = record['availability'].to(data[7].device)
            view = trainer._prepare_view_from_primary_masks(data, availability, availability, bundle['dimensions'])
            result = forward_result(model, view, mode)
            values = dict(meta, availability=availability.transpose(0, 1)[data[7].bool()].cpu().numpy())
            for key in ('prediction', 'candidates', 'weights', 'eligible'):
                values[key] = result[key].transpose(0, 1)[data[7].bool()].cpu().numpy()
            if not np.isfinite(values['prediction']).all():
                raise RuntimeError('Nonfinite scalar prediction')
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
    arrays = {key: np.concatenate(values) for key, values in pieces.items()}
    metrics = summarize_arrays(arrays, trainer)
    metrics['mask_sha256'] = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
    return metrics, arrays


def aggregate(evaluation):
    return {tag: {scope: {key: sum(evaluation[rate][tag][key] for rate in rates) / len(rates)
                         for key in ('ACC', 'WF1', 'MSE')}
                  for scope, rates in (('mean', RATES), ('high', RATES[5:]))} for tag in TAGS}


def setup(gpu_uuid, output):
    """Verify archived inputs, then bridge protected weights using CURRENT code."""
    import torch
    import sklearn
    from experiments.osram_mosi_recent_history_20261018 import run as original
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if (torch.__version__ != '2.2.2+cu121' or np.__version__ != '1.24.3' or sklearn.__version__ != '1.3.2'):
        raise ValueError('MOSI requires the archived s0 Torch/NumPy/sklearn environment')
    historical = original.verify_source(original.SOURCE)
    if h.sha(original.DATA) != original.DATA_SHA:
        raise ValueError('Original MOSI manifest hash changed')
    manifest = json.loads(original.DATA.read_text())
    for name, digest in manifest['files'].items():
        if h.sha(name) != digest:
            raise ValueError('Original MOSI data changed: ' + name)
    os.environ['GCNET_DATASET_ROOT'] = manifest['dataset_root']
    os.environ['GCNET_CACHE_ROOT'] = str(output.parent / 'feature_cache' / 'CMUMOSI')
    # Never prepend original.SOURCE: the current implementation must pass parity.
    from experiments.osram_meld_write_retention_20261014.run import guard_gpu
    gpu = guard_gpu(gpu_uuid)
    from gcnet_missing_m3 import train_gcnet as trainer
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import _build_model
    raw = json.loads((original.REFERENCE / 'config.json').read_text())
    if raw['epochs'] != 150 or raw['learning_rate'] != 1e-4:
        raise ValueError('Expected the accepted original MOSI150 reference')
    raw.update(epochs=100, learning_rate=1e-3)
    first, _ = original.stage_configs(raw)
    first['missing_mask_protocol'] = 'conversation-mask-v2'
    cfg = replace(trainer.TrainConfig(**first), epochs=150, learning_rate=1e-3,
                  checkpoint_selection='validation', checkpoint_selection_metric='weighted_f1', device='cuda:0')
    train, validation, test, *dimensions = trainer.get_loaders(*manifest['feature_roots'],
        dataset='CMUMOSI', num_folder=1, batch_size=cfg.batch_size, num_workers=0,
        seed=66, validation_fraction=cfg.validation_fraction, evaluation_protocol=cfg.evaluation_protocol)
    bundle = dict(trainer=trainer, cfg=cfg, dimensions=tuple(dimensions), train_loader=train[0],
                  validation_loader=validation[0], test_loader=test[0])
    device = torch.device('cuda:0')
    # Original MOSI run_experiment explicitly used fold:5 for initialization even
    # though data/protocol fold is1. Change this factory-only value, not the split.
    def fresh():
        return _build_model(replace(cfg, fold=5), bundle['dimensions'], device).requires_grad_(True)
    protected = fresh().eval()
    if sum(p.numel() for p in protected.parameters()) != 13509793:
        raise ValueError('Current MOSI Flat parameter count differs from the accepted baseline')
    checkpoint = original.REFERENCE / 'best_miss_0p3.pt'
    reference_path = original.REFERENCE / 'predictions_miss_0p3.npz'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    protected.load_state_dict(state['model'], strict=True)
    test[0].sampler.set_epoch(0)
    with h.isolated_rng(66 + 12291), torch.no_grad():
        bridge, predictions = trainer.evaluate_rate(protected, test[0], trainer._schedules(cfg, 'test')[.3],
            'CMUMOSI', bundle['dimensions'], device, collect=True)
    with np.load(reference_path, allow_pickle=False) as reference:
        for key in ('predictions', 'labels', 'availability'):
            np.testing.assert_array_equal(predictions[key], reference[key], err_msg='Current-source original MOSI parity failed: ' + key)
    reference_metrics = json.loads((original.REFERENCE / 'metrics.json').read_text())
    expected_mask = reference_metrics['mask_sha256']['0.3']
    if bridge['mask_sha256'] != expected_mask:
        raise ValueError('Original MOSI replay mask hash differs')
    for key in ('accuracy', 'weighted_f1'):
        if bridge[key] != reference_metrics['test']['0.3'][key]:
            raise ValueError('Original MOSI replay metric differs: ' + key)
    proof = dict(passed=True, strict_state_dict=True, exact_prediction_labels_availability=True,
        current_source_used=True, historical_source=historical,
        checkpoint=str(checkpoint), checkpoint_sha256=h.sha(checkpoint),
        prediction_reference=str(reference_path), prediction_reference_sha256=h.sha(reference_path),
        metrics=bridge, historical_data_manifest_sha256=original.DATA_SHA)
    protected_digest = h.backbone_digest(protected)
    del protected, state
    gc.collect()
    torch.cuda.empty_cache()
    backbone = fresh().train()
    initial_digest = h.backbone_digest(backbone)
    if initial_digest == protected_digest:
        raise RuntimeError('Protected weights leaked into fresh initialization')
    bundle.update(model=backbone, bridge=proof, initial_backbone_sha256=initial_digest,
        protected_backbone_sha256=protected_digest, gpu=gpu,
        environment=dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
        split={name: loader.protocol_metadata for name, loader in
               (('train', train[0]), ('validation', validation[0]), ('test', test[0]))})
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('smoke', 'train'))
    parser.add_argument('--mode', required=True, choices=('flat', 'joint'))
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_20261019.run import checkpoint_payload, restore
    if args.output.exists() and not args.resume:
        raise FileExistsError('Independent output directory required')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary_path, tick = args.output / 'SUMMARY.json', time.monotonic()
    if args.resume and args.phase != 'train':
        raise ValueError('Only complete training runs support exact resume')
    if args.resume and summary_path.exists() and json.loads(summary_path.read_text()).get('status') == 'complete':
        raise ValueError('Completed run cannot be repeated silently')
    summary = dict(status='preflight', dataset='CMUMOSI', mode=args.mode, phase=args.phase, seed=66,
                   source_commit=args.source_commit, pid=os.getpid())
    try:
        torch.set_num_threads(1)
        bundle = setup(args.gpu_uuid, args.output)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        model = wrap_model(bundle['model'], args.mode)
        optimizer = make_optimizer(model, bundle['cfg'], args.mode)
        masks = {split: build_masks(bundle, split) for split in ('validation', 'test')}
        from experiments.osram_mosi_recent_history_20261018 import run as original
        old_metrics = json.loads((original.REFERENCE / 'metrics.json').read_text())
        historical_test_mask_hashes = verify_test_mask_hashes(masks['test'], old_metrics['mask_sha256'], bundle['trainer'])
        hashes = {split: {rate: hashlib.sha256(b''.join(row['availability'].numpy().tobytes() for row in bank)).hexdigest()
                          for rate, bank in rates.items()} for split, rates in masks.items()}
        torch.save(masks, args.output / 'fixed_masks.pt')
        source_files = ('gcnet_missing_m3/osram.py', 'gcnet_missing_m3/tasr.py', 'gcnet_missing_m3/model.py',
            'gcnet_missing_m3/train_gcnet.py', 'gcnet_modality_jepa/mask_schedule.py',
            'gcnet_modality_jepa/train_gcnet.py', 'experiments/osram_text_path_router_joint_20261019/joint.py',
            'experiments/osram_text_path_router_joint_mosi_20261019/router.py',
            'experiments/osram_text_path_router_joint_mosi_20261019/run.py')
        identity = dict(source_commit=args.source_commit, source_hashes={p: h.sha(ROOT / p) for p in source_files},
            mode=args.mode, config=asdict(bundle['cfg']), split=bundle['split'], mask_hashes=hashes,
            historical_test_mask_parity=historical_test_mask_hashes,
            initialization='fresh-original-hardcoded-missing_m3_model_init:fold:5',
            selection='single-validation-eight-rate-WF1-then-lower-MSE; no test selection',
            backbone_learning_rates=dict(first_100=1e-3, final_50=1e-4),
            training_rng_namespace='training_stochasticity:fold:' + str(bundle['cfg'].fold),
            router_learning_rates=[3e-4, 3e-5] if args.mode == 'joint' else None,
            historical_bridge=bundle['bridge'])
        summary.update(identity=identity, config=identity['config'], environment=bundle['environment'], gpu=bundle['gpu'],
            initial_backbone_sha256=bundle['initial_backbone_sha256'],
            protected_weights_excluded_from_initialization=True, anchor_parity_gate=bundle['bridge'],
            mask_hashes=hashes, total_parameters=sum(p.numel() for p in model.parameters()),
            router_parameters=sum(p.numel() for p in model.router.parameters()) if args.mode == 'joint' else 0,
            status='running', historical_80p747_is_not_paired_score=True)
        h.write_json(args.output / 'config.json', identity['config'])
        h.write_json(summary_path, summary)
        reset_training_stochasticity(66, bundle['cfg'].fold)
        torch.cuda.reset_peak_memory_stats()
        if args.phase == 'smoke':
            before = h.backbone_digest(model)
            row = train_epoch(bundle, model, optimizer, 0, args.mode, max_batches=3)
            if h.backbone_digest(model) == before:
                raise RuntimeError('Smoke made no trainable parameter update')
            if args.mode == 'joint' and not row['gradients']['router']['nonzero']:
                raise RuntimeError('Smoke did not reach Router parameters')
            metrics, _ = evaluate(bundle, model, args.mode, 'validation', masks['validation']['0.3'], max_batches=1)
            state = checkpoint_payload(model, optimizer, epoch=1, identity=identity, history=[row], best_key=None)
            h.atomic_torch_save(args.output / 'smoke.pt', state)
            stored = torch.load(args.output / 'smoke.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, stored, identity)
            if not all(torch.equal(value.cpu(), stored['model'][name]) for name, value in model.state_dict().items()):
                raise RuntimeError('Complete model checkpoint reload failed')
            summary.update(smoke=row, smoke_validation=metrics, smoke_only=True, full_checkpoint_reloaded=True)
        else:
            start, history, best_key = 0, [], None
            if args.resume:
                state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
                restore(model, optimizer, state, identity)
                start, history, best_key = state['epoch'], state['history'], state['best_key']
            for epoch in range(start, 150):
                set_epoch_lrs(optimizer, epoch)
                row = train_epoch(bundle, model, optimizer, epoch, args.mode)
                validation = {rate: evaluate(bundle, model, args.mode, 'validation', masks['validation'][rate])[0] for rate in RATES}
                key = selection_key(validation)
                improved = best_key is None or tuple(key) > tuple(best_key)
                if improved:
                    best_key = key
                row.update(validation=validation, validation_key=key)
                history.append(row)
                state = checkpoint_payload(model, optimizer, epoch=epoch + 1, identity=identity, history=history, best_key=best_key)
                if improved:
                    h.atomic_torch_save(args.output / 'best.pt', state)
                h.atomic_torch_save(args.output / 'last.pt', state)
                h.write_json(args.output / 'history.json', history)
                summary.update(epoch=epoch + 1, best_validation_key=best_key, seconds=time.monotonic() - tick,
                    peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
                h.write_json(summary_path, summary)
                print(json.dumps(dict(event='epoch', epoch=epoch + 1, task_MSE=row['task_MSE'],
                    validation=key, best=best_key, learning_rates=row['learning_rates'])), flush=True)
            state = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            output = args.output / 'evaluation'
            output.mkdir(exist_ok=True)
            summary.update(selected_epoch=state['epoch'], evaluation={})
            for rate in RATES:
                metrics, arrays = evaluate(bundle, model, args.mode, 'test', masks['test'][rate])
                artifact = output / ('random_' + rate.replace('.', 'p') + '.npz')
                np.savez_compressed(artifact, **arrays)
                metrics['output_sha256'] = h.sha(artifact)
                summary['evaluation'][rate] = metrics
                h.write_json(summary_path, summary)
            summary['aggregate'] = aggregate(summary['evaluation'])
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
