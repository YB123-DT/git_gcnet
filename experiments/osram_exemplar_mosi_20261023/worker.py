"""MOSI66 TRAIN-exemplar retrieval: two finite from-scratch task-only arms."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace, MISSING
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(f'{i / 10:.1f}' for i in range(8))
VARIANTS = ('feature-only', 'label-aware')
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003')


def set_learning_rates(optimizer, epoch):
    if not 0 <= epoch < 150:
        raise ValueError('Only the approved MOSI150 budget')
    for group in optimizer.param_groups:
        if group['name'] != 'backbone':
            raise ValueError('One joint original-LR optimizer group required')
        group['lr'] = 3e-4 if epoch < 100 else 3e-5


def check_training_parity(actual, reference):
    for key in ('epoch', 'mask_trace_sha256', 'optimizer_steps'):
        if actual[key] != reference[key]:
            raise ValueError('Reference training protocol differs: ' + key)
    if len(actual['learning_rates']) != len(reference['learning_rates']) or any(
            not math.isclose(a, b, rel_tol=1e-14, abs_tol=0.)
            for a, b in zip(actual['learning_rates'], reference['learning_rates'])):
        raise ValueError('Reference learning rates differ')


def validate_reference_config(current, reference, defaults):
    """Permit newly declared default fields, never changing existing settings."""
    expected = dict(reference)
    for key in set(current) - set(reference):
        if key not in defaults or current[key] != defaults[key]:
            raise ValueError('Non-default new config field: ' + key)
        expected[key] = defaults[key]
    if json.loads(json.dumps(expected)) != json.loads(json.dumps(current)):
        raise ValueError('Original Flat config differs')


def build_train_bank(bundle):
    """Read TRAIN indices directly; never advance the shuffled loader or RNG."""
    import torch
    from gcnet_missing_m3.exemplar import TrainSupportBank
    loader, trainer = bundle['train_loader'], bundle['trainer']
    features, labels, ids, indices = [], [], [], []
    for index in loader.sampler.indices:
        raw = loader.dataset.collate_fn([loader.dataset[int(index)]])
        data = trainer._move_batch(raw, torch.device('cpu'))
        full = torch.ones(data[0].shape[:2] + (3,), dtype=data[0].dtype)
        view = trainer._prepare_view_from_primary_masks(data, full, full, bundle['dimensions'])
        valid = view['umask'].T.bool()
        features.append(view['complete'][valid].detach().cpu())
        labels.append(view['labels'].T[valid].detach().cpu())
        count = int(valid.sum())
        ids.extend([str(data[-1][0])] * count)
        indices.extend(range(count))
    if not features:
        raise ValueError('Empty TRAIN support bank')
    return TrainSupportBank(torch.cat(features), torch.cat(labels), ids, indices)


def setup(uuid, output, variant):
    import torch
    from gcnet_missing_m3.exemplar import ExemplarAugmentedModel
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as original
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import testbest_config
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if variant not in VARIANTS:
        raise ValueError('Unapproved retrieval arm')
    bundle = original.setup(uuid, output, 66)
    cfg = replace(testbest_config(bundle['cfg']), learning_rate=3e-4)
    if (cfg.dataset != 'CMUMOSI' or cfg.batch_size != 32 or cfg.epochs != 150
            or cfg.weight_decay != 1e-5 or cfg.dropout != 0.5 or cfg.missing_mask_protocol != 'gcnet-random-mask-v1'
            or cfg.osram_readout_fusion != 'flat' or cfg.training_objective != 'emotion-only'):
        raise ValueError('Require unchanged accepted MOSI66 protocol')
    bank = build_train_bank(bundle)
    original_digest = h.backbone_digest(bundle['model'])
    model = ExemplarAugmentedModel(bundle['model'], bank, use_labels=variant == 'label-aware', seed=66)
    model.to(torch.device('cuda:0')).requires_grad_(True).train()
    if h.backbone_digest(model.backbone) != original_digest:
        raise RuntimeError('Added retrieval altered original initialization')
    bundle.update(model=model, cfg=cfg, bank=bank)
    return bundle


def result_for(model, view):
    import torch
    prediction = model(view)
    # The historical selection publisher expects this stable scalar schema.
    # Its old router/uniform labels are compatibility aliases, NOT new heads.
    weights = prediction.new_zeros(*prediction.shape[:2], 3)
    weights[..., 0] = 1
    return dict(prediction=prediction, candidates=prediction.unsqueeze(-2).expand(
        *prediction.shape[:2], 3, 1), weights=weights,
        eligible=torch.zeros(prediction.shape[:2], dtype=torch.bool, device=prediction.device))


def train_epoch(bundle, optimizer, epoch, max_batches=None):
    import torch
    from gcnet_missing_m3.mixed_rate import BalancedBatchRateSchedule
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    model, cfg, trainer, loader = (bundle[k] for k in ('model', 'cfg', 'trainer', 'train_loader'))
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
        optimizer.zero_grad(set_to_none=True)
        prediction = model(view)
        loss = scalar.scalar_loss(trainer, cfg, prediction, view)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite original scalar MSE')
        loss.backward()
        gradients = {}
        for name, module in (('backbone', model.backbone), ('retrieval', model.retrieval)):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Nonfinite gradient in ' + name)
            gradients[name] = dict(connected=len(grads), nonzero=sum(bool(g.ne(0).any()) for g in grads))
        if not gradients['backbone']['nonzero'] or not gradients['retrieval']['nonzero']:
            raise RuntimeError('Original task gradient must reach both original and new parameters')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite joint gradient norm')
        optimizer.step()
        n = int(view['umask'].sum())
        rows.append(dict(task_MSE=float(loss.detach()), N=n, gradients=gradients, gradient_norm=float(norm)))
        digest.update(json.dumps([str(v) for v in data[-1]], separators=(',', ':')).encode())
        digest.update(view['availability'].detach().cpu().numpy().tobytes())
    if not rows:
        raise RuntimeError('No training batches')
    return dict(epoch=epoch + 1, batches=len(rows), optimizer_steps=len(rows), N=sum(r['N'] for r in rows),
        task_MSE=sum(r['task_MSE'] * r['N'] for r in rows) / sum(r['N'] for r in rows),
        gradient_norm_max=max(r['gradient_norm'] for r in rows),
        gradients={group: {key: max(r['gradients'][group][key] for r in rows)
                          for key in ('connected', 'nonzero')} for group in ('backbone', 'retrieval')},
        mask_trace_sha256=digest.hexdigest(), learning_rates=[g['lr'] for g in optimizer.param_groups])


def evaluate(bundle, bank, max_batches=None):
    import numpy as np
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    model, loader, trainer = bundle['model'], bundle['test_loader'], bundle['trainer']
    loader.sampler.set_epoch(0)
    model.eval()
    pieces = {}
    with h.isolated_rng(66 + 99193), torch.no_grad():
        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            data = trainer._move_batch(raw, next(model.parameters()).device)
            meta = scalar.metadata(data)
            record = bank[index]
            for key, value in meta.items():
                np.testing.assert_array_equal(value, record[key])
            availability = record['availability'].to(data[7].device)
            view = trainer._prepare_view_from_primary_masks(data, availability, availability, bundle['dimensions'])
            result = result_for(model, view)
            values = dict(meta, availability=availability.transpose(0, 1)[data[7].bool()].cpu().numpy())
            for key in ('prediction', 'candidates', 'weights', 'eligible'):
                values[key] = result[key].transpose(0, 1)[data[7].bool()].cpu().numpy()
            if not np.isfinite(values['prediction']).all():
                raise RuntimeError('Nonfinite test prediction')
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
    arrays = {key: np.concatenate(values) for key, values in pieces.items()}
    metrics = scalar.summarize_arrays(arrays, trainer)
    metrics['mask_sha256'] = hashlib.sha256(arrays['availability'].tobytes()).hexdigest()
    return metrics, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--seed', type=int, choices=(66,), default=66)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import numpy as np
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    if args.output.exists() and not args.resume:
        raise FileExistsError('Require fresh independent output')
    if args.resume and args.phase != 'train':
        raise ValueError('Only complete-state training resume supported')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick = args.output / 'SUMMARY.json', time.monotonic()
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError('Completed run must not repeat')
    summary = dict(status='preflight', phase=args.phase, variant=args.variant, dataset='CMUMOSI', seed=66,
        source_commit=args.source_commit, pid=os.getpid(), label='INTERNAL TEST-ORACLE SCREEN')
    scan_hook = None
    try:
        torch.set_num_threads(1)
        bundle = setup(args.gpu_uuid, args.output, args.variant)
        cfg, model = bundle['cfg'], bundle['model']
        masks = {split: scalar.build_masks(bundle, split) for split in ('validation', 'test')}
        hashes = {split: {rate: hashlib.sha256(b''.join(r['availability'].numpy().tobytes()
            for r in bank)).hexdigest() for rate, bank in rates.items()} for split, rates in masks.items()}
        prior = json.loads((REFERENCE / 'SUMMARY.json').read_text())
        reference_history = json.loads((REFERENCE / 'history.json').read_text())
        if prior.get('status') != 'complete' or prior.get('epoch') != 150 or len(reference_history) != 150:
            raise ValueError('Completed accepted original MOSI reference required')
        defaults = {f.name: f.default for f in fields(cfg) if f.default is not MISSING}
        validate_reference_config(asdict(cfg), prior['identity']['config'], defaults)
        for key in ('split', 'seed_protocol', 'data_files'):
            if bundle.get(key) != prior['identity'].get(key):
                raise ValueError('Reference differs: ' + key)
        if bundle['initial_backbone_sha256'] != prior['initial_backbone_sha256']:
            raise ValueError('Reference original initialization differs')
        old_hashes = prior['identity']['mask_hashes']
        if hashes['test'] != old_hashes.get('test', old_hashes):
            raise ValueError('Reference exact test masks differ')
        source_files = tuple(prior['identity']['source_hashes']) + (
            'gcnet_missing_m3/exemplar.py', 'experiments/osram_exemplar_mosi_20261023/worker.py',
            'experiments/osram_exemplar_mosi_20261023/run.py', 'experiments/osram_exemplar_mosi_20261023/report.py')
        identity = dict(source_commit=args.source_commit, source_hashes={p: h.sha(ROOT / p) for p in source_files},
            config=asdict(cfg), seed=66, variant=args.variant, split=bundle['split'],
            seed_protocol=bundle['seed_protocol'], mask_hashes=hashes, data_files=bundle['data_files'],
            support_bank=bundle['bank'].manifest(),
            reference=dict(path=str(REFERENCE), summary_sha256=h.sha(REFERENCE / 'SUMMARY.json')),
            selection='per-rate TEST weighted_f1; strict improvement; earliest ties',
            one_stage=True, inference_requires_train_feature_label_bank=True)
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=cfg.learning_rate, name='backbone')],
                                     weight_decay=cfg.weight_decay)
        torch.save(masks, args.output / 'fixed_masks.pt')
        h.write_json(args.output / 'support_manifest.json', identity['support_bank'])
        h.write_json(args.output / 'config.json', identity['config'])
        summary.update(status='running', identity=identity, config=identity['config'], gpu=bundle['gpu'],
            environment=bundle['environment'], total_parameters=sum(p.numel() for p in model.parameters()),
            retrieval_parameters=sum(p.numel() for p in model.retrieval.parameters()),
            initial_backbone_sha256=bundle['initial_backbone_sha256'], initialization_exact=True,
            reference_test_masks_exact=True)
        h.write_json(path, summary)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        reset_training_stochasticity(66, cfg.fold)
        torch.cuda.reset_peak_memory_stats()
        history, best, start = [], {}, 0
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            history, best, start = state['history'], state['best_by_rate'], state['epoch']
            selected.validate_selected(args.output, best)
        before = h.backbone_digest(model.retrieval)
        scans = [0]
        scan_hook = model.backbone.osram.register_forward_hook(lambda *unused: scans.__setitem__(0, scans[0] + 1))
        for epoch in (range(2) if args.phase == 'smoke' else range(start, 150)):
            set_learning_rates(optimizer, epoch)
            scans[0] = 0
            row = train_epoch(bundle, optimizer, epoch, max_batches=3 if args.phase == 'smoke' else None)
            if scans[0] != row['optimizer_steps']:
                raise RuntimeError('Require exactly one CHM scan per training batch')
            row['actual_memory_scans'] = scans[0]
            if args.phase == 'train':
                check_training_parity(row, reference_history[epoch])
            evaluated = {r: evaluate(bundle, masks['test'][r], max_batches=1 if args.phase == 'smoke' else None)
                         for r in RATES}
            row['test'] = {r: v[0] for r, v in evaluated.items()}
            history.append(row)
            improved = selected.persist_epoch(args.output, model, optimizer, epoch=epoch + 1,
                identity=identity, history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch + 1, best_by_rate=best,
                selected_epoch_by_rate={r: v['epoch'] for r, v in best.items()},
                seconds=time.monotonic() - tick, peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
            h.write_json(path, summary)
            print(json.dumps(dict(event='epoch', variant=args.variant, epoch=epoch + 1,
                task_MSE=row['task_MSE'], improved_rates=improved,
                test_WF1={r: v[0]['router']['WF1'] for r, v in evaluated.items()})), flush=True)
        if args.phase == 'smoke':
            if before == h.backbone_digest(model.retrieval):
                raise RuntimeError('Retrieval parameters did not update')
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            expected, expected_arrays = evaluate(bundle, masks['test']['0.3'], max_batches=1)
            restore(model, optimizer, state, identity)
            if any(not torch.equal(value.cpu(), state['model'][name])
                   for name, value in model.state_dict().items()):
                raise RuntimeError('Full model state did not restore exactly')

            def compare_state(actual, expected):
                if torch.is_tensor(expected):
                    if not torch.equal(actual.cpu(), expected.cpu()):
                        raise RuntimeError('Restored Adam tensor differs')
                elif isinstance(expected, dict):
                    if actual.keys() != expected.keys():
                        raise RuntimeError('Restored Adam keys differ')
                    for key in expected:
                        compare_state(actual[key], expected[key])
                elif isinstance(expected, (list, tuple)):
                    if len(actual) != len(expected):
                        raise RuntimeError('Restored Adam sequence differs')
                    for first, second in zip(actual, expected):
                        compare_state(first, second)
                elif actual != expected:
                    raise RuntimeError('Restored Adam scalar differs')

            compare_state(optimizer.state_dict(), state['optimizer'])
            from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
            require_same_rng(state['rng'], h.rng_state())
            actual, actual_arrays = evaluate(bundle, masks['test']['0.3'], max_batches=1)
            if expected != actual:
                raise RuntimeError('Checkpoint reload changed metrics')
            for key in expected_arrays:
                np.testing.assert_array_equal(expected_arrays[key], actual_arrays[key])
            selected.validate_selected(args.output, best)
            summary.update(smoke_only=True, smoke_optimizer_steps=sum(r['optimizer_steps'] for r in history),
                optimizer_rng_prediction_reload_exact=True, smoke=history[-1])
        else:
            summary['evaluation'] = selected.publish_selected(args.output, best)
            summary['aggregate'] = scalar.aggregate(summary['evaluation'])
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT / name) != digest:
                raise RuntimeError('Immutable source changed: ' + name)
        summary.update(status='complete', seconds=time.monotonic() - tick)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        if scan_hook is not None:
            scan_hook.remove()
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
