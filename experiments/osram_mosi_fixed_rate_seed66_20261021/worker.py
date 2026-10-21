"""One fresh original MOSI Flat per fixed missing rate; no cyclic-rate training."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(i / 10 for i in range(8))
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003')


def rate_key(rate):
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate not in RATES:
        raise ValueError('Only missing rates0.0 through0.7 are authorized')
    return f'{rate:.1f}'


def fixed_config(cfg, rate):
    rate_key(rate)
    required = dict(seed=66, epochs=150, batch_size=32, learning_rate=3e-4,
        weight_decay=1e-5, train_rate_mode='cyclic', training_objective='emotion-only',
        missing_mask_protocol='gcnet-random-mask-v1', osram_readout_fusion='flat',
        osram_meaningful_block='none')
    if any(getattr(cfg, name) != value for name, value in required.items()):
        raise ValueError('Require the accepted MOSI66 cyclic Flat configuration')
    flags = ('osram_source_role', 'osram_source_relation', 'osram_source_role_v3',
             'osram_tasr', 'paired_history_views', 'osram_relation_block')
    if any(getattr(cfg, name) for name in flags) or tuple(cfg.train_missing_rates) != RATES:
        raise ValueError('No extra block, view or altered original rate set')
    return replace(cfg, train_rate_mode='fixed', fixed_missing_rate=float(rate),
                   train_missing_rates=(float(rate),))


def validate_reference_config(cfg, old, rate):
    rate_key(rate)
    expected = dict(old)
    for name, default in dict(osram_source_role=False, osram_source_role_mode='residual',
            osram_source_relation=False, osram_source_relation_layout='source-mean',
            osram_source_role_v3=False).items():
        expected.setdefault(name, default)
    expected.update(train_rate_mode='fixed', fixed_missing_rate=float(rate),
                    train_missing_rates=[float(rate)])
    if expected != json.loads(json.dumps(asdict(cfg))):
        raise ValueError('Unexpected configuration change beyond fixed training rate')


def strict_improvement(previous, metrics):
    if not all(math.isfinite(metrics[k]) for k in ('ACC', 'WF1', 'MSE')):
        raise ValueError('Selected metrics must be finite')
    return previous is None or metrics['WF1'] > previous['metrics']['WF1']


def validate_best(output, best):
    from experiments.osram_tasr_20261018 import runner_helpers as h
    for name in ('checkpoint', 'predictions'):
        if h.sha(output / best[name]) != best[name + '_sha256']:
            raise ValueError('Selected artifact integrity failure: ' + name)


def persist(output, model, optimizer, epoch, identity, history, best, measured, arrays):
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import _atomic_arrays
    from experiments.osram_text_path_router_joint_20261019.run import checkpoint_payload
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if (not np.isfinite(arrays['prediction']).all()
            or hashlib.sha256(arrays['availability'].tobytes()).hexdigest() != measured['mask_sha256']):
        raise ValueError('Nonfinite or mismatched same-call selected predictions')
    improved = strict_improvement(best, measured['router'])
    if improved:
        _atomic_arrays(output / 'predictions_best.npz', arrays)
        best = dict(epoch=epoch, metrics=copy.deepcopy(measured['router']),
            rate=identity['fixed_rate'], checkpoint='best.pt', predictions='predictions_best.npz',
            predictions_sha256=h.sha(output / 'predictions_best.npz'),
            mask_sha256=measured['mask_sha256'], same_evaluation_call=True,
            criterion='same-rate TEST weighted_f1; strict improvement; earliest ties')
    state = checkpoint_payload(model, optimizer, epoch=epoch, identity=identity,
                               history=history, best_key=best['metrics']['WF1'])
    state['config'] = copy.deepcopy(identity['config'])
    if improved:
        state['selection'] = copy.deepcopy(best)
        h.atomic_torch_save(output / 'best.pt', state)
        best['checkpoint_sha256'] = h.sha(output / 'best.pt')
        state.pop('selection')
    state['testbest'] = copy.deepcopy(best)
    h.atomic_torch_save(output / 'last.pt', state)
    h.write_json(output / 'history.json', history)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--seed', type=int, choices=(66,), default=66)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); key = rate_key(args.rate)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as original
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import testbest_config
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosi_router_lr1e5_20261020.worker import set_epoch_lrs
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh fixed-rate output required')
    if args.resume and args.phase != 'train':
        raise ValueError('Only complete-state training resume is supported')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output / 'SUMMARY.json'; tick = time.monotonic(); prepare = None
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError('Completed fixed-rate run cannot silently repeat')
    summary = dict(status='preflight', dataset='CMUMOSI', seed=66, mode='flat',
        fixed_rate=args.rate, phase=args.phase, source_commit=args.source_commit, pid=os.getpid(),
        label='INTERNAL TEST-ORACLE SCREEN; EIGHT INDEPENDENT RATE-SPECIFIC MODELS')
    try:
        torch.set_num_threads(1); h.write_json(path, summary)
        bundle = original.setup(args.gpu_uuid, args.output, 66)
        cyclic = replace(testbest_config(bundle['cfg']), learning_rate=3e-4)
        cfg = fixed_config(cyclic, args.rate); bundle['cfg'] = cfg
        prior = json.loads((REFERENCE / 'SUMMARY.json').read_text())
        if prior['status'] != 'complete' or prior['epoch'] != 150:
            raise ValueError('Completed150-epoch cyclic reference is required')
        validate_reference_config(cfg, prior['identity']['config'], args.rate)
        if prior['initial_backbone_sha256'] != bundle['initial_backbone_sha256']:
            raise ValueError('Original Flat initialization changed')
        for name in ('split', 'seed_protocol'):
            if prior['identity'][name] != bundle[name]:
                raise ValueError('Original ' + name + ' differs')
        if bundle['trainer']._protocol_rates(cfg) != (args.rate,):
            raise ValueError('Fixed train/test rate protocol was not activated')
        model = scalar.wrap_model(bundle['model'], 'flat')
        if sum(p.numel() for p in model.parameters()) != 13509793:
            raise ValueError('Original Flat parameter count changed')
        # Generate the complete original bank in its original RNG order, then
        # select ONE bank; generating only this rate would change its RNG stream.
        banks = scalar.build_masks(bundle, 'test')
        hashes = {r: hashlib.sha256(b''.join(v['availability'].numpy().tobytes()
                  for v in bank)).hexdigest() for r, bank in banks.items()}
        if hashes != prior['identity']['mask_hashes']['test']:
            raise ValueError('Exact cyclic reference test masks differ')
        bank = banks[key]
        if not args.resume:
            torch.save({'test': {key: bank}}, args.output / 'fixed_masks.pt')
        added = ('experiments/osram_mosi_fixed_rate_seed66_20261021/worker.py',
                 'experiments/osram_mosi_fixed_rate_seed66_20261021/run.py',
                 'experiments/osram_mosi_fixed_rate_seed66_20261021/report.py')
        sources = tuple(dict.fromkeys(tuple(prior['identity']['source_hashes']) + added))
        identity = dict(source_commit=args.source_commit, source_hashes={p: h.sha(ROOT / p) for p in sources},
            config=asdict(cfg), fixed_rate=args.rate, seed=66, split=bundle['split'],
            seed_protocol=bundle['seed_protocol'], mask_protocol=cfg.missing_mask_protocol,
            test_mask_hashes=hashes, selected_test_mask_hash=hashes[key],
            dataset_manifest_sha256=bundle['data_manifest_sha256'], data_manifest=bundle['data_manifest'],
            data_files=bundle['data_files'], initialization='fresh original factory; legacy parity weights discarded',
            cyclic_reference=dict(path=str(REFERENCE), summary_sha256=h.sha(REFERENCE / 'SUMMARY.json')),
            original_initialization_sha256=bundle['initial_backbone_sha256'],
            learning_rates=dict(first100=3e-4, final50=3e-5),
            selection='same-rate TEST weighted_f1; strict improvement; earliest ties')
        optimizer = scalar.make_optimizer(model, cfg, 'flat')
        summary.update(status='running', identity=identity, config=asdict(cfg), environment=bundle['environment'],
            gpu=bundle['gpu'], total_parameters=13509793, initial_backbone_sha256=bundle['initial_backbone_sha256'])
        h.write_json(args.output / 'config.json', asdict(cfg)); h.write_json(path, summary)
        reset_training_stochasticity(66, cfg.fold); torch.cuda.reset_peak_memory_stats()
        best, history, start = None, [], 0
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            best, history, start = state['testbest'], state['history'], state['epoch']
            validate_best(args.output, best)
        prepare = bundle['trainer']._prepare_view; observations = []
        def observed_prepare(data, schedule, epoch, dimensions):
            if schedule.requested_missing_rate != args.rate:
                raise RuntimeError('A different rate entered fixed-rate training')
            view = prepare(data, schedule, epoch, dimensions)
            valid = view['umask'].T.bool(); masks = view['availability'].detach()
            active = masks[valid]
            if (not active.numel() or not ((active == 0) | (active == 1)).all()
                    or not active.bool().any(-1).all() or masks[~valid].ne(0).any()):
                raise RuntimeError('Invalid GCNet observed/padding masks')
            if args.rate == 0 and not active.eq(1).all():
                raise RuntimeError('Rate0 must observe all modalities')
            if args.rate == .7 and not active.sum(-1).eq(1).all():
                raise RuntimeError('GCNet rate.7 must be one-hot')
            code = (active.long() * active.new_tensor([1, 2, 4], dtype=torch.long)).sum(-1)
            observations.append(dict(N=len(active), observed=int(active.sum()),
                patterns=torch.bincount(code, minlength=8).cpu().tolist()))
            return view
        bundle['trainer']._prepare_view = observed_prepare
        initial = h.backbone_digest(model)
        for epoch in (range(1) if args.phase == 'smoke' else range(start, 150)):
            observations.clear()
            set_epoch_lrs(optimizer, epoch, backbone_lr=3e-4, final_backbone_lr=3e-5)
            row = scalar.train_epoch(bundle, model, optimizer, epoch, 'flat')
            if len(observations) != row['optimizer_steps'] or row['optimizer_steps'] != 2:
                raise RuntimeError('Expected exactly2 same-rate original training batches')
            n = sum(v['N'] for v in observations)
            if n != row['N']:
                raise RuntimeError('Observed mask/utterance count mismatch')
            row.update(rate_batch_counts={key: len(observations)},
                actual_missing_fraction=1 - sum(v['observed'] for v in observations) / (3 * n),
                availability_pattern_counts={tag: sum(v['patterns'][i] for v in observations)
                    for i, tag in enumerate(('NONE', 'A', 'T', 'AT', 'V', 'AV', 'TV', 'ATV'))})
            measured, arrays = scalar.evaluate(bundle, model, 'flat', 'test', bank)
            row['test'] = {key: measured}; history.append(row)
            best = persist(args.output, model, optimizer, epoch + 1, identity, history, best, measured, arrays)
            summary.update(epoch=epoch + 1, testbest=best, seconds=time.monotonic() - tick,
                           peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
            h.write_json(path, summary)
            h.write_json(args.output / 'EPOCH_STATUS.json', dict(epoch=epoch + 1,
                fixed_rate=args.rate, peak_memory_mib=summary['peak_memory_mib'],
                rate_batch_counts=row['rate_batch_counts'], optimizer_steps=row['optimizer_steps']))
            print(json.dumps(dict(event='epoch', rate=args.rate, epoch=epoch + 1,
                task_MSE=row['task_MSE'], learning_rates=row['learning_rates'],
                test=measured['router'], best_epoch=best['epoch'],
                actual_missing_fraction=row['actual_missing_fraction'])), flush=True)
        validate_best(args.output, best)
        if args.phase == 'smoke':
            if h.backbone_digest(model) == initial:
                raise RuntimeError('Original Flat failed to update')
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
            require_same_rng(state['rng'], h.rng_state())
            after, replay = scalar.evaluate(bundle, model, 'flat', 'test', bank)
            if after != measured:
                raise RuntimeError('Checkpoint metric replay differs')
            for name in arrays:
                np.testing.assert_array_equal(arrays[name], replay[name])
            summary.update(smoke_only=True, full_checkpoint_reloaded=True, restoration_predictions_exact=True)
        summary['evaluation'] = {key: copy.deepcopy(best['metrics'])}
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT / name) != digest:
                raise RuntimeError('Source changed during training: ' + name)
        summary.update(status='complete', seconds=time.monotonic() - tick)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic() - tick)
        raise
    finally:
        if prepare is not None:
            bundle['trainer']._prepare_view = prepare
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
