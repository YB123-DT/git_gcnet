"""Six authorized V3 connection runs, reusing the accepted scalar protocol."""
import argparse
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from experiments.osram_mosi_v3_capacity_20261022.worker import GradientCapture, source_catalog as prior_catalog
from experiments.osram_mosei_v3_capacity_20261022.worker import digest_without, same_state
from experiments.osram_source_relation_attention_20261021.worker import (
    REFERENCES, RATES, settings, set_learning_rates, check_training_parity,
)

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('head-preserving', 'source-slots', 'flat-hidden')
# Only reviewed constructor additions may fill fields missing in old references.
# Never use TrainConfig(**old) to silently normalize an arbitrary schema change.
REVIEWED_SCHEMA_DEFAULTS = dict(osram_adapter_hidden_dim=0,
    osram_source_role=False, osram_source_role_mode='residual',
    osram_source_relation=False, osram_source_relation_layout='source-mean',
    osram_source_role_v3=False, osram_source_role_v3_dim=64,
    osram_source_role_v3_pair_hidden=128, osram_source_role_v3_mode='residual',
    osram_source_role_v3_variant='standard')


def candidate_config(cfg, variant):
    if variant not in VARIANTS:
        raise ValueError('Only the three approved V3 connection variants')
    spec = settings(cfg.dataset)
    required = dict(fold=1, seed=66, epochs=spec['epochs'], learning_rate=spec['first_lr'],
        weight_decay=1e-5, batch_size=32, optimizer='adam', lr_schedule='constant',
        backbone_type='osram', missing_mask_protocol='gcnet-random-mask-v1',
        training_objective='emotion-only', mosi_task_mode='regression',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
        evaluate_test=True, osram_readout_fusion='flat', osram_output_dim=1600,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64, latent_dim=256,
        osram_bidirectional=False, osram_write_step=.6, osram_adapter_hidden_dim=0,
        osram_source_role_v3=False, osram_source_role_v3_variant='standard',
        osram_source_role_v3_mode='residual', osram_source_role_v3_dim=64,
        osram_source_role_v3_pair_hidden=128, osram_source_role=False,
        osram_source_relation=False, osram_tasr=False, osram_meaningful_block='none',
        osram_relation_block=False, osram_post_grn=False, osram_base_first_gap=False,
        osram_memory_only_adapter=False, osram_history_recency=False,
        osram_causal_gru_history=False, osram_history_query_adapter=False,
        osram_history_input_gate=False, osram_local_evidence_gate=False,
        osram_hierarchical_evidence_gate=False, osram_local_skip_gate=False,
        osram_disable_local_skip=False, osram_decision_correction=False,
        osram_gap_increment_filter=False, osram_ced_block=False,
        paired_history_views=False, completion_path='none', classification_completion=False,
        initial_backbone_checkpoint=None, train_rate_mode='cyclic', fixed_missing_rate=None,
        task_regression_loss='mse')
    for name, expected in required.items():
        if getattr(cfg, name) != expected:
            raise ValueError('Accepted Flat config changed: ' + name)
    if tuple(cfg.train_missing_rates) != tuple(i / 10 for i in range(8)):
        raise ValueError('Accepted Flat config changed: train_missing_rates')
    return replace(cfg, osram_source_role_v3=True, osram_source_role_v3_variant=variant)


def validate_reference_config(cfg, old):
    expected = dict(old)
    for name, default in REVIEWED_SCHEMA_DEFAULTS.items():
        expected.setdefault(name, default)
    if expected['osram_source_role_v3'] or expected['osram_source_role_v3_variant'] != 'standard':
        raise ValueError('Reference must be the accepted original Flat')
    expected.update(osram_source_role_v3=True,
                    osram_source_role_v3_variant=cfg.osram_source_role_v3_variant)
    actual = json.loads(json.dumps(asdict(cfg)))
    if expected != actual:
        changed = sorted(name for name in set(expected) | set(actual)
                         if expected.get(name) != actual.get(name) or (name in expected) != (name in actual))
        raise ValueError('Unexpected config differences beyond V3 switch/variant: ' + ', '.join(changed))


def smoke_limits(dataset):
    settings(dataset)
    # Original MOSI batch32 yields two updates/epoch; MOSEI needs four batches.
    return (2, 2) if dataset == 'CMUMOSI' else (1, 4)


def validate_gradients(module, observed):
    groups = tuple(module.expected_gradient_groups()) + ('input.source_base', 'input.source_gap')
    if len(groups) < 3:
        raise RuntimeError('Variant omitted its gradient contract')
    missing = [group for group in groups
               if not any(value and name.startswith(group) for name, value in observed.items())]
    if missing:
        raise RuntimeError('Missing finite nonzero V3 gradient groups: ' + ', '.join(missing))


def source_catalog(prior_sources):
    paths = set(prior_catalog(prior_sources))
    paths.update('experiments/osram_v3_connection_screen_20261022/' + name
                 for name in ('worker.py', 'run.py', 'report.py'))
    paths.update(('experiments/osram_mosei_gcnet_router_20261020/worker.py',
                  'experiments/osram_meld_flat_source_combinations_20261013/evaluate.py'))
    return sorted(paths)


def setup(dataset, uuid, output, variant):
    import torch
    from gcnet_missing_m3.model import MissingM3GraphModel
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import testbest_config
    if 'osram_source_role_v3_variant' not in inspect.signature(MissingM3GraphModel).parameters:
        raise RuntimeError('The shared fresh factory cannot forward the new variant')
    if dataset == 'CMUMOSI':
        from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as old
        bundle = old.setup(uuid, output, 66)
        original_cfg = replace(testbest_config(bundle['cfg']), learning_rate=3e-4)
    else:
        from experiments.osram_mosei_gcnet_router_20261020 import worker as old
        bundle = old.setup(uuid, output, 66)
        original_cfg = bundle['cfg']
    cfg = candidate_config(original_cfg, variant)
    del bundle['model']
    device = torch.device('cuda:0')
    baseline = old.fresh_backbone(original_cfg, bundle['dimensions'], device).eval()
    flat_rng = h.rng_state()
    if h.backbone_digest(baseline) != bundle['initial_backbone_sha256']:
        raise RuntimeError('Fresh Flat initialization changed')
    model = old.fresh_backbone(cfg, bundle['dimensions'], device)
    require_same_rng(flat_rng, h.rng_state())
    if digest_without(model, ('osram.source_role_v3.',)) != bundle['initial_backbone_sha256']:
        raise RuntimeError('Variant changed original Flat parameters')
    if model.osram.source_role_v3.variant != variant:
        raise RuntimeError('Fresh factory did not apply the requested variant')
    bundle.update(cfg=cfg, model=model)
    return bundle, baseline


def _verify_masks(current, saved):
    if current.keys() != saved.keys():
        raise ValueError('Saved mask splits differ')
    for split, rates in current.items():
        if rates.keys() != saved[split].keys():
            raise ValueError('Saved mask rates differ')
        for rate, bank in rates.items():
            old_bank = saved[split][rate]
            if len(bank) != len(old_bank):
                raise ValueError('Saved mask batch counts differ')
            for actual, expected in zip(bank, old_bank):
                if actual.keys() != expected.keys():
                    raise ValueError('Saved mask metadata differs')
                for name in actual:
                    np.testing.assert_array_equal(actual[name], expected[name])


def sync_restored_summary(summary, *, start, best):
    """last.pt is authoritative when its epoch was saved before SUMMARY.json."""
    summary.update(epoch=start, best_by_rate=best)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(REFERENCES), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--seed', type=int, choices=(66,), default=66)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    if args.resume and (args.phase != 'train' or not (args.output / 'last.pt').is_file()):
        raise ValueError('Explicit resume requires a full training last.pt')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent output required')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path, tick = args.output / 'SUMMARY.json', time.monotonic()
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get('status') == 'complete':
        raise ValueError('Completed experiments cannot repeat')
    summary = dict(previous, status='preflight', dataset=args.dataset, variant=args.variant,
        seed=args.seed, source_commit=args.source_commit, phase=args.phase,
        pid=os.getpid(), hostname=socket.gethostname(), gpu_uuid=args.gpu_uuid,
        command=[sys.executable] + sys.argv, label='INTERNAL TEST-ORACLE SCREEN')
    capture = scan_hook = step_hook = None
    try:
        torch.set_num_threads(1)
        from experiments.osram_source_relation_attention_20261021.run import gpu_free
        gpu_free(args.gpu_uuid)  # Reject host GPU4 before initializing CUDA.
        torch.cuda.reset_peak_memory_stats()
        h.write_json(path, summary)
        bundle, baseline = setup(args.dataset, args.gpu_uuid, args.output, args.variant)
        cfg, model = bundle['cfg'], bundle['model']
        module = model.osram.source_role_v3
        masks = {'test': scalar.build_masks(bundle, 'test')}
        hashes = {'test': {rate: hashlib.sha256(b''.join(record['availability'].numpy().tobytes()
            for record in bank)).hexdigest() for rate, bank in masks['test'].items()}}
        reference, spec = REFERENCES[args.dataset], settings(args.dataset)
        prior = json.loads((reference / 'SUMMARY.json').read_text())
        if prior.get('status') != 'complete' or prior.get('epoch') != spec['epochs']:
            raise ValueError('Completed same-budget Flat66 reference required')
        validate_reference_config(cfg, prior['identity']['config'])
        if prior['initial_backbone_sha256'] != bundle['initial_backbone_sha256']:
            raise ValueError('Accepted reference initialization differs')
        for name in ('split', 'seed_protocol', 'data_files'):
            if prior['identity'][name] != bundle[name]:
                raise ValueError('Accepted reference differs: ' + name)
        old_hashes = prior['identity']['mask_hashes']
        if hashes['test'] != old_hashes.get('test', old_hashes):
            raise ValueError('Accepted reference exact test masks differ')
        reference_history = json.loads((reference / 'history.json').read_text())
        if len(reference_history) != spec['epochs']:
            raise ValueError('Complete reference training history required')
        # Compare complete model predictions, including the original Flat output.
        # This is a fresh zero-final-layer Flat, never an already-trained Flat.
        for rate in RATES:
            _, expected = scalar.evaluate(bundle, baseline, 'flat', 'test', masks['test'][rate], max_batches=1)
            _, actual = scalar.evaluate(bundle, model, 'flat', 'test', masks['test'][rate], max_batches=1)
            for name in expected:
                np.testing.assert_array_equal(actual[name], expected[name])
        del baseline, expected, actual
        identity = dict(source_commit=args.source_commit,
            source_hashes={name: h.sha(ROOT / name) for name in source_catalog(prior['identity']['source_hashes'])},
            config=asdict(cfg), dataset=args.dataset, seed=66, variant=args.variant,
            split=bundle['split'], seed_protocol=bundle['seed_protocol'], mask_hashes=hashes,
            data_files=bundle['data_files'], data_manifest=bundle['data_manifest'],
            dataset_manifest_sha256=bundle['data_manifest_sha256'], environment=bundle['environment'],
            initialization='fresh original Flat factory; no pretrained initialization',
            original_initialization_and_rng_exact=True,
            initial_model_predictions_exact=dict(rates=list(RATES), test_batches_per_rate=1),
            learning_rates=spec,
            reference=dict(path=str(reference), summary_sha256=h.sha(reference / 'SUMMARY.json'),
                history_sha256=h.sha(reference / 'history.json'), masks_exact=True,
                config_only_v3_switch_and_variant_changed=True),
            selection='per-rate TEST weighted_f1; strict improvement, earliest tie')
        optimizer = scalar.make_optimizer(model, cfg, 'flat')
        if args.resume:
            _verify_masks(masks, torch.load(args.output / 'fixed_masks.pt', map_location='cpu', weights_only=False))
        else:
            torch.save(masks, args.output / 'fixed_masks.pt')
        summary.update(status='running', identity=identity, config=asdict(cfg),
            environment=bundle['environment'], gpu=bundle['gpu'],
            fresh_flat_full_model_predictions_exact=True,
            initial_backbone_sha256=bundle['initial_backbone_sha256'],
            parameters=dict(total=sum(p.numel() for p in model.parameters()),
                flat_adapter=sum(p.numel() for p in model.osram.emotion_adapter.parameters()),
                v3=sum(p.numel() for p in module.parameters())))
        h.write_json(args.output / 'config.json', asdict(cfg))
        reset_training_stochasticity(66, cfg.fold)
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            if len(history) != start or not 0 <= start <= cfg.epochs:
                raise ValueError('Invalid full-state epoch/history')
            selected.validate_selected(args.output, best)
            for epoch, row in enumerate(history):
                check_training_parity(row, reference_history[epoch])
            sync_restored_summary(summary, start=start, best=best)
            h.write_json(path, summary)
        cumulative = {}
        for row in history:
            cumulative.update(row.get('v3_gradient_paths', {}))
        updates = [sum(row['optimizer_steps'] for row in history)]
        if updates[0] >= 4:
            validate_gradients(module, cumulative)
        else:
            capture = GradientCapture(module)
            capture.observed.update(cumulative)

            def after_step(*unused):
                updates[0] += 1
                if updates[0] == 4:
                    cumulative.update(capture.observed)
                    validate_gradients(module, cumulative)
                    capture.remove()
            step_hook = optimizer.register_step_post_hook(after_step)
        scans = [0]
        scan_hook = model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0, scans[0] + 1))
        initial_original = digest_without(model, ('osram.source_role_v3.',))
        initial_added = h.backbone_digest(module)
        epochs, batches = smoke_limits(args.dataset)
        epoch_range = range(epochs) if args.phase == 'smoke' else range(start, cfg.epochs)
        for epoch in epoch_range:
            scans[0] = 0
            set_learning_rates(optimizer, args.dataset, epoch)
            instrumented = capture is not None and bool(capture.handles)
            row = scalar.train_epoch(bundle, model, optimizer, epoch, 'flat',
                                     max_batches=batches if args.phase == 'smoke' else None)
            if scans[0] != row['optimizer_steps']:
                raise RuntimeError('Exactly one CHM scan per training batch required')
            if args.phase == 'train' or row['optimizer_steps'] == reference_history[epoch]['optimizer_steps']:
                check_training_parity(row, reference_history[epoch])
                row['reference_training_parity'] = 'exact mask trace, update count, and LR'
            else:
                row['reference_training_parity'] = 'limited smoke prefix; full-epoch parity checked in formal train'
            if capture is not None:
                cumulative.update(capture.observed)
            if step_hook is not None and updates[0] >= 4:
                step_hook.remove()
                step_hook = None
            from experiments.osram_coupled_source_role_20261020.worker import serial
            row.update(actual_memory_scans=scans[0], v3_gradient_instrumented=instrumented,
                v3_gradient_paths=dict(cumulative) if instrumented else {},
                v3_diagnostics={name: serial(value) for name, value in module.last_diagnostics.items()
                                if not name.endswith('attention_weights')})
            evaluated = {rate: scalar.evaluate(bundle, model, 'flat', 'test', masks['test'][rate],
                max_batches=1 if args.phase == 'smoke' else None) for rate in RATES}
            row['test'] = {rate: value[0] for rate, value in evaluated.items()}
            history.append(row)
            improved = selected.persist_epoch(args.output, model, optimizer, epoch=epoch + 1,
                identity=identity, history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch + 1, best_by_rate=best, v3_gradient_paths=cumulative,
                v3_gradient_checked_after_updates=4 if updates[0] >= 4 else None,
                v3_gradient_hooks_removed=not capture or not capture.handles,
                seconds=float(previous.get('seconds', 0)) + time.monotonic() - tick,
                peak_memory_mib=max(float(previous.get('peak_memory_mib', 0)),
                                    torch.cuda.max_memory_reserved() / 2**20),
                peak_memory_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_memory_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            h.write_json(path, summary)
            print(json.dumps(dict(event='epoch', dataset=args.dataset, variant=args.variant,
                epoch=epoch + 1, task_MSE=row['task_MSE'], improved_rates=improved,
                test_WF1={rate: value[0]['router']['WF1'] for rate, value in evaluated.items()})), flush=True)
        if args.phase == 'smoke':
            validate_gradients(module, cumulative)
            steps = sum(row['optimizer_steps'] for row in history)
            if steps < 4 or initial_added == h.backbone_digest(module):
                raise RuntimeError('Smoke requires four actual updates and changed V3 parameters')
            if initial_original == digest_without(model, ('osram.source_role_v3.',)):
                raise RuntimeError('Smoke did not update original Flat parameters')
            state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
            before_metrics, before = scalar.evaluate(bundle, model, 'flat', 'test', masks['test']['0.3'], max_batches=1)
            restore(model, optimizer, state, identity)
            same_state(model.state_dict(), state['model'])
            same_state(optimizer.state_dict(), state['optimizer'])
            require_same_rng(state['rng'], h.rng_state())
            selected.validate_selected(args.output, best)
            after_metrics, after = scalar.evaluate(bundle, model, 'flat', 'test', masks['test']['0.3'], max_batches=1)
            if before_metrics != after_metrics:
                raise RuntimeError('Restored checkpoint metrics differ')
            for name in before:
                np.testing.assert_array_equal(before[name], after[name])
            summary.update(smoke_only=True, full_checkpoint_reloaded=True,
                optimizer_rng_prediction_reload_exact=True, smoke_optimizer_steps=steps)
        else:
            summary['evaluation'] = selected.publish_selected(args.output, best)
            summary['aggregate'] = scalar.aggregate(summary['evaluation'])
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT / name) != digest:
                raise RuntimeError('Immutable source changed: ' + name)
        summary['status'] = 'complete'
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        if capture is not None:
            capture.remove()
        for handle in (scan_hook, step_hook):
            if handle is not None:
                handle.remove()
        summary['seconds'] = float(previous.get('seconds', 0)) + time.monotonic() - tick
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
