"""Finite MOSI capacity arms under their protected per-seed training protocol."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.osram_mosei_v3_capacity_20261022.worker import digest_without, same_state
from experiments.osram_source_relation_attention_20261021.worker import (
    REFERENCES, RATES, settings, set_learning_rates, check_training_parity,
)
from experiments.osram_source_role_v3_20261021.worker import gradient_check_epoch, validate_gradients

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REFERENCES['CMUMOSI']
MODES = {'f512_plain': (512, 0, 0), 'f512_v128': (512, 128, 256),
         'f512_v256': (512, 256, 512), 'f1024_plain': (1024, 0, 0),
         'f1024_v128': (1024, 128, 256), 'f1024_v256': (1024, 256, 512)}


def candidate_config(cfg, mode):
    if mode not in MODES:
        raise ValueError('Only the six approved MOSI capacity arms')
    required = dict(dataset='CMUMOSI', fold=1, seed=66, epochs=150, learning_rate=3e-4,
        weight_decay=1e-5, batch_size=32, optimizer='adam', lr_schedule='constant',
        backbone_type='osram', missing_mask_protocol='gcnet-random-mask-v1',
        training_objective='emotion-only', mosi_task_mode='regression',
        checkpoint_selection='test-oracle-per-rate', checkpoint_selection_metric='weighted_f1',
        evaluate_test=True, osram_readout_fusion='flat', osram_output_dim=1600,
        osram_num_heads=8, osram_key_dim=64, osram_value_dim=64,
        osram_bidirectional=False, osram_write_step=.6, osram_source_role_v3=False,
        osram_source_role=False, osram_source_relation=False, osram_tasr=False,
        osram_meaningful_block='none', paired_history_views=False,
        train_rate_mode='cyclic', fixed_missing_rate=None, task_regression_loss='mse',
        osram_adapter_hidden_dim=0)
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError('Original MOSI config changed: ' + key)
    if tuple(cfg.train_missing_rates) != tuple(i/10 for i in range(8)):
        raise ValueError('Original MOSI config changed: train_missing_rates')
    flat, width, pair = MODES[mode]
    return replace(cfg, osram_adapter_hidden_dim=flat, osram_source_role_v3=bool(width),
        osram_source_role_v3_dim=width or cfg.osram_source_role_v3_dim,
        osram_source_role_v3_pair_hidden=pair or cfg.osram_source_role_v3_pair_hidden)


def smoke_epochs():
    return gradient_check_epoch('CMUMOSI') + 1


class GradientCapture:
    """Cumulative first-two-epoch evidence, detached after validation."""
    def __init__(self, module):
        self.observed = {}
        self.handles = []
        for name, parameter in module.named_parameters():
            self.handles.append(parameter.register_hook(self.capture(name)))
        self.handles.append(module.register_forward_pre_hook(self.watch_inputs))

    def capture(self, name):
        def hook(grad):
            import torch
            if not torch.isfinite(grad).all():
                raise RuntimeError('Nonfinite V3 gradient: ' + name)
            self.observed[name] = self.observed.get(name, False) or bool(grad.ne(0).any())
        return hook

    def watch_inputs(self, unused, inputs):
        for name, value in zip(('source_base', 'source_gap'), inputs[1:3]):
            if value.requires_grad:
                self.handles.append(value.register_hook(self.capture('input.' + name)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def source_catalog(prior_sources):
    sources = set(prior_sources)
    for package in ('gcnet_missing_m3', 'gcnet_modality_jepa'):
        sources.update(str(f.relative_to(ROOT)) for f in (ROOT/package).rglob('*.py'))
    sources.update('experiments/osram_mosi_v3_capacity_20261022/' + name
                   for name in ('worker.py', 'run.py', 'report.py'))
    sources.update(('experiments/osram_mosei_v3_capacity_20261022/worker.py',
        'experiments/osram_source_relation_attention_20261021/worker.py',
        'experiments/osram_source_role_v3_20261021/worker.py',
        'experiments/osram_mosi_gcnet_router_3seed_20261019/worker.py',
        'experiments/osram_mosi_gcnet_testbest_20261020/worker.py',
        'experiments/osram_text_path_router_joint_mosi_20261019/run.py',
        'experiments/osram_text_path_router_joint_20261019/run.py',
        'experiments/osram_coupled_source_role_20261020/worker.py',
        'experiments/osram_12_source_evidence_audit_20261019/run.py',
        'experiments/osram_tasr_20261018/runner_helpers.py'))
    return sorted(sources)


def default_reference(seed):
    if seed != 66:
        raise ValueError('Original capacity screen supports only seed66')
    return REFERENCE


def main(argv=None, *, allowed_seeds=(66,), allowed_modes=MODES,
         config_factory=candidate_config, reference_factory=default_reference,
         source_catalog_factory=source_catalog):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=allowed_modes, required=True)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--seed', type=int, choices=allowed_seeds,
                        default=66 if 66 in allowed_seeds else None, required=66 not in allowed_seeds)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    reference = reference_factory(args.seed)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as old
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent output required')
    if args.resume and args.phase != 'train':
        raise ValueError('Only complete-state train resume is supported')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'SUMMARY.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get('status') == 'complete':
        raise ValueError('Completed experiment cannot repeat')
    tick = time.monotonic()
    capture = scan_hook = None
    summary = dict(status='preflight', dataset='CMUMOSI', mode=args.mode, seed=args.seed,
        source_commit=args.source_commit, phase=args.phase, pid=os.getpid(),
        label='INTERNAL TEST-ORACLE SCREEN')
    try:
        torch.set_num_threads(1)
        h.write_json(path, summary)
        bundle = old.setup(args.gpu_uuid, args.output, args.seed)
        from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
        from gcnet_missing_m3.train_gcnet import TrainConfig
        large = bundle.pop('model')
        original_cfg = replace(selected.testbest_config(bundle['cfg']), learning_rate=3e-4)
        cfg = config_factory(original_cfg, args.mode)
        plain_mode = 'f' + str(MODES[args.mode][0]) + '_plain'
        plain = old.fresh_backbone(config_factory(original_cfg, plain_mode),
                                   bundle['dimensions'], torch.device('cuda:0'))
        model = (old.fresh_backbone(cfg, bundle['dimensions'], torch.device('cuda:0'))
                 if cfg.osram_source_role_v3 else plain)
        untouched = ('osram.emotion_adapter.', 'osram.source_role_v3.')
        if digest_without(large, untouched) != digest_without(model, untouched):
            raise RuntimeError('Capacity changed unrelated original initialization')
        plain_initial = h.backbone_digest(plain)
        if digest_without(model, ('osram.source_role_v3.',)) != plain_initial:
            raise RuntimeError('V3 changed same-width Flat initialization')
        del large
        bundle.update(cfg=cfg, model=model)
        masks = {'test': old.build_masks(bundle, 'test')}
        hashes = {'test': {rate: hashlib.sha256(b''.join(row['availability'].numpy().tobytes()
            for row in bank)).hexdigest() for rate, bank in masks['test'].items()}}
        prior = json.loads((reference/'SUMMARY.json').read_text())
        if prior.get('status') != 'complete' or prior.get('epoch') != 150:
            raise ValueError(f'Complete original MOSI150 Flat{args.seed} reference required')
        normalize = lambda value: json.loads(json.dumps(asdict(value)))
        if normalize(TrainConfig(**prior['identity']['config'])) != normalize(original_cfg):
            raise ValueError('Current original config differs from accepted reference')
        if bundle['initial_backbone_sha256'] != prior['initial_backbone_sha256']:
            raise ValueError('Original large Flat initialization differs')
        if hashes['test'] != prior['identity']['mask_hashes'].get('test', prior['identity']['mask_hashes']):
            raise ValueError('Exact original test masks differ')
        for key in ('split', 'seed_protocol', 'data_files'):
            if bundle[key] != prior['identity'][key]:
                raise ValueError('Original reference differs: ' + key)
        reference_history = json.loads((reference/'history.json').read_text())
        if len(reference_history) != 150:
            raise ValueError('Complete reference training history required')
        _, a = scalar.evaluate(bundle, plain.eval(), 'flat', 'test', masks['test']['0.3'], max_batches=1)
        _, b = scalar.evaluate(bundle, model.eval(), 'flat', 'test', masks['test']['0.3'], max_batches=1)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        if cfg.osram_source_role_v3:
            del plain
        instrumentation = dict(v3_gradient_epochs=[1, 2] if cfg.osram_source_role_v3 else [],
            cumulative_until_validated=True, remove_gradient_hooks_after_validation=True,
            memory_scan_counter='all training epochs', scalar_train_epoch_diagnostics='unchanged')
        identity = dict(source_commit=args.source_commit,
            source_hashes={f: h.sha(ROOT/f) for f in source_catalog_factory(prior['identity']['source_hashes'])},
            config=asdict(cfg), dataset='CMUMOSI', seed=args.seed, mode=args.mode,
            split=bundle['split'], seed_protocol=bundle['seed_protocol'], mask_hashes=hashes,
            data_files=bundle['data_files'], data_manifest=bundle['data_manifest'],
            dataset_manifest_sha256=bundle['data_manifest_sha256'], environment=bundle['environment'],
            initialization='fresh joint training; protected bridge weights discarded',
            original_large_flat_initial_sha256=bundle['initial_backbone_sha256'],
            same_width_flat_initial_sha256=plain_initial,
            model_initial_sha256=h.backbone_digest(model),
            historical_bridge=bundle['bridge'], instrumentation=instrumentation,
            reference=dict(path=str(reference), summary_sha256=h.sha(reference/'SUMMARY.json'),
                history_sha256=h.sha(reference/'history.json'), masks_exact=True,
                baseline_config_exact=True, untouched_original_initialization_exact=True),
            learning_rates=settings('CMUMOSI'),
            selection='per-rate TEST weighted_f1; strict improvement, earliest tie')
        optimizer = scalar.make_optimizer(model, cfg, 'flat')
        if args.resume:
            stored_masks = torch.load(args.output/'fixed_masks.pt', map_location='cpu', weights_only=False)
            if stored_masks.keys() != masks.keys() or stored_masks['test'].keys() != masks['test'].keys():
                raise ValueError('Stored fixed mask banks differ')
            for rate, bank in masks['test'].items():
                stored = stored_masks['test'][rate]
                if len(bank) != len(stored):
                    raise ValueError('Stored fixed mask batch count differs')
                for current, old_record in zip(bank, stored):
                    if current.keys() != old_record.keys():
                        raise ValueError('Stored fixed mask metadata differs')
                    for key in current:
                        np.testing.assert_array_equal(current[key], old_record[key])
        else:
            torch.save(masks, args.output/'fixed_masks.pt')
        module = model.osram.source_role_v3 if cfg.osram_source_role_v3 else None
        counts = dict(total=sum(p.numel() for p in model.parameters()),
            flat_adapter=sum(p.numel() for p in model.osram.emotion_adapter.parameters()),
            v3=sum(p.numel() for p in module.parameters()) if module else 0)
        summary.update(status='running', identity=identity, config=asdict(cfg), parameters=counts,
            environment=bundle['environment'], gpu=bundle['gpu'], zero_init_predictions_exact=True,
            initial_backbone_sha256=bundle['initial_backbone_sha256'],
            instrumentation=instrumentation)
        h.write_json(args.output/'config.json', asdict(cfg))
        h.write_json(path, summary)
        scans = [0]
        scan_hook = model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0, scans[0]+1))
        reset_training_stochasticity(args.seed, cfg.fold)
        torch.cuda.reset_peak_memory_stats()
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            if len(history) != start or not 0 <= start <= cfg.epochs:
                raise ValueError('Invalid full-state epoch/history')
            selected.validate_selected(args.output, best)
            for index, row in enumerate(history):
                check_training_parity(row, reference_history[index])
        cumulative = {}
        for row in history:
            for name, value in row.get('v3_gradient_paths', {}).items():
                cumulative[name] = cumulative.get(name, False) or value
        if module is not None:
            if start >= smoke_epochs():
                validate_gradients(cumulative)
            else:
                capture = GradientCapture(module)
                capture.observed.update(cumulative)
        summary.update(epoch=start, best_by_rate=best, v3_gradient_paths=cumulative,
                       v3_gradient_hooks_removed=not capture or not capture.handles)
        initial_model = h.backbone_digest(model) if args.phase == 'smoke' else None
        initial_original = digest_without(model, ('osram.source_role_v3.',)) if args.phase == 'smoke' else None
        initial_v3 = h.backbone_digest(module) if args.phase == 'smoke' and module is not None else None
        for epoch in (range(smoke_epochs()) if args.phase == 'smoke' else range(start, cfg.epochs)):
            scans[0] = 0
            instrumented = capture is not None and bool(capture.handles)
            set_learning_rates(optimizer, 'CMUMOSI', epoch)
            row = scalar.train_epoch(bundle, model, optimizer, epoch, 'flat',
                                     max_batches=3 if args.phase == 'smoke' else None)
            if scans[0] != row['optimizer_steps']:
                raise RuntimeError('Exactly one CHM scan per train batch required')
            check_training_parity(row, reference_history[epoch])
            if instrumented:
                cumulative.update(capture.observed)
                if epoch >= gradient_check_epoch('CMUMOSI'):
                    validate_gradients(cumulative)
                    capture.remove()
            row.update(actual_memory_scans=scans[0], v3_gradient_instrumented=instrumented,
                       v3_gradient_paths=dict(cumulative) if instrumented else {})
            if module is not None:
                from experiments.osram_coupled_source_role_20261020.worker import serial
                row['v3_diagnostics'] = {k: serial(v) for k, v in module.last_diagnostics.items()
                                         if k != 'cross_attention_weights'}
            evaluated = {rate: scalar.evaluate(bundle, model, 'flat', 'test', masks['test'][rate],
                max_batches=1 if args.phase == 'smoke' else None) for rate in RATES}
            row['test'] = {rate: value[0] for rate, value in evaluated.items()}
            history.append(row)
            improved = selected.persist_epoch(args.output, model, optimizer, epoch=epoch+1,
                identity=identity, history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch+1, best_by_rate=best,
                v3_gradient_paths=cumulative, v3_gradient_hooks_removed=not capture or not capture.handles,
                seconds=time.monotonic()-tick, peak_memory_mib=max(float(previous.get('peak_memory_mib', 0)),
                    torch.cuda.max_memory_allocated()/2**20))
            h.write_json(path, summary)
            print(json.dumps(dict(event='epoch', mode=args.mode, epoch=epoch+1, task_MSE=row['task_MSE'],
                improved_rates=improved, test_WF1={r: v[0]['router']['WF1'] for r, v in evaluated.items()})), flush=True)
        if args.phase == 'smoke':
            steps = sum(row['optimizer_steps'] for row in history)
            if steps < 4 or initial_model == h.backbone_digest(model):
                raise RuntimeError('Smoke requires four real updates and changed task parameters')
            if initial_original == digest_without(model, ('osram.source_role_v3.',)):
                raise RuntimeError('Smoke did not update original Flat parameters')
            if module is not None:
                validate_gradients(cumulative)
                if initial_v3 == h.backbone_digest(module):
                    raise RuntimeError('Smoke did not update V3 parameters')
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            before_metrics, before = scalar.evaluate(bundle, model, 'flat', 'test', masks['test']['0.3'], max_batches=1)
            restore(model, optimizer, state, identity)
            same_state(model.state_dict(), state['model'])
            same_state(optimizer.state_dict(), state['optimizer'])
            from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
            require_same_rng(state['rng'], h.rng_state())
            selected.validate_selected(args.output, best)
            after_metrics, after = scalar.evaluate(bundle, model, 'flat', 'test', masks['test']['0.3'], max_batches=1)
            if before_metrics != after_metrics:
                raise RuntimeError('Restored checkpoint metrics differ')
            for key in before:
                np.testing.assert_array_equal(before[key], after[key])
            summary.update(smoke_only=True, full_checkpoint_reloaded=True,
                optimizer_rng_prediction_reload_exact=True, smoke_optimizer_steps=steps)
        else:
            summary['evaluation'] = selected.publish_selected(args.output, best)
            summary['aggregate'] = scalar.aggregate(summary['evaluation'])
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT/name) != digest:
                raise RuntimeError('Immutable source changed: ' + name)
        summary.update(status='complete', seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        if capture is not None:
            capture.remove()
        if scan_hook is not None:
            scan_hook.remove()
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
