"""Finite MOSEI capacity arms; no CHM, data, mask or objective changes."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODES = ('small_flat', 'wide_v3')
REFERENCE = Path('/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat')
RATES = tuple(f'{i/10:.1f}' for i in range(8))


def candidate_config(cfg, mode):
    if mode not in MODES:
        raise ValueError('Only the two approved capacity arms')
    required = dict(dataset='CMUMOSEI', fold=1, seed=66, epochs=60, learning_rate=.001,
        weight_decay=1e-5, batch_size=32, optimizer='adam', lr_schedule='constant',
        missing_mask_protocol='gcnet-random-mask-v1', training_objective='emotion-only',
        osram_readout_fusion='flat', osram_output_dim=1600, osram_num_heads=8,
        osram_key_dim=64, osram_value_dim=64, osram_bidirectional=False,
        osram_write_step=.6, osram_source_role_v3=False, osram_source_role=False,
        osram_source_relation=False, osram_tasr=False, osram_meaningful_block='none',
        paired_history_views=False, train_rate_mode='cyclic', task_regression_loss='mse',
        osram_adapter_hidden_dim=0)
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError('Original MOSEI config changed: ' + key)
    if tuple(cfg.train_missing_rates) != tuple(i/10 for i in range(8)):
        raise ValueError('Require original cyclic eight rates')
    return replace(cfg, osram_adapter_hidden_dim=1024, osram_source_role_v3=mode=='wide_v3',
        osram_source_role_v3_dim=256 if mode=='wide_v3' else 64,
        osram_source_role_v3_pair_hidden=512 if mode=='wide_v3' else 128)


def expected_counts(mode):
    if mode not in MODES:
        raise ValueError('Only the two approved capacity arms')
    return dict(total=12397503 if mode == 'wide_v3' else 10080865,
                flat_adapter=6106176, v3=2316638 if mode == 'wide_v3' else 0)


class GradientCapture:
    """Own parameter and input hooks so a finite capture can be fully removed."""
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


def digest_without(model, prefixes):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(prefixes):
            continue
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def same_state(actual, expected):
    import torch
    if torch.is_tensor(expected):
        if not torch.equal(actual.cpu(), expected.cpu()):
            raise RuntimeError('Restored tensor differs')
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise RuntimeError('Restored state keys differ')
        for key in expected:
            same_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise RuntimeError('Restored state sequence differs')
        for a, b in zip(actual, expected):
            same_state(a, b)
    elif actual != expected:
        raise RuntimeError('Restored state scalar differs')


def main(argv=None, *, allowed_modes=MODES, config_factory=candidate_config,
         expected_counts_factory=expected_counts, source_catalog_additions=(),
         capture_first_epoch_only=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=allowed_modes, required=True)
    p.add_argument('--phase', choices=('smoke', 'train'), required=True)
    p.add_argument('--seed', type=int, choices=(66,), default=66)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_mosei_gcnet_router_20261020 import worker as old
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import persist_epoch, validate_selected, publish_selected
    from experiments.osram_source_role_v3_20261021.worker import validate_gradients
    from experiments.osram_source_relation_attention_20261021.worker import check_training_parity
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent output required')
    if args.resume and args.phase != 'train':
        raise ValueError('Only complete-state train resume is supported')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'SUMMARY.json'; tick = time.monotonic(); hooks = []; gradient_capture = None
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        raise ValueError('Completed experiment cannot repeat')
    summary = dict(status='preflight', dataset='CMUMOSEI', mode=args.mode, seed=66,
        source_commit=args.source_commit, phase=args.phase, pid=os.getpid(), label='INTERNAL TEST-ORACLE SCREEN')
    try:
        torch.set_num_threads(1); h.write_json(path, summary)
        bundle = old.setup(args.gpu_uuid, args.output, 66)
        large = bundle.pop('model')
        original_cfg = bundle['cfg']; cfg = config_factory(original_cfg, args.mode)
        small_cfg = replace(cfg, osram_source_role_v3=False,
                            osram_source_role_v3_dim=64, osram_source_role_v3_pair_hidden=128)
        small = old.fresh_backbone(small_cfg, bundle['dimensions'], torch.device('cuda:0'))
        model = old.fresh_backbone(cfg, bundle['dimensions'], torch.device('cuda:0')) if cfg.osram_source_role_v3 else small
        untouched = ('osram.emotion_adapter.', 'osram.source_role_v3.')
        if digest_without(large, untouched) != digest_without(model, untouched):
            raise RuntimeError('Shrinking Flat altered unrelated original initialization')
        small_initial = h.backbone_digest(small)
        if digest_without(model, ('osram.source_role_v3.',)) != small_initial:
            raise RuntimeError('V3 changed same-width Flat initialization')
        del large
        bundle.update(cfg=cfg, model=model)
        masks = {'test': scalar.build_masks(bundle, 'test')}
        hashes = {'test': {r: hashlib.sha256(b''.join(v['availability'].numpy().tobytes() for v in bank)).hexdigest()
                           for r, bank in masks['test'].items()}}
        prior = json.loads((REFERENCE/'SUMMARY.json').read_text())
        if prior.get('status') != 'complete' or prior.get('epoch') != 60:
            raise ValueError('Complete original Flat66 reference required')
        from gcnet_missing_m3.train_gcnet import TrainConfig
        if asdict(TrainConfig(**prior['identity']['config'])) != asdict(original_cfg):
            raise ValueError('Current original config differs from accepted reference')
        if bundle['initial_backbone_sha256'] != prior['initial_backbone_sha256']:
            raise ValueError('Original large Flat initialization differs')
        if hashes['test'] != prior['identity']['mask_hashes'].get('test', prior['identity']['mask_hashes']):
            raise ValueError('Exact original test masks differ')
        for key in ('split', 'seed_protocol', 'data_files'):
            if bundle[key] != prior['identity'][key]:
                raise ValueError('Original reference differs: ' + key)
        _, a = scalar.evaluate(bundle, small.eval(), 'flat', 'test', masks['test']['0.3'], max_batches=1)
        _, b = scalar.evaluate(bundle, model.eval(), 'flat', 'test', masks['test']['0.3'], max_batches=1)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        if cfg.osram_source_role_v3:
            del small
        sources = set(prior['identity']['source_hashes'])
        for package in ('gcnet_missing_m3', 'gcnet_modality_jepa'):
            sources.update(str(f.relative_to(ROOT)) for f in (ROOT/package).rglob('*.py'))
        sources.update('experiments/osram_mosei_v3_capacity_20261022/' + n for n in ('worker.py','run.py','report.py'))
        sources.update(source_catalog_additions)
        identity = dict(source_commit=args.source_commit, source_hashes={f:h.sha(ROOT/f) for f in sorted(sources)},
            config=asdict(cfg), dataset='CMUMOSEI', seed=66, mode=args.mode,
            split=bundle['split'], seed_protocol=bundle['seed_protocol'], mask_hashes=hashes,
            data_files=bundle['data_files'], data_manifest=bundle['data_manifest'],
            dataset_manifest_sha256=bundle['data_manifest_sha256'],
            initialization='fresh joint training; no trained weight loaded', small_flat_initial_sha256=small_initial,
            reference=dict(path=str(REFERENCE), summary_sha256=h.sha(REFERENCE/'SUMMARY.json'),
                           masks_exact=True, untouched_original_initialization_exact=True),
            selection='per-rate TEST weighted_f1; strict improvement, earliest tie')
        if capture_first_epoch_only:
            identity['instrumentation'] = dict(v3_gradient_epochs=[1] if cfg.osram_source_role_v3 else [],
                remove_gradient_hooks_after_validation=True, memory_scan_counter='all training epochs',
                scalar_train_epoch_diagnostics='unchanged')
        optimizer = scalar.make_optimizer(model, cfg, 'flat')
        torch.save(masks, args.output/'fixed_masks.pt')
        module = model.osram.source_role_v3 if cfg.osram_source_role_v3 else None
        counts = dict(total=sum(p.numel() for p in model.parameters()),
            flat_adapter=sum(p.numel() for p in model.osram.emotion_adapter.parameters()),
            v3=sum(p.numel() for p in module.parameters()) if module else 0)
        if counts != expected_counts_factory(args.mode):
            raise RuntimeError('Actual capacity/wiring differs: ' + str(counts))
        summary.update(status='running', identity=identity, config=asdict(cfg), parameters=counts,
            environment=bundle['environment'], gpu=bundle['gpu'], zero_init_predictions_exact=True)
        if capture_first_epoch_only:
            summary['instrumentation'] = identity['instrumentation']
        h.write_json(args.output/'config.json', asdict(cfg)); h.write_json(path, summary)
        observed = {}; scans = [0]
        hooks.append(model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0, scans[0]+1)))
        reset_training_stochasticity(66, 1); torch.cuda.reset_peak_memory_stats()
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            validate_selected(args.output, best)
        if len(history) != start or not 0 <= start <= 60:
            raise ValueError('Invalid full-state epoch/history')
        if module is not None:
            if capture_first_epoch_only and start:
                validate_gradients(history[0]['v3_gradient_paths'])
            else:
                gradient_capture = GradientCapture(module)
                observed = gradient_capture.observed
        summary.update(epoch=start, best_by_rate=best)
        if capture_first_epoch_only:
            summary['first_epoch_v3_gradient_paths'] = history[0]['v3_gradient_paths'] if history else {}
            summary['v3_gradient_hooks_removed'] = gradient_capture is None
        reference_history = json.loads((REFERENCE/'history.json').read_text())
        initial_model = h.backbone_digest(model) if args.phase=='smoke' else None
        for epoch in (range(1) if args.phase=='smoke' else range(start, 60)):
            scans[0] = 0; observed.clear()
            row = scalar.train_epoch(bundle, model, optimizer, epoch, 'flat', max_batches=3 if args.phase=='smoke' else None)
            if scans[0] != row['optimizer_steps']:
                raise RuntimeError('Exactly one CHM scan per train batch required')
            if args.phase=='train':
                check_training_parity(row, reference_history[epoch])
            if module and epoch==0:
                validate_gradients(observed)
                if capture_first_epoch_only:
                    gradient_capture.remove()
            row.update(actual_memory_scans=scans[0], v3_gradient_paths=dict(observed))
            if module:
                from experiments.osram_coupled_source_role_20261020.worker import serial
                row['v3_diagnostics'] = {k:serial(v) for k,v in module.last_diagnostics.items() if k!='cross_attention_weights'}
            evaluated = {r:scalar.evaluate(bundle,model,'flat','test',masks['test'][r],
                max_batches=1 if args.phase=='smoke' else None) for r in RATES}
            row['test'] = {r:v[0] for r,v in evaluated.items()}; history.append(row)
            improved = persist_epoch(args.output,model,optimizer,epoch=epoch+1,identity=identity,
                                     history=history,best=best,evaluated=evaluated)
            summary.update(epoch=epoch+1,best_by_rate=best,
                seconds=time.monotonic()-tick,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            if capture_first_epoch_only:
                summary['first_epoch_v3_gradient_paths'] = history[0]['v3_gradient_paths']
                summary['v3_gradient_hooks_removed'] = not gradient_capture or not gradient_capture.handles
            h.write_json(path,summary)
            print(json.dumps(dict(event='epoch',mode=args.mode,epoch=epoch+1,task_MSE=row['task_MSE'],
                improved_rates=improved,test_WF1={r:v[0]['router']['WF1'] for r,v in evaluated.items()})),flush=True)
        if args.phase=='smoke':
            if initial_model==h.backbone_digest(model):
                raise RuntimeError('Smoke made no task update')
            state = torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
            _, before = scalar.evaluate(bundle,model,'flat','test',masks['test']['0.3'],max_batches=1)
            restore(model,optimizer,state,identity)
            same_state(model.state_dict(),state['model']); same_state(optimizer.state_dict(),state['optimizer'])
            from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
            require_same_rng(state['rng'],h.rng_state()); validate_selected(args.output,best)
            _, after = scalar.evaluate(bundle,model,'flat','test',masks['test']['0.3'],max_batches=1)
            for key in before:
                np.testing.assert_array_equal(before[key],after[key])
            summary.update(smoke_only=True,full_checkpoint_reloaded=True,optimizer_rng_prediction_reload_exact=True,
                           gradient_paths=observed,smoke_optimizer_steps=history[0]['optimizer_steps'])
        else:
            summary['evaluation']=publish_selected(args.output,best)
            summary['aggregate']=scalar.aggregate(summary['evaluation'])
        for name,digest in identity['source_hashes'].items():
            if h.sha(ROOT/name)!=digest:
                raise RuntimeError('Immutable source changed: ' + name)
        summary.update(status='complete',seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed',error=repr(error),seconds=time.monotonic()-tick)
        raise
    finally:
        if gradient_capture is not None:
            gradient_capture.remove()
        for hook in hooks:
            hook.remove()
        h.write_json(path,summary)


if __name__=='__main__':
    main()
