"""Finite one-stage source-relation attention: MOSI66 / MOSEI66 only."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RATES = tuple(f'{i/10:.1f}' for i in range(8))
REFERENCES = {
    'CMUMOSI': Path('/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003'),
    'CMUMOSEI': Path('/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat')}
LAYOUTS = ('source-mean', 'role-tokens')


def source_catalog(prior_sources, layout):
    if layout not in LAYOUTS: raise ValueError('Unknown source relation layout')
    additions = ('gcnet_missing_m3/source_relation.py',
                 'experiments/osram_source_relation_attention_20261021/worker.py')
    if layout == 'role-tokens':
        additions += ('gcnet_missing_m3/source_role_tokens.py',
                      'experiments/osram_source_relation_attention_20261021/run.py',
                      'experiments/osram_source_relation_attention_20261021/report.py')
    return tuple(dict.fromkeys(tuple(prior_sources) + additions))


def settings(dataset):
    if dataset not in REFERENCES:
        raise ValueError('Only the approved MOSI66/MOSEI66 screen')
    return dict(epochs=150, first_lr=3e-4, final_lr=3e-5) if dataset == 'CMUMOSI' else dict(
        epochs=60, first_lr=1e-3, final_lr=1e-3)


def set_learning_rates(optimizer, dataset, epoch):
    spec = settings(dataset)
    if not 0 <= epoch < spec['epochs'] or any(g['name'] != 'backbone' for g in optimizer.param_groups):
        raise ValueError('Require the original joint backbone group and finite epoch budget')
    for group in optimizer.param_groups:
        group['lr'] = spec['final_lr'] if dataset == 'CMUMOSI' and epoch >= 100 else spec['first_lr']


def original_digest(model):
    """Original parameter byte digest, excluding only the added module."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('osram.source_relation.'): continue
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_reference_config(cfg, old):
    # The original MOSEI run predates these two default-disabled fields.
    # Normalize only this reviewed schema evolution, never arbitrary defaults.
    expected = dict(old, osram_source_relation=True)
    expected.setdefault('osram_source_role', False)
    expected.setdefault('osram_source_role_mode', 'residual')
    if hasattr(cfg, 'osram_source_relation_layout'):
        if cfg.osram_source_relation_layout not in LAYOUTS:
            raise ValueError('Unknown source relation layout')
        expected['osram_source_relation_layout'] = cfg.osram_source_relation_layout
    if expected != json.loads(json.dumps(asdict(cfg))):
        raise ValueError('Unexpected changes beyond new module switch')


def check_training_parity(row, reference):
    for key in ('epoch', 'mask_trace_sha256', 'optimizer_steps'):
        if row[key] != reference[key]: raise RuntimeError('Reference training '+key+' changed')
    actual, expected = row['learning_rates'], reference['learning_rates']
    if len(actual) != len(expected) or any(not math.isclose(a, b, rel_tol=1e-14, abs_tol=0.)
                                         for a, b in zip(actual, expected)):
        raise RuntimeError('Reference training learning_rates changed')


def same_manifest_content(before, after):
    if before == after: return True
    old, new = dict(before), dict(after)
    first, second = old.pop('manifest_origin', None), new.pop('manifest_origin', None)
    expected = old.get('combined_manifest_sha256')
    if old != new or not expected or not first or not second: return False
    return all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected
               for p in (first, second))


def validate_audit_resume(state, identity):
    """One explicit migration for the audited epoch101 float-comparison failure."""
    old = state['identity']; path = 'experiments/osram_source_relation_attention_20261021/worker.py'
    before = '64b17f0c5cca283b2b9f2c737457e48f73bd28833556f1b2f515b22f5536ce35'
    if (state['epoch'] != 100 or old.get('dataset') != 'CMUMOSI' or old.get('seed') != 66
            or old.get('source_commit') != '6dbcc986c451688ea33f33d3ffe6785e0be775f5'
            or old['source_hashes'].get(path) != before):
        raise ValueError('Only the known MOSI66 epoch100 audit-failure checkpoint may migrate')
    if set(old) != set(identity): raise ValueError('Resume identity schema changed')
    for key in old:
        if key == 'data_manifest' and same_manifest_content(old[key], identity[key]): continue
        if key not in ('source_commit', 'source_hashes') and old[key] != identity[key]:
            raise ValueError('Resume changed '+key)
    if set(old['source_hashes']) != set(identity['source_hashes']):
        raise ValueError('Resume source catalog changed')
    for name, digest in old['source_hashes'].items():
        if name != path and digest != identity['source_hashes'][name]:
            raise ValueError('Resume changed non-audit source: '+name)
    return dict(normal_training_math_unchanged=True, source_commit_before=old['source_commit'],
        source_commit_after=identity['source_commit'], restored_epoch=100,
        manifest_path_relocated_with_content_hash_verified=old.get('data_manifest') != identity.get('data_manifest'),
        only_source_hash_change=dict(path=path, before=before, after=identity['source_hashes'][path]),
        correction='LR parity comparison only: rel_tol1e-14,abs_tol0; actual scheduler unchanged')


def validate_gradients(observed, layout='source-mean'):
    if layout not in LAYOUTS: raise ValueError('Unknown source relation layout')
    prefixes = ('base_projection.', 'gap_projections.', 'relation_mlp.',
        'source_self_attention.', 'local_projection.', 'local_cross_attention.',
        'source_embedding', 'output_projection.', 'input.source_base', 'input.source_gap')
    missing = [p for p in prefixes if not any(v and n.startswith(p) for n, v in observed.items())]
    if layout == 'role-tokens':
        if not any(v and n.startswith('role_embedding') for n, v in observed.items()):
            missing.append('role_embedding')
        # Slots are registered only if genuinely active in an actual batch.
        # Zero-input or all-masked slots must not masquerade as tested paths.
        missing += [n for n, value in observed.items()
                    if n.startswith(('input.source_base.source_', 'input.source_gap.target_')) and not value]
    if missing: raise RuntimeError('Missing nonzero end-to-end task gradients: '+', '.join(missing))


def compact_diagnostics(module):
    from experiments.osram_coupled_source_role_20261020.worker import serial
    d = module.last_diagnostics
    result = {k: serial(v) for k, v in d.items() if k != 'cross_attention_weights'}
    weights = d['cross_attention_weights']
    if weights.ndim == 4 and tuple(weights.shape[-2:]) == (4, 3):
        selected = weights.sum((-2, -1)) > 0
        result['role_source_attention_mean'] = serial(weights[selected].mean(0)) if selected.any() else [[0.]*3 for _ in range(4)]
    elif weights.ndim == 3 and weights.shape[-1] == 3:
        selected = weights.sum(-1) > 0
        result['source_attention_mean'] = serial(weights[selected].mean(0)) if selected.any() else [0., 0., 0.]
    else:
        raise ValueError('Unexpected source relation attention weight layout')
    return result


def setup(dataset, uuid, output, layout='source-mean'):
    if layout not in LAYOUTS: raise ValueError('Unknown source relation layout')
    import torch
    from experiments.osram_mosi_gcnet_testbest_20261020.worker import testbest_config
    if dataset == 'CMUMOSI':
        from experiments.osram_mosi_gcnet_router_3seed_20261019 import worker as old
        bundle = old.setup(uuid, output, 66)
        cfg = replace(testbest_config(bundle['cfg']), learning_rate=3e-4)
    else:
        from experiments.osram_mosei_gcnet_router_20261020 import worker as old
        bundle = old.setup(uuid, output, 66)
        cfg = bundle['cfg']
    if (cfg.osram_source_role or cfg.osram_tasr or cfg.osram_source_relation
            or cfg.missing_mask_protocol != 'gcnet-random-mask-v1'
            or cfg.training_objective != 'emotion-only' or cfg.osram_readout_fusion != 'flat'):
        raise ValueError('Plain no-JEPA Flat required; no old gamma or overlapping methods')
    baseline = bundle.pop('model').eval()
    cfg = replace(cfg, osram_source_relation=True, osram_source_relation_layout=layout)
    model = old.fresh_backbone(cfg, bundle['dimensions'], torch.device('cuda:0')).train()
    if original_digest(model) != bundle['initial_backbone_sha256']:
        raise ValueError('New block changed original parameter initialization')
    bundle.update(model=model, cfg=cfg)
    return bundle, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(REFERENCES), required=True)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--seed', type=int, choices=(66,), default=66)
    parser.add_argument('--layout', choices=LAYOUTS, default='source-mean')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-audit-only-resume', action='store_true',
                        help='Explicit audited epoch100 MOSI66 recovery after LR comparison roundoff failure.')
    args = parser.parse_args(); spec = settings(args.dataset)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    if args.output.exists() and not args.resume: raise FileExistsError('Fresh output required')
    if args.resume and args.phase != 'train': raise ValueError('Full-state train resume only')
    if args.allow_audit_only_resume and (not args.resume or args.dataset != 'CMUMOSI'):
        raise ValueError('Audit-only migration requires explicit MOSI full-state resume')
    if args.allow_audit_only_resume and args.layout != 'source-mean':
        raise ValueError('Legacy audit migration cannot change module layout')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'SUMMARY.json'; tick = time.monotonic(); hooks = []
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get('status') == 'complete': raise ValueError('Completed run must not repeat')
    summary = dict(status='preflight', source_commit=args.source_commit, dataset=args.dataset,
        seed=66, phase=args.phase, layout=args.layout, pid=os.getpid(), label='INTERNAL TEST-ORACLE SCREEN')
    try:
        torch.set_num_threads(1); h.write_json(path, summary)
        bundle, baseline = setup(args.dataset, args.gpu_uuid, args.output, args.layout)
        cfg, model = bundle['cfg'], bundle['model']
        masks = {s: scalar.build_masks(bundle, s) for s in ('validation', 'test')}
        hashes = {s: {r: hashlib.sha256(b''.join(x['availability'].numpy().tobytes() for x in bank)).hexdigest()
            for r, bank in rates.items()} for s, rates in masks.items()}
        reference = REFERENCES[args.dataset]; prior = json.loads((reference/'SUMMARY.json').read_text())
        if prior['status'] != 'complete' or prior['epoch'] != spec['epochs']:
            raise ValueError('Completed same-dataset original Flat reference required')
        validate_reference_config(cfg, prior['identity']['config'])
        if prior['initial_backbone_sha256'] != bundle['initial_backbone_sha256']:
            raise ValueError('Reference initialization differs')
        if prior['identity']['split'] != bundle['split'] or prior['identity']['seed_protocol'] != bundle['seed_protocol']:
            raise ValueError('Reference data split or loader seed differs')
        old_hashes = prior['identity']['mask_hashes']
        if hashes['test'] != old_hashes.get('test', old_hashes):
            raise ValueError('Reference test masks differ')
        reference_history = json.loads((reference/'history.json').read_text())
        if len(reference_history) != spec['epochs']: raise ValueError('Incomplete reference history')
        _, base_arrays = scalar.evaluate(bundle, baseline, 'flat', 'test', masks['test']['0.3'], max_batches=1)
        _, new_arrays = scalar.evaluate(bundle, model, 'flat', 'test', masks['test']['0.3'], max_batches=1)
        for key in ('prediction', 'labels', 'availability'):
            np.testing.assert_array_equal(base_arrays[key], new_arrays[key])
        del baseline, base_arrays, new_arrays
        prior_sources = tuple(prior['identity']['source_hashes'])
        sources = source_catalog(prior_sources, args.layout)
        identity = dict(source_commit=args.source_commit, source_hashes={p: h.sha(ROOT/p) for p in sources},
            config=asdict(cfg), dataset=args.dataset, seed=66, mode='flat', split=bundle['split'],
            seed_protocol=bundle['seed_protocol'], mask_hashes=hashes, data_manifest=bundle['data_manifest'],
            data_files=bundle.get('data_files'), dataset_manifest_sha256=bundle.get('data_manifest_sha256'),
            initialization='fresh original factory; no trained checkpoint loaded', one_stage=True,
            reference=dict(path=str(reference), summary_sha256=h.sha(reference/'SUMMARY.json'),
                initialization_exact=True, masks_exact=True, config_only_module_switch_changed=True),
            learning_rates=spec, selection='per-rate TEST weighted_f1; earliest strict-best ties')
        optimizer = scalar.make_optimizer(model, cfg, 'flat')
        torch.save(masks, args.output/'fixed_masks.pt')
        summary.update(status='running', identity=identity, config=identity['config'],
            environment=bundle['environment'], gpu=bundle['gpu'], zero_init_predictions_exact=True,
            total_parameters=sum(p.numel() for p in model.parameters()),
            source_relation_parameters=sum(p.numel() for p in model.osram.source_relation.parameters()),
            initial_backbone_sha256=bundle['initial_backbone_sha256'])
        h.write_json(args.output/'config.json', identity['config']); h.write_json(path, summary)
        observed = {}; scans = [0]
        def capture(name):
            def hook(grad):
                if not torch.isfinite(grad).all(): raise RuntimeError('Nonfinite source relation gradient: '+name)
                observed[name] = observed.get(name, False) or bool(grad.ne(0).any())
            return hook
        for name, p in model.osram.source_relation.named_parameters(): hooks.append(p.register_hook(capture(name)))
        def watch_inputs(module, inputs):
            for name, value in zip(('source_base', 'source_gap'), inputs[1:3]):
                if value.requires_grad: value.register_hook(capture('input.'+name))
            if args.layout == 'role-tokens':
                trajectory, availability, umask = inputs[3:6]
                seen = trajectory[..., 3].gt(0) & umask.T.bool().unsqueeze(-1)
                masks = (seen, ~availability.bool().unsqueeze(-1) & seen.unsqueeze(2))
                for name, value, active in zip(('source_base', 'source_gap'), inputs[1:3], masks):
                    if not value.requires_grad: continue
                    for index in np.ndindex(tuple(active.shape[2:])):
                        slot_active = active[(slice(None), slice(None)) + index]
                        if not slot_active.any(): continue
                        suffix = ('source_'+str(index[0]) if name == 'source_base'
                                  else 'target_'+str(index[0])+'.source_'+str(index[1]))
                        key = 'input.'+name+'.'+suffix
                        observed.setdefault(key, False)
                        def slot_hook(grad, slot=index, mask=slot_active, key=key):
                            selected_grad = grad[(slice(None), slice(None)) + slot][mask]
                            if not torch.isfinite(selected_grad).all():
                                raise RuntimeError('Nonfinite active source gradient: '+key)
                            observed[key] = observed.get(key, False) or bool(selected_grad.ne(0).any())
                        value.register_hook(slot_hook)
        hooks.append(model.osram.source_relation.register_forward_pre_hook(watch_inputs))
        hooks.append(model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0, scans[0]+1)))
        reset_training_stochasticity(66, cfg.fold); torch.cuda.reset_peak_memory_stats()
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            if args.allow_audit_only_resume:
                summary['audit_only_resume'] = validate_audit_resume(state, identity)
                restore(model, optimizer, state, state['identity'])
                if any(not torch.equal(v.cpu(), state['model'][n]) for n, v in model.state_dict().items()):
                    raise RuntimeError('Migrated full-state weights differ')
                from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
                require_same_rng(state['rng'], h.rng_state())
                summary['audit_only_resume'].update(model_exact=True, optimizer_restored=True, rng_exact=True)
            else:
                restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            selected.validate_selected(args.output, best)
        before_added = h.backbone_digest(model.osram.source_relation)
        before_original = original_digest(model)
        for epoch in (range(1) if args.phase == 'smoke' else range(start, cfg.epochs)):
            set_learning_rates(optimizer, args.dataset, epoch); observed.clear(); scans[0] = 0
            row = scalar.train_epoch(bundle, model, optimizer, epoch, 'flat', max_batches=3 if args.phase == 'smoke' else None)
            if scans[0] != row['optimizer_steps']: raise RuntimeError('One Memory scan per training batch required')
            if not any(observed.values()): raise RuntimeError('No nonzero new-module task gradients')
            if args.phase == 'smoke' or epoch == 0: validate_gradients(observed, args.layout)
            if args.phase == 'train':
                check_training_parity(row, reference_history[epoch])
            row.update(source_relation_gradients=dict(connected=len(observed), nonzero=sum(observed.values())),
                gradient_paths={k: v for k, v in observed.items()},
                actual_memory_scans=scans[0], source_relation_diagnostics=compact_diagnostics(model.osram.source_relation))
            evaluated = {r: scalar.evaluate(bundle, model, 'flat', 'test', masks['test'][r],
                max_batches=1 if args.phase == 'smoke' else None) for r in RATES}
            row['test'] = {r: value[0] for r, value in evaluated.items()}; history.append(row)
            improved = selected.persist_epoch(args.output, model, optimizer, epoch=epoch+1,
                identity=identity, history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch+1, best_by_rate=best, selected_epoch_by_rate={r: v['epoch'] for r, v in best.items()},
                seconds=time.monotonic()-tick, peak_memory_mib=max(float(previous.get('peak_memory_mib', 0)), torch.cuda.max_memory_allocated()/2**20))
            h.write_json(path, summary)
            print(json.dumps(dict(event='epoch', dataset=args.dataset, epoch=epoch+1, task_MSE=row['task_MSE'],
                improved_rates=improved, gradients=row['source_relation_gradients'], test_WF1={r: v[0]['router']['WF1'] for r, v in evaluated.items()})), flush=True)
        if args.phase == 'smoke':
            if h.backbone_digest(model.osram.source_relation) == before_added or original_digest(model) == before_original:
                raise RuntimeError('Both original and new parameters must update')
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            expected_rng = state['rng']; restore(model, optimizer, state, identity)
            if any(not torch.equal(v.cpu(), state['model'][n]) for n, v in model.state_dict().items()):
                raise RuntimeError('Model recovery differs')
            from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
            require_same_rng(expected_rng, h.rng_state()); selected.validate_selected(args.output, best)
            summary.update(smoke_only=True, full_checkpoint_reloaded=True, smoke=history[0])
        else:
            summary['evaluation'] = selected.publish_selected(args.output, best)
            summary['aggregate'] = scalar.aggregate(summary['evaluation'])
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT/name) != digest: raise RuntimeError('Source changed during run: '+name)
        summary.update(status='complete', seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick); raise
    finally:
        for hook in hooks: hook.remove()
        h.write_json(path, summary)


if __name__ == '__main__': main()
