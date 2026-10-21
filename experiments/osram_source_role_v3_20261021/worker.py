"""Finite V3 source-role screen; original task, masks, data and optimizer."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.osram_source_relation_attention_20261021.worker import (
    REFERENCES, RATES, settings, set_learning_rates, check_training_parity,
)

ROOT = Path(__file__).resolve().parents[2]


def source_catalog(prior_sources):
    additions = ('gcnet_missing_m3/source_role_v3.py', 'gcnet_missing_m3/osram.py',
        'gcnet_missing_m3/model.py', 'gcnet_missing_m3/train_gcnet.py',
        'gcnet_missing_m3/tasr.py',
        'experiments/osram_source_relation_attention_20261021/worker.py',
        'experiments/osram_source_relation_attention_20261021/run.py',
        'experiments/osram_text_path_router_joint_mosi_20261019/run.py',
        'experiments/osram_mosi_gcnet_testbest_20261020/worker.py',
        'experiments/osram_source_role_v3_20261021/worker.py',
        'experiments/osram_source_role_v3_20261021/run.py',
        'experiments/osram_source_role_v3_20261021/report.py')
    return tuple(dict.fromkeys(tuple(prior_sources) + additions))


def original_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('osram.source_role_v3.'):
            continue
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_reference_config(cfg, old):
    expected = dict(old)
    for key, value in dict(osram_source_role=False, osram_source_role_mode='residual',
            osram_source_relation=False, osram_source_relation_layout='source-mean').items():
        expected.setdefault(key, value)
    expected['osram_source_role_v3'] = True
    if expected != json.loads(json.dumps(asdict(cfg))):
        raise ValueError('Unexpected config difference beyond opt-in V3 and reviewed disabled schema')


def validate_gradients(observed):
    prefixes = ('memory_projection.', 'local_projection.', 'pair_mlp.',
        'source_embedding.', 'gap_role_embedding.', 'attention_source_embedding.',
        'attention_role_embedding.', 'source_norm.', 'source_attention.',
        'output_attention.', 'role_queries.', 'decoders.',
        'input.source_base', 'input.source_gap')
    missing = [p for p in prefixes if not any(v and n.startswith(p) for n, v in observed.items())]
    missing += [f'{group}.{role}.' for group in ('role_queries', 'decoders') for role in range(4)
                if not any(v and n.startswith(f'{group}.{role}.') for n, v in observed.items())]
    if missing:
        raise RuntimeError('Missing nonzero V3 task-gradient groups: ' + ', '.join(missing))


def gradient_check_epoch(dataset):
    settings(dataset)
    # MOSI has only two batch32 updates/epoch. Original Flat's adapter and
    # V3's decoder both start at zero; four actual updates open upstream paths.
    return 1 if dataset == 'CMUMOSI' else 0


def setup(dataset, uuid, output):
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
            or cfg.osram_source_role_v3 or cfg.missing_mask_protocol != 'gcnet-random-mask-v1'
            or cfg.training_objective != 'emotion-only' or cfg.osram_readout_fusion != 'flat'):
        raise ValueError('Require original no-JEPA Flat without overlapping modules')
    baseline = bundle.pop('model').eval()
    cfg = replace(cfg, osram_source_role_v3=True)
    model = old.fresh_backbone(cfg, bundle['dimensions'], torch.device('cuda:0')).train()
    if original_digest(model) != bundle['initial_backbone_sha256']:
        raise ValueError('V3 changed original initialization')
    bundle.update(model=model, cfg=cfg)
    return bundle, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(REFERENCES), required=True)
    parser.add_argument('--phase', choices=('smoke', 'train'), required=True)
    parser.add_argument('--seed', type=int, choices=(66,), default=66)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); spec = settings(args.dataset)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import fcntl
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    from experiments.osram_text_path_router_joint_20261019.run import restore
    from experiments.osram_mosi_gcnet_testbest_20261020 import worker as selected
    from gcnet_modality_jepa.train_gcnet import reset_training_stochasticity
    if args.output.exists() and not args.resume:
        raise FileExistsError('Fresh independent output required')
    if args.resume and args.phase != 'train':
        raise ValueError('Only full-state training resume supported')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output/'RUN.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.output/'SUMMARY.json'; tick = time.monotonic(); hooks = []
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get('status') == 'complete':
        raise ValueError('Completed experiment cannot silently repeat')
    summary = dict(status='preflight', dataset=args.dataset, seed=66, phase=args.phase,
        source_commit=args.source_commit, pid=os.getpid(), label='INTERNAL TEST-ORACLE SCREEN')
    try:
        torch.set_num_threads(1); h.write_json(path, summary)
        bundle, baseline = setup(args.dataset, args.gpu_uuid, args.output)
        cfg, model = bundle['cfg'], bundle['model']; module = model.osram.source_role_v3
        masks = {s: scalar.build_masks(bundle, s) for s in ('validation', 'test')}
        hashes = {s: {r: hashlib.sha256(b''.join(x['availability'].numpy().tobytes() for x in bank)).hexdigest()
            for r, bank in rates.items()} for s, rates in masks.items()}
        reference = REFERENCES[args.dataset]; prior = json.loads((reference/'SUMMARY.json').read_text())
        if prior['status'] != 'complete' or prior['epoch'] != spec['epochs']:
            raise ValueError('Completed same-budget original Flat reference required')
        validate_reference_config(cfg, prior['identity']['config'])
        if prior['initial_backbone_sha256'] != bundle['initial_backbone_sha256']:
            raise ValueError('Reference initialization mismatch')
        for key in ('split', 'seed_protocol'):
            if prior['identity'][key] != bundle[key]:
                raise ValueError('Reference ' + key + ' differs')
        old_hashes = prior['identity']['mask_hashes']
        if hashes['test'] != old_hashes.get('test', old_hashes):
            raise ValueError('Reference exact test masks differ')
        reference_history = json.loads((reference/'history.json').read_text())
        if len(reference_history) != spec['epochs']:
            raise ValueError('Incomplete reference history')
        _, original = scalar.evaluate(bundle, baseline, 'flat', 'test', masks['test']['0.3'], max_batches=1)
        _, initial = scalar.evaluate(bundle, model, 'flat', 'test', masks['test']['0.3'], max_batches=1)
        for key in ('prediction', 'labels', 'availability'):
            np.testing.assert_array_equal(original[key], initial[key])
        del baseline, original, initial
        identity = dict(source_commit=args.source_commit,
            source_hashes={p: h.sha(ROOT/p) for p in source_catalog(prior['identity']['source_hashes'])},
            config=asdict(cfg), dataset=args.dataset, seed=66, mode='flat', split=bundle['split'],
            seed_protocol=bundle['seed_protocol'], mask_hashes=hashes,
            data_manifest=bundle['data_manifest'], data_files=bundle.get('data_files'),
            dataset_manifest_sha256=bundle.get('data_manifest_sha256'),
            initialization='fresh original factory, no trained weights', one_stage=True,
            reference=dict(path=str(reference), summary_sha256=h.sha(reference/'SUMMARY.json'),
                initialization_exact=True, masks_exact=True, config_only_v3_changed=True),
            learning_rates=spec, selection='per-rate TEST weighted_f1; strict-best, earliest ties')
        optimizer = scalar.make_optimizer(model, cfg, 'flat'); torch.save(masks, args.output/'fixed_masks.pt')
        summary.update(status='running', identity=identity, config=identity['config'],
            environment=bundle['environment'], gpu=bundle['gpu'], zero_init_predictions_exact=True,
            total_parameters=sum(p.numel() for p in model.parameters()),
            source_role_v3_parameters=sum(p.numel() for p in module.parameters()),
            initial_backbone_sha256=bundle['initial_backbone_sha256'])
        h.write_json(args.output/'config.json', identity['config']); h.write_json(path, summary)
        observed = {}; scans = [0]
        def capture(name):
            def hook(grad):
                if not torch.isfinite(grad).all():
                    raise RuntimeError('Nonfinite V3 gradient: ' + name)
                observed[name] = observed.get(name, False) or bool(grad.ne(0).any())
            return hook
        for name, parameter in module.named_parameters():
            hooks.append(parameter.register_hook(capture(name)))
        def watch_inputs(unused, inputs):
            for name, value in zip(('source_base', 'source_gap'), inputs[1:3]):
                if value.requires_grad:
                    value.register_hook(capture('input.'+name))
        hooks.append(module.register_forward_pre_hook(watch_inputs))
        hooks.append(model.osram.register_forward_hook(lambda *unused: scans.__setitem__(0, scans[0]+1)))
        reset_training_stochasticity(66, cfg.fold); torch.cuda.reset_peak_memory_stats()
        best, history, start = {}, [], 0
        if args.resume:
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            restore(model, optimizer, state, identity)
            start, history, best = state['epoch'], state['history'], state['best_by_rate']
            selected.validate_selected(args.output, best)
        before_added = h.backbone_digest(module); before_original = original_digest(model)
        cumulative = {}
        for previous_row in history:
            for name, value in previous_row.get('gradient_paths', {}).items():
                cumulative[name] = cumulative.get(name, False) or value
        smoke_epochs = gradient_check_epoch(args.dataset) + 1
        for epoch in (range(smoke_epochs) if args.phase == 'smoke' else range(start, cfg.epochs)):
            set_learning_rates(optimizer, args.dataset, epoch); observed.clear(); scans[0] = 0
            row = scalar.train_epoch(bundle, model, optimizer, epoch, 'flat',
                max_batches=3 if args.phase == 'smoke' else None)
            if scans[0] != row['optimizer_steps']:
                raise RuntimeError('Require exactly one Memory scan per training batch')
            for name, value in observed.items():
                cumulative[name] = cumulative.get(name, False) or value
            if epoch == gradient_check_epoch(args.dataset):
                validate_gradients(cumulative)
            if args.phase == 'train':
                check_training_parity(row, reference_history[epoch])
            from experiments.osram_coupled_source_role_20261020.worker import serial
            diagnostics = {k: serial(v) for k, v in module.last_diagnostics.items()
                           if k != 'cross_attention_weights'}
            weights = module.last_diagnostics['cross_attention_weights']
            active_roles = weights.sum(-1) > 0
            denominator = active_roles.sum((0, 1)).clamp_min(1)[..., None]
            diagnostics['role_source_attention_mean'] = serial(weights.sum((0, 1)) / denominator)
            row.update(source_role_v3_gradients=dict(connected=len(observed), nonzero=sum(observed.values())),
                gradient_paths=dict(observed), actual_memory_scans=scans[0],
                source_role_v3_diagnostics=diagnostics)
            evaluated = {r: scalar.evaluate(bundle, model, 'flat', 'test', masks['test'][r],
                max_batches=1 if args.phase == 'smoke' else None) for r in RATES}
            row['test'] = {r: v[0] for r, v in evaluated.items()}; history.append(row)
            improved = selected.persist_epoch(args.output, model, optimizer, epoch=epoch+1,
                identity=identity, history=history, best=best, evaluated=evaluated)
            summary.update(epoch=epoch+1, best_by_rate=best,
                selected_epoch_by_rate={r: v['epoch'] for r, v in best.items()},
                seconds=time.monotonic()-tick, peak_memory_mib=max(float(previous.get('peak_memory_mib', 0)),
                    torch.cuda.max_memory_allocated()/2**20))
            h.write_json(path, summary)
            print(json.dumps(dict(event='epoch', dataset=args.dataset, epoch=epoch+1,
                task_MSE=row['task_MSE'], improved_rates=improved,
                gradients=row['source_role_v3_gradients'],
                test_WF1={r: v[0]['router']['WF1'] for r, v in evaluated.items()})), flush=True)
        if args.phase == 'smoke':
            validate_gradients(cumulative)
            minimum_steps = 4 if args.dataset == 'CMUMOSI' else 3
            if sum(row['optimizer_steps'] for row in history) < minimum_steps:
                raise RuntimeError('Smoke did not execute enough genuine optimizer updates')
            if h.backbone_digest(module) == before_added or original_digest(model) == before_original:
                raise RuntimeError('Both original and V3 parameters must update')
            state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            before_metrics, before_arrays = scalar.evaluate(bundle, model, 'flat', 'test', masks['test']['0.3'], max_batches=1)
            restore(model, optimizer, state, identity)
            if any(not torch.equal(v.cpu(), state['model'][n]) for n, v in model.state_dict().items()):
                raise RuntimeError('Checkpoint model restoration mismatch')
            def compare_state(actual, expected):
                if torch.is_tensor(expected):
                    if not torch.equal(actual.cpu(), expected.cpu()):
                        raise RuntimeError('Checkpoint Adam tensor differs')
                elif isinstance(expected, dict):
                    if actual.keys() != expected.keys():
                        raise RuntimeError('Checkpoint Adam keys differ')
                    for key in expected:
                        compare_state(actual[key], expected[key])
                elif isinstance(expected, (tuple, list)):
                    if len(actual) != len(expected):
                        raise RuntimeError('Checkpoint Adam sequence differs')
                    for first, second in zip(actual, expected):
                        compare_state(first, second)
                elif actual != expected:
                    raise RuntimeError('Checkpoint Adam scalar differs')
            compare_state(optimizer.state_dict(), state['optimizer'])
            from experiments.osram_12_source_evidence_audit_20261019.run import require_same_rng
            require_same_rng(state['rng'], h.rng_state())
            after_metrics, after_arrays = scalar.evaluate(bundle, model, 'flat', 'test', masks['test']['0.3'], max_batches=1)
            if before_metrics != after_metrics:
                raise RuntimeError('Restored checkpoint metrics differ')
            for name in before_arrays:
                np.testing.assert_array_equal(before_arrays[name], after_arrays[name])
            selected.validate_selected(args.output, best)
            summary.update(smoke_only=True, full_checkpoint_reloaded=True,
                restored_adam_exact=True, restoration_predictions_exact=True, smoke=history[-1],
                smoke_optimizer_steps=sum(row['optimizer_steps'] for row in history),
                cumulative_gradient_paths=cumulative)
        else:
            summary['evaluation'] = selected.publish_selected(args.output, best)
            summary['aggregate'] = scalar.aggregate(summary['evaluation'])
        for name, digest in identity['source_hashes'].items():
            if h.sha(ROOT/name) != digest:
                raise RuntimeError('Source changed during run: ' + name)
        summary.update(status='complete', seconds=time.monotonic()-tick)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        for hook in hooks:
            hook.remove()
        h.write_json(path, summary)


if __name__ == '__main__':
    main()
