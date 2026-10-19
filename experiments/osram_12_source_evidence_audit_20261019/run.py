"""Finite frozen12-source audit on protected original Flat checkpoints."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def capture_flat(model):
    captured = {}

    def observe(name):
        def hook(module, inputs):
            if name in captured:
                raise RuntimeError('Expected exactly one original Flat invocation')
            captured[name] = inputs[0].detach().clone()
        return hook

    hooks = [model.osram.local_skip.register_forward_pre_hook(observe('local')),
             model.osram.emotion_adapter.register_forward_pre_hook(observe('emotion_input'))]
    try:
        yield captured
    finally:
        for hook in hooks:
            hook.remove()


def require_same_rng(before, after):
    import torch
    assert before['python'] == after['python']
    for a, b in zip(before['numpy'], after['numpy']):
        np.testing.assert_equal(a, b)
    assert torch.equal(before['torch'], after['torch'])
    assert len(before['cuda']) == len(after['cuda'])
    assert all(torch.equal(a, b) for a, b in zip(before['cuda'], after['cuda']))


def collect(bundle, *, saved, pattern, output, check_selected_logits=False):
    import torch
    from .readout import replay_readouts
    from experiments.osram_tasr_20261018 import runner_helpers as helper
    from experiments.osram_persistent_patterns_20261018.evaluate import fixed_availability

    model, trainer = bundle['model'], bundle['trainer']
    loader = bundle['test_loader']
    loader.sampler.set_epoch(0)
    parts, diagnostics, cursor, scans = {}, [], 0, 0
    tick = time.monotonic()

    def append(name, value):
        parts.setdefault(name, []).append(value)

    with torch.no_grad():
        for raw in loader:
            data = trainer._move_batch(raw, next(model.parameters()).device)
            valid = data[7].bool()
            meta = helper.metadata(data)
            n = len(meta['labels'])
            for name, values in meta.items():
                np.testing.assert_array_equal(values, saved[name][cursor:cursor+n])
            if pattern is None:
                batch_av = torch.zeros((*valid.shape, 3), device=valid.device)
                batch_av[valid] = torch.as_tensor(saved['availability'][cursor:cursor+n],
                                                 device=valid.device, dtype=batch_av.dtype)
                av = batch_av.transpose(0, 1)
            else:
                av = fixed_availability(data[7], pattern)
            view = trainer._prepare_view_from_primary_masks(data, av, av, bundle['dimensions'])
            if not torch.equal(view['availability'], av):
                raise RuntimeError('GCNet mask changed during view preparation')
            rng = helper.rng_state()
            with capture_flat(model) as captured:
                logits = helper.forward(model, view)
            scans += 1
            trace = model.osram.last_source_trace
            sources = torch.cat((trace['source_base'].unsqueeze(2), trace['source_gap']), 2)
            replay = replay_readouts(model.osram, model.smax_fc, captured['local'],
                captured['emotion_input'], sources, data[7], av, logits)
            require_same_rng(rng, helper.rng_state())
            fd = model.osram.num_heads * model.osram.value_dim
            contexts = captured['emotion_input'][..., model.osram.latent_dim:].reshape(
                *av.shape[:2], 4, 2 * fd)
            torch.testing.assert_close(sources.sum(-2), contexts[..., :fd], rtol=5e-5, atol=1e-5)
            if not torch.isfinite(sources).all():
                raise RuntimeError('Nonfinite source reads')
            flat = lambda value: value.transpose(0, 1)[valid].detach().cpu().numpy()
            arrays = dict(logits_full=flat(logits), source_reads=flat(sources),
                original_reads=flat(contexts[..., :fd]), trajectory=flat(trace['trajectory']),
                state_source_norms=flat(trace['state_source_norms']),
                state_joint_norm=flat(trace['state_joint_norm']), availability=flat(av),
                **{k:flat(v) for k,v in replay.items()})
            for name, values in {**meta, **arrays}.items():
                append(name, values)
            diagnostics.append(dict(valid=n, **trace['diagnostics']))
            cursor += n
    if cursor != len(saved['labels']):
        raise RuntimeError('Saved evaluation cohort not completely consumed')
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    arrays['predictions'] = arrays['logits_full'].argmax(1)
    if check_selected_logits:
        np.testing.assert_array_equal(arrays['logits_full'], saved['logits'])
    np.savez_compressed(output, **arrays)
    return arrays, dict(scans=scans, diagnostics=diagnostics, seconds=time.monotonic()-tick,
        availability_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest(),
        predictions_sha256=hashlib.sha256(arrays['predictions'].tobytes()).hexdigest(),
        selected_anchor_logits_exact=bool(check_selected_logits),
        no_rng_change_during_forward_and_readouts=True,
        original_readout_reproduction_max_abs_error=0.,
        source_sum_max_abs_error=float(np.abs(arrays['source_reads'].sum(-2)-arrays['original_reads']).max()),
        output_sha256=helper.sha(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('MELD', 'IEMOCAPSix'), required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2')
    import torch
    from .analysis import analyze_arrays
    from experiments.osram_tasr_20261018 import runner_helpers as helper
    from experiments.osram_tasr_20261018.protocol import RATES

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output/'SUMMARY.json'
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if prior.get('source_commit') != args.source_commit or prior.get('dataset') != args.dataset:
            raise ValueError('Existing immutable run identity differs')
        if prior['status'] == 'complete':
            from .report import summarize_run
            summarize_run(args.output)
            print('Already complete; not rerunning', flush=True)
            return
    tick = time.monotonic()
    report = dict(status='running', dataset=args.dataset, source_commit=args.source_commit,
        protocol='INTERNAL DIAGNOSTIC ONLY', no_training=True, conditions={}, error=None)
    helper.write_json(report_path, report)
    try:
        bundle = helper.setup_case(args.dataset, args.gpu_uuid, args.output)
        model = bundle['model']
        if model.osram.osram_tasr or model.osram.bidirectional:
            raise ValueError('Original causal Flat only; TASR must be off')
        if (model.osram.num_heads, model.osram.value_dim, model.osram.latent_dim) != (8, 64, 256):
            raise ValueError('Not protected cfg84 dimensions')
        if model.osram.osram_readout_fusion != 'flat':
            raise ValueError('Original Flat required')
        digest = helper.backbone_digest(model)
        model.osram.osram_source_tracking = True
        model.osram.osram_source_audit = True
        report.update(anchor=bundle['case'], anchor_parity=bundle['anchor_parity'],
                      environment=bundle['environment'], gpu=bundle['gpu'],
                      backbone_sha256=digest,
                      source_files={str(p.relative_to(ROOT)):helper.sha(p) for p in (
                          ROOT/'gcnet_missing_m3/osram.py', ROOT/'gcnet_missing_m3/tasr.py',
                          Path(__file__), Path(__file__).with_name('readout.py'),
                          Path(__file__).with_name('analysis.py'))})
        torch.cuda.reset_peak_memory_stats()
        conditions = [('random', r, bundle['selected'][r], None) for r in RATES]
        conditions += [('persistent', p, bundle['selected']['0.3'], p)
                       for p in ('A','T','V','AT','AV','TV','ATV')]
        for group, name, saved, pattern in conditions:
            key = group+'_'+name.replace('.', 'p')
            artifact = args.output/(key+'.npz')
            condition_json = args.output/(key+'.json')
            # Only reuse atomically completed per-condition stats with matching identity.
            if condition_json.exists():
                result = json.loads(condition_json.read_text())
                if (result['source_commit'] != args.source_commit
                        or result['anchor_sha256'] != bundle['case']['checkpoint_sha256']
                        or result['checks']['output_sha256'] != helper.sha(artifact)):
                    raise RuntimeError('Condition resume artifact mismatch')
            else:
                arrays, checks = collect(bundle, saved=saved, pattern=pattern, output=artifact,
                                         check_selected_logits=(group == 'random' and name == '0.3'))
                result = dict(source_commit=args.source_commit,
                    anchor_sha256=bundle['case']['checkpoint_sha256'], group=group, condition=name,
                    checks=checks, statistics=analyze_arrays(arrays, bundle['case']['classes']))
                helper.write_json(condition_json, result)
            report['conditions'][key] = dict(path=condition_json.name, group=group, condition=name,
                full_metrics=result['statistics']['full_metrics'], checks=result['checks'])
            helper.write_json(report_path, report)
            print(json.dumps(dict(dataset=args.dataset, condition=key,
                metrics=result['statistics']['full_metrics'], seconds=time.monotonic()-tick)), flush=True)
        if helper.backbone_digest(model) != digest or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError('Original frozen model changed')
        report.update(status='complete', backbone_unchanged=True,
                      peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                      seconds=time.monotonic()-tick)
    except Exception as error:
        report.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        helper.write_json(report_path, report)
        raise
    helper.write_json(report_path, report)
    from .report import summarize_run
    summarize_run(args.output)


if __name__ == '__main__':
    main()
