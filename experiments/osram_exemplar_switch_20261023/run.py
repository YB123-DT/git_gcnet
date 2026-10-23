"""Frozen MOSI66 exemplar ON/OFF readouts; no training or fresh mask sampling."""
import argparse
import csv
from dataclasses import asdict, fields, MISSING
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.osram_exemplar_mosi_20261023.worker import RATES, VARIANTS, REFERENCE

TRAINING = Path('/data2/yb/remote_experiments/osram_exemplar_mosi_20261023/execution_v1/runs')


def assert_aligned(actual, expected):
    for name in ('labels', 'speakers', 'conversation_ids', 'utterance_indices', 'availability'):
        np.testing.assert_array_equal(actual[name], expected[name], err_msg='Target/mask differs: ' + name)


def metrics(labels, prediction):
    from experiments.osram_value_covariance_20261023.report import binary_metrics
    y, p = np.asarray(labels).reshape(-1), np.asarray(prediction).reshape(-1)
    if y.shape != p.shape or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite aligned nonempty predictions required')
    result = binary_metrics(y, p)
    return dict(ACC=result['ACC'], WF1=result['W_F1'], MSE=float(np.square(y-p).mean()))


def paired_metrics(labels, on, off):
    y, on, off = [np.asarray(x).reshape(-1) for x in (labels, on, off)]
    eligible = y != 0
    good_on, good_off = (on > 0) == (y > 0), (off > 0) == (y > 0)
    return dict(N=len(y), N_binary=int(eligible.sum()), ON=metrics(y, on), OFF=metrics(y, off),
        corrections=int((good_on & ~good_off & eligible).sum()),
        harms=int((~good_on & good_off & eligible).sum()),
        mean_abs_prediction_change=float(np.abs(on-off).mean()),
        max_abs_prediction_change=float(np.abs(on-off).max()))


def frozen_readouts(model, view):
    """Capture one original forward, then reclassify its unchanged Flat anchor."""
    import torch
    if model.training or any(p.requires_grad for p in model.parameters()) or not model.enabled:
        raise ValueError('Frozen eval model with exemplar enabled is required')
    osram = model.backbone.osram
    captured, handles = {}, []

    def output(name):
        def capture(_module, _inputs, value):
            if name in captured:
                raise RuntimeError('Expected a single original Flat call')
            captured[name] = value.detach().clone()
        return capture

    def norm_input(_module, inputs):
        if 'on_anchor' in captured:
            raise RuntimeError('Expected one original normalization')
        captured['on_anchor'] = inputs[0].detach().clone()

    try:
        handles = [osram.local_skip.register_forward_hook(output('skip')),
                   osram.emotion_adapter.register_forward_hook(output('adapter')),
                   osram.emotion_norm.register_forward_pre_hook(norm_input)]
        with torch.no_grad():
            on = model(view)
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        anchor = captured['skip'] + captured['adapter']
        valid = view['umask'].T.bool()
        hidden_off = osram.emotion_norm(anchor) * valid[..., None].to(anchor.dtype)
        off = model.backbone.smax_fc(hidden_off)
        delta = captured['on_anchor'] - anchor
        return dict(ON=torch.where(valid[..., None], on, 0.),
            OFF=torch.where(valid[..., None], off, 0.),
            residual_norm=delta[valid].norm(dim=-1).mean(),
            anchor_norm=anchor[valid].norm(dim=-1).mean())


def evaluate(bundle, mask_bank, saved, flat, output):
    import torch
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_text_path_router_joint_mosi_20261019 import run as scalar
    model, loader, trainer = bundle['model'], bundle['test_loader'], bundle['trainer']
    loader.sampler.set_epoch(0)
    pieces, counts, residuals, anchor_norms = {}, [], [], []
    parity = disabled_parity = 0.
    with h.isolated_rng(66 + 99193), torch.no_grad():
        for index, raw in enumerate(loader):
            data = trainer._move_batch(raw, next(model.parameters()).device)
            meta, record = scalar.metadata(data), mask_bank[index]
            for name, value in meta.items():
                np.testing.assert_array_equal(value, record[name])
            availability = record['availability'].to(data[7].device)
            view = trainer._prepare_view_from_primary_masks(data, availability, availability, bundle['dimensions'])
            valid = view['umask'].T.bool()
            if bool(availability[~valid].any()) or bool((availability.sum(-1)[valid] < 1).any()):
                raise ValueError('Saved GCNet mask violates padding/nonempty observations')
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            calls = []
            hook = model.backbone.osram.register_forward_hook(lambda *unused: calls.append(1))
            try:
                result = frozen_readouts(model, view)
            finally:
                hook.remove()
            if len(calls) != 1:
                raise RuntimeError('Primary ON/OFF must use one causal OSRAM forward')
            # Verification only: existing disabled implementation must produce
            # the same OFF valid outputs. It does not supply a second candidate.
            model.enabled = False
            try:
                explicit = model(view)
            finally:
                model.enabled = True
            torch.testing.assert_close(result['OFF'][valid], explicit[valid], rtol=0, atol=0)
            disabled_parity = max(disabled_parity, float((result['OFF'][valid]-explicit[valid]).abs().max()))
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                raise RuntimeError('Frozen replay unexpectedly changed Torch RNG')
            n = int(valid.sum()); counts.append(n)
            residuals.append(float(result['residual_norm'])); anchor_norms.append(float(result['anchor_norm']))
            values = dict(meta, availability=availability.transpose(0, 1)[data[7].bool()].cpu().numpy())
            for name in ('ON', 'OFF'):
                values[name] = result[name].transpose(0, 1)[data[7].bool()].cpu().numpy().reshape(-1)
            for name, value in values.items():
                pieces.setdefault(name, []).append(value)
    if len(counts) != len(mask_bank):
        raise RuntimeError('Incomplete exact-mask coverage')
    arrays = {k: np.concatenate(v) for k, v in pieces.items()}
    assert_aligned(arrays, saved)
    assert_aligned(arrays, flat)
    if len(arrays['labels']) != 686:
        raise RuntimeError('Require all686 valid original MOSI test targets')
    expected = saved['prediction'].reshape(-1)
    np.testing.assert_allclose(arrays['ON'], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(arrays['ON'] > 0, expected > 0)
    parity = float(np.abs(arrays['ON'] - expected).max())
    arrays['Flat'] = flat['prediction'].reshape(-1)
    np.savez_compressed(output, **arrays)
    row = paired_metrics(arrays['labels'], arrays['ON'], arrays['OFF'])
    row.update(Flat=metrics(arrays['labels'], arrays['Flat']),
        on_saved_prediction_max_error=parity, off_explicit_disabled_max_error=disabled_parity,
        exact_mask_sha256=hashlib.sha256(arrays['availability'].tobytes()).hexdigest(),
        primary_memory_scans=len(counts), verification_disabled_memory_scans=len(counts),
        residual_norm=float(np.average(residuals, weights=counts)),
        anchor_norm=float(np.average(anchor_norms, weights=counts)),
        predictions_sha256=h.sha(output))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    from experiments.osram_exemplar_mosi_20261023 import worker
    from experiments.osram_tasr_20261018 import runner_helpers as h
    if args.output.exists():
        raise FileExistsError('Use a fresh independent diagnostic output')
    args.output.mkdir(parents=True)
    tick = time.monotonic()
    summary = dict(status='preflight', label='INTERNAL TEST-ORACLE DIAGNOSTIC', source_commit=args.source_commit,
        training_steps=0, server='biggpu', pid=os.getpid(), variants={}, primary_single_memory_scan=True)
    status = args.output/'SUMMARY.json'
    h.write_json(status, summary)
    try:
        torch.set_num_threads(1)
        prior_flat = json.loads((REFERENCE/'SUMMARY.json').read_text())
        for variant in VARIANTS:
            folder = TRAINING / ('MOSI66_'+variant)
            prior = json.loads((folder/'SUMMARY.json').read_text())
            if (prior['status'] != 'complete' or prior['epoch'] != 150 or prior['seed'] != 66
                    or prior['variant'] != variant or prior['phase'] != 'train'):
                raise ValueError('Completed approved training run required')
            bundle = worker.setup(args.gpu_uuid, args.output, variant)
            worker.validate_reference_config(asdict(bundle['cfg']), prior['config'],
                {f.name:f.default for f in fields(bundle['cfg']) if f.default is not MISSING})
            for key in ('split', 'seed_protocol', 'data_files'):
                if bundle[key] != prior['identity'][key]:
                    raise ValueError('Protected training identity differs: '+key)
            if bundle['bank'].manifest() != prior['identity']['support_bank']:
                raise ValueError('Protected TRAIN bank changed')
            # These are original completed training dependency hashes, not an
            # unrelated old source snapshot used as a model implementation.
            for path, digest in prior['identity']['source_hashes'].items():
                if h.sha(Path(__file__).resolve().parents[2]/path) != digest:
                    raise ValueError('Training implementation source changed: '+path)
            masks_path = folder/'fixed_masks.pt'
            masks = torch.load(masks_path, map_location='cpu', weights_only=False)['test']
            hashes = {r:hashlib.sha256(b''.join(b['availability'].numpy().tobytes() for b in masks[r])).hexdigest() for r in RATES}
            if hashes != prior['identity']['mask_hashes']['test']:
                raise ValueError('Original test mask bank changed')
            model = bundle['model'].eval().requires_grad_(False)
            out = args.output/variant; out.mkdir()
            record = dict(status='running', training_source=prior['source_commit'],
                training_summary_sha256=h.sha(folder/'SUMMARY.json'), config=prior['config'],
                support_bank=prior['identity']['support_bank'], data_manifest=bundle['data_manifest'],
                test_mask_hashes=hashes, fixed_masks_sha256=h.sha(masks_path), environment=bundle['environment'],
                gpu=bundle['gpu'], rates={})
            summary['variants'][variant] = record
            summary['status'] = 'running'
            h.write_json(status, summary)
            torch.cuda.reset_peak_memory_stats()
            for rate in RATES:
                selected, reference = prior['best_by_rate'][rate], prior_flat['best_by_rate'][rate]
                checkpoint = folder/selected['checkpoint']
                for path, digest in [(checkpoint, selected['checkpoint_sha256']),
                                     (folder/selected['predictions'], selected['prediction_sha256']),
                                     (REFERENCE/reference['predictions'], reference['prediction_sha256'])]:
                    if h.sha(path) != digest:
                        raise ValueError('Selected artifact hash mismatch: '+str(path))
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if (state['epoch'] != selected['epoch'] or
                        json.loads(json.dumps(state['identity'])) != prior['identity']):
                    raise ValueError('Wrong checkpoint epoch/trajectory identity')
                model.load_state_dict(state['model'], strict=True)
                del state
                model.eval().requires_grad_(False)
                before = h.backbone_digest(model)
                with np.load(folder/selected['predictions'], allow_pickle=False) as a:
                    saved = {k:a[k].copy() for k in a.files}
                with np.load(REFERENCE/reference['predictions'], allow_pickle=False) as a:
                    flat = {k:a[k].copy() for k in a.files}
                row = evaluate(bundle, masks[rate], saved, flat, out/('rate_'+rate+'.npz'))
                if h.backbone_digest(model) != before or any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Frozen weights/gradients changed')
                row.update(checkpoint=str(checkpoint), checkpoint_sha256=selected['checkpoint_sha256'],
                    selected_epoch=selected['epoch'], weights_unchanged=True, saved_prediction_sha256=selected['prediction_sha256'])
                record['rates'][rate] = row
                h.write_json(status, summary)
                print(json.dumps(dict(variant=variant, rate=rate, on=row['ON'], off=row['OFF'],
                    corrections=row['corrections'], harms=row['harms'])), flush=True)
            record.update(status='complete', peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            del bundle, model
            torch.cuda.empty_cache()
        summary.update(status='complete', seconds=time.monotonic()-tick)
        from .report import publish
        publish(summary, args.output)
    except BaseException as error:
        summary.update(status='failed', error=repr(error), seconds=time.monotonic()-tick)
        raise
    finally:
        h.write_json(status, summary)


if __name__ == '__main__':
    main()
