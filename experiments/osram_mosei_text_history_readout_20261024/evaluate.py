"""Same-checkpoint, single-scan Flat source replay on fixed Text-history cohorts.

INTERNAL DIAGNOSTIC ONLY. No training, new masks, changed queries, or writes.
Numerical model and loaders are imported from a verified immutable source tree.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def cohort_masks(availability, ids, times):
    seen = np.zeros(len(ids), dtype=bool)
    state, last = {}, {}
    for i, (cid, t) in enumerate(zip(ids.tolist(), times.tolist())):
        if t != last.get(cid, -1) + 1:
            raise ValueError('Require contiguous ordered utterances within each conversation')
        seen[i] = state.get(cid, False)
        state[cid] = seen[i] or bool(availability[i, 1])
        last[cid] = t
    missing = availability[:, 1] == 0
    return dict(T_missing_history_seen_T=missing & seen,
                T_missing_history_never_T=missing & ~seen)


def paired_vs(reference, candidate, labels, selected):
    use = selected & (labels != 0)
    old = (reference[use] > 0) == (labels[use] > 0)
    new = (candidate[use] > 0) == (labels[use] > 0)
    return dict(N=int(use.sum()), corrections=int((~old & new).sum()),
                harms=int((old & ~new).sum()), both_correct=int((old & new).sum()),
                both_wrong=int((~old & ~new).sum()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required; preserve previous artifacts')
    args.output.mkdir(parents=True)
    source_summary = json.loads((args.reference / 'SUMMARY.json').read_text())
    if source_summary['status'] != 'complete' or source_summary['method'] != 'none':
        raise ValueError('Expected complete original Flat control')
    source_hashes = source_summary['identity']['source_hashes']
    for relative, digest in source_hashes.items():
        if relative.startswith(('gcnet_missing_m3/', 'gcnet_modality_jepa/')):
            if sha(args.source / relative) != digest:
                raise ValueError('Numerical source mismatch: ' + relative)
    best = source_summary['best_by_rate']['0.7']
    checkpoint = args.reference / best['checkpoint']
    pred_file = args.reference / best['predictions']
    if sha(checkpoint) != best['checkpoint_sha256'] or sha(pred_file) != best['prediction_sha256']:
        raise ValueError('Selected checkpoint/predictions changed')
    sys.path.insert(0, str(args.source))
    import torch
    from experiments.osram_nested_cmerc_tuning_20261023.worker import setup
    from experiments.osram_text_path_router_joint_mosi_20261019.run import metadata
    from experiments.osram_tasr_20261018 import runner_helpers as h
    from experiments.osram_meld_flat_source_combinations_20261013.evaluate import (
        SOURCE_COMBINATIONS, FlatReadoutCapture, replay_flat_hidden)

    torch.set_num_threads(1)
    tick = time.monotonic()
    # This selected run's immutable worker predates the later seed/fold kwargs.
    bundle, model = setup('CMUMOSEI', 'none', args.gpu_uuid, args.output)
    if (bundle['cfg'].seed, bundle['cfg'].fold) != (66, 1):
        raise ValueError('This diagnostic is restricted to MOSEI seed66/fold1')
    if bundle['environment'] != source_summary['environment']:
        raise ValueError('Environment must match the selected cmerc Flat')
    effective = json.loads(json.dumps(asdict(bundle['cfg'])))
    if effective != source_summary['config']:
        raise ValueError('Diagnostic model configuration differs from the selected checkpoint')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['epoch'] != best['epoch']:
        raise ValueError('Wrong selected epoch')
    model.load_state_dict(state['model'], strict=True)
    del state
    model.eval().requires_grad_(False)
    initial_digest = h.backbone_digest(model)
    with np.load(pred_file, allow_pickle=False) as stored:
        saved = {key: stored[key].copy() for key in stored.files}
    bank = bundle['masks']['0.7']
    trainer, loader = bundle['trainer'], bundle['test_loader']
    loader.sampler.set_epoch(0)
    cursor, scans, batches, max_error = 0, [0], 0, 0.
    pieces = {name: [] for name in SOURCE_COMBINATIONS}
    original_scan = model.osram._scan

    def counted_scan(*positional, **keywords):
        scans[0] += 1
        return original_scan(*positional, **keywords)

    model.osram._scan = counted_scan
    torch.cuda.reset_peak_memory_stats()
    try:
        with h.isolated_rng(66 + 99193), torch.no_grad():
            for index, raw in enumerate(loader):
                data = trainer._move_batch(raw, next(model.parameters()).device)
                meta = metadata(data)
                n = len(meta['labels'])
                for key, value in meta.items():
                    np.testing.assert_array_equal(value, saved[key][cursor:cursor+n])
                    np.testing.assert_array_equal(value, bank[index][key])
                availability = bank[index]['availability'].to(data[7].device)
                np.testing.assert_array_equal(availability.transpose(0, 1)[data[7].bool()].cpu().numpy(),
                                              saved['availability'][cursor:cursor+n])
                view = trainer._prepare_view_from_primary_masks(
                    data, availability, availability, bundle['dimensions'])
                before = scans[0]
                with FlatReadoutCapture(model.osram) as capture:
                    full = h.forward(model, view)
                if scans[0] != before + 1 or capture.local_calls != 1 or capture.adapter_calls != 1:
                    raise RuntimeError('Expected exactly one Memory scan and original Flat per batch')
                osram, valid = model.osram, view['umask'].T.bool()
                local_dim, context_dim = osram.latent_dim, osram.context_dim
                torch.testing.assert_close(capture.local, capture.flat_input[..., :local_dim], rtol=0, atol=0)
                contexts = capture.flat_input[..., local_dim:].reshape(*valid.shape, 4, context_dim)
                if contexts[..., context_dim//2:].ne(0).any():
                    raise RuntimeError('Causal backward half must remain zero')
                if contexts[..., 1:, :][availability.bool()].ne(0).any():
                    raise RuntimeError('Observed-target Gap must be inactive')
                original = full.transpose(0, 1)[data[7].bool()].cpu().numpy().reshape(-1)
                expected = saved['prediction'][cursor:cursor+n].reshape(-1)
                np.testing.assert_allclose(original, expected, atol=1e-6, rtol=1e-6)
                np.testing.assert_array_equal(original > 0, expected > 0)
                max_error = max(max_error, float(np.max(np.abs(original - expected))))
                for name in SOURCE_COMBINATIONS:
                    hidden = replay_flat_hidden(local_skip=osram.local_skip,
                        emotion_adapter=osram.emotion_adapter, emotion_norm=osram.emotion_norm,
                        local=capture.local, flat_input=capture.flat_input, umask=view['umask'],
                        local_dim=local_dim, context_dim=context_dim, combination=name)
                    logits = model.smax_fc(hidden)
                    prediction = logits.transpose(0, 1)[data[7].bool()].cpu().numpy().reshape(-1)
                    if name == 'LBG':
                        np.testing.assert_allclose(prediction, original, atol=1e-6, rtol=1e-6)
                        np.testing.assert_array_equal(prediction > 0, original > 0)
                        max_error = max(max_error, float(np.max(np.abs(prediction - original))))
                    if not np.isfinite(prediction).all():
                        raise RuntimeError('Nonfinite replay prediction')
                    pieces[name].append(prediction)
                if scans[0] != before + 1:
                    raise RuntimeError('Readout replay must not run another Memory scan')
                cursor += n
                batches += 1
                print(json.dumps(dict(batch=batches, utterances=cursor, max_full_error=max_error)), flush=True)
    finally:
        del model.osram._scan
    if cursor != len(saved['labels']) or batches != len(bank):
        raise ValueError('Incomplete exact selected cohort')
    if initial_digest != h.backbone_digest(model):
        raise RuntimeError('Frozen parameters changed')
    predictions = {name: np.concatenate(parts) for name, parts in pieces.items()}
    labels = saved['labels'].reshape(-1)
    cohorts = cohort_masks(saved['availability'], saved['conversation_ids'], saved['utterance_indices'])
    cohorts['ALL'] = np.ones(len(labels), dtype=bool)
    reports = {}
    for cohort, selected in cohorts.items():
        rows = {}
        for name, prediction in predictions.items():
            binary = selected & (labels != 0)
            measured = trainer._metrics('CMUMOSEI', labels[binary], prediction[binary], 'regression')
            rows[name] = dict(N_all=int(selected.sum()), N_binary=int(binary.sum()),
                ACC=100*measured['accuracy'], WF1=100*measured['weighted_f1'],
                MSE=float(np.mean((prediction[selected] - labels[selected])**2)),
                paired_vs_Local=paired_vs(predictions['L'], prediction, labels, selected),
                paired_vs_Full=paired_vs(predictions['LBG'], prediction, labels, selected))
        for key in ('ACC', 'WF1', 'MSE'):
            if cohort == 'ALL' and abs(rows['LBG'][key] - best['metrics']['router'][key]) > 1e-5:
                raise RuntimeError('Original aggregate not reproduced')
        reports[cohort] = rows
    result = dict(status='complete', protocol='INTERNAL DIAGNOSTIC ONLY; selected TEST-best checkpoint',
        source=str(args.source), source_git_sha=source_summary['source_commit'],
        reference=str(args.reference), rate='0.7', epoch=best['epoch'],
        checkpoint_sha256=best['checkpoint_sha256'], prediction_sha256=best['prediction_sha256'],
        mask_sha256=best['metrics']['mask_sha256'], environment=bundle['environment'], gpu=bundle['gpu'],
        config=effective, evaluator_sha256=sha(__file__), metrics=reports,
        checks=dict(strict_checkpoint_load=True, model_parameters_unchanged=True,
            exact_metadata_and_availability=True, original_prediction_polarity_exact=True,
            full_prediction_max_abs_error=max_error, memory_scans=scans[0], batches=batches,
            inactive_gap_zero=True, causal_backward_half_zero=True),
        cost=dict(seconds=time.monotonic()-tick, peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20))
    np.savez_compressed(args.output/'predictions.npz', labels=labels,
        availability=saved['availability'], conversation_ids=saved['conversation_ids'],
        utterance_indices=saved['utterance_indices'], **predictions)
    h.write_json(args.output/'RESULT.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    run(parser.parse_args())
